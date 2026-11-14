"""Patient-clustered evaluation of hourly risk predictions.

Point estimates are exact. Confidence intervals come from a patient-level
bootstrap that is evaluated on per-patient sufficient statistics (score
histograms and sums), so 1,000 resamples cost seconds rather than hours.
"""
import numpy as np
from scipy import sparse
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-6


def pct_ci(x, level=0.95):
    lo, hi = np.percentile(x, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(lo), float(hi)


def rcs_basis(x, knots):
    """Harrell's restricted cubic spline basis (linear tails)."""
    k = np.asarray(knots, float)
    kk, k1 = k[-1], k[-2]
    cols = [x]
    for j in range(len(k) - 2):
        cols.append(np.maximum(x - k[j], 0) ** 3
                    - np.maximum(x - k1, 0) ** 3 * (kk - k[j]) / (kk - k1)
                    + np.maximum(x - kk, 0) ** 3 * (k1 - k[j]) / (kk - k1))
    return np.column_stack(cols) / (kk - k[0]) ** 2 * np.r_[(kk - k[0]) ** 2, np.ones(len(k) - 2)]


def batched_logistic(Xd, POS, NEG, offset=None, iters=50):
    """Weighted logistic regression for B bootstrap replicates at once.

    Xd: (K, p) design shared by all replicates; POS, NEG: (B, K) event and
    non-event counts per bin. Returns (B, p) coefficients.
    """
    B, p = POS.shape[0], Xd.shape[1]
    n = POS + NEG
    off = 0.0 if offset is None else offset[None, :]
    beta = np.zeros((B, p))
    for _ in range(iters):
        mu = expit(beta @ Xd.T + off)
        g = (POS - n * mu) @ Xd
        w = n * mu * (1 - mu)
        H = (w[:, :, None] * Xd[None]).transpose(0, 2, 1) @ Xd + 1e-8 * np.eye(p)[None]
        step = np.linalg.solve(H, g[..., None])[..., 0]
        beta += step
        if np.abs(step).max() < 1e-9:
            break
    return beta


class SiteEval:
    """Evaluation context for one cohort (rows sorted by patient)."""

    def __init__(self, y, pidx, n_boot=1000, seed=0, patient_mask=None):
        y = np.asarray(y).astype(np.int8)
        pidx = np.asarray(pidx)
        if patient_mask is not None:                # subgroup: re-index patients
            keep_rows = patient_mask[pidx]
            new_id = np.cumsum(patient_mask) - 1
            self.rows = np.flatnonzero(keep_rows)
            y, pidx = y[self.rows], new_id[pidx[self.rows]]
        else:
            self.rows = None
        self.y, self.pidx = y, pidx.astype(np.int64)
        self.n_pat = int(self.pidx.max()) + 1
        rng = np.random.default_rng(seed)
        self.W = rng.multinomial(self.n_pat, np.full(self.n_pat, 1.0 / self.n_pat),
                                 size=n_boot).astype(np.float32)      # (B, n_pat)
        self.n_pp = np.bincount(self.pidx, minlength=self.n_pat).astype(float)
        self.y_pp = np.bincount(self.pidx, weights=self.y, minlength=self.n_pat)

    def sub(self, s):
        return s if self.rows is None else s[self.rows]

    # ---------- sufficient statistics -------------------------------------
    def hist(self, s, K=2000):
        s = self.sub(np.asarray(s, float))
        edges = np.unique(np.quantile(s, np.linspace(0, 1, K + 1)[1:-1]))
        b = np.searchsorted(edges, s, side="right")
        nb = len(edges) + 1
        pos = self.y == 1
        Hpos = sparse.csr_matrix((np.ones(pos.sum()), (self.pidx[pos], b[pos])),
                                 shape=(self.n_pat, nb))
        Hneg = sparse.csr_matrix((np.ones((~pos).sum()), (self.pidx[~pos], b[~pos])),
                                 shape=(self.n_pat, nb))
        tot = np.maximum(np.bincount(b, minlength=nb), 1)
        sc = np.clip(s, EPS, 1 - EPS)
        return dict(
            POS0=np.asarray(Hpos.sum(0)).ravel(), NEG0=np.asarray(Hneg.sum(0)).ravel(),
            POS=np.asarray((Hpos.T @ self.W.T).T), NEG=np.asarray((Hneg.T @ self.W.T).T),
            mean_s=np.bincount(b, weights=s, minlength=nb) / tot,
            mean_lp=np.bincount(b, weights=logit(sc), minlength=nb) / tot,
            s_pp=np.bincount(self.pidx, weights=s, minlength=self.n_pat),
            sq_pp=np.bincount(self.pidx, weights=(s - self.y) ** 2, minlength=self.n_pat),
            s=s)

    # ---------- statistics on (B, K) histograms ----------------------------
    @staticmethod
    def _auc(POS, NEG):
        below = np.cumsum(NEG, axis=-1) - NEG
        return (POS * (below + 0.5 * NEG)).sum(-1) / (POS.sum(-1) * NEG.sum(-1))

    @staticmethod
    def _ap(POS, NEG):
        tp = np.cumsum(POS[..., ::-1], axis=-1)
        fp = np.cumsum(NEG[..., ::-1], axis=-1)
        prec = tp / np.maximum(tp + fp, 1e-12)
        return (prec * POS[..., ::-1]).sum(-1) / POS.sum(-1)

    def _calibration(self, h, POS, NEG, knots):
        """Returns O:E, intercept (slope fixed at 1), slope, ICI for each replicate."""
        lp, ms = h["mean_lp"], h["mean_s"]
        n = POS + NEG
        oe = POS.sum(-1) / (n * ms).sum(-1)
        citl = batched_logistic(np.ones((len(lp), 1)), POS, NEG, offset=lp)[:, 0]
        slope = batched_logistic(np.column_stack([np.ones_like(lp), lp]), POS, NEG)[:, 1]
        Xs = np.column_stack([np.ones_like(lp), rcs_basis(lp, knots)])
        beta = batched_logistic(Xs, POS, NEG)
        fitted = expit(beta @ Xs.T)
        ici = (n * np.abs(fitted - ms)).sum(-1) / n.sum(-1)
        return oe, citl, slope, ici, fitted

    def _knots(self, h):
        w = h["POS0"] + h["NEG0"]
        order = np.argsort(h["mean_lp"])
        cw = np.cumsum(w[order]) / w.sum()
        return np.interp([0.05, 0.275, 0.5, 0.725, 0.95], cw, h["mean_lp"][order])

    # ---------- public API --------------------------------------------------
    def discrimination(self, s, h=None):
        h = h or self.hist(s)
        auc, ap = roc_auc_score(self.y, h["s"]), average_precision_score(self.y, h["s"])
        return dict(auroc=float(auc), auroc_ci=pct_ci(self._auc(h["POS"], h["NEG"])),
                    auprc=float(ap), auprc_ci=pct_ci(self._ap(h["POS"], h["NEG"])),
                    auroc_binned=float(self._auc(h["POS0"], h["NEG0"])),
                    prevalence=float(self.y.mean()))

    def calibration(self, s, h=None):
        h = h or self.hist(s)
        knots = self._knots(h)
        oe0, citl0, slope0, ici0, fit0 = self._calibration(
            h, h["POS0"][None], h["NEG0"][None], knots)
        oe, citl, slope, ici, _ = self._calibration(h, h["POS"], h["NEG"], knots)
        Wn = self.W @ self.n_pp
        brier = float(h["sq_pp"].sum() / self.n_pp.sum())
        prev = self.y.mean()
        return dict(oe=float(oe0[0]), oe_ci=pct_ci(oe), citl=float(citl0[0]), citl_ci=pct_ci(citl),
                    slope=float(slope0[0]), slope_ci=pct_ci(slope),
                    ici=float(ici0[0]), ici_ci=pct_ci(ici),
                    brier=brier, brier_ci=pct_ci((self.W @ h["sq_pp"]) / Wn),
                    brier_scaled=float(1 - brier / (prev * (1 - prev))),
                    curve=dict(pred=h["mean_s"].tolist(), obs=fit0[0].tolist(),
                               n=(h["POS0"] + h["NEG0"]).tolist()))

    def delta_auc(self, s1, s2, h1=None, h2=None):
        """Paired bootstrap for AUROC(s1) - AUROC(s2) on the same patients."""
        h1, h2 = h1 or self.hist(s1), h2 or self.hist(s2)
        d = self._auc(h1["POS"], h1["NEG"]) - self._auc(h2["POS"], h2["NEG"])
        point = roc_auc_score(self.y, h1["s"]) - roc_auc_score(self.y, h2["s"])
        dap = self._ap(h1["POS"], h1["NEG"]) - self._ap(h2["POS"], h2["NEG"])
        return dict(delta=float(point), ci=pct_ci(d), boot=d,
                    p_two_sided=float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))),
                    delta_auprc=float(average_precision_score(self.y, h1["s"])
                                      - average_precision_score(self.y, h2["s"])),
                    delta_auprc_ci=pct_ci(dap))

    def net_benefit(self, s, thresholds):
        s = self.sub(np.asarray(s, float))
        out = []
        n = self.n_pp.sum()
        Wn = self.W @ self.n_pp
        prev = self.y.mean()
        for pt in thresholds:
            hit = s >= pt
            tp = np.bincount(self.pidx, weights=hit & (self.y == 1), minlength=self.n_pat)
            fp = np.bincount(self.pidx, weights=hit & (self.y == 0), minlength=self.n_pat)
            w = pt / (1 - pt)
            nb = (tp.sum() - w * fp.sum()) / n
            nb_b = (self.W @ tp - w * (self.W @ fp)) / Wn
            all_b = (self.W @ self.y_pp - w * (Wn - self.W @ self.y_pp)) / Wn
            out.append(dict(pt=pt, nb=float(nb), nb_ci=pct_ci(nb_b),
                            nb_all=float(prev - w * (1 - prev)),
                            nb_minus_all_ci=pct_ci(nb_b - all_b),
                            alert_rate=float(hit.mean())))
        return out


# ---------------- Challenge utility score -------------------------------------
def utility_weights(y, pidx, t, dt_early=-12, dt_optimal=-6, dt_late=3.0,
                    max_u_tp=1.0, min_u_fn=-2.0, u_fp=-0.05, u_tn=0.0):
    """Per-row utility of predicting 1 (u1) or 0 (u0), and of the best policy.

    Vectorised re-implementation of the official evaluate_sepsis_score.py.
    """
    n_pat = int(pidx.max()) + 1
    first = np.full(n_pat, np.iinfo(np.int64).max, np.int64)
    pos = np.flatnonzero(y == 1)
    np.minimum.at(first, pidx[pos], t[pos])
    septic_p = first < np.iinfo(np.int64).max
    septic = septic_p[pidx]
    t_sepsis = np.where(septic, first[pidx] - dt_optimal, 0).astype(float)
    d = t - t_sepsis
    m1 = max_u_tp / (dt_optimal - dt_early); b1 = -m1 * dt_early
    m2 = -max_u_tp / (dt_late - dt_optimal); b2 = -m2 * dt_late
    m3 = min_u_fn / (dt_late - dt_optimal); b3 = -m3 * dt_optimal
    u1 = np.full(len(y), u_fp, float)
    u0 = np.full(len(y), u_tn, float)
    early, mid = septic & (d <= dt_optimal), septic & (d > dt_optimal) & (d <= dt_late)
    late = septic & (d > dt_late)
    u1[early] = np.maximum(m1 * d[early] + b1, u_fp)
    u1[mid] = m2 * d[mid] + b2
    u0[septic] = 0.0
    u0[mid] = m3 * d[mid] + b3
    u1[late] = 0.0
    best = septic & (d >= dt_early) & (d <= dt_late)
    return u1, u0, best


def normalised_utility(pred, u1, u0, best):
    obs = np.where(pred, u1, u0).sum()
    inaction = u0.sum()
    return float((obs - inaction) / (np.where(best, u1, u0).sum() - inaction))


def best_utility_threshold(s, u1, u0, best, grid=None):
    grid = np.unique(np.quantile(s, np.linspace(0.5, 0.999, 250))) if grid is None else grid
    order = np.argsort(-s)
    gain = np.cumsum((u1 - u0)[order])             # utility gained by alerting the top-k rows
    ss = s[order]
    inaction = u0.sum()
    denom = np.where(best, u1, u0).sum() - inaction
    vals = []
    for th in grid:
        k = np.searchsorted(-ss, -th, side="right")   # rows with s >= th
        vals.append((gain[k - 1] if k > 0 else 0.0) / denom)
    i = int(np.argmax(vals))
    return float(grid[i]), float(vals[i])


def utility_ci(s, th, u1, u0, best, ev):
    """Bootstrap interval for the normalised utility at a fixed threshold."""
    s = ev.sub(s)
    pred = s >= th
    obs = np.bincount(ev.pidx, weights=np.where(pred, u1, u0), minlength=ev.n_pat)
    ina = np.bincount(ev.pidx, weights=u0, minlength=ev.n_pat)
    bst = np.bincount(ev.pidx, weights=np.where(best, u1, u0), minlength=ev.n_pat)
    W = ev.W
    b = (W @ obs - W @ ina) / (W @ bst - W @ ina)
    return float((obs.sum() - ina.sum()) / (bst.sum() - ina.sum())), pct_ci(b)


# ---------------- patient-level alerting ---------------------------------------
class AlertEval:
    """First-alert analysis. Septic patients can only be alerted at or before
    onset (first positive hour + 6 h); non-septic patients at any hour."""

    def __init__(self, y, pidx, t, W=None):
        self.pidx, self.t = np.asarray(pidx, np.int64), np.asarray(t, np.int64)
        self.n_pat = int(self.pidx.max()) + 1
        big = np.iinfo(np.int64).max
        first = np.full(self.n_pat, big, np.int64)
        pos = np.flatnonzero(np.asarray(y) == 1)
        np.minimum.at(first, self.pidx[pos], self.t[pos])
        self.septic = first < big
        self.first_pos = np.where(self.septic, first, -1)
        self.onset = np.where(self.septic, first + 6, big)
        self.eligible = self.t <= self.onset[self.pidx]
        self.W = W

    def max_eligible(self, s):
        m = np.full(self.n_pat, -np.inf)
        e = self.eligible
        np.maximum.at(m, self.pidx[e], np.asarray(s, float)[e])
        return m

    def threshold_for_sensitivity(self, s, target):
        m = np.sort(self.max_eligible(s)[self.septic])
        k = int(np.floor((1 - target) * len(m)))      # at most k septic patients missed
        return float(m[k])

    def evaluate(self, s, tau):
        s = np.asarray(s, float)
        m = self.max_eligible(s)
        alerted = m >= tau
        big = np.iinfo(np.int64).max
        first_alert = np.full(self.n_pat, big, np.int64)
        hit = np.flatnonzero((s >= tau) & self.eligible)
        np.minimum.at(first_alert, self.pidx[hit], self.t[hit])
        lead_ok = self.septic & alerted & (self.first_pos > 0)
        lead = (self.onset[lead_ok] - first_alert[lead_ok]).astype(float)
        sp, ns = self.septic, ~self.septic
        a_s, a_n = (alerted & sp).astype(float), (alerted & ns).astype(float)
        res = dict(tau=float(tau), sensitivity=float(a_s.sum() / sp.sum()),
                   false_alert_prop=float(a_n.sum() / ns.sum()),
                   ppv=float(a_s.sum() / max(alerted.sum(), 1)),
                   n_alerted=int(alerted.sum()), n_septic=int(sp.sum()),
                   lead_median=float(np.median(lead)) if len(lead) else float("nan"),
                   lead_iqr=[float(x) for x in np.percentile(lead, [25, 75])] if len(lead) else None,
                   lead_ge6_prop=float((lead >= 6).mean()) if len(lead) else float("nan"))
        if self.W is not None:
            W = self.W
            ws, wn = W @ sp.astype(np.float32), W @ ns.astype(np.float32)
            was, wan = W @ a_s.astype(np.float32), W @ a_n.astype(np.float32)
            res["sensitivity_ci"] = pct_ci(was / ws)
            res["false_alert_prop_ci"] = pct_ci(wan / wn)
            res["ppv_ci"] = pct_ci(was / np.maximum(was + wan, 1e-9))
        return res
