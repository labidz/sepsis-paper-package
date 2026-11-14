"""Assemble the manuscript: number citations in order of first appearance,
build the reference list in the journal's numbered style, count words and
check journal limits. Writes manuscript/build/<section>.md and a report."""
import json
import re
import sys
from pathlib import Path

M = Path(__file__).resolve().parents[1] / "manuscript"
B = M / "build"
B.mkdir(exist_ok=True)
refs = json.load(open(M / "refs.json"))
CITE = re.compile(r"\[((?:@[\w]+)(?:;\s*@[\w]+)*)\]")


def sections(path):
    out, name = {}, None
    for line in path.read_text().splitlines():
        m = re.match(r"<!--SECTION:(\w+)-->", line)
        if m:
            name = m.group(1); out[name] = []
        elif name:
            out[name].append(line)
    return {k: "\n".join(v).strip() for k, v in out.items()}


def compress(nums):
    nums = sorted(set(nums)); out = []; i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(str(nums[i]) if j == i else (f"{nums[i]},{nums[j]}" if j == i + 1 else f"{nums[i]}-{nums[j]}"))
        i = j + 1
    return ",".join(out)


def fmt_ref(n, r):
    if r.get("dataset"):
        return f"[{n}] [dataset] {r['a']} {r['t']}. {r['j']}; {r['y']}. https://doi.org/{r['doi']}"
    t = r['t'] if r['t'].endswith(("?", "!")) else r['t'] + "."
    s = f"[{n}] {r['a']} {t} {r['j']} {r['y']};{r['v']}:{r['p']}."
    return s + (f" https://doi.org/{r['doi']}" if r.get("doi") else "")


def words(text):
    text = re.sub(r"\[[\d,\-]+\]", " ", text)                 # numeric citations
    text = re.sub(r"^#+ .*$", " ", text, flags=re.M)         # headings
    text = re.sub(r"[*_|]", " ", text)
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9\-'.,%:×/]*", text))


def main():
    front, main_, back = sections(M / "front.md"), sections(M / "main.md"), sections(M / "back.md")
    extra = {}
    for f in sorted((M / "parts").glob("*.md")) if (M / "parts").exists() else []:
        extra[f.stem] = f.read_text()
    order = {}

    def repl(m):
        keys = [k.strip()[1:] for k in m.group(1).split(";")]
        for k in keys:
            if k not in refs:
                raise KeyError(f"unknown citation {k}")
            order.setdefault(k, len(order) + 1)
        return "[" + compress([order[k] for k in keys]) + "]"

    built = {}
    for name in ("intro", "methods", "results", "discussion", "conclusions"):
        text = main_[name]
        for k, v in extra.items():
            text = text.replace(k.upper() + "_PLACEHOLDER", v.strip())
        built[name] = CITE.sub(repl, text)
    for name, text in {**front, **back}.items():
        built[name] = CITE.sub(repl, text)
    for f in sorted((M / "tables").glob("*.md")):
        built[f.stem] = CITE.sub(repl, f.read_text())
    reflist = "\n\n".join(fmt_ref(n, refs[k]) for k, n in sorted(order.items(), key=lambda kv: kv[1]))
    built["references"] = "## References\n\n" + reflist
    for k, v in built.items():
        (B / f"{k}.md").write_text(v + "\n")
    wc = {k: words(built[k]) for k in ("intro", "methods", "results", "discussion", "conclusions")}
    total = sum(wc.values())
    abstract_words = words(re.sub(r"\*\*Keywords:.*", "", built["abstract"]))
    hl = [ln[2:] for ln in built["title"].splitlines() if ln.startswith("- ")]
    unused = [k for k in refs if k not in order]
    report = dict(main_text_words=total, by_section=wc, abstract_words=abstract_words,
                  highlights=[(len(h), h) for h in hl], n_references=len(order), unused_refs=unused)
    print(json.dumps(report, indent=1))
    ok = 2500 <= total <= 3500 and abstract_words <= 250 and all(len(h) <= 85 for h in hl) and 3 <= len(hl) <= 5
    print("JOURNAL LIMITS:", "OK" if ok else "VIOLATED")
    (B / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
