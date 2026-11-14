"""Assemble manuscript/build/full.md (single-file manuscript for review) from the
section files written by build_manuscript.py. Run after build_manuscript.py."""
from pathlib import Path

M = Path(__file__).resolve().parents[1] / "manuscript"
B = M / "build"


def main():
    caps = dict(l.split("\t", 1) for l in (M / "tablecaptions.md").read_text().splitlines() if l.strip())
    parts = [B / "title.md", B / "abstract.md", B / "intro.md", B / "methods.md", B / "results.md",
             B / "discussion.md", B / "conclusions.md"]
    out = [p.read_text().strip() for p in parts]
    tab = ["## Tables and figures (inline in the journal file)"]
    for k in ("1", "2", "3"):
        tab.append(caps[k].strip() + "\n" + (B / f"table{k}.md").read_text().strip())
    tab.append((M / "figcaptions.md").read_text().strip())
    out.append("\n\n".join(tab))
    out.append((B / "declarations.md").read_text().strip())
    out.append((B / "references.md").read_text().strip())
    (B / "full.md").write_text("\n\n\n".join(out) + "\n")


if __name__ == "__main__":
    main()
