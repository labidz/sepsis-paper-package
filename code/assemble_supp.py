"""Assemble manuscript/build/supplement_full.md: the supplement text with the
generated Tables C.1 to C.9 inserted at their placeholders."""
import re
from pathlib import Path

M = Path(__file__).resolve().parents[1] / "manuscript"


def main():
    s = (M / "supplement.md").read_text()
    s = re.sub(r"^TABLE_C(\d+)$", lambda m: (M / "tables" / f"tableC{m.group(1)}.md").read_text().strip(), s, flags=re.M)
    assert "TABLE_C" not in s
    (M / "build" / "supplement_full.md").write_text(s)


if __name__ == "__main__":
    main()
