"""One table of every scored Hypothesis Agent on every arm (hypothesis H2).

Reads ``<scores>/<HA>/overview.json`` written by ``python -m src.score_ha`` and prints
accuracy, exact one-vs-rest AUROC, ECE and pancreatitis recall per arm and HA. Paired
tests and intervals come from scripts/test_stats.py (see docs/HYPOTHESIS_H2.md).

    python scripts/h2_table.py                       # outputs/ha_sft/scores
    python scripts/h2_table.py --scores outputs/ha_sft/scores_smoke

Standard library only. Writes ``<scores>/h2_table.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ORDER = ("base", "old2", "old3")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, default=Path("outputs/ha_sft/scores"))
    args = parser.parse_args()

    found = {p.parent.name: json.loads(p.read_text()) for p in sorted(args.scores.glob("*/overview.json"))}
    if not found:
        raise SystemExit(f"No {args.scores}/*/overview.json yet")
    has = sorted(found, key=lambda h: (ORDER.index(h) if h in ORDER else len(ORDER), h))
    arms = list(dict.fromkeys(arm for h in has for arm in found[h]))

    lines = [f"# H2: Hypothesis Agents on fixed test sets ({args.scores})", ""]
    for title, cell in (
        ("Accuracy (exact one-vs-rest AUROC)",
         lambda s: f"{100 * s['accuracy']:.1f}% ({s['auroc_ovr_exact']:.3f})"),
        ("ECE (10 bins) / Brier", lambda s: f"{100 * s['ece_10_bins']:.1f}% / {s['brier']:.3f}"),
        ("Pancreatitis recall", lambda s: f"{100 * s['recall']['pancreatitis']:.1f}%"),
    ):
        lines += [f"## {title}", "", "| Arm | " + " | ".join(has) + " |", "|---|" + "---|" * len(has)]
        for arm in arms:
            cells = [cell(found[h][arm]) if arm in found[h] else "-" for h in has]
            lines.append(f"| {arm} | " + " | ".join(cells) + " |")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    (args.scores / "h2_table.md").write_text(text + "\n", encoding="utf-8")
    print(f"Wrote {args.scores / 'h2_table.md'}")


if __name__ == "__main__":
    main()
