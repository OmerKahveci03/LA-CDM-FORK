"""A3 (docs/FINAL_ANALYSES.md): combined HAs, the equal-weight mean of HAs' probability vectors.

Reads the existing ``ha_probs.csv`` of each HA on the four H4a record sets (the same patients and
records for every HA), averages the probability vectors (weights fixed in advance, not tuned), and
compares each combination with the separately fine-tuned HA: accuracy (exact McNemar, Holm over the
combinations, primary record set A_da2), ECE, Brier and AUROC (confidence vs correct).
Standard library only.

    python scripts/combined_ha.py --out-dir outputs/final_analyses
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ha.prompts import DISEASES  # noqa: E402
from src.ha.records import probability_summary  # noqa: E402

RECORD_SETS = ["A_da2", "B_none", "C_random2", "D_all"]
HAS = {
    "separate": "outputs/ha_sft/scores/new0",          # separately fine-tuned HA (H2, seed 0)
    "clm_finetuned": "outputs/contrastive/scores/h2",  # CLM HA, fine-tuned backbone (chosen)
    "clm_plain": "outputs/contrastive/scores/base",    # CLM HA, plain Llama
    "baseline": "outputs/ha_sft/scores/old2",          # baseline (shared-adapter) HA
}
COMBOS = {  # fixed in docs/FINAL_ANALYSES.md
    "a_separate+clm_finetuned": ["separate", "clm_finetuned"],
    "b_separate+clm_plain": ["separate", "clm_plain"],
    "c_separate+baseline": ["separate", "baseline"],
    "d_all_four": ["separate", "clm_finetuned", "clm_plain", "baseline"],
}
REFERENCE = "separate"


def read_probs(path: Path) -> dict[str, tuple[str, list[float]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["patient_id"]: (row["label"].lower(), [float(row[f"p_{d}"]) for d in DISEASES])
                for row in csv.DictReader(handle)}


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    order = sorted(pvalues, key=pvalues.get)
    running, out = 0.0, {}
    for i, name in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * pvalues[name]))
        out[name] = running
    return out


def top(p: list[float]) -> int:
    return max(range(len(p)), key=p.__getitem__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/final_analyses"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for rs in RECORD_SETS:
        probs = {ha: read_probs(Path(folder) / rs / "ha_probs.csv") for ha, folder in HAS.items()
                 if (Path(folder) / rs / "ha_probs.csv").is_file()}
        if REFERENCE not in probs:
            print(f"[combined] {rs}: no {REFERENCE} probabilities, skipped")
            continue
        patients = sorted(set.intersection(*(set(p) for p in probs.values())))
        labels = [probs[REFERENCE][pid][0] for pid in patients]
        truth = [DISEASES.index(label) for label in labels]
        vectors = {ha: [probs[ha][pid][1] for pid in patients] for ha in probs}
        for name, members in COMBOS.items():
            if all(m in vectors for m in members):
                vectors[name] = [[sum(vectors[m][i][d] for m in members) / len(members) for d in range(len(DISEASES))]
                                 for i in range(len(patients))]
        ref_correct = [top(p) == t for p, t in zip(vectors[REFERENCE], truth)]
        out = {}
        for name, vecs in vectors.items():
            summary = probability_summary(labels, vecs)
            correct = [top(p) == t for p, t in zip(vecs, truth)]
            wins = sum(c and not r for c, r in zip(correct, ref_correct))
            losses = sum(r and not c for c, r in zip(correct, ref_correct))
            out[name] = {"n": len(patients), "accuracy": round(100 * summary["accuracy"], 1),
                         "ece": round(100 * summary["ece_10_bins"], 1), "brier": round(summary["brier"], 3),
                         "auroc_correct": round(summary["auroc_correct_exact"] or float("nan"), 3),
                         "vs_separate": {"wins": wins, "losses": losses, "mcnemar_p": round(mcnemar_exact(wins, losses), 4)}}
        if rs == "A_da2":
            for name, p in holm({c: out[c]["vs_separate"]["mcnemar_p"] for c in COMBOS if c in out}).items():
                out[name]["vs_separate"]["p_holm"] = round(p, 4)
        results[rs] = out

    (args.out_dir / "a3_combined_ha.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = ["# A3: combined HAs (equal-weight mean of probabilities)", "",
             "Accuracy and ECE in %. vs separate: records the HA gets right and the separately fine-tuned HA "
             "wrong / the reverse, exact McNemar (Holm over the 4 combinations on A_da2).", ""]
    for rs, out in results.items():
        lines += [f"## {rs}", "", "| HA | n | accuracy | ECE | Brier | AUROC (correct) | vs separate |",
                  "|---|---|---|---|---|---|---|"]
        for name, r in out.items():
            v = r["vs_separate"]
            vs = "-" if name == REFERENCE else (f"+{v['wins']}/-{v['losses']}, p={v['mcnemar_p']}"
                                               + (f", Holm {v['p_holm']}" if "p_holm" in v else ""))
            lines.append(f"| {name} | {r['n']} | {r['accuracy']} | {r['ece']} | {r['brier']} | {r['auroc_correct']} | {vs} |")
        lines.append("")
    (args.out_dir / "a3_combined_ha.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
