"""A6 (docs/FINAL_ANALYSES.md): one run's score against another run's score, paired per patient.

scripts/test_stats.py compares the same score across runs (DA vs DA, or HA vs HA). This compares any
two (run, score) pairs, for example H3a's HA final hypothesis ("H3a + defer") against the baseline's
DA diagnosis, on the same patients: accuracy of each, paired difference with a bootstrap 95% CI,
exact McNemar, and the mean test cost of each run. Standard library only.

    python scripts/defer_vs_baseline.py --out-dir outputs/final_analyses
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.episode_log import read_rows  # noqa: E402

R = "outputs/cost_sweep/cost_1_attempt2"
E = "outputs/ext_v2_trained"
# (name, (run folder, score) of the candidate, (run folder, score) of the reference, role)
COMPARISONS = [
    ("MIMIC: H3a + defer vs baseline", ("outputs/h3/h3a_ha0/frozen_ha_seed0_step_0700", "final_hypothesis"),
     (f"{R}/test_step_0600", "diagnosis"), "primary"),
    ("External: H3a + defer vs baseline", (f"{E}/h3a", "final_hypothesis"), (f"{E}/baseline", "diagnosis"), "secondary"),
    ("MIMIC: step2 + defer vs baseline", (f"{R}/frozen_ha_seed0_step_0600", "final_hypothesis"),
     (f"{R}/test_step_0600", "diagnosis"), "secondary (chosen after seeing A1)"),
]


def first_episodes(path: Path) -> dict[str, dict]:
    rows = {}
    for row in read_rows(path):
        rows.setdefault(str(row["patient_id"]), row)
    return rows


def correct(row: dict, score: str) -> bool:
    answer = (row.get("final_hypothesis") if score == "final_hypothesis" else row.get("diagnosis")) or ""
    return answer.strip().lower() == str(row["label"]).strip().lower()


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def compare(candidate: tuple[str, str], reference: tuple[str, str], resamples: int, seed: int) -> dict:
    a_rows, b_rows = (first_episodes(Path(folder) / "test_episodes.csv") for folder, _ in (candidate, reference))
    patients = sorted(set(a_rows) & set(b_rows))
    a = [correct(a_rows[p], candidate[1]) for p in patients]
    b = [correct(b_rows[p], reference[1]) for p in patients]
    n = len(patients)
    wins, losses = sum(x and not y for x, y in zip(a, b)), sum(y and not x for x, y in zip(a, b))
    rng = random.Random(seed)
    diffs = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(100 * (sum(a[i] for i in idx) - sum(b[i] for i in idx)) / n)
    diffs.sort()
    cost = lambda rows: sum(float(rows[p]["cost_usd"] or 0) for p in patients) / n
    return {"n": n, "candidate_accuracy": round(100 * sum(a) / n, 1), "reference_accuracy": round(100 * sum(b) / n, 1),
            "difference": round(100 * (sum(a) - sum(b)) / n, 1),
            "ci95": [round(diffs[int(0.025 * resamples)], 1), round(diffs[int(0.975 * resamples) - 1], 1)],
            "candidate_right_reference_wrong": wins, "reference_right_candidate_wrong": losses,
            "mcnemar_p": round(mcnemar_exact(wins, losses), 4),
            "candidate_cost": round(cost(a_rows), 0), "reference_cost": round(cost(b_rows), 0)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/final_analyses"))
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--h3c-dir", default=None,
                        help="H3c test folder (docs/HYPOTHESIS_H3C.md); adds its pre-registered comparisons")
    parser.add_argument("--h3d-dir", default=None,
                        help="H3d test folder (docs/HYPOTHESIS_H3D.md); adds its comparisons with H3a + defer and the baseline")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.h3d_dir:  # fixed in docs/HYPOTHESIS_H3D.md before training
        COMPARISONS.extend([
            ("MIMIC: H3d vs H3a + defer", (args.h3d_dir, "diagnosis"),
             ("outputs/h3/h3a_ha0/frozen_ha_seed0_step_0700", "final_hypothesis"), "H3d secondary"),
            ("MIMIC: H3d vs baseline", (args.h3d_dir, "diagnosis"), (f"{R}/test_step_0600", "diagnosis"), "H3d secondary"),
        ])
    if args.h3c_dir:  # fixed in docs/HYPOTHESIS_H3C.md before training
        COMPARISONS.extend([
            ("MIMIC: H3c vs H3a + defer", (args.h3c_dir, "diagnosis"),
             ("outputs/h3/h3a_ha0/frozen_ha_seed0_step_0700", "final_hypothesis"), "H3c primary"),
            ("MIMIC: H3c vs baseline", (args.h3c_dir, "diagnosis"), (f"{R}/test_step_0600", "diagnosis"), "H3c secondary"),
            ("MIMIC: H3c vs H3a", (args.h3c_dir, "diagnosis"),
             ("outputs/h3/h3a_ha0/frozen_ha_seed0_step_0700", "diagnosis"), "H3c secondary"),
        ])
    results, lines = {}, ["# A6: H3a + defer (and step2 + defer) vs the baseline", "",
                          "Accuracy in %; difference = candidate minus reference, paired bootstrap 95% CI; "
                          "wins/losses = patients only the candidate / only the reference got right; exact McNemar.", "",
                          "| Comparison | role | n | candidate | baseline | difference [95% CI] | wins/losses | p | cost candidate / baseline |",
                          "|---|---|---|---|---|---|---|---|---|"]
    for name, candidate, reference, role in COMPARISONS:
        if not all((Path(folder) / "test_episodes.csv").is_file() for folder, _ in (candidate, reference)):
            lines.append(f"| {name} | {role} | missing files | | | | | | |")
            continue
        r = compare(candidate, reference, args.resamples, seed=0) | {"role": role, "candidate": candidate, "reference": reference}
        results[name] = r
        lines.append(f"| {name} | {role} | {r['n']} | {r['candidate_accuracy']} | {r['reference_accuracy']} | "
                     f"{r['difference']:+.1f} [{r['ci95'][0]:+.1f}, {r['ci95'][1]:+.1f}] | "
                     f"+{r['candidate_right_reference_wrong']}/-{r['reference_right_candidate_wrong']} | {r['mcnemar_p']} | "
                     f"${r['candidate_cost']:,.0f} / ${r['reference_cost']:,.0f} |")
    (args.out_dir / "a6_defer_vs_baseline.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (args.out_dir / "a6_defer_vs_baseline.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
