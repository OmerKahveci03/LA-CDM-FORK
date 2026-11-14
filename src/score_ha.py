"""Score a Hypothesis Agent on fixed test sets (hypothesis H2, docs/HYPOTHESIS_H2.md).

For every arm, each patient's tests are copied from an existing ``test_episodes.csv`` (the
trained DA's run or an H1 scripted arm), so every HA sees exactly the same records. The HA
does not generate: it scores the four diseases after ``Hypothesis:`` and answers with the
most probable one (see src/ha/scoring.py). The same procedure is used for the old adapters
and the new standalone HA.

    python -m src.score_ha --adapter outputs/ha_sft/seed_0/adapter --out-dir outputs/ha_sft/seed_0/scores \
        --arm D_all=outputs/cost_sweep/cost_1_attempt2/scripted_all_step_0600/test_episodes.csv

``--adapter none`` scores the base model. Output per arm: ``<out-dir>/<arm>/test_episodes.csv``
(readable by scripts/test_stats.py), ``ha_probs.csv`` and ``summary.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.ha.prompts import ANSWER_PREFIX, DISEASES, candidate_ids, chat_prompt, encode, ha_user_message
from src.ha.records import probability_summary, scored_row
from src.ha.scoring import candidate_logprobs, softmax
from src.train_ha_sft import load_split
from src.utils.episode_log import read_rows, write_rows

REPO = Path(__file__).resolve().parents[1]


def first_episodes(path: Path) -> list[dict]:
    seen, rows = set(), []
    for row in read_rows(path):
        if row["patient_id"] not in seen:
            seen.add(row["patient_id"])
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", required=True, help="LoRA adapter folder, or 'none' for the base model")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=EPISODES_CSV")
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N patients per arm (smoke test)")
    args = parser.parse_args()

    dataset_cfg = OmegaConf.load(REPO / "configs/dataset/defaults.yaml")
    model_cfg = OmegaConf.load(REPO / "configs/model/defaults.yaml")
    env_cfg = OmegaConf.load(REPO / "configs/environment/la_cdm.yaml")
    template = (REPO / env_cfg.hypothesis_prompt_template_file).read_text()
    patients = {str(p["patient_id"]): p for p in load_split(dataset_cfg, args.split, None)}

    arms = []
    for item in args.arm:
        name, sep, path = item.partition("=")
        if not sep or not Path(path).is_file():
            raise SystemExit(f"--arm needs NAME=EPISODES_CSV with an existing file: {item}")
        arms.append((name, Path(path)))

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa"
    )
    if args.adapter != "none":
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    candidates = candidate_ids(tokenizer)
    answer_prefix = encode(tokenizer, ANSWER_PREFIX)

    cache: dict[tuple[int, ...], list[float]] = {}  # identical prompts across arms are scored once
    overview = {}
    for name, path in arms:
        sources = first_episodes(path)[: args.limit]
        rows, prob_rows, started = [], [], time.time()
        for source in sources:
            patient = patients.get(str(source["patient_id"]))
            if patient is None:
                raise SystemExit(f"{path}: patient {source['patient_id']} is not in the {args.split} split")
            message = ha_user_message(template, patient["patient_history"], patient["test_results"], source["tests"])
            prompt = encode(tokenizer, chat_prompt(tokenizer, message)) + answer_prefix
            key = tuple(prompt)
            if key not in cache:
                cache[key] = softmax(candidate_logprobs(model, prompt, candidates))
            probs = cache[key]
            rows.append(scored_row(source, probs))
            prob_rows.append([source["patient_id"], rows[-1]["label"], len(source["tests"]), len(prompt), *probs])
        arm_dir = args.out_dir / name
        arm_dir.mkdir(parents=True, exist_ok=True)
        write_rows(arm_dir / "test_episodes.csv", rows)
        with (arm_dir / "ha_probs.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["patient_id", "label", "num_tests", "prompt_tokens", *(f"p_{d}" for d in DISEASES)])
            writer.writerows(prob_rows)
        summary = probability_summary([r["label"] for r in rows], [p[4:] for p in prob_rows])
        summary |= {"arm": name, "source": str(path), "adapter": args.adapter, "split": args.split,
                    "mean_tests": sum(r["num_tests"] for r in rows) / len(rows),
                    "mean_cost_usd": sum(r["cost_usd"] or 0 for r in rows) / len(rows),
                    "seconds": round(time.time() - started, 1)}
        (arm_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        overview[name] = summary
        print(f"[score_ha] {name}: accuracy {summary['accuracy']:.3f}, exact AUROC {summary['auroc_ovr_exact']:.3f}, "
              f"ECE {summary['ece_10_bins']:.3f}, n={summary['n']} ({summary['seconds']}s)")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "overview.json").write_text(json.dumps(overview, indent=2))


if __name__ == "__main__":
    main()
