"""Run the CLM HA over fixed test sequences: stage 2 (H4a) and test scoring of stage 3 (H4b).

Each patient's tests come from an existing ``test_episodes.csv`` (the baseline DA's run or an
H1 scripted arm), as in ``src/score_ha.py``. The HA gets them one at a time, in that order:
turn 0 is the history alone, then one test per turn. In ``edit`` mode it may rewrite its
context at every turn; in ``append`` mode it never does, which reproduces the H2 scorer.

    python -m src.clm.replay --model none --mode edit --out-dir outputs/clm/replay/base_edit \\
        --arm D_all=outputs/cost_sweep/cost_1_attempt2/scripted_all_step_0600/test_episodes.csv

Per arm, in ``<out-dir>/<arm>/``:

- ``test_episodes.csv``: one row per patient, the final hypothesis as the answer (readable by
  scripts/test_stats.py; per-turn hypotheses and confidences are kept);
- ``ha_probs.csv``: final probabilities, context tokens and rewrite counts per patient;
- ``notes.jsonl``: every turn's action, context (after any rewrite) and reply;
- ``summary.json``: accuracy, AUROC, ECE etc. (src/ha/records.py) plus context and edit statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.clm.agent import CLMHA
from src.clm.context import CLMState, blocks_for
from src.ha.prompts import DISEASES
from src.ha.records import probability_summary, scored_row

REPO = Path(__file__).resolve().parents[2]
EDIT_TEMPLATE = REPO / "prompts/clm_hypothesis_agent.txt"


def run_sequences(agent: CLMHA, items: list[dict], sample: bool = False, batch_size: int = 8,
                  progress: str | None = None) -> list[dict]:
    """Run the HA over every item (``{"history", "blocks", ...}``) in lock-step batches.

    Returns ``{"item", "turns": [TurnRecord], "context"}`` per item, in order.
    """
    results = []
    started = time.time()
    for start in range(0, len(items), batch_size):
        group = items[start: start + batch_size]
        states = [CLMState.start(item["history"]) for item in group]
        turns: list[list] = [[] for _ in group]
        for t in range(1 + max(len(item["blocks"]) for item in group)):
            active = [k for k, item in enumerate(group) if t <= len(item["blocks"])]
            for k in active:
                if t > 0:
                    states[k].observe(group[k]["blocks"][t - 1])
            for k, record in zip(active, agent.turn([states[k] for k in active], sample, batch_size)):
                turns[k].append(record)
        results += [{"item": item, "turns": tr, "context": s.context} for item, tr, s in zip(group, turns, states)]
        if progress:
            print(f"[{progress}] {len(results)}/{len(items)} patients ({time.time() - started:.0f}s)", flush=True)
    return results


def turn_statistics(results: list[dict]) -> dict:
    """Edit and context-length statistics over all turns."""
    actions = Counter(t.action for r in results for t in r["turns"])
    replies = sum(actions[a] for a in ("keep", "rewrite", "malformed"))
    n = max(1, len(results))
    return {
        "turns": sum(actions.values()),
        "actions": dict(actions),
        "rewrite_rate": actions["rewrite"] / max(1, replies),
        "malformed_rate": actions["malformed"] / max(1, replies),
        "patients_with_rewrite": sum(any(t.action == "rewrite" for t in r["turns"]) for r in results) / n,
        "mean_final_context_tokens": sum(r["turns"][-1].context_tokens for r in results) / n,
        "max_final_context_tokens": max((r["turns"][-1].context_tokens for r in results), default=0),
        "mean_total_context_tokens": sum(sum(t.context_tokens for t in r["turns"]) for r in results) / n,
        "mean_turns": sum(len(r["turns"]) for r in results) / n,
    }


def episode_row(source: dict, turns: list) -> dict:
    row = scored_row(source, turns[-1].probs)
    hypotheses = [DISEASES[max(range(len(DISEASES)), key=t.probs.__getitem__)] for t in turns]
    row.update(num_turns=len(turns), hypotheses=hypotheses,
               confidences=[round(10 * max(t.probs)) for t in turns])
    return row


def write_arm(out: Path, results: list[dict], extra: dict) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    from src.utils.episode_log import write_rows

    write_rows(out / "test_episodes.csv", [episode_row(r["item"]["source"], r["turns"]) for r in results])
    with (out / "ha_probs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["patient_id", "label", "num_tests", "turns", "rewrites", "final_context_tokens",
                         "total_context_tokens", *(f"p_{d}" for d in DISEASES)])
        for r in results:
            writer.writerow([r["item"]["patient_id"], r["item"]["label"], len(r["item"]["blocks"]), len(r["turns"]),
                             sum(t.action == "rewrite" for t in r["turns"]), r["turns"][-1].context_tokens,
                             sum(t.context_tokens for t in r["turns"]), *r["turns"][-1].probs])
    with (out / "notes.jsonl").open("w", encoding="utf-8") as handle:
        for r in results:
            handle.write(json.dumps({"patient_id": r["item"]["patient_id"], "label": r["item"]["label"],
                                     "tests": r["item"]["tests"], "turns": [t.log() for t in r["turns"]]}) + "\n")
    summary = probability_summary([r["item"]["label"] for r in results], [r["turns"][-1].probs for r in results])
    summary |= turn_statistics(results) | extra
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_model(model_name: str, adapter: str):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto",
                                                 attn_implementation="sdpa")
    if adapter != "none":
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    return model.eval()


def main() -> None:
    from src.score_ha import first_episodes
    from src.train_ha_sft import load_split

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="LoRA adapter folder, or 'none' for the untrained model")
    parser.add_argument("--mode", required=True, choices=("edit", "append"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=EPISODES_CSV")
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--limit", type=int, default=None, help="First N patients per arm (smoke test)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-words", type=int, default=250)
    args = parser.parse_args()

    dataset_cfg = OmegaConf.load(REPO / "configs/dataset/defaults.yaml")
    model_cfg = OmegaConf.load(REPO / "configs/model/defaults.yaml")
    env_cfg = OmegaConf.load(REPO / "configs/environment/la_cdm.yaml")
    score_template = (REPO / env_cfg.hypothesis_prompt_template_file).read_text()
    patients = {str(p["patient_id"]): p for p in load_split(dataset_cfg, args.split, None)}
    arms = []
    for spec in args.arm:
        name, sep, path = spec.partition("=")
        if not sep or not Path(path).is_file():
            raise SystemExit(f"--arm needs NAME=EPISODES_CSV with an existing file: {spec}")
        arms.append((name, Path(path)))

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name)
    model = load_model(model_cfg.model_name, args.model)
    agent = CLMHA(model, tokenizer, score_template, EDIT_TEMPLATE.read_text(), edit=args.mode == "edit",
                  max_new_tokens=args.max_new_tokens, max_words=args.max_words)
    print(f"[clm replay] model={args.model} mode={args.mode} arms={[a for a, _ in arms]} -> {args.out_dir}")

    overview = {}
    for name, path in arms:
        items = []
        for source in first_episodes(path)[: args.limit]:
            patient = patients.get(str(source["patient_id"]))
            if patient is None:
                raise SystemExit(f"{path}: patient {source['patient_id']} is not in the {args.split} split")
            items.append({"patient_id": source["patient_id"], "label": str(patient["condition"]).lower(),
                          "history": patient["patient_history"], "tests": list(source["tests"]),
                          "blocks": blocks_for(patient, source["tests"]), "source": source})
        started = time.time()
        results = run_sequences(agent, items, sample=False, batch_size=args.batch_size, progress=name)
        summary = write_arm(args.out_dir / name, results, {
            "arm": name, "source": str(path), "model": args.model, "mode": args.mode, "split": args.split,
            "seconds": round(time.time() - started, 1)})
        overview[name] = summary
        print(f"[clm replay] {name}: accuracy {summary['accuracy']:.3f}, rewrite rate {summary['rewrite_rate']:.2f}, "
              f"final context {summary['mean_final_context_tokens']:.0f} tokens, n={summary['n']}", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "overview.json").write_text(json.dumps(overview, indent=2))


if __name__ == "__main__":
    main()
