"""Stage 3 of H4 (H4b): train the CLM Hypothesis Agent alone, on replayed test sequences.

No Decision Agent: each step takes one training patient and a test sequence (all available
tests, or a random subset, in random order; ``src/clm/context.random_sequence``). The HA runs
``--group-size`` sampled rollouts over it (rewrite or keep at every turn) and is updated once
(src/clm/learning.py: stepwise GRPO for the rewrites, SFT for the diagnosis, mixed replay).

    python -m src.clm.train_ha --output-dir outputs/clm/ha_train/h2init_seed0 --init outputs/ha_sft/seed_0/adapter
    python -m src.clm.train_ha --output-dir outputs/clm/ha_train/smoke --init none --smoke

Output (all under ``--output-dir``):

- ``checkpoint/``: adapter + optimizer + state of the newest save (resumed automatically);
- ``adapters/step_XXXX/``: the adapter at every validation;
- ``val/step_XXXX/``: validation notes and summary (fixed sequences, greedy); ``val_curve.csv``;
- ``updates.jsonl``: one line of training statistics per step;
- ``adapter/``: the best adapter by validation accuracy (ties: fewer context tokens), written
  at the end together with ``training_complete.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from src.clm.agent import CLMHA
from src.clm.context import blocks_for, fit_tests, random_sequence
from src.clm.learning import ha_update, replay_rows
from src.clm.replay import EDIT_TEMPLATE, run_sequences, write_arm
from src.ha.prompts import available_tests, chat_prompt, encode

REPO = Path(__file__).resolve().parents[2]


def build_model(model_name: str, init: str, resume: Path | None, lora_cfg):
    from peft import LoraConfig, PeftModel, get_peft_model

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto",
                                                 attn_implementation="sdpa")
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    source = resume if resume is not None else (None if init == "none" else Path(init))
    if source is not None:
        model = PeftModel.from_pretrained(model, str(source), is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=lora_cfg.lora_r, lora_alpha=lora_cfg.lora_alpha, target_modules=list(lora_cfg.lora_target_modules),
            lora_dropout=lora_cfg.lora_dropout, bias=lora_cfg.lora_bias, task_type=lora_cfg.lora_task_type))
    return model


def make_items(patients: list[dict], sequences: list[list[str]]) -> list[dict]:
    return [{"patient_id": p["patient_id"], "label": str(p["condition"]).lower(), "history": p["patient_history"],
             "tests": list(seq), "blocks": blocks_for(p, seq),
             "source": {"patient_id": p["patient_id"], "label": str(p["condition"]).lower(), "tests": list(seq),
                        "cost_usd": 0.0}}
            for p, seq in zip(patients, sequences)]


def validate(agent: CLMHA, items: list[dict], out: Path, step: int, batch_size: int) -> dict:
    started = time.time()
    results = run_sequences(agent, items, sample=False, batch_size=batch_size, progress=f"val step {step}")
    summary = write_arm(out, results, {"step": step, "seconds": round(time.time() - started, 1)})
    return summary


def save_checkpoint(model, optimizer, out: Path, state: dict) -> None:
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    model.save_pretrained(str(tmp))
    torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
    (tmp / "state.json").write_text(json.dumps(state, indent=2))
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)


def main() -> None:
    from src.train_ha_sft import load_split

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init", required=True, help="Starting adapter folder (e.g. the H2 HA), or 'none'")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--w-eff", type=float, default=0.5, help="Weight of the success-gated efficiency advantage")
    parser.add_argument("--sft-weight", type=float, default=None, help="Default: 1/B (docs/HYPOTHESIS_H4.md D6)")
    parser.add_argument("--replay", type=int, default=8, help="Mixed-disease SFT records per step")
    parser.add_argument("--p-all", type=float, default=0.5, help="Share of training sequences with all tests")
    parser.add_argument("--max-tokens", type=int, default=7000, help="Budget of the append-only HA prompt")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-words", type=int, default=250)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--val-limit", type=int, default=None, help="Validate on the first N val patients")
    parser.add_argument("--batch-size", type=int, default=8, help="Validation batch size")
    parser.add_argument("--smoke", action="store_true", help="8 train / 8 val patients, 4 steps")
    args = parser.parse_args()
    if args.smoke:
        args.steps, args.eval_every, args.save_every, args.max_new_tokens = 4, 2, 2, 128
        args.val_limit = 8

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    dataset_cfg = OmegaConf.load(REPO / "configs/dataset/defaults.yaml")
    model_cfg = OmegaConf.load(REPO / "configs/model/defaults.yaml")
    env_cfg = OmegaConf.load(REPO / "configs/environment/la_cdm.yaml")
    score_template = (REPO / env_cfg.hypothesis_prompt_template_file).read_text()
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name)

    train_patients = load_split(dataset_cfg, "train", 8 if args.smoke else None)
    val_patients = load_split(dataset_cfg, "val", args.val_limit)

    def count(message: str) -> int:
        return len(encode(tokenizer, chat_prompt(tokenizer, message)))

    # Fixed validation sequences: one random non-empty subset per patient, seed 10_000 (as in H2).
    val_rng = random.Random(10_000)
    val_sequences = []
    for p in val_patients:
        available = available_tests(p["test_results"])
        seq = val_rng.sample(available, val_rng.randint(1, len(available))) if available else []
        val_sequences.append(fit_tests(count, score_template, p, seq, args.max_tokens, val_rng))
    val_items = make_items(val_patients, val_sequences)

    resume = out / "checkpoint" if (out / "checkpoint" / "state.json").is_file() else None
    state = json.loads((resume / "state.json").read_text()) if resume else {"step": 0, "best": None}
    model = build_model(model_cfg.model_name, args.init, resume, model_cfg)
    agent = CLMHA(model, tokenizer, score_template, EDIT_TEMPLATE.read_text(), edit=True,
                  max_new_tokens=args.max_new_tokens, max_words=args.max_words, temperature=args.temperature)
    params = agent.trainable_parameters()
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=0.0)
    if resume:
        optimizer.load_state_dict(torch.load(resume / "optimizer.pt", map_location="cpu"))
    print(f"[clm train_ha] init={args.init} resume={resume} step={state['step']} trainable={sum(p.numel() for p in params):,}"
          f" train={len(train_patients)} val={len(val_items)} -> {out}", flush=True)
    (out / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    curve_path = out / "val_curve.csv"

    def record_validation(step: int) -> None:
        summary = validate(agent, val_items, out / "val" / f"step_{step:04d}", step, args.batch_size)
        model.save_pretrained(str(out / "adapters" / f"step_{step:04d}"))
        new = not curve_path.exists()
        with curve_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if new:
                writer.writerow(["step", "accuracy", "auroc_ovr_exact", "ece", "rewrite_rate", "malformed_rate",
                                 "mean_final_context_tokens", "mean_total_context_tokens"])
            writer.writerow([step, summary["accuracy"], summary["auroc_ovr_exact"], summary["ece_10_bins"],
                             summary["rewrite_rate"], summary["malformed_rate"],
                             summary["mean_final_context_tokens"], summary["mean_total_context_tokens"]])
        key = (summary["accuracy"], -summary["mean_total_context_tokens"])
        if state["best"] is None or key > tuple(state["best"]["key"]):
            state["best"] = {"step": step, "key": list(key), "accuracy": summary["accuracy"]}
        print(f"[clm train_ha] val step {step}: accuracy {summary['accuracy']:.3f}, rewrite rate "
              f"{summary['rewrite_rate']:.2f}, context {summary['mean_total_context_tokens']:.0f} tokens "
              f"(best: step {state['best']['step']})", flush=True)

    if state["step"] == 0 and not (out / "val" / "step_0000" / "summary.json").exists():
        record_validation(0)

    order_rng = random.Random(args.seed)
    order = list(range(len(train_patients)))
    epochs = []  # deterministic patient order, reproducible on resume
    while len(epochs) * len(order) < args.steps + 1:
        perm = order[:]
        order_rng.shuffle(perm)
        epochs.append(perm)
    schedule = [i for perm in epochs for i in perm]

    with (out / "updates.jsonl").open("a", encoding="utf-8") as log:
        while state["step"] < args.steps:
            step = state["step"] + 1
            rng = random.Random(args.seed * 1_000_003 + step)
            torch.manual_seed(args.seed * 1_000_003 + step)
            lr = args.learning_rate * min(1.0, step / max(1, args.warmup_steps))
            for g in optimizer.param_groups:
                g["lr"] = lr
            patient = train_patients[schedule[step - 1]]
            seq = fit_tests(count, score_template, patient, random_sequence(patient, rng, args.p_all),
                            args.max_tokens, rng)
            item = make_items([patient], [seq])[0]
            started = time.time()
            rollouts = run_sequences(agent, [item] * args.group_size, sample=True, batch_size=args.group_size)
            group = [{"label": item["label"], "turns": r["turns"]} for r in rollouts]
            stats = ha_update(agent, [group], replay_rows(agent, train_patients, rng, args.replay, args.max_tokens),
                              optimizer, w_eff=args.w_eff, sft_weight=args.sft_weight)
            stats |= {"step": step, "patient_id": item["patient_id"], "label": item["label"], "tests": len(seq),
                      "lr": lr, "seconds": round(time.time() - started, 1)}
            log.write(json.dumps(stats) + "\n")
            log.flush()
            state["step"] = step
            if step % 10 == 0 or args.smoke:
                print(f"[clm train_ha] step {step}: success {stats['success_rate']:.2f}, rewrite rate "
                      f"{stats['rewrite_rate']:.2f}, sft_nll {stats['sft_nll']:.3f}, {stats['seconds']}s", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                record_validation(step)
            if step % args.save_every == 0 or step == args.steps:
                save_checkpoint(model, optimizer, out / "checkpoint", state)

    best = state["best"]
    best_dir = out / "adapters" / f"step_{best['step']:04d}"
    if (out / "adapter").exists():
        shutil.rmtree(out / "adapter")
    shutil.copytree(best_dir, out / "adapter")
    (out / "training_complete.json").write_text(json.dumps({
        "steps": state["step"], "best_step": best["step"], "best_val_accuracy": best["accuracy"],
        "adapter": str(out / "adapter"), "init": args.init}, indent=2))
    print(f"[clm train_ha] done: best step {best['step']} (val accuracy {best['accuracy']:.3f}) -> {out / 'adapter'}")


if __name__ == "__main__":
    main()
