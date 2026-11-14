"""Train the standalone Hypothesis Agent (hypothesis H2, docs/HYPOTHESIS_H2.md).

A separate LoRA adapter on the same base model, trained alone by supervised fine-tuning to
answer ``Hypothesis: <disease>`` from the patient history plus full or partial test results.
The Decision Agent plays no part. The adapter with the lowest validation loss is saved to
``<output-dir>/adapter`` and is then used frozen.

    python -m src.train_ha_sft --output-dir outputs/ha_sft/seed_0 --seed 0
    python -m src.train_ha_sft --output-dir outputs/ha_sft/smoke --smoke     # a few minutes

Resubmitting with the same ``--output-dir`` resumes from the newest checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

from src.dataset.mimic_cdm_dataset import MIMICCDMDataset
from src.ha.scoring import suffix_nll
from src.ha.sft_data import build_examples

REPO = Path(__file__).resolve().parents[1]


class SuffixDataset(torch.utils.data.Dataset):
    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> dict:
        e = self.examples[i]
        return {"input_ids": e["input_ids"], "labels": e["labels"]}


def collate(batch: list[dict]) -> dict:
    if len(batch) != 1:
        raise ValueError("Use per_device_train_batch_size=1 (rows are unpadded)")
    ids = torch.tensor([batch[0]["input_ids"]])
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": torch.tensor([batch[0]["labels"]])}


class SuffixTrainer(Trainer):
    """Scores only the answer tokens, with logits for those positions only."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The loss is a per-row mean: let the Trainer average over gradient accumulation.
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = suffix_nll(model, inputs["input_ids"], inputs["attention_mask"], inputs["labels"])
        return (loss, {"loss": loss}) if return_outputs else loss


def load_split(dataset_cfg, split: str, limit: int | None) -> list[dict]:
    files = {"train": dataset_cfg.train_data_file, "val": dataset_cfg.val_data_file, "test": dataset_cfg.test_data_file}
    data = MIMICCDMDataset(
        data_file=files[split],
        lab_test_mapping_file=dataset_cfg.lab_test_mapping_file,
        lab_tests=list(dataset_cfg.lab_tests),
        imaging_tests=list(dataset_cfg.imaging_tests),
        other_tests=list(dataset_cfg.other_tests),
    )
    patients = [data[i] for i in range(len(data))]
    return patients[:limit] if limit else patients


def write_manifest(path: Path, examples: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["patient_id", "label", "num_tests", "tests", "dropped", "num_tokens"])
        for e in examples:
            writer.writerow([e["patient_id"], e["label"], len(e["tests"]), "|".join(e["tests"]),
                             "|".join(e["dropped"]), len(e["input_ids"])])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-random", type=int, default=2, help="Random partial records per training patient")
    parser.add_argument("--val-n-random", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=7000)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=240, help="Optimizer steps between validations")
    parser.add_argument("--lora-r", type=int, default=None, help="Default: configs/model/defaults.yaml (8)")
    parser.add_argument("--smoke", action="store_true", help="16 train / 8 val patients, 10 steps")
    args = parser.parse_args()

    set_seed(args.seed)
    dataset_cfg = OmegaConf.load(REPO / "configs/dataset/defaults.yaml")
    model_cfg = OmegaConf.load(REPO / "configs/model/defaults.yaml")
    env_cfg = OmegaConf.load(REPO / "configs/environment/la_cdm.yaml")
    template = (REPO / env_cfg.hypothesis_prompt_template_file).read_text()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name)

    train_patients = load_split(dataset_cfg, "train", 16 if args.smoke else None)
    val_patients = load_split(dataset_cfg, "val", 8 if args.smoke else None)
    train = build_examples(tokenizer, template, train_patients, args.seed, args.n_random, args.max_tokens)
    # Validation views are fixed (seed 10_000) so every run selects its checkpoint on the same rows.
    val = build_examples(tokenizer, template, val_patients, 10_000, args.val_n_random, args.max_tokens)
    write_manifest(out / "sft_examples_train.csv", train)
    write_manifest(out / "sft_examples_val.csv", val)
    print(f"[ha_sft] {len(train_patients)} train patients -> {len(train)} rows; "
          f"{len(val_patients)} val patients -> {len(val)} rows; "
          f"{sum(bool(e['dropped']) for e in train)} train rows dropped tests to fit {args.max_tokens} tokens")

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa"
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    lora = LoraConfig(
        r=args.lora_r or model_cfg.lora_r,
        lora_alpha=model_cfg.lora_alpha if args.lora_r is None else 2 * args.lora_r,
        target_modules=list(model_cfg.lora_target_modules),
        lora_dropout=model_cfg.lora_dropout,
        bias=model_cfg.lora_bias,
        task_type=model_cfg.lora_task_type,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=10 if args.smoke else -1,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=1 if args.smoke else 10,
        eval_strategy="steps",
        eval_steps=5 if args.smoke else args.eval_every,
        save_strategy="steps",
        save_steps=5 if args.smoke else args.eval_every,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        prediction_loss_only=True,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        label_names=["labels"],
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = SuffixTrainer(
        model=model, args=training_args, train_dataset=SuffixDataset(train),
        eval_dataset=SuffixDataset(val), data_collator=collate,
    )
    resume = any((out / "checkpoints").glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=True if resume else None)

    trainer.save_model(str(out / "adapter"))  # the best checkpoint (load_best_model_at_end)
    tokenizer.save_pretrained(str(out / "adapter"))
    (out / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    (out / "training_complete.json").write_text(json.dumps({
        "global_step": trainer.state.global_step,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_eval_loss": trainer.state.best_metric,
        "args": {k: str(v) for k, v in vars(args).items()},
        "train_rows": len(train),
        "val_rows": len(val),
    }, indent=2))
    print(f"[ha_sft] saved {out / 'adapter'} (best {trainer.state.best_model_checkpoint}, "
          f"eval_loss {trainer.state.best_metric})")


if __name__ == "__main__":
    main()
