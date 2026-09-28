"""Summarize patient histories using an LLM.

Reads a MIMIC-CDM CSV with a ``Patient History`` column, generates concise
summaries using the specified model, and writes the result with a new
``Patient History Summary`` column.

Requires a GPU.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SYSTEM_PROMPT = (
    "You are a medical expert. Summarize the following passage with respect to "
    "relevant information to the symptoms. For privacy purposes it has been "
    "anonymised by replacing personal information with underscores. Completely "
    "ignore this anonymised information when it does not provide further "
    "information in its anonymised form and do not hallucinate new information "
    "if it is not given in the text. Include symptoms, patient history and "
    "family history. Be brief and concise. Only include information that is in "
    "the text and relevant to the symptoms.\n\nHere is the passage:\n\n"
)


def load_summarizer(model_name: str):
    """Load a 4-bit quantized model and tokenizer for summarization."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def summarize_histories(
    input_file: Path,
    output_file: Path,
    model_name: str,
    max_new_tokens: int = 512,
    model=None,
    tokenizer=None,
    checkpoint_every: int = 25,
) -> None:
    """Add a ``Patient History Summary`` column to a CSV using an LLM.

    Args:
        input_file: CSV with a ``Patient History`` column.
        output_file: Path to write the augmented CSV.
        model_name: HuggingFace model name or local path (used only if
            ``model``/``tokenizer`` are not provided).
        max_new_tokens: Maximum tokens to generate per summary.
        model: Optional pre-loaded model to reuse across calls.
        tokenizer: Optional pre-loaded tokenizer to reuse across calls.
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_summarizer(model_name)

    df = pd.read_csv(input_file)
    print(f"Summarizing {len(df)} patient histories from {input_file}")

    if "Patient History Summary" not in df.columns:
        df["Patient History Summary"] = pd.NA

    completed = df["Patient History Summary"].fillna("").astype(str).str.strip().ne("")
    print(f"Resuming with {int(completed.sum())}/{len(df)} summaries already complete")

    generated_since_checkpoint = 0
    for index, row in tqdm(df.iterrows(), total=len(df)):
        if completed.loc[index]:
            continue
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["Patient History"]},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(
            outputs[0, input_ids.shape[-1]:], skip_special_tokens=True
        )
        df.at[index, "Patient History Summary"] = response.strip()
        generated_since_checkpoint += 1
        if generated_since_checkpoint >= checkpoint_every:
            df.to_csv(output_file, index=False)
            generated_since_checkpoint = 0

    df.to_csv(output_file, index=False)
    print(f"Saved {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add patient history summaries to a MIMIC-CDM CSV."
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Hugging Face model name or local Transformers model path.",
    )
    parser.add_argument(
        "--input_file", type=str, required=True,
        help="Input CSV with a Patient History column.",
    )
    parser.add_argument(
        "--output_file", type=str, default=None,
        help="Output CSV path (default: overwrite input file).",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512,
        help="Maximum tokens per summary (default: 512).",
    )
    parser.add_argument(
        "--checkpoint_every", type=int, default=25,
        help="Save progress after this many new summaries (default: 25).",
    )
    args = parser.parse_args()

    output_file = Path(args.output_file) if args.output_file else Path(args.input_file)
    summarize_histories(
        Path(args.input_file),
        output_file,
        args.model,
        args.max_new_tokens,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
