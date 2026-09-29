"""Validate prepared LA-CDM CSV files before evaluation or training."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
EXPECTED_SIZES = {"train": 1920, "val": 240, "test": 240}
EXPECTED_LABELS = {
    "appendicitis",
    "cholecystitis",
    "diverticulitis",
    "pancreatitis",
}
REQUIRED_COLUMNS = {
    "Patient ID",
    "Patient History",
    "Patient History Summary",
    "Physical Examination",
    "Laboratory Tests",
    "Radiology",
    "Label",
}


def parse_as(value: object, expected_type: type) -> bool:
    try:
        return isinstance(ast.literal_eval(str(value)), expected_type)
    except (ValueError, SyntaxError):
        return False


def main() -> None:
    errors: list[str] = []
    ids_by_split: dict[str, set[int]] = {}

    for split, expected_size in EXPECTED_SIZES.items():
        path = DATA_DIR / f"{split}.csv"
        if not path.is_file():
            errors.append(f"missing file: {path}")
            continue

        frame = pd.read_csv(path)
        missing_columns = REQUIRED_COLUMNS - set(frame.columns)
        if missing_columns:
            errors.append(
                f"{split}: missing columns {', '.join(sorted(missing_columns))}"
            )
            continue

        summaries = frame["Patient History Summary"].fillna("").astype(str).str.strip()
        blank_summaries = int(summaries.eq("").sum())
        duplicate_ids = int(frame["Patient ID"].duplicated().sum())
        invalid_labels = sorted(set(frame["Label"].dropna()) - EXPECTED_LABELS)
        invalid_labs = sum(
            not parse_as(value, dict) for value in frame["Laboratory Tests"]
        )
        invalid_radiology = sum(
            not parse_as(value, list) for value in frame["Radiology"]
        )
        ids_by_split[split] = set(frame["Patient ID"].astype(int))

        counts = frame["Label"].value_counts().sort_index().to_dict()
        print(
            f"{split:5} rows={len(frame):4} blank_summaries={blank_summaries} "
            f"duplicate_ids={duplicate_ids} invalid_labs={invalid_labs} "
            f"invalid_radiology={invalid_radiology} labels={counts}"
        )

        if len(frame) != expected_size:
            errors.append(f"{split}: expected {expected_size} rows, found {len(frame)}")
        if blank_summaries:
            errors.append(f"{split}: {blank_summaries} blank summaries")
        if duplicate_ids:
            errors.append(f"{split}: {duplicate_ids} duplicate patient IDs")
        if invalid_labels:
            errors.append(f"{split}: invalid labels {invalid_labels}")
        if invalid_labs:
            errors.append(f"{split}: {invalid_labs} malformed lab fields")
        if invalid_radiology:
            errors.append(f"{split}: {invalid_radiology} malformed radiology fields")

    split_names = list(ids_by_split)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = ids_by_split[left] & ids_by_split[right]
            if overlap:
                errors.append(
                    f"{left}/{right}: {len(overlap)} overlapping patient IDs"
                )

    mapping_path = DATA_DIR / "lab_test_mapping.csv"
    if not mapping_path.is_file():
        errors.append(f"missing file: {mapping_path}")
    else:
        mapping = pd.read_csv(mapping_path)
        missing = {"itemid", "label", "corresponding_ids"} - set(mapping.columns)
        print(f"lab mapping rows={len(mapping)}")
        if missing:
            errors.append(f"lab mapping missing columns: {sorted(missing)}")

    if errors:
        print("\nVALIDATION FAILED", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        raise SystemExit(1)
    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
