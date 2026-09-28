"""Step 3: Validate the final MIMIC-CDM cohort before data preparation."""

from __future__ import annotations

import csv
import pickle
import sys
from pathlib import Path


COHORT_DIR = Path("/blue/prismap-ai-core/omerkahveci/mimic_cdm_final")
EXPECTED_COUNTS = {
    "appendicitis": 957,
    "cholecystitis": 648,
    "diverticulitis": 257,
    "pancreatitis": 538,
}
LEAKAGE_TERMS = {
    "appendicitis": ("acute appendicitis", "appendicitis", "appendectomy"),
    "cholecystitis": (
        "acute cholecystitis",
        "cholecystitis",
        "cholecystostomy",
    ),
    "diverticulitis": ("acute diverticulitis", "diverticulitis"),
    "pancreatitis": ("acute pancreatitis", "pancreatitis", "pancreatectomy"),
}
REQUIRED_FIELDS = (
    "Patient History",
    "Physical Examination",
    "Laboratory Tests",
    "Radiology",
    "Discharge Diagnosis",
)


def main() -> None:
    errors: list[str] = []
    all_ids: set[int] = set()

    for condition, expected_count in EXPECTED_COUNTS.items():
        path = COHORT_DIR / f"{condition}_hadm_info_first_diag.pkl"
        if not path.is_file():
            errors.append(f"missing file: {path}")
            continue

        with path.open("rb") as file:
            records = pickle.load(file)
        if not isinstance(records, dict):
            errors.append(f"{condition}: pickle is not a dictionary")
            continue

        duplicate_ids = set(records) & all_ids
        if duplicate_ids:
            errors.append(
                f"{condition}: {len(duplicate_ids)} admission IDs overlap another disease"
            )
        all_ids.update(records)

        invalid = 0
        leakage = 0
        for hadm_id, record in records.items():
            if not isinstance(record, dict):
                invalid += 1
                continue
            if any(field not in record for field in REQUIRED_FIELDS):
                invalid += 1
                continue
            history = str(record["Patient History"] or "")
            physical_exam = str(record["Physical Examination"] or "")
            labs = record["Laboratory Tests"]
            radiology = record["Radiology"]
            abdominal_image = any(
                isinstance(report, dict)
                and report.get("Region") == "Abdomen"
                and report.get("Modality") is not None
                for report in radiology
            ) if isinstance(radiology, list) else False

            if (
                not history.strip()
                or len(physical_exam) < 40
                or not isinstance(labs, dict)
                or not labs
                or not abdominal_image
            ):
                invalid += 1
            if any(term in history.lower() for term in LEAKAGE_TERMS[condition]):
                leakage += 1

        print(
            f"{condition:16} cases={len(records):4} expected={expected_count:4} "
            f"invalid={invalid} leakage={leakage}"
        )
        if len(records) != expected_count:
            errors.append(
                f"{condition}: expected {expected_count} cases, found {len(records)}"
            )
        if invalid:
            errors.append(f"{condition}: {invalid} invalid records")
        if leakage:
            errors.append(f"{condition}: {leakage} histories contain leakage terms")

    mapping_path = COHORT_DIR / "lab_test_mapping.csv"
    if not mapping_path.is_file():
        errors.append(f"missing file: {mapping_path}")
    else:
        with mapping_path.open(encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            mapping_rows = list(reader)
            columns = set(reader.fieldnames or [])
        required_columns = {"itemid", "label", "corresponding_ids"}
        missing_columns = required_columns - columns
        print(f"lab mapping      rows={len(mapping_rows)}")
        if missing_columns:
            errors.append(
                "lab mapping missing columns: " + ", ".join(sorted(missing_columns))
            )

    print(f"unique admissions: {len(all_ids)}")
    if errors:
        print("\nVALIDATION FAILED", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        raise SystemExit(1)
    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
