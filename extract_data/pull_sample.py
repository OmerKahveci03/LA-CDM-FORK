"""
A simple but efficient script to extract a sample of patients from MIMIC dataset.
"""

import os
import glob
import warnings


import pandas as pd

from io import StringIO
from typing import Optional
from concurrent.futures import Future, ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

pid = 'subject_id'

def process_file(dir_type: str, file: str, write_dir: str, all_pids: set[int]) -> Optional[str]:
    """ Returns the file type if successful. None otherwise """

    output_chunks: list[pd.DataFrame] = []

    _basename: str = f"{dir_type}_{os.path.basename(file).removesuffix('.csv.gz')}"
    buffer: StringIO = StringIO()

    buffer.write(f" - Pulling {_basename}\n")

    for chunk in pd.read_csv(file, compression="gzip", chunksize=100_000):
        if pid in chunk.columns:
            matching: pd.DataFrame = chunk[chunk[pid].isin(all_pids)]

        else:
            matching: pd.DataFrame = chunk

        if not matching.empty:
            output_chunks.append(matching)

    if not output_chunks:
        buffer.write("  - Found nothing.\n")

        print(buffer.getvalue())
        return None

    filtered = pd.concat(output_chunks, ignore_index=True)
    buffer.write(f"{_basename}: {len(filtered):,} matching rows\n")

    write_fp = os.path.join(write_dir, f"{_basename}.parquet")

    # Normalize mixed-type text columns so PyArrow can write them safely.
    for column in filtered.select_dtypes(include=["object"]).columns:
        filtered[column] = filtered[column].astype("string")

    filtered.to_parquet(write_fp, index=False)

    buffer.write(f"  - Wrote {len(filtered):,} rows to '{write_fp}'\n")
    print(buffer.getvalue())
    return _basename

def pull_sample(input_dir: str, write_dir: str, patient_count: int = 2000) -> None:
    """ Pulls the sample patients and writes them to the write directory. """

    os.makedirs(write_dir, exist_ok=True)
    print(f"Extracting {patient_count:,} patients from {input_dir} -> {write_dir}")

    # --- 1. Pick the patients of interest ---
    pat_fp: str = os.path.join(input_dir, "hosp", "patients.csv.gz")

    # Assuming 1 row per patient, but we'll play it safe
    all_pids: set[int] = set()
    for pat_chunk in pd.read_csv(pat_fp, compression="gzip", chunksize=patient_count, usecols=[pid]):
        pids: set[int] = set(pat_chunk[pid].astype("UInt64").unique())

        all_pids |= pids

        if len(all_pids) >= patient_count:
            all_pids = set(sorted(all_pids)[:patient_count])
            break

    print(f"Obtained {len(all_pids):,} unique patients to extract.")

    # --- 2. Extract every file type ---
    extracted_file_types: list[str] = []
    for dir_type in ["hosp", "icu"]:
        print(f"Pulling {dir_type} files.")

        files: list[str] = glob.glob(os.path.join(input_dir, dir_type, "*.csv.gz"))

        with ProcessPoolExecutor() as executor:
            futures: list[Future] = [
                executor.submit(process_file, dir_type=dir_type, file=file, write_dir=write_dir, all_pids=all_pids)
                for file in files
            ]
            for future in as_completed(futures):
                result: Optional[str] = future.result()
                if isinstance(result, str):
                    extracted_file_types.append(result)
            

    with open(os.path.join(write_dir, "metadata.txt"), mode="w", encoding="utf-8") as meta_file:
        meta_file.write(
            f"Patients extracted: {len(all_pids):,}\n"
            f"Extracted file types: {sorted(extracted_file_types)}\n"
            f"Extraction source: {input_dir}\n"
        )

if __name__ == "__main__":
    input_dir: str = "/orange/prismap-data-core/MIMIC/physionet.org/files/mimiciv/3.1"
    write_dir: str = "/blue/prismap-ai-core/omerkahveci/mimic_sample"

    pull_sample(input_dir=input_dir, write_dir=write_dir)
