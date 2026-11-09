#!/usr/bin/env python3
"""Download guanning-ai/dapo14k and write the exact VERL RL parquet schema."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset, load_dataset


REQUIRED_COLUMNS = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
SOURCE_DATASET = "guanning-ai/dapo14k"
SOURCE_REVISION = "42edfa38f7f2ac452eb12e20049b2b3e2efc6651"
EXPECTED_TRAIN_ROWS = 13_542
BOXED_INSTRUCTION = r"Please reason step by step, and put your final answer within \boxed{}."
PROMPT_PROTOCOL_VERSION = "boxed-only-v1"


def add_boxed_instruction(problem: str) -> str:
    problem = str(problem).rstrip()
    if problem.endswith(BOXED_INSTRUCTION):
        return problem
    return problem + "\n" + BOXED_INSTRUCTION


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def validate_existing(path: Path) -> int:
    metadata = pq.read_metadata(path)
    columns = set(metadata.schema.names)
    # Nested Arrow leaves do not necessarily expose the top-level names, so
    # perform the authoritative schema check through datasets below.
    dataset = load_dataset("parquet", data_files=str(path), split="train")
    missing = REQUIRED_COLUMNS.difference(dataset.column_names)
    if missing:
        raise ValueError(f"Prepared parquet is missing columns: {sorted(missing)}; Arrow leaves={sorted(columns)}")
    if len(dataset) <= 0:
        raise ValueError("Prepared dapo14k parquet is empty")
    if len(dataset) != EXPECTED_TRAIN_ROWS:
        raise ValueError(
            "Prepared parquet has the wrong row count for guanning-ai/dapo14k: "
            f"{len(dataset)} != {EXPECTED_TRAIN_ROWS}"
        )
    invalid_prompt_rows = []
    invalid_source_rows = []
    for index, row in enumerate(dataset):
        messages = row["prompt"]
        valid = (
            isinstance(messages, list)
            and len(messages) == 1
            and messages[0].get("role") == "user"
            and str(messages[0].get("content", "")).rstrip().endswith(BOXED_INSTRUCTION)
        )
        if not valid:
            invalid_prompt_rows.append(index)
            if len(invalid_prompt_rows) >= 10:
                break
        extra_info = row.get("extra_info") or {}
        source_valid = (
            row.get("data_source") == "math_dapo"
            and extra_info.get("source_dataset") == SOURCE_DATASET
            and extra_info.get("source_revision") == SOURCE_REVISION
        )
        if not source_valid:
            invalid_source_rows.append(index)
            if len(invalid_source_rows) >= 10:
                break
    if invalid_prompt_rows:
        raise ValueError(
            "Prepared parquet uses the obsolete raw-problem prompt protocol; "
            f"first invalid rows: {invalid_prompt_rows}"
        )
    if invalid_source_rows:
        raise ValueError(
            "Prepared parquet was not built from guanning-ai/dapo14k; "
            f"first invalid rows: {invalid_source_rows}"
        )
    return len(dataset)


def main():
    args = parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.is_file():
        try:
            rows = validate_existing(output)
        except ValueError as error:
            print(json.dumps({"phase": "rebuild_dapo14k", "path": str(output), "reason": str(error)}))
        else:
            print(json.dumps({"phase": "reuse_dapo14k", "path": str(output), "rows": rows}))
            return

    source = load_dataset(SOURCE_DATASET, split="train", revision=SOURCE_REVISION)
    if len(source) != EXPECTED_TRAIN_ROWS:
        raise RuntimeError(
            f"Pinned {SOURCE_DATASET}@{SOURCE_REVISION} changed unexpectedly: "
            f"{len(source)} != {EXPECTED_TRAIN_ROWS}"
        )
    required_source = {"problem", "answer"}
    missing_source = required_source.difference(source.column_names)
    if missing_source:
        raise ValueError(
            f"{SOURCE_DATASET} is missing required columns {sorted(missing_source)}; "
            f"found {source.column_names}"
        )

    rows = []
    for index, item in enumerate(source):
        problem = add_boxed_instruction(item["problem"])
        answer = str(item["answer"])
        rows.append(
            {
                "data_source": "math_dapo",
                "prompt": [{"role": "user", "content": problem}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "index": index,
                    "source_dataset": SOURCE_DATASET,
                    "source_revision": SOURCE_REVISION,
                    "source_fingerprint": str(source._fingerprint),
                    "source_datasource": str(item.get("datasource", "dapo14k")),
                    "prompt_protocol": PROMPT_PROTOCOL_VERSION,
                },
            }
        )

    prepared = Dataset.from_list(rows)
    temporary = output.with_suffix(output.suffix + ".tmp")
    prepared.to_parquet(str(temporary))
    temporary.replace(output)
    count = validate_existing(output)
    if count != len(source):
        raise RuntimeError(f"Row-count mismatch after conversion: source={len(source)}, output={count}")
    print(
        json.dumps(
            {
                "phase": "prepared_dapo14k",
                "source": SOURCE_DATASET,
                "source_revision": SOURCE_REVISION,
                "source_fingerprint": str(source._fingerprint),
                "path": str(output),
                "rows": count,
                "columns": prepared.column_names,
            }
        )
    )


if __name__ == "__main__":
    main()
