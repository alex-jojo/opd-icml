"""Append the pinned 40-question AMC23 test split to the evaluation suite."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import urllib.request
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from project_paths import project_absolute, project_relative
SOURCE_DATASET = "math-ai/amc23"
SOURCE_REVISION = "80815d37005feb82cd7f8fbc6901d5d3eff43057"
SOURCE_FILE = "test-00000-of-00001.parquet"
SOURCE_SHA256 = "b696e87ba47be4e879a60fd4ef1d4aa522ba78c5ae013c6aff5cc9788a397c5e"
SOURCE_URL = f"https://huggingface.co/datasets/{SOURCE_DATASET}/resolve/{SOURCE_REVISION}/{SOURCE_FILE}"
EXPECTED_ROWS = 40


def benchmark_rows(payload):
    import pyarrow.parquet as pq

    if hashlib.sha256(payload).hexdigest() != SOURCE_SHA256:
        raise ValueError("AMC23 source checksum mismatch")
    source = pq.read_table(io.BytesIO(payload)).to_pylist()
    if len(source) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} AMC23 questions, got {len(source)}")
    if len({row["id"] for row in source}) != EXPECTED_ROWS or len({row["question"] for row in source}) != EXPECTED_ROWS:
        raise ValueError("Duplicate AMC23 source IDs or questions")
    result = []
    for index, row in enumerate(source):
        problem = row["question"]
        answer = float(row["answer"])
        if not problem.strip() or not math.isfinite(answer) or not answer.is_integer():
            raise ValueError(f"Invalid AMC23 question or answer at source ID {row['id']}")
        result.append(dict(benchmark="amc23", benchmark_index=index, source_id=str(row["id"]),
                           source_dataset=SOURCE_DATASET, source_revision=SOURCE_REVISION,
                           problem=problem, answer=str(int(answer))))
    return result


def append_benchmark(output, additions):
    original = output.read_bytes()
    rows = [json.loads(line) for line in original.decode("utf-8").splitlines() if line.strip()]
    existing = [row for row in rows if row.get("benchmark") == "amc23"]
    if existing:
        if existing != additions:
            raise ValueError("Existing AMC23 records differ from the pinned source; refusing to overwrite")
        return False
    suffix = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in additions).encode("utf-8")
    separator = b"" if not original or original.endswith(b"\n") else b"\n"
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_bytes(original + separator + suffix)
    temporary.replace(output)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/benchmarks.jsonl"))
    parser.add_argument("--source-parquet", type=Path, help="Use a downloaded source file with the pinned checksum")
    args = parser.parse_args()
    args.output = project_absolute(args.output)
    if args.source_parquet:
        args.source_parquet = project_absolute(args.source_parquet)
    if not args.output.is_file():
        parser.error(f"Existing benchmark suite not found: {args.output}")
    if args.source_parquet:
        payload = args.source_parquet.read_bytes()
    else:
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
            payload = response.read()
    additions = benchmark_rows(payload)
    changed = append_benchmark(args.output, additions)
    print(json.dumps(dict(output=project_relative(args.output), benchmark="amc23", questions=len(additions),
                          appended=changed, source_dataset=SOURCE_DATASET, source_revision=SOURCE_REVISION)))


if __name__ == "__main__":
    main()
