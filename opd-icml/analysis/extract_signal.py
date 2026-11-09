"""Export RQ1 token and loss measurements from paired FULL training output."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_run(path, model):
    records = sorted((path / "training_records").glob("post_update_*.json"))
    if not records:
        raise ValueError(f"No training records found in {path}")
    for record in records:
        batch = json.loads(record.read_text(encoding="utf8"))
        step = batch["step"]
        audit = path / f"replication/token_audit/step_{step:03}_inner_1.json"
        frequency = share = ""
        if audit.is_file():
            groups = json.loads(audit.read_text(encoding="utf8"))["groups"]
            frequency = 100 * groups["kl_above_step1_p90"]["fraction"]
            share = 100 * groups["kl_top10"]["quantities"]["raw_gradient_energy"]["share"]
        yield [step, model, batch["metrics"]["actor/opd_top16_kl"], frequency, share]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-17", type=Path, default=ROOT / "runs/1.7b/paired-full")
    parser.add_argument("--run-4", type=Path, default=ROOT / "runs/4b/paired-full")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/rq1.csv")
    args = parser.parse_args()
    rows = sorted([*read_run(args.run_17, "1.7B"), *read_run(args.run_4, "4B")])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step","model","train_kl","above_initial_kl_threshold_pct","top_kl_decile_hidden_energy_pct"])
        writer.writerows(rows)
