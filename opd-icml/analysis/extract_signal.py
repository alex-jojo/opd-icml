"""Export RQ1 token and loss measurements from paired FULL training output."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from project_paths import project_absolute
FIELDS = ["step", "model", "train_kl", "above_initial_kl_threshold_pct", "top_kl_decile_hidden_energy_pct",
          "top_kl_decile_parameter_energy_pct", "parameter_grad_norm", "hidden_gradient_rms",
          "parameter_gradient_rms", "hidden_gradient_frobenius", "parameter_gradient_frobenius",
          "above_initial_kl_parameter_energy_pct", "parameter_token_extra_backwards",
          "parameter_token_gradient_seconds", "parameter_token_mode", "parameter_token_sampled",
          "parameter_token_sample_count", "parameter_token_population_count",
          "top_kl_decile_parameter_energy_ci95_low_pct", "top_kl_decile_parameter_energy_ci95_high_pct",
          "step1_kl_p90", "hidden_token_sampled", "hidden_token_sample_count", "hidden_token_population_count",
          "top_kl_decile_hidden_energy_ci95_low_pct", "top_kl_decile_hidden_energy_ci95_high_pct"]


def read_run(path, model):
    records = sorted((path / "training_records").glob("post_update_*.json"))
    if not records:
        raise ValueError(f"No training records found in {path}")
    for record in records:
        batch = json.loads(record.read_text(encoding="utf8"))
        step = batch["step"]
        audit = record.with_name(f"rq1_step_{step:04d}.json")
        if not audit.is_file():
            audit = path / f"replication/token_audit/step_{step:03}_inner_1.json"
        frequency = share = ""
        parameter_share = fixed_share = ""
        report = {}
        if audit.is_file():
            report = json.loads(audit.read_text(encoding="utf8"))
            groups = report["groups"]
            frequency = 100 * groups["kl_above_step1_p90"]["fraction"]
            value = groups["kl_top10"]["quantities"]["raw_gradient_energy"]["share"]
            share = 100 * value if value is not None else ""
            value = groups["kl_top10"]["quantities"].get("parameter_gradient_energy", {}).get("share")
            parameter_share = 100 * value if value is not None else ""
            value = groups["kl_above_step1_p90"]["quantities"].get("parameter_gradient_energy", {}).get("share")
            fixed_share = 100 * value if value is not None else ""
        meta = report.get("parameter_gradient_measurement", {})
        interval = report.get('groups', {}).get('kl_top10', {}).get('quantities', {}).get(
            'parameter_gradient_energy', {}).get('share_ci95')
        hidden_meta = report.get('token_statistics_measurement', {})
        hidden_interval = report.get('groups', {}).get('kl_top10', {}).get('quantities', {}).get(
            'raw_gradient_energy', {}).get('share_ci95')
        yield [step, model, batch["metrics"]["actor/opd_top16_kl"], frequency, share,
               parameter_share, batch["metrics"].get("actor/grad_norm", ""),
               report.get("gradient_rms_full", ""), report.get("parameter_gradient_rms", ""),
               report.get("gradient_frobenius_full", ""), report.get("parameter_gradient_frobenius", ""),
               fixed_share, meta.get("extra_backwards", ""), meta.get("seconds", ""),
               meta.get('mode', ''), meta.get('sampled', ''), meta.get('token_count', ''),
               meta.get('population_token_count', meta.get('token_count', '')),
               100 * interval[0] if interval is not None else '',
               100 * interval[1] if interval is not None else '',
               report.get('thresholds', {}).get('kl_p90', batch["metrics"].get('token_audit/step1_kl_p90', '')),
               hidden_meta.get('estimated', False), hidden_meta.get('sample_count', ''),
               hidden_meta.get('population_token_count', ''),
               100 * hidden_interval[0] if hidden_interval is not None else '',
               100 * hidden_interval[1] if hidden_interval is not None else '']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-17", type=Path, default=ROOT / "runs/1.7b/paired-full")
    parser.add_argument("--run-4", type=Path, default=ROOT / "runs/4b/paired-full")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/rq1.csv")
    args = parser.parse_args()
    for name in ('run_17', 'run_4', 'output'):
        setattr(args, name, project_absolute(getattr(args, name)))
    rows = sorted([*read_run(args.run_17, "1.7B"), *read_run(args.run_4, "4B")])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        writer.writerows(rows)
