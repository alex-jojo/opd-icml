"""Create figures from the paired FULL observations and RQ4 benchmark summaries."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from project_paths import project_absolute
FIGURES = ROOT / "outputs/figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def rows(path):
    with path.open(encoding="utf8") as source:
        return list(csv.DictReader(source))


models = [("1.7B", "#426A92", "o"), ("4B", "#E68A45", "s")]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--rq1", type=Path, default=ROOT / "data/rq1.csv")
parser.add_argument("--rq4", type=Path, default=ROOT / "data/rq4.csv")
args = parser.parse_args()
args.rq1, args.rq4 = project_absolute(args.rq1), project_absolute(args.rq4)
signal = rows(args.rq1)
fig, axes = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
for ax, key, title in zip(axes, ["train_kl", "above_initial_kl_threshold_pct", "top_kl_decile_hidden_energy_pct"],
                          ["Training KL", "Above initial KL threshold (%)", "Top-decile hidden energy (%)"]):
    for model, color, marker in models:
        selected = [r for r in signal if r["model"] == model and r[key]]
        ax.plot([int(r["step"]) for r in selected], [float(r[key]) for r in selected],
                color=color, label=model, lw=1.3, marker=marker, ms=2, markevery=max(1,len(selected)//12))
    ax.set(xlabel="Training step", title=title)
    ax.grid(alpha=.25)
axes[0].legend()
fig.savefig(FIGURES / "rq1_signal.pdf")
plt.close(fig)

if any(r.get('top_kl_decile_parameter_energy_pct') for r in signal):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), constrained_layout=True)
    for ax, (model, color, marker) in zip(axes, models):
        for key, label, style in [('top_kl_decile_hidden_energy_pct', 'Hidden state', '--'),
                                  ('top_kl_decile_parameter_energy_pct', 'Full parameters', '-')]:
            selected = [r for r in signal if r['model'] == model and r.get(key)]
            ax.plot([int(r['step']) for r in selected], [float(r[key]) for r in selected],
                    color=color, linestyle=style, label=label, marker=marker, ms=2)
        ax.set(xlabel='Training step', ylabel='Top-KL decile energy (%)', title=model, ylim=(0, 100))
        ax.legend(); ax.grid(alpha=.25)
    fig.savefig(FIGURES / 'rq1_parameter_energy.pdf')
    plt.close(fig)

if any(r.get('parameter_grad_norm') for r in signal):
    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
    for model, color, marker in models:
        selected = [r for r in signal if r['model'] == model and r.get('parameter_grad_norm')]
        ax.plot([int(r['step']) for r in selected], [float(r['parameter_grad_norm']) for r in selected],
                color=color, label=model)
    ax.set(xlabel='Training step', ylabel='Parameter gradient norm (before clipping)')
    ax.legend(); ax.grid(alpha=.25)
    fig.savefig(FIGURES / 'rq1_gradient_norm.pdf')
    plt.close(fig)

costs = rows(args.rq4)
fig, axes = plt.subplots(1, 2, figsize=(7, 3), constrained_layout=True)
for ax, (model, color, marker) in zip(axes, models):
    for r in (r for r in costs if r["model"] == model):
        x, y = float(r["minutes"]), float(r["avg"])
        ax.scatter(x, y, color=color, marker=marker)
        ax.annotate(r["method"], (x,y), xytext=(0,7), textcoords="offset points", ha="center", fontsize=8)
    ax.margins(x=.4, y=.3)
    ax.set(xlabel="Training time (min)", ylabel="Five-benchmark average (%)", title=model)
    ax.grid(alpha=.25)
fig.savefig(FIGURES / "rq4_efficiency.pdf")
plt.close(fig)
