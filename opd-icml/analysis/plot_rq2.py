from pathlib import Path
import re, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/"outputs"
(PAPER/"figures").mkdir(parents=True,exist_ok=True)
data=json.loads((ROOT/"data/rq2.json").read_text(encoding="utf8"))
for model in data.values():
    for key, series in model.items(): model[key]={int(t): v for t,v in series.items()}
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 10, 'axes.titlesize': 11,
    'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 10, 'axes.linewidth': .65, 'lines.linewidth': 1.35,
    'grid.color': '#dedede', 'grid.linewidth': .45, 'grid.alpha': .7,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})
COLORS = ['#426A92', '#E68A45']
MARKERS = ['o', 's']

def style(axis):
    axis.grid(True, linestyle=':', zorder=0)
    axis.tick_params(width=.6, length=3)
    axis.set_axisbelow(True)

def trajectory(axis, x, y, model, index):
    axis.plot(x, y, color=COLORS[index], marker=MARKERS[index], markersize=3,
              markevery=3, label=model)
    axis.set_xlim(0, 108)
    axis.set_xticks([0, 20, 40, 60, 80, 100])
    axis.set_xlabel('Training step')

def save(figure, name, legend=True):
    if legend:
        handles, labels = figure.axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, 1.0),
                      ncol=2, frameon=True, edgecolor='#888888', fancybox=False,
                      borderpad=.3, handlelength=2.1)
    figure.savefig(PAPER / 'figures' / f'{name}.pdf', bbox_inches='tight', pad_inches=.04)
    figure.savefig(PAPER / 'figures' / f'{name}.png', dpi=220, bbox_inches='tight', pad_inches=.04)
    plt.close(figure)

fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.55))
fig.subplots_adjust(left=.07, right=.99, bottom=.23, top=.72, wspace=.39)
for index, (model, values) in enumerate(data.items()):
    motion = values['motion']
    steps = sorted(motion)
    trajectory(axes[0], steps, [float(motion[t]['单位hidden位移RMS']) for t in steps], model, index)
    ranks = [int(re.search(r'r(\d+)', key)[1]) for key in motion[106] if key.startswith('r')]
    axes[1].plot(ranks, [float(motion[106][f'r{r}覆盖%']) for r in ranks], color=COLORS[index], marker=MARKERS[index], markersize=3)
    fixed = [t for t in steps if t >= 10]
    trajectory(axes[2], fixed, [float(motion[t]['观察U覆盖比例']) * 100 for t in fixed], model, index)
axes[0].set(title='(a) Change magnitude', ylabel='Unit-hidden RMS', ylim=(0, .56))
axes[1].set(title='(b) Change spectrum', xlabel='Leading directions $r$', ylabel='Energy covered (%)', ylim=(0, 103), xscale='log', xlim=(1, 3000))
axes[1].set_xticks([1, 16, 128, 1024], ['1', '16', '128', '1024'])
axes[1].minorticks_off()
axes[2].set(title='(c) Early-basis coverage', ylabel='Change covered (%)', ylim=(70, 100))
for axis in axes: style(axis)
save(fig, 'rq2_adaptation')

fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.55))
fig.subplots_adjust(left=.07, right=.99, bottom=.23, top=.72, wspace=.41)
for index, (model, values) in enumerate(data.items()):
    steps = sorted(values['full'])
    trajectory(axes[0], steps, [float(values['full'][t]['rms']) for t in steps], model, index)
    trajectory(axes[1], steps, [float(values['full'][t]['effective_rank']) for t in steps], model, index)
    spectrum = values['spectra'][106]
    ranks = list(map(int, spectrum))
    axes[2].plot(ranks, np.array(list(spectrum.values())) * 100, color=COLORS[index], marker=MARKERS[index], markersize=3)
axes[0].set(title='(a) Hidden-gradient strength', ylabel='Gradient RMS', ylim=(0, .8))
axes[1].set(title='(b) Gradient dimensionality', ylabel='Energy effective rank', ylim=(0, 450))
axes[2].set(title='(c) Gradient spectrum', xlabel='Leading directions $r$', ylabel='Energy covered (%)', ylim=(0, 103), xscale='log', xlim=(1, 3000))
axes[2].set_xticks([1, 16, 128, 1024], ['1', '16', '128', '1024'])
axes[2].minorticks_off()
for axis in axes: style(axis)
save(fig, 'rq2_gradient_dynamics')

fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.65), sharex=True)
fig.subplots_adjust(left=.31, right=.99, bottom=.2, top=.84, wspace=.18)
for index, (model, values) in enumerate(data.items()):
    axis = axes[index]
    shares = [float(values['motion'][106]['观察U覆盖比例']) * 100,
              values['coverage'][106] * 100, values['spectra'][106]['128'] * 100]
    axis.barh([2, 1, 0], shares, height=.48, color=COLORS[index], edgecolor='white', linewidth=.5)
    axis.set_yticks([2, 1, 0], ['Representation change\nin early directions', 'Gradient energy\nin early directions', 'Gradient energy in its\nown top 128 directions'])
    axis.set(xlim=(0, 104), xlabel='Energy covered (%)', title=f'({chr(97 + index)}) {model} at step 106')
    axis.set_xticks([0, 25, 50, 75, 100])
    if index == 1:
        axis.tick_params(axis='y', labelleft=False)
    style(axis)
    axis.grid(False, axis='y')
    for y, share in zip([2, 1, 0], shares):
        axis.text(share + 1.3, y, f'{share:.2f}', va='center', fontsize=9)
save(fig, 'rq2_gradient_coverage', legend=False)

