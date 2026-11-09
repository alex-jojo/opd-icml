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
data=json.loads((ROOT/"data/rq3.json").read_text(encoding="utf8"))
for model in data.values():
    for key, series in model.items(): model[key]={int(t): v for t,v in series.items()}
plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'], 'mathtext.fontset':'stix',
    'font.size':10,'axes.titlesize':11,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,
    'legend.fontsize':9,'axes.linewidth':.65,'lines.linewidth':1.35,'pdf.fonttype':42,'ps.fonttype':42})
COLORS=['#426A92','#E68A45']; MARKERS=['o','s']
def style(ax):
    ax.set_axisbelow(True); ax.grid(True,ls=':',lw=.5,color='#dedede')
    ax.tick_params(width=.6,length=3)
    ax.set_xlim(-2,109);ax.set_xticks([0,20,40,60,80,100]);ax.set_xlabel('Training step')
def models(fig):
    fig.legend([Line2D([],[],c=c,marker=m,ms=4) for c,m in zip(COLORS,MARKERS)],list(data),
        loc='upper center',bbox_to_anchor=(.5,1),ncol=2,fancybox=False,edgecolor='#999',borderpad=.3)
def save(fig,name):
    for ext in ['pdf','png']:
        fig.savefig(PAPER/'figures'/f'{name}.{ext}',bbox_inches='tight',pad_inches=.04,dpi=220)
    plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(7.05,2.6))
fig.subplots_adjust(left=.08,right=.99,bottom=.21,top=.76,wspace=.30)
for j,(model,d) in enumerate(data.items()):
    x=sorted(d['vals']); kl=[d['vals'][t][0] for t in x]; acc=[d['vals'][t][1] for t in x]
    axes[0].plot(x,kl,c=COLORS[j],marker=MARKERS[j],ms=3)
    axes[1].plot(x,acc,c=COLORS[j],alpha=.25,lw=1,marker=MARKERS[j],ms=3)
    smooth=[np.mean(acc[max(0,i-2):i+1]) for i in range(len(acc))]
    axes[1].plot(x,smooth,c=COLORS[j],marker=MARKERS[j],ms=3)
axes[0].set(title='(a) Validation KL loss',ylabel='Mean response KL',yscale='log')
axes[1].set(title='(b) Validation accuracy',ylabel='Correct answers (%)',ylim=(0,35))
for ax in axes:style(ax)
models(fig);save(fig,'rq3_validation')

fig,axes=plt.subplots(1,3,figsize=(7.05,2.65))
fig.subplots_adjust(left=.07,right=.99,bottom=.22,top=.75,wspace=.43)
for j,(model,d) in enumerate(data.items()):
    x=sorted(d['steps']); r=[d['steps'][t] for t in x]
    axes[j].plot(x,[float(v[2]) for v in r],c=COLORS[j],ls='--',marker='o',mfc='white',ms=3,label='Before update')
    axes[j].plot(x,[float(v[3]) for v in r],c=COLORS[j],ls='-',marker='s',ms=3,label='After update')
    axes[j].set(title=f'({chr(97+j)}) {model}: same responses',yscale='log',ylabel='Validation KL loss')
    axes[2].plot(x,[float(v[10])*100 for v in r],c=COLORS[j],marker=MARKERS[j],ms=3,label=model)
axes[2].set(title='(c) Responses with lower KL',ylabel='Responses (%)',ylim=(35,104))
for ax in axes:style(ax);ax.set_xlabel('Checkpoint before update')
handles=[Line2D([],[],c='#444',ls='--',marker='o',mfc='white',ms=4,label='Before update'),Line2D([],[],c='#444',ls='-',marker='s',ms=4,label='After update')]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.34,1),ncol=2,fancybox=False,edgecolor='#999',borderpad=.3)
axes[2].legend(loc='lower left',frameon=False,fontsize=8)
save(fig,'rq3_single_update')

fig,axes=plt.subplots(1,3,figsize=(7.05,2.75))
fig.subplots_adjust(left=.07,right=.99,bottom=.22,top=.74,wspace=.37)
for j,(model,d) in enumerate(data.items()):
    for ax,key in zip(axes[:2],['tv','batch']):
        x=sorted(d[key]);ax.plot(x,[d[key][t] for t in x],c=COLORS[j],marker=MARKERS[j],ms=3)
    x=sorted(d['hidden'])
    axes[2].plot(x,[d['hidden'][t] for t in x],c=COLORS[j],marker=MARKERS[j],ms=3)
    axes[2].plot(x,[d['overlap'][t] for t in x],c=COLORS[j],ls='--',marker=MARKERS[j],mfc='white',ms=3)
axes[0].set(title='(a) Training vs. validation',ylabel='Parameter-gradient cosine')
axes[1].set(title='(b) Different training subsets',ylabel='Parameter-gradient cosine')
axes[2].set(title='(c) Hidden-gradient structure',ylabel='Cosine / subspace overlap')
for ax in axes:style(ax);ax.set_ylim(-.04,1.04)
axes[2].legend([Line2D([],[],c='#444',ls='-'),Line2D([],[],c='#444',ls='--',marker='o',mfc='white',ms=3)],
    ['Mean directions','Top-4 axes'],loc='lower left',frameon=False,fontsize=8,handlelength=1.8)
models(fig);save(fig,'rq3_gradient_agreement')

