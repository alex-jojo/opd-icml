from pathlib import Path
import csv, json
import numpy as np

import argparse
ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from project_paths import project_absolute
parser = argparse.ArgumentParser(description="Build the parameter appendix from two completed paired FULL runs.")
parser.add_argument("--run-17", type=Path, default=ROOT/"runs/1.7b/paired-full")
parser.add_argument("--run-4", type=Path, default=ROOT/"runs/4b/paired-full")
parser.add_argument("--output", type=Path, default=ROOT/"outputs/parameter_appendix")
args = parser.parse_args()
for name in ('run_17', 'run_4', 'output'):
    setattr(args, name, project_absolute(getattr(args, name)))
P = args.output
for name in ["tables", "evidence"]:
    (P/name).mkdir(parents=True, exist_ok=True)
roots = {"1.7B": args.run_17, "4B": args.run_4}
data={};cross={};spectra=[];rawrows=[];crossrows=[]
for model,root in roots.items():
    data[model]={};cross[model]={}
    for t in range(0,101,10):
        d=root/f'replication/gradient_study/checkpoint_{t:03}'
        v=json.loads((d/'validation_transfer.json').read_text());g=v['groups'];data[model][t]=g
        assert np.isclose(g['all']['update_norm']**2,g['backbone']['update_norm']**2+g['shared_embedding_head']['update_norm']**2,rtol=1e-9)
        for name,r in g.items():
            rawrows.append(dict(model=model,checkpoint=t,group=name,**r,
                update_squared_norm_share=r['update_norm']**2/g['all']['update_norm']**2))
        if t not in [10,30,60,100]:continue
        groups=json.loads((d/'parameter_cross_batch.json').read_text())['groups'];cross[model][t]=groups
        for name,r in groups.items():
            k=np.array(r['gram']);ev=np.linalg.eigvalsh(k)[::-1]
            assert ev.min()>-1e-10*ev.max()
            ev=ev.clip(0);q=ev/ev.sum();den=np.sqrt(np.outer(np.diag(k),np.diag(k)))
            corr=k/den;ec=np.linalg.eigvalsh(corr).clip(0);qc=ec/ec.sum()
            entry=dict(model=model,checkpoint=t,group=name,
                eigenvalue_shares=q.tolist(),participation_rank=float(1/(q*q).sum()),
                unit_norm_participation_rank=float(1/(qc*qc).sum()),
                mean_cosine=r['mean_cosine'],negative_fraction=r['negative_fraction'],
                mean_gradient_energy=r['mean_gradient_energy'],variance=r['variance'])
            crossrows.append(entry)
            if name=='all':spectra.append(entry)

def start(caption,label,cols,header,size=r'\small',sep=5):
    return [r'\begin{table}[!htbp]',r'\centering',r'\caption{'+caption+'}',r'\label{tab:'+label+'}',
        size+rf'\setlength{{\tabcolsep}}{{{sep}pt}}',r'\begin{tabular}{'+cols+'}',r'\toprule',header+r' \\',r'\midrule']
def finish(name,lines):
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (P/'tables'/f'{name}.tex').write_text('\n'.join(lines)+'\n',encoding='utf8')

lines=start(r'Parameter-scale measurements at all FULL update diagnostics. $g_T$ is the recorded training gradient at the optimizer boundary; $c_{T,\Delta}=\cos(-g_T,\Delta\theta)$ and $c_{T,V}=\cos(g_T,g_V)$. BB denotes backbone; SH denotes the shared embedding/output matrix. SH share is its fraction of the full squared update norm. Shared weights are counted once. Norms are absolute, with no parameter-count normalization.',
    'parameter_update_groups','llrrrrrrr',r'Model & Step & $\|g_T\|$ & $\|\Delta\theta\|$ & BB norm & SH norm & SH (\%) & $c_{T,\Delta}$ & $c_{T,V}$')
for model,steps in data.items():
    if model=='4B':lines.append(r'\midrule')
    for i,(t,g) in enumerate(steps.items()):
        a,b,s=g['all'],g['backbone'],g['shared_embedding_head']
        lab=r'\multirow{11}{*}{'+model+'}' if i==0 else ''
        vals=[a['train_norm'],a['update_norm'],b['update_norm'],s['update_norm'],100*s['update_norm']**2/a['update_norm']**2,a['train_descent_update_cosine'],a['train_validation_cosine']]
        lines.append(f'{lab} & {t} & '+' & '.join(f'{v:.5f}' for v in vals)+r' \\')
finish('parameter_update_groups',lines)

for model,steps in data.items():
    short='17' if model=='1.7B' else '4b'
    lines=start(model+r' FULL: every Transformer layer at the four checkpoints with complete cross-batch measurements. Update columns are $10^3\|\Delta\theta_\ell\|$; drop is the percentage decrease from step 10 to 100. Share uses $\|\Delta\theta_\ell\|^2/\|\Delta\theta\|^2$ at step 100. $g_{T,\ell}$ is the recorded training gradient; $c_{B,\ell}$ averages six cross-subset parameter-gradient cosines. Layer indices start at zero. These absolute group norms reflect both coordinate count and update magnitude.',
        'parameter_layers_'+short,'rrrrrrrrrrr',
        r'Layer & $\Delta_{10}$ & $\Delta_{30}$ & $\Delta_{60}$ & $\Delta_{100}$ & Drop (\%) & Share (\%) & $g_{T,10}$ & $g_{T,100}$ & $c_{B,10}$ & $c_{B,100}$',r'\fontsize{8.5}{10.5}\selectfont',4)
    layers=sorted((k for k in steps[10] if k.startswith('layer_')),key=lambda x:int(x.split('_')[1]))
    for layer in layers:
        u=[steps[t][layer]['update_norm'] for t in [10,30,60,100]]
        vals=[*(1000*x for x in u),100*(1-u[-1]/u[0]),100*u[-1]**2/steps[100]['all']['update_norm']**2,
            steps[10][layer]['train_norm'],steps[100][layer]['train_norm'],cross[model][10][layer]['mean_cosine'],cross[model][100][layer]['mean_cosine']]
        lines.append(layer.split('_')[1]+' & '+' & '.join(f'{v:.4f}' for v in vals)+r' \\')
    finish('parameter_layers_'+short,lines)

lines=start(r'Spectrum of four sampled parameter gradients in FULL, recomputed from their exact $4\times4$ Gram matrix. Columns $p_1$--$p_4$ are descending eigenvalue energy shares. $r_{\mathrm{PR}}=1/\sum_i p_i^2$ measures concentration across these four vectors. The unit-norm column repeats the calculation after normalizing each gradient, reducing sensitivity to differences in gradient magnitude. Both dimensions are bounded by four and describe this sampled gradient panel.',
    'parameter_gradient_spectrum','llrrrrrr',r'Model & Step & $p_1$ (\%) & $p_2$ (\%) & $p_3$ (\%) & $p_4$ (\%) & $r_{\mathrm{PR}}$ & Unit-norm $r_{\mathrm{PR}}$')
for i,r in enumerate(spectra):
    if i==4:lines.append(r'\midrule')
    lab=r'\multirow{4}{*}{'+r['model']+'}' if i%4==0 else ''
    vals=[*(100*x for x in r['eigenvalue_shares']),r['participation_rank'],r['unit_norm_participation_rank']]
    lines.append(f"{lab} & {r['checkpoint']} & "+' & '.join(f'{v:.4f}' for v in vals)+r' \\')
finish('parameter_gradient_spectrum',lines)

with (P/'evidence/parameter_group_dynamics_all.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(rawrows[0]));w.writeheader();w.writerows(rawrows)
flat=[]
for r in crossrows:
    c={k:v for k,v in r.items() if k!='eigenvalue_shares'}
    c.update({f'eigenvalue_share_{i+1}':v for i,v in enumerate(r['eigenvalue_shares'])});flat.append(c)
with (P/'evidence/parameter_gradient_spectra_all.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
(P/'evidence/parameter_appendix_audit.json').write_text(json.dumps(dict(spectra=spectra,group_rows=len(rawrows),gradient_group_rows=len(crossrows),
    sources={k:str(v) for k,v in roots.items()},shared_weights_counted_once=True,
    actual_weight_update_matrix_spectrum_available=False,coordinate_change_counts_available=False,
    cross_time_update_vectors_available=False),indent=2,ensure_ascii=False),encoding='utf8')
print(json.dumps({'group_rows':len(rawrows),'gradient_group_rows':len(crossrows),'spectra':spectra},indent=2))
