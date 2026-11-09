"""Full-rank, rotation-invariant stability check for cumulative hidden PCA.

This observer never changes gradients, optimizer state or the sampling RNG.
All thresholds are predeclared and a budget cap is NOT a stability decision.
"""
import json
from pathlib import Path
import numpy as np
import torch
from .opd_plasticity import atomic_json, atomic_npz

DEFAULTS = dict(interval=5, min_step=15, window=3, adjacent_overlap=.98,
                window_overlap=.97, p10_principal_cos2=.90, max_energy_gap=.005)


def compare(previous, current, second):
    """Full-space mean, lower-decile/worst cos² and energy loss, not axis cosine."""
    from .opd_distributed import primary_call
    return primary_call(lambda:_compare_local(previous,current,second))


def _compare_local(previous,current,second):
    cross=previous.double().T @ current.double()
    cos2=torch.linalg.svdvals(cross).square().clamp(0,1)
    c=second.double();den=torch.trace(c).clamp_min(1e-30)
    old=torch.trace(previous.double().T@c@previous.double())/den
    new=torch.trace(current.double().T@c@current.double())/den
    return dict(overlap=float(cos2.mean()), min_cos2=float(cos2.min()),
                p10_cos2=float(torch.quantile(cos2,.1)),energy_gap=max(0.,float(new-old)))


class Stability:
    def __init__(self, output, config):
        self.output=Path(output)/'stability';self.output.mkdir(parents=True,exist_ok=True)
        self.cfg={**DEFAULTS,**config};self.history=[];self.initialized=False;self.stable=False
        if self.cfg['window']<3 or self.cfg['interval']<1:raise ValueError('stability needs >=3 observations')
        for k in ('adjacent_overlap','window_overlap','p10_principal_cos2'):
            if not 0<self.cfg[k]<=1:raise ValueError('invalid stability threshold: '+k)
        if self.cfg['max_energy_gap']<0:raise ValueError('negative energy tolerance')

    def resume_before(self, step):
        if self.initialized:return
        # Files written after the restored checkpoint are observations of an
        # uncommitted future. Never use them to trigger an earlier stop.
        for path in sorted(self.output.glob('basis_*.npz')):
            s=int(path.stem.split('_')[-1])
            if s>=step:continue
            with np.load(path,allow_pickle=False) as f:u=torch.from_numpy(f['basis'].copy())
            self.history.append((s,u))
        self.history=self.history[-(self.cfg['window']-1):]
        self.initialized=True
        atomic_json(self.output/'resume.json',dict(next_step=step,accepted_steps=[s for s,_ in self.history]))

    def observe(self, step, u, second):
        if not self.initialized:self.resume_before(step)
        history=(self.history+[(step,u.cpu())])[-self.cfg['window']:]
        adjacent=[]
        for (_,a),(_,b) in zip(history,history[1:]):
            adjacent.append(compare(a,b,second))
        anchor=compare(history[0][1],u,second)
        ready=len(history)==self.cfg['window'] and step>=self.cfg['min_step']
        consecutive=all(b[0]-a[0]==self.cfg['interval'] for a,b in zip(history,history[1:]))
        self.stable=bool(ready and consecutive and
            all(p['overlap']>=self.cfg['adjacent_overlap'] and p['p10_cos2']>=self.cfg['p10_principal_cos2'] for p in adjacent) and
            anchor['overlap']>=self.cfg['window_overlap'] and
            anchor['p10_cos2']>=self.cfg['p10_principal_cos2'] and
            anchor['energy_gap']<=self.cfg['max_energy_gap'])
        record=dict(step=step,stable=self.stable,config=self.cfg,steps=[s for s,_ in history],
                    adjacent=adjacent,anchor=anchor,ready=ready,consecutive=consecutive,
                    decision='stable' if self.stable else 'continue')
        atomic_npz(self.output/f'basis_{step:03d}.npz',dict(basis=u.cpu().numpy()))
        atomic_json(self.output/f'step_{step:03d}.json',record)
        atomic_json(self.output/'latest.json',record)
        self.history=history[-(self.cfg['window']-1):]
        from .opd_distributed import primary
        if primary():print(f'[U128 stability] step={step} window={record["steps"]} '
            f'overlap={anchor["overlap"]:.5f} worst_cos2={anchor["min_cos2"]:.5f} '
            f'energy_gap={anchor["energy_gap"]:.5f} stable={self.stable}',flush=True)
        return record
