"""Exact KL quantile group spectra from existing backward tensors, no replay.

Transient rank-local CPU G is freed each measured update; never serialized.
Group boundaries are GLOBAL token quantiles, not per-microbatch quantiles.
"""
import numpy as np
import torch
from . import opd_distributed as dp
from .opd_gradient_scopes import summarize


def partitions(kl):
    order=np.argsort(np.asarray(kl),kind='stable');k=max(1,int(np.ceil(len(order)*.1)))
    if 2*k>=len(order):raise ValueError('High/middle/low spectrum requires at least three tokens')
    return {'low':order[:k],'middle':order[k:-k],'high':order[-k:]}


def moment_scopes(second,basis,count):
    c=torch.as_tensor(second,dtype=torch.float64)
    u=torch.linalg.qr(torch.as_tensor(basis,dtype=torch.float64),mode='reduced').Q
    cu=c@u;inside=u.T@cu;outside=c-cu@u.T-u@cu.T+u@inside@u.T
    report={};arrays={}
    for name,m,d in [('full',c,len(c)),('u',inside,u.shape[1]),('outside',outside,len(c)-u.shape[1])]:
        e=torch.linalg.eigvalsh((m+m.T)/2).flip(0)[:d].clamp_min(0).numpy()
        arrays[name+'_eigenvalues']=e;report[name]=summarize(e,count)
    closure=abs(report['full']['energy_per_token']-report['u']['energy_per_token']-report['outside']['energy_per_token'])
    if closure>3e-5*max(report['full']['energy_per_token'],1e-30):raise ValueError('Group energy closure failed')
    return report,arrays


def compute(local_gradient_parts,global_kl,basis,device):
    """Local parts retain original rank/part order, matching merged token fields."""
    from .opd_gradient_study import HiddenMoments
    n=sum(len(x) for x in local_gradient_parts);counts=dp.gather(n);offset=sum(counts[:dp.rank()])
    if sum(counts)!=len(global_kl):raise ValueError('Group gradient/token alignment mismatch')
    groups=partitions(global_kl);matched=min(map(len,groups.values()));report={};arrays={}
    for name,indices in groups.items():
        picks=indices[np.linspace(0,len(indices)-1,matched).round().astype(int)]
        entry=dict(count=len(indices),fraction=len(indices)/len(global_kl),
            kl_min=float(np.min(global_kl[indices])),kl_max=float(np.max(global_kl[indices])),matched_count=matched)
        for kind,selected in [('all',indices),('matched',picks)]:
            if kind=='matched' and len(indices)==matched:
                entry[kind]=entry['all']
                for scope in ('full','u','outside'):
                    arrays[f'{name}_{kind}_{scope}_eigenvalues']=arrays[f'{name}_all_{scope}_eigenvalues']
                continue
            chosen=np.zeros(len(global_kl),bool);chosen[selected]=True;local=chosen[offset:offset+n]
            moment=HiddenMoments(basis);moment.add(torch.empty((0,basis.shape[0]),device=device))
            cursor=0
            for part in local_gradient_parts:
                # Bounded H2D copies; no second large CPU concatenation.
                for start in range(0,len(part),4096):
                    block=part[start:start+4096];mask=local[cursor+start:cursor+start+len(block)]
                    if mask.any():moment.add(torch.from_numpy(block[mask]).to(device))
                cursor+=len(part)
            saved=moment.finish()
            if int(saved['count'])!=len(selected):raise ValueError('Incomplete KL group')
            stats,spectra=dp.primary_call(lambda:moment_scopes(saved['second'],saved['basis'],len(selected)))
            entry[kind]=stats
            arrays.update({f'{name}_{kind}_{k}':v for k,v in spectra.items()})
        report[name]=entry
    return dict(definition='global current-batch KL bottom10%, middle80%, top10%; stable tie breaking',
        normalization='per-token original hidden loss gradients; G^T G/N_group',
        matched_definition='equal token counts, deterministic evenly spaced indices in KL-sorted groups',
        exact_all_tokens=True,groups=report,transformer_replays=0),arrays
