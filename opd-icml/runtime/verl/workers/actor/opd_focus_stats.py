"""Concentration/allocation statistics; zeros are missing energy, not uniformity."""
import numpy as np


def concentration(values):
    x=np.asarray(values,dtype=np.float64).reshape(-1)
    if not np.isfinite(x).all() or (x < -1e-8).any():
        raise ValueError('concentration requires finite nonnegative masses')
    x=x.clip(0); n=len(x); total=float(x.sum())
    result=dict(count=n,total=total,available=bool(n and total>1e-30))
    if not result['available']:
        return {**result,'gini':None,'effective_fraction':None,'entropy_effective_fraction':None,
                'relative_hhi':None,'top_shares':{}}
    p=x/total; ordered=np.sort(p); hhi=float(p@p)
    entropy=float(-(p[p>0]*np.log(p[p>0])).sum())
    result.update(gini=float((2*np.arange(1,n+1)-n-1)@ordered/n),
                  effective_fraction=1/(n*hhi), entropy_effective_fraction=float(np.exp(entropy)/n),
                  relative_hhi=n*hhi,
                  top_shares={str(f):float(ordered[-max(1,int(np.ceil(f*n))):].sum()) for f in (.01,.05,.1,.2)})
    return result


def allocation(signal, change):
    """Freeze PRE-update ranking; never rank by realized benefit. Signed changes."""
    x=np.asarray(signal,dtype=float).reshape(-1); y=np.asarray(change,dtype=float).reshape(-1)
    if x.shape!=y.shape or not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('invalid aligned signal/response')
    order=np.argsort(x,kind='stable')
    positive=y.clip(0); negative=(-y).clip(0)
    result=dict(count=len(x), signal_min=float(x.min()),signal_max=float(x.max()),
                tied_signal_fraction=float(1-len(np.unique(x))/len(x)),
                signal_constant=bool(np.ptp(x)<=1e-12), net=float(y.sum()),
                positive_total=float(positive.sum()),negative_total=float(negative.sum()),top={})
    for fraction in (.01,.05,.1,.2):
        ix=order[-max(1,int(np.ceil(fraction*len(x)))):]
        result['top'][str(fraction)]=dict(count=len(ix),net=float(y[ix].sum()),
            positive_share=float(positive[ix].sum()/positive.sum()) if positive.sum()>1e-30 else None,
            negative_share=float(negative[ix].sum()/negative.sum()) if negative.sum()>1e-30 else None,
            enrichment=(float(positive[ix].sum()/positive.sum())/(len(ix)/len(x))) if positive.sum()>1e-30 else None)
    edges=(-np.inf,1e-4,1e-3,1e-2,.1,1.,np.inf)
    result['absolute_signal_bins']=[]
    for lo,hi in zip(edges,edges[1:]):
        mask=(x>=lo)&(x<hi)
        result['absolute_signal_bins'].append(dict(lower=None if not np.isfinite(lo) else lo,
            upper=None if not np.isfinite(hi) else hi,count=int(mask.sum()),
            net=float(y[mask].sum()),positive=float(positive[mask].sum()),negative=float(negative[mask].sum())))
    return result


def distribution_pair(signal,response):
    """Normalized shape alignment, not an estimate of causality or usefulness."""
    x=np.asarray(signal,dtype=float).clip(0).reshape(-1);y=np.asarray(response,dtype=float).clip(0).reshape(-1)
    if x.shape!=y.shape:raise ValueError('unaligned allocations')
    if x.sum()<=1e-30 or y.sum()<=1e-30:return dict(available=False)
    p=x/x.sum();q=y/y.sum()
    return dict(available=True,total_variation=float(np.abs(p-q).sum()/2),
                bhattacharyya=float(np.sqrt(p*q).sum()))
