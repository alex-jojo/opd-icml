"""Shared spectral summaries; legacy score-based diagnostics were removed."""
import numpy as np
from .opd_focus_stats import concentration


def spectrum(matrix):
    eigen, q = np.linalg.eigh((matrix+matrix.T)*.5)
    eigen = eigen[::-1].clip(0); q = q[:, ::-1]
    total = float(eigen.sum())
    stats = concentration(eigen)
    stats['top_modes'] = {str(k): float(eigen[:k].sum()/total) if total > 1e-30 else None
                          for k in (1, 4, 8, 16, 32, 64, 128) if k <= len(eigen)}
    stats['effective_rank'] = (stats['entropy_effective_fraction']*len(eigen)
                               if stats['available'] else None)
    stats['stable_rank']=float(total/eigen[0]) if total>1e-30 else None
    stats['participation_rank']=float(total**2/(eigen@eigen)) if total>1e-30 else None
    singular=np.sqrt(eigen)
    ps=singular/singular.sum() if singular.sum()>1e-30 else singular
    stats['singular_entropy_rank']=float(np.exp(-(ps[ps>0]*np.log(ps[ps>0])).sum())) if total>1e-30 else None
    stats['effective_rank_definition']='entropy rank of energy eigenvalues; singular_entropy_rank uses singular values'
    stats['relative_eigengaps'] = {str(k):float((eigen[k-1]-eigen[k])/max(float(eigen[0]),1e-30))
                                  for k in (4,16,32) if k<len(eigen)}
    return eigen, q, stats
