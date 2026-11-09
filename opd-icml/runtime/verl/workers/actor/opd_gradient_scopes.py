"""Read-only all-token spectra of full/U/complement final-hidden gradients."""
import time
import numpy as np
import torch


def summarize(e,count):
    e=np.maximum(np.asarray(e,dtype=np.float64),0);total=float(e.sum())
    s=np.sqrt(e);nuclear=float(s.sum())
    def rank(v):
        if v.sum()<=1e-30:return None
        p=v[v>0]/v.sum();return float(np.exp(-(p*np.log(p)).sum()))
    return dict(dimension=len(e),tokens=count,effective_rank=rank(e),singular_entropy_rank=rank(s),
        stable_rank=total/float(e[0]) if total>1e-30 else None,
        participation_rank=total**2/float(np.square(e).sum()) if total>1e-30 else None,
        nuclear_norm=nuclear*np.sqrt(count),nuclear_norm_token_normalized=nuclear,
        rms=np.sqrt(total),frobenius_norm=np.sqrt(count*total),energy_per_token=total,
        top_energy={str(k):float(e[:k].sum()/total) if total>1e-30 else None
                    for k in sorted(set([1,4,8,16,32,64,128,256,512,1024,2048,len(e)])) if k<=len(e)})


class GradientScopes:
    def __init__(self,basis):
        self.basis=basis.detach().float().cpu().numpy().copy()
        if self.basis.ndim!=2 or self.basis.shape[0]<=self.basis.shape[1]:raise ValueError('Invalid scope basis')
        self.gram=None;self.count=0;self.dispatch_seconds=0.

    @torch.no_grad()
    def add(self,g):
        started=time.perf_counter();g=g.detach().float()
        if g.ndim!=2 or g.shape[1]!=len(self.basis):raise ValueError('Gradient dimension mismatch')
        if self.gram is None:self.gram=torch.zeros((g.shape[1],g.shape[1]),device=g.device,dtype=torch.float32)
        with torch.autocast(device_type=g.device.type,enabled=False):
            for part in g.split(4096):self.gram.addmm_(part.T,part)
        self.count+=len(g);self.dispatch_seconds+=time.perf_counter()-started

    @torch.no_grad()
    def finish(self,expected):
        from .opd_distributed import sum_gram,primary_call
        self.gram,self.count=sum_gram(self.gram,self.count)
        if self.gram is None or self.count!=expected:raise ValueError('Incomplete all-token gradient statistic')
        result=primary_call(lambda:self._finish_local(expected))
        self.gram=None
        return result

    def _finish_local(self,expected):
        start=time.perf_counter()
        # Keep the SAME all-token Gram and FP64 statistics, but avoid the
        # NumPy/SciPy OpenBLAS dsyr2k/SKYLAKEX route that SIGSEGV'd at round5.
        # CPU tensors only: no additional GPU memory or transformer computation.
        c=self.gram.detach().to(device='cpu',dtype=torch.float64)/self.count;self.gram=None
        u=torch.linalg.qr(torch.from_numpy(self.basis).to(dtype=torch.float64),mode='reduced').Q
        cu=c@u;inside=u.T@cu;outside=c-cu@u.T-u@cu.T+u@inside@u.T
        arrays={};stats={}
        for name,m in [('full',c),('u',inside),('outside',outside)]:
            e=torch.linalg.eigvalsh(((m+m.T)*.5).contiguous()).flip(0)
            if float(e[-1])<-2e-5*max(float(e[0]),1e-30):raise ValueError('Non-PSD gradient moment: '+name)
            if name=='outside':e=e[:len(u)-u.shape[1]]
            e=e.clamp_min(0).numpy().copy();arrays[name+'_eigenvalues']=e;stats[name]=summarize(e,self.count)
        closure=abs(stats['full']['energy_per_token']-stats['u']['energy_per_token']-stats['outside']['energy_per_token'])
        if closure>2e-5*max(stats['full']['energy_per_token'],1e-30):raise ValueError('Full/U/outside energy mismatch')
        return dict(schema='full-u-out-hidden-gradient-v1',scopes=stats,
            distributed_world_size=torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1,
            definition='Original per-token final-hidden loss gradient before projection, batch mean and clipping',
            covariance='G^T G/N, uncentered, ALL valid training tokens',
            effective_rank_definition='exp entropy of normalized covariance eigenvalues',
            singular_entropy_rank_definition='exp entropy of normalized singular values',
            nuclear_definition='sum singular values of G; normalized nuclear is divided by sqrt(N)',
            gram_dtype='float32',eigen_dtype='float64',eigen_backend='torch_cpu_float64',observation_U_qr_only=True,
            extra_forwards=0,extra_backwards=0,extra_teacher_calls=0,token_subsampling=False,
            energy_closure_error=closure,gram_dispatch_seconds=self.dispatch_seconds,
            finalize_seconds=time.perf_counter()-start),arrays


def metric_values(report):
    return {f'gradient_scope/{name}/{k}':float(v) for name,s in report['scopes'].items()
        for k,v in s.items() if k in ('rms','frobenius_norm','nuclear_norm','nuclear_norm_token_normalized',
            'effective_rank','singular_entropy_rank','stable_rank','participation_rank') and v is not None}


def save(report,arrays,directory,step,inner,reuse,reference):
    from .opd_distributed import primary
    from .opd_plasticity import atomic_json,atomic_npz
    report.update(round=step,inner=inner,optimizer_update=(step-1)*reuse+inner,basis_reference=reference)
    path=directory/f'round_{step:03d}_inner_{inner}'
    atomic_npz(path.with_suffix('.npz'),arrays);atomic_json(path.with_suffix('.json'),report)
    for name,s in report['scopes'].items() if primary() else []:
        print(f'[gradient scope] round={step} inner={inner} scope={name} RMS={s["rms"]:.6g} '
            f'nuclear/sqrtN={s["nuclear_norm_token_normalized"]:.6g} effective_rank={s["effective_rank"]}',flush=True)
