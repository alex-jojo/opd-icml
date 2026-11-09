"""Ordinary FULL OPD with passive U128 mode demand/progress measurements.

No selected tokens, weighted loss, projected backward, A_t, candidate updates
or optimizer preview. Frozen-bank response is observational, not a causal
measurement of the benefit of independently accelerating a mode.
"""
from contextlib import contextmanager
import time
import numpy as np
import torch
import torch.nn.functional as F

from .opd_response_geometry import Observer, readout, fingerprint
from .opd_focus_stats import concentration
from .opd_priority_stats import spectrum
from .opd_top16 import reverse_kl
from .opd_plasticity import atomic_json, atomic_npz


@contextmanager
def observe_only(actor):
    # No gradients are computed, so do not copy gigabytes of parameter/grads.
    modes=[(m,m.training) for m in actor.actor_module.modules()]
    device=next(actor.actor_module.parameters()).device
    with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
        try:
            actor.actor_module.eval()
            yield
        finally:
            for m,flag in modes:m.training=flag


def mode_progress(before, after, basis, q):
    """Hold the *anchor* batch eigenbasis and teacher support fixed at all lags.

    C is the loss gradient on these states, D their observed representation
    response. No regression asserts that a held-out C caused an update.
    """
    u=basis.float().cpu(); q=torch.as_tensor(np.ascontiguousarray(q),dtype=torch.float32)
    c0=before['g_u']@q; ct=after['g_u']@q
    dh=after['h']-before['h']
    d=(dh@u)@q
    du=((F.normalize(after['h'],dim=-1)-F.normalize(before['h'],dim=-1))@u)@q
    e0=c0.square().mean(0).numpy(); et=ct.square().mean(0).numpy()
    cross=(c0*ct).mean(0).numpy()
    valid=e0>max(float(e0.max())*1e-8,1e-30)
    def ratio(numerator,denominator):
        return [float(a/b) if ok and b>1e-30 else None for a,b,ok in zip(numerator,denominator,valid)]
    row=before['zrow'];pos=before['zpos']
    # Pair sampled hidden positions to ALL-valid-token logits without assuming
    # every response has the same length or begins at position zero.
    lookup={(int(r),int(p)):i for i,(r,p) in enumerate(zip(row,pos))}
    ix=torch.tensor([lookup[(int(r),int(p))] for r,p in zip(before['row'],before['pos'])])
    p0=before['z'].softmax(-1); pt=after['z'].softmax(-1); teacher=before['q'].softmax(-1)
    tv0=.5*(p0-teacher).abs().sum(-1); tvt=.5*(pt-teacher).abs().sum(-1)
    kl0=reverse_kl(before['z'],before['q']); klt=reverse_kl(after['z'],before['q'])
    weights=c0.square(); denom=weights.sum(0).clamp_min(1e-30)
    def weighted(values):return ((weights*values[ix,None]).sum(0)/denom).tolist()
    result=dict(valid_initial_mode=valid.tolist(), initial_gradient_energy=e0.tolist(),
        remaining_gradient_energy=et.tolist(), remaining_energy_ratio=ratio(et,e0),
        signed_demand_persistence=ratio(cross,e0),
        demand_pattern_cosine=ratio(cross,np.sqrt(e0*et)),
        raw_response_energy=d.square().mean(0).tolist(), unit_response_energy=du.square().mean(0).tolist(),
        negative_initial_gradient_dot_raw_response=(-(c0*d).mean(0)).tolist(),
        gradient_change_energy=(ct-c0).square().mean(0).tolist(),
        mode_associated_tv_improvement=weighted(tv0-tvt),
        mode_associated_kl_improvement=weighted(kl0-klt),
        mode_associated_remaining_tv=weighted(tvt),
        teacher_kl_before=before['kl'],teacher_kl_after=after['kl'],
        tv_before=float(tv0.mean()),tv_after=float(tvt.mean()),
        teacher_top1_agreement_before=float((p0.argmax(-1)==teacher.argmax(-1)).float().mean()),
        teacher_top1_agreement_after=float((pt.argmax(-1)==teacher.argmax(-1)).float().mean()),
        student_entropy_before=float(-(p0*p0.clamp_min(1e-30).log()).sum(-1).mean()),
        student_entropy_after=float(-(pt*pt.clamp_min(1e-30).log()).sum(-1).mean()),
        policy_kl=float(reverse_kl(before['z'],after['z']).mean()),
        response_outside_u_fraction=float((dh.square().sum()-(dh@u).square().sum()).clamp_min(0)/dh.square().sum().clamp_min(1e-30)),
        scope='Mode-associated changes are overlapping weighted associations, NOT additive causal contributions. '
              'A smaller gradient is NOT proof of absorption; inspect remaining disagreement and entropy. '
              'Only first update uses the anchor training trajectories; later updates use fresh batches.')
    arrays=dict(g_u=after['g_u'].numpy(),h_u=(after['h']@u).numpy(),
                z=after['z'].numpy(),raw_response_modes=d.numpy(),unit_response_modes=du.numpy())
    return result,arrays


class ModeObserver(Observer):
    def before_update(self):
        start=time.perf_counter()
        fields={k:torch.cat(v).numpy() for k,v in self.token_fields.items()}
        z=torch.cat(self.raw_z).numpy();self.raw_z.clear()
        if len(z)!=int(self.train.batch['response_mask'].sum()) or any(len(v)!=len(z) for v in fields.values()):
            raise RuntimeError('mode observation token alignment mismatch')
        matrix=(z.T@z/len(z)).astype(np.float64)
        eigen,q,stats=spectrum(matrix)
        coef=z@q.astype(np.float32)
        mean=coef.mean(0,dtype=np.float64)
        coherence=np.divide(mean*mean,eigen,out=np.zeros_like(eigen),where=eigen>1e-30)
        boundary=fields['boundary'].astype(bool); energy=np.square(coef)
        mode_energy=energy.sum(0,dtype=np.float64)
        boundary_share=np.divide(energy[boundary].sum(0,dtype=np.float64),mode_energy,
                                 out=np.zeros_like(mode_energy),where=mode_energy>1e-30)
        report=dict(step=self.step,training='FULL OPD; all training tokens retain original weights',
            mode_spectrum=stats,fixed_u_axis_energy=concentration(np.diag(matrix)),
            raw_gradient_energy=concentration(fields['raw_gradient_energy']),
            u_gradient_energy=concentration(np.square(z).sum(1)),
            signed_hidden_mode_coherence=coherence.clip(0,1).tolist(),
            boundary_mode_energy_share=boundary_share.tolist(),
            cross_batch_comparison='Different states: not per-state absorption or convergence')
        previous=getattr(self,'previous_matrix',None)
        if previous is not None:
            report['previous_batch_matrix_cosine']=float(np.sum(previous*matrix)/max(np.linalg.norm(previous)*np.linalg.norm(matrix),1e-30))
        self.previous_matrix=matrix
        atomic_json(self.directory/f'modes_{self.step:03d}.json',report)
        atomic_npz(self.directory/f'step_{self.step:03d}.npz',dict(step=np.asarray(self.step),
            gradient_second=matrix,eigenvalues=eigen,Q=q,
            **{'token_'+k:v for k,v in fields.items()}))
        self.metrics['modes/u_gradient_top10_share']=report['u_gradient_energy']['top_shares'].get('0.1',0.)
        if stats['effective_rank'] is not None:self.metrics['modes/effective_rank']=stats['effective_rank']
        if self.step in self.anchors:
            if self.pending is not None:raise RuntimeError('unfinished frozen-bank observation')
            banks,indices=self.banks()
            with observe_only(self.actor):
                before={k:readout(self.actor,v,self.stats_basis,capture_gradient=True) for k,v in banks.items()}
            self.pending=dict(step=self.step,banks=banks,before=before,q=q)
            directory=self.directory/f'window_{self.step:03d}'
            arrays=dict(Q=q,eigenvalues=eigen,basis=self.stats_basis.cpu().numpy(),train_indices=indices)
            for name,v in before.items():
                for key in ('z','q','h','g_u','row','pos','zrow','zpos','token','boundary'):
                    arrays[name+'_'+key]=v[key].numpy()
            atomic_npz(directory/'before.npz',arrays)
            atomic_json(directory/'protocol.json',dict(anchor_before_update=self.step,
                train_sequences=len(banks['train']),heldout_sequences=len(banks['heldout']),
                sampled_positions_per_sequence=32,mode_source='entire actual anchor training batch gradient',
                bank_sha256={k:fingerprint(v) for k,v in banks.items()},
                lags=list(self.lags),teacher_calls_extra=0,backwards_extra=0,temporary_updates=0))
        self.metrics['timing/mode_statistics_before_seconds']=time.perf_counter()-start

    def after_update(self):
        start=time.perf_counter()
        if self.pending is not None:
            lag=self.step-self.pending['step']+1
            if lag in self.lags:
                records={};arrays={}
                with observe_only(self.actor):
                    for name,data in self.pending['banks'].items():
                        after=readout(self.actor,data,self.stats_basis,capture_gradient=True)
                        record,payload=mode_progress(self.pending['before'][name],after,self.stats_basis,self.pending['q'])
                        records[name]=record
                        for k,v in payload.items():arrays[name+'_'+k]=v
                directory=self.directory/f"window_{self.pending['step']:03d}"
                atomic_json(directory/f'after_{lag}.json',dict(step=self.step,lag=lag,records=records))
                atomic_npz(directory/f'after_{lag}.npz',arrays)
                self.metrics['modes/fixed_bank_lag']=float(lag)
                self.metrics['modes/heldout_tv_improvement']=records['heldout']['tv_before']-records['heldout']['tv_after']
                if lag==max(self.lags):self.pending=None
        self.metrics['timing/mode_statistics_after_seconds']=time.perf_counter()-start


def get_observer(actor):
    if not hasattr(actor,'_response_geometry'):
        actor._response_geometry=ModeObserver(actor,actor.config.policy_loss['response_geometry'])
    if not isinstance(actor._response_geometry,ModeObserver):raise ValueError('cannot change observer protocol mid-run')
    return actor._response_geometry
