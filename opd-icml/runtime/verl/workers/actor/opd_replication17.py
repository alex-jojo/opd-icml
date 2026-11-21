"""Opt-in 1.7B replication: passive FULL audit, fresh cumulative U, reuse readouts.

No optimizer transformation here. FULL uses unmodified OPD, reuse uses the
existing fresh-forward/backward AdamW loop. Diagnostic caches never train.
"""
import json
import hashlib
import time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .opd_top16 import forward, _final_norm_module, _BASIS_CACHE, _BASIS_DEVICE_CACHE
from .opd_mode_absorption import observe_only
from .opd_response_geometry import select, fingerprint
from .opd_plasticity import atomic_json, atomic_npz
from .opd_token_audit_runtime import TokenAudit, read_bank, differences, compact_bank,capture_bank
from .opd_token_audit import response_summary
from . import opd_distributed as dp


def cumulative_basis(initial, current, rank):
    """Uncentered cumulative UNIT hidden displacement, not an EMA of increments."""
    return dp.primary_call(lambda:_cumulative_basis_local(initial,current,rank))


def _cumulative_basis_local(initial,current,rank,*,allow_zero=False):
    delta=F.normalize(current.float(),dim=-1)-F.normalize(initial.float(),dim=-1)
    second=delta.T@delta/max(len(delta),1)
    eigen,vectors=torch.linalg.eigh((second+second.T)*.5)
    eigen=eigen.flip(0).clamp_min(0);u=vectors.flip(1)[:,:rank].contiguous()
    total=float(eigen.sum())
    if total<=1e-20:
        if not allow_zero:raise ValueError('No measurable cumulative hidden displacement; cannot discover U')
        return u[:, :0],second,eigen,dict(unit_displacement_rms=total**.5,
            own_top128_energy=None,positions=len(delta))
    return u,second,eigen,dict(unit_displacement_rms=total**.5,
        own_top128_energy=float(eigen[:rank].sum()/eigen.sum()),positions=len(delta))


def panel_positions(mask, per_sequence=256):
    """Response-position indices, in the order used for the fixed probe rows."""
    result = []
    for row in mask:
        valid = row.nonzero().flatten()
        if len(valid) > per_sequence:
            selected = torch.linspace(0, len(valid)-1, per_sequence, device=valid.device).round().long()
            valid = valid[selected]
        result.append(valid)
    return result


@torch.no_grad()
def panel_hidden(actor,data,per_sequence=256):
    device=next(actor.actor_module.parameters()).device;values=[]
    with observe_only(actor):
        local,_=dp.shard(data)
        for part in local.split(actor.config.ppo_micro_batch_size_per_gpu):
            x={k:v.to(device) for k,v in part.batch.items()};n=x['responses'].shape[1]
            positions=panel_positions(x['response_mask'],per_sequence)
            def capture(_m,_a,h):
                h=h[:,-n-1:-1]
                for row,ix in enumerate(positions):
                    if len(ix):values.append(h[row,ix].float().cpu())
            handle=_final_norm_module(actor.actor_module).register_forward_hook(capture)
            try:forward(actor,x,float(data.meta_info['temperature']),x['opd_top16_ids'],diagnostics=False,
                        hidden_only=bool(actor.config.policy_loss.get('replication17',{}).get('reuse_forward',False)))
            finally:handle.remove()
    if not values:raise ValueError('empty discovery panel')
    return torch.cat(dp.gather(torch.cat(values)))


def install_basis(path,u):
    atomic_npz(path,{'basis':u.cpu().numpy()})
    dp.barrier()
    key=(str(Path(path).resolve()),u.shape[1])
    _BASIS_CACHE.pop(key,None)
    for k in list(_BASIS_DEVICE_CACHE):
        if k[0]==key:del _BASIS_DEVICE_CACHE[k]


class Audit(TokenAudit):
    def observed_bank(self,candidate,name,bank):
        if candidate=='ordinary' and name=='fixed' and self.cfg.get('reuse_forward',False):
            self.fixed_panel_after=sample_panel(bank)

    def probe(self, fields):
        self.heavy=self.step in self.cfg.get('heavy_steps',[30,60])
        if self.heavy:
            print(f'[OPD audit] step {self.step}: scheduled HEAVY intervention',flush=True)
            return super().probe(fields)
        # Normal measurement: observe the ONE actual training update. No
        # optimizer snapshot, replay backward, temporary update or full hash.
        if dp.primary():print(f'[OPD audit] step {self.step}: LIGHT before/after readout; no trial updates',flush=True)
        self.light_banks,self.light_indices=self.banks();self.light_banks['fixed']=self.fixed
        self.probe_directory=self.directory/f'probe_{self.step:03d}'
        with observe_only(self.actor):
            self.light_before={k:self.decorate(self.training_bank if k=='train' and self.cfg.get('reuse_forward',False)
                else read_bank(self.actor,v,self.stats_basis,scope_label=(k,'before',1) if k!='train' else None)) for k,v in self.light_banks.items()}
        self.training_bank=None
        for k,v in self.light_before.items():atomic_npz(self.probe_directory/f'{k}_before.npz',compact_bank(v))

    def before_update(self):
        super().before_update()
        dp.barrier()
        path=self.directory/f'step_{self.step:03d}.json'
        report=json.loads(path.read_text())
        report['basis_reference']=self.reference
        atomic_json(path,report)
        for key in ('nuclear_norm_token_normalized','gradient_frobenius_full','gradient_frobenius_u'):
            self.metrics['token_audit/'+key]=report[key]
        for key in ('stable_rank','participation_rank','singular_entropy_rank'):
            self.metrics['token_audit/u_'+key]=report['spectrum'][key] or 0.
        if dp.primary():print(f'[gradient stats] step={self.step} full_RMS={report["gradient_rms_full"]:.6g} '
              f'U_RMS={report["gradient_rms_u"]:.6g} '
              f'U_nuclear={report["nuclear_norm_token_normalized"]:.6g} '
              f'U_effective_rank={report["spectrum"]["effective_rank"]} '
              f'U_stable_rank={report["spectrum"]["stable_rank"]}',flush=True)

    def after_update(self):
        started=time.perf_counter()
        if self.step in self.anchors:
            if self.heavy:
                super().after_update()
            else:
                records={}
                with observe_only(self.actor):
                    for k,data in self.light_banks.items():
                        after=self.decorate(read_bank(self.actor,data,self.stats_basis,scope_label=(k,'after',1) if k!='train' else None))
                        self.observed_bank('ordinary',k,after)
                        differences(self.light_before[k],after)
                        records[k]=response_summary(self.light_before[k],after,self.thresholds)
                        atomic_npz(self.probe_directory/f'{k}_ordinary.npz',compact_bank(after))
                atomic_json(self.probe_directory/'results.json',dict(step=self.step,
                    records={'ordinary':dict(banks=records)},observation_only=True,
                    actual_committed_update=True,committed_verified=None,
                    train_indices=self.light_indices.tolist(),
                    bank_sha256={k:fingerprint(v) for k,v in self.light_banks.items()},
                    extra_backwards=0,temporary_optimizer_steps=0,teacher_calls_extra=0,
                    definition='Measured actual before/after update, not a disposable preview. No shadow comparison was performed.'))
                self.light_before=None;self.light_banks=None
                atomic_json(self.directory/'token_dictionary.json',{str(k):v[-1] for k,v in self.labels.items()})
                self.metrics['token_audit/light_observation']=1.
        self.metrics['token_audit/heavy_intervention']=float(self.step in self.anchors and self.heavy)
        self.metrics['timing/token_audit_after_seconds']=time.perf_counter()-started


class Replication:
    def __init__(self,actor):
        self.actor=actor;self.cfg=dict(actor.config.policy_loss['replication17'])
        self.output=Path(self.cfg['output_dir']);self.output.mkdir(parents=True,exist_ok=True)
        self.asset=Path(actor.config.policy_loss['hidden_projection']['asset_path'])
        self.full=self.cfg['mode']=='full';self.rank=int(self.cfg.get('rank',128))
        self.freeze=int(self.cfg.get('freeze_step',10));self.interval=int(self.cfg.get('interval',5))
        self.dimension=int(_final_norm_module(actor.actor_module).weight.numel())
        self.audit=None
        self.scope_accumulator=None
        self.stability=None
        if self.cfg.get('adaptive_stability',{}).get('enabled',False):
            if not self.full:raise ValueError('adaptive discovery must use FULL OPD')
            from .opd_adaptive_stability import Stability
            self.stability=Stability(self.output,self.cfg['adaptive_stability'])
        self.reuse_forward=bool(self.cfg.get('reuse_forward',False))
        if self.reuse_forward and any(isinstance(m,torch.nn.Dropout) and m.p>0 for m in actor.actor_module.modules()):
            raise ValueError('forward reuse requires dropout=0 for identical train/eval readouts')

    @contextmanager
    def watch_gradient_scopes(self,inputs,inner,projection_state):
        if not self.cfg.get('all_gradient_scopes',False) or not self.measure:
            yield;return
        from .opd_gradient_scopes import GradientScopes
        if self.scope_accumulator is None:
            u=self.audit.stats_basis if self.full else self.u
            self.scope_accumulator=GradientScopes(u)
        mask=inputs['response_mask'].bool();n=mask.shape[1]
        total=int(self.train.batch['response_mask'].sum())
        def consume(g):self.scope_accumulator.add(g[:,-n-1:-1][mask].float()*(total/dp.size()))
        old=projection_state.get('raw_gradient_observer');handle=None
        if self.cfg['mode']=='r128':projection_state['raw_gradient_observer']=consume
        else:
            def hook(_m,_a,h):
                def backward(g):consume(g);return g
                h.register_hook(backward)
            handle=_final_norm_module(self.actor.actor_module).register_forward_hook(hook)
        try:yield
        finally:
            if handle is not None:handle.remove()
            projection_state['raw_gradient_observer']=old

    @contextmanager
    def watch_training(self,inputs,inner,cursor):
        if not self.capture_train or inner!=0:
            yield;return
        device=inputs['responses'].device
        labels=torch.tensor([self.train_row_labels.get(i,-1) for i in range(cursor,cursor+len(inputs['responses']))],device=device)
        if bool((labels>=0).any()):
            u=self.audit.stats_basis if self.full else self.u
            with capture_bank(self.actor,inputs,u,labels,self.capture_cost) as bank:yield
            self.train_captures.append(bank)
            # Forward/backward time itself is NOT charged to diagnostics here.
        else:yield

    def before_update(self,inner):
        if self.scope_accumulator is not None:
            scope_started=time.perf_counter()
            from .opd_gradient_scopes import save,metric_values
            report,arrays=self.scope_accumulator.finish(int(self.train.batch['response_mask'].sum()))
            self.scope_accumulator=None
            reference=self.audit.reference if self.full else 'FULL frozen step10 U128'
            save(report,arrays,self.output/'gradient_scopes',self.step,inner+1,int(self.cfg.get('reuse',1)),reference)
            self.scope_metrics.update(metric_values(report))
            scope_seconds=time.perf_counter()-scope_started+report['gram_dispatch_seconds']
            self.extra+=scope_seconds
            self.scope_metrics['timing/gradient_scope_seconds']=self.scope_metrics.get('timing/gradient_scope_seconds',0.)+scope_seconds
        if not self.capture_train or inner!=0:return
        started=time.perf_counter()
        local=({k:np.concatenate([b[k] for b in self.train_captures]) for k in self.train_captures[0]}
               if self.train_captures else None)
        bank=dp.merge_fields(dp.gather(local))
        self.train_captures=[]
        if self.full:self.audit.training_bank=bank
        else:
            self.bank_before['train']=bank
            self.thresholds=dict(kl_p90=float(np.quantile(bank['kl'],.9)),kl_median=float(np.median(bank['kl'])))
            for k,v in self.bank_before.items():atomic_npz(self.output/f'round_{self.step:03d}'/f'{k}_before.npz',compact_bank(v))
        self.extra+=time.perf_counter()-started+self.capture_cost.pop('seconds',0.)

    def begin(self,train,heldout,basis,step):
        started=time.perf_counter();self.step=step;self.train=train;self.heldout=heldout
        self.extra=0.;self.records=[];self.bank_before=None
        self.scope_metrics={};self.scope_accumulator=None
        self.capture_train=False;self.train_captures=[];self.capture_cost={}
        self.measure=(step%self.interval==0 or step==int(self.cfg['rounds']))
        if self.stability is not None:self.stability.resume_before(step)
        if self.full:
            if basis is not None:raise ValueError('FULL discovery must not project training gradients')
            cache=self.output/'discovery_panel.pt'
            if not hasattr(self,'panel'):
                if dp.exists(cache):
                    saved=torch.load(cache,map_location='cpu',weights_only=False)
                    self.panel,self.h0=saved['panel'],saved['h0']
                else:
                    if step!=1:raise ValueError('missing initial discovery panel on resume')
                    self.panel=select(heldout,range(min(32,len(heldout))))
                    self.h0=panel_hidden(self.actor,self.panel)
                    if dp.primary():
                        temporary=cache.with_suffix('.tmp');torch.save(dict(panel=self.panel,h0=self.h0),temporary);temporary.replace(cache)
                    dp.barrier()
                    atomic_json(self.output/'discovery_panel.json',dict(sha256=fingerprint(self.panel),
                        positions=len(self.h0),hidden_size=self.dimension,source='step1 heldout, before first update'))
            if step!=1 and not self.measure:
                self.extra+=time.perf_counter()-started
                return None
            if step==1:
                self.asset.with_suffix('.json').unlink(missing_ok=True)
                install_basis(self.asset,torch.eye(self.dimension)[:,:self.rank])
                reference=dict(status='placeholder_for_step1_thresholds_only',valid_u128=False)
            elif self.stability is not None:
                # Reuse the last completed observation for PASSIVE statistics.
                # Discovery is computed exactly once, after this update below.
                if self.stability.history:
                    observed_step,observed_u=self.stability.history[-1]
                    install_basis(self.asset,observed_u)
                    reference=dict(status='adaptive_observation',valid_u128=True,cumulative_from=0,cumulative_to=observed_step)
                else:
                    reference=dict(status='placeholder_until_first_observation',valid_u128=False)
            elif step<=self.freeze:
                self.asset.with_suffix('.json').unlink(missing_ok=True)
                current=panel_hidden(self.actor,self.panel)
                u,_,_,_=cumulative_basis(self.h0,current,self.rank)
                install_basis(self.asset,u)
                reference=dict(status='provisional',valid_u128=True,cumulative_from=0,cumulative_to=step-1)
            else:
                meta=json.loads(self.asset.with_suffix('.json').read_text())
                if meta['freeze_step']!=self.freeze or meta['hidden_size']!=self.dimension:raise ValueError('wrong frozen basis')
                reference=dict(status='frozen',valid_u128=True,cumulative_from=0,cumulative_to=self.freeze)
            if step==1 or self.measure:
                if self.audit is None:
                    anchors=sorted(set(range(self.interval,int(self.cfg['rounds'])+1,self.interval))|{int(self.cfg['rounds'])})
                    config=dict(output_dir=str(self.output/'token_audit'),anchors=anchors,
                        lags=[],train_readout=32,tokenizer_path=self.cfg.get('tokenizer_path'),
                        heavy_steps=self.cfg.get('heavy_steps',[30,60]),reuse_forward=self.reuse_forward,
                        cache_only_moments=True)
                    self.audit=Audit(self.actor,config)
                self.audit.cfg['probe_r128']=self.stability is None and step>self.freeze
                self.audit.reference=reference
                self.audit.begin(train,heldout,None,step)
                self.audit.fixed_panel_after=None
                if self.reuse_forward and self.measure and step not in self.cfg.get('heavy_steps',[30,60]):
                    _,ix=self.audit.banks()
                    self.train_row_labels={int(row):i for i,row in enumerate(ix)};self.capture_train=True
                self.extra+=time.perf_counter()-started
                return self.audit
        elif self.measure:
            if not self.asset.is_file():raise ValueError('Run FULL discovery before reuse arms')
            with np.load(self.asset,allow_pickle=False) as f:self.u=torch.from_numpy(f['basis'].copy()).to(next(self.actor.actor_module.parameters()).device)
            cache=self.output/'fixed_bank.pt'
            if not hasattr(self,'fixed'):
                # Created on round 1 below, so this is never a post-hoc heldout.
                if not dp.exists(cache):raise ValueError('missing first-round fixed bank')
                self.fixed=torch.load(cache,map_location='cpu',weights_only=False)
            self.banks={'train':select(train,range(min(32,len(train)))),
                        'heldout':select(heldout,range(len(heldout))),'fixed':self.fixed}
            self.capture_train=self.reuse_forward
            self.train_row_labels={i:i for i in range(min(32,len(train)))}
            with observe_only(self.actor):self.bank_before={k:read_bank(self.actor,v,self.u,scope_label=(k,'before',1) if k!='train' else None) for k,v in self.banks.items()
                if not self.capture_train or k!='train'}
            if not self.capture_train:
                self.thresholds=dict(kl_p90=float(np.quantile(self.bank_before['train']['kl'],.9)),
                    kl_median=float(np.median(self.bank_before['train']['kl'])))
                for k,v in self.bank_before.items():atomic_npz(self.output/f'round_{step:03d}'/f'{k}_before.npz',compact_bank(v))
        if not self.full and step==1:
            self.fixed=select(heldout,range(len(heldout)))
            if dp.primary():
                cache=self.output/'fixed_bank.pt';tmp=cache.with_suffix('.tmp');torch.save(self.fixed,tmp);tmp.replace(cache)
            dp.barrier()
        self.extra+=time.perf_counter()-started
        return None

    def after_inner(self,inner):
        started=time.perf_counter()
        if self.full and (self.measure or self.step==self.freeze):
            cached=getattr(self.audit,'fixed_panel_after',None) if self.reuse_forward else None
            current=cached if cached is not None else panel_hidden(self.actor,self.panel)
            u,second,eigen,record=cumulative_basis(self.h0,current,self.rank)
            if self.stability is not None:
                decision=self.stability.observe(self.step,u,second)
                install_basis(self.asset,u)
                atomic_json(self.asset.with_suffix('.json'),dict(freeze_step=self.step if decision['stable'] else None,
                    observed_step=self.step,stable=decision['stable'],hidden_size=self.dimension,
                    basis_sha256=hashlib.sha256(self.asset.read_bytes()).hexdigest(),rank=self.rank,
                    panel_sha256=fingerprint(self.panel),definition='top eigenvectors of mean unit cumulative displacement outer products',**record))
            elif self.step==self.freeze:
                install_basis(self.asset,u)
                atomic_json(self.asset.with_suffix('.json'),dict(freeze_step=self.freeze,hidden_size=self.dimension,
                    basis_sha256=hashlib.sha256(self.asset.read_bytes()).hexdigest(),
                    rank=self.rank,panel_sha256=fingerprint(self.panel),definition='top eigenvectors of mean unit cumulative displacement outer products',**record))
            with np.load(self.asset,allow_pickle=False) as f:observed=torch.from_numpy(f['basis'].copy())
            record.update(step=self.step,raw_displacement_rms=float((current-self.h0).square().sum(-1).mean().sqrt()),
                observation_basis_energy=float(torch.trace(observed.T@second@observed)/eigen.sum()),
                top128_overlap_with_observation_basis=float((u.T@observed).square().sum()/self.rank),
                frozen_basis=(self.stability.stable if self.stability is not None else self.step>=self.freeze))
            atomic_npz(self.output/'motion'/f'step_{self.step:03d}.npz',dict(second=second.numpy(),eigenvalues=eigen.numpy()))
            atomic_json(self.output/'motion'/f'step_{self.step:03d}.json',record)
        if self.bank_before is not None:
            summary={}
            with observe_only(self.actor):
                for k,data in self.banks.items():
                    after=read_bank(self.actor,data,self.u,scope_label=(k,'after',inner) if k!='train' else None);differences(self.bank_before[k],after)
                    summary[k]=response_summary(self.bank_before[k],after,self.thresholds)
                    atomic_npz(self.output/f'round_{self.step:03d}'/f'{k}_after_{inner}.npz',compact_bank(after))
            self.records.append(dict(inner=inner,banks=summary))
        self.extra+=time.perf_counter()-started

    def finish(self,losses,norms):
        record=dict(round=self.step,reuse=len(losses),optimizer_updates=self.step*len(losses),
            preupdate_train_kl=losses,gradient_norms=norms,extra_seconds=self.extra,readouts=self.records,
            diagnostic_revision='compact-head-forward-reuse-v3')
        if self.audit is not None and (self.step==1 or self.measure):
            record['audit_metrics']=self.audit.metrics
        atomic_json(self.output/f'round_{self.step:03d}.json',record)
        if dp.primary():print(f'[OPD] {self.cfg["mode"]} rollout={self.step} real_updates={record["optimizer_updates"]} KL(before inner steps)={losses}',flush=True)
        audit_seconds=sum(record.get('audit_metrics',{}).get(k,0.) for k in
            ('timing/token_audit_before_seconds','timing/token_audit_after_seconds'))
        if self.measure and dp.primary():
            print(f'[OPD timing] diagnostics={self.extra+audit_seconds:.2f}s '
                  f'(audit={audit_seconds:.2f}s, readout/discovery={self.extra:.2f}s); '
                  'these are observation costs, not optimizer-step time',flush=True)
        metrics={'timing/replication_readout_seconds':self.extra,
                 'timing/replication_extra_total_seconds':self.extra+audit_seconds}
        metrics.update(self.scope_metrics)
        if self.stability is not None:metrics['replication/adaptive_stop']=float(self.stability.stable)
        return metrics


def sample_panel(bank,per_sequence=256):
    values=[]
    for row in np.unique(bank['row']):
        h=torch.from_numpy(bank['h'][bank['row']==row])
        ix=torch.linspace(0,len(h)-1,min(per_sequence,len(h))).round().long()
        values.append(h[ix])
    return torch.cat(values)


def get_runtime(actor):
    if not hasattr(actor,'_replication17'):actor._replication17=Replication(actor)
    return actor._replication17
