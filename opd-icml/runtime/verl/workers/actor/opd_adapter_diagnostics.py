"""Adapter-phase statistics in the actual A parameter space plus full hidden geometry.

Frozen feature replays are cached only for the active update, never on disk.
Both ranks execute the same unique diagnostic forwards; A gradients are replicated.
"""
import hashlib
import pickle
import time
from pathlib import Path
import numpy as np
import torch
from . import opd_distributed as dp
from .opd_top16 import forward,reverse_kl,_unwrap_actor_module
from .opd_fixed_u_adapter import selected_logits
from .opd_gradient_study import HiddenMoments,reshard_after_readout
from .opd_study_replication import StudyReplication
from .opd_token_audit import output_scores
from .opd_replication17 import cumulative_basis
from .opd_plasticity import atomic_npz,atomic_json


def row_keys(data):
    """Identity includes targets/masks. Never reuse across changed teacher targets."""
    digests=[hashlib.sha256() for _ in range(len(data))]
    for name in ('input_ids','attention_mask','position_ids','response_mask','opd_top16_ids','opd_teacher_top16_logits'):
        if name not in data.batch:continue
        value=data.batch[name].detach().cpu().contiguous()
        for i,digest in enumerate(digests):
            digest.update(str((name,tuple(value[i].shape),value.dtype)).encode())
            digest.update(value[i].view(torch.uint8).numpy().tobytes())
    return tuple(d.hexdigest() for d in digests)


def subset_features(record,keys,tokens):
    """Reuse locally owned frozen rows, even when the new random batch has a
    different DP partition. Only small A gradients are all-reduced afterwards.
    Identical duplicate rows have one owner and are repeated for each new row.
    """
    if record['local'] or record['replicated'] or not set(keys)<=set(record['row_keys']):return None
    source={}
    for i,key in enumerate(record['row_keys']):source.setdefault(key,i)
    wanted={}
    for new,key in enumerate(keys):wanted.setdefault(source[key],[]).append(new)
    items=[]
    for item in record['items']:
        for old in item['row'].unique(sorted=True).tolist():
            for new in wanted.get(old,[]):
                mask=item['row']==old
                part={k:v[mask] for k,v in item.items()}
                part['row']=torch.full_like(part['row'],new);items.append(part)
    return dict(items=items,replicated=False,local=False,count=len(keys),tokens=tokens,row_keys=keys,subset_reused=True)


def features(actor,data,already_local=False):
    cache=actor._adapter_feature_cache;keys=row_keys(data);key=(keys,already_local)
    if key in cache:return cache[key]
    if not already_local:
        for source in list(cache.values()):
            reused=subset_features(source,keys,int(data.batch['response_mask'].sum()))
            if reused is not None:
                cache[key]=reused
                actor._adapter_subset_cache_hits=getattr(actor,'_adapter_subset_cache_hits',0)+1
                return reused
    started=time.perf_counter();replicated=len(data)%dp.size()!=0
    local,offset=(data,0) if already_local or replicated else dp.shard(data)
    device=next(actor.actor_module.parameters()).device;adapter=actor.fixed_u_adapter
    head=_unwrap_actor_module(_unwrap_actor_module(actor.actor_module).lm_head)
    items=[];cursor=offset;adapter.suspended=True
    try:
        with torch.no_grad():
            for part in local.split(actor.config.ppo_micro_batch_size_per_gpu):
                x={k:v.to(device) for k,v in part.batch.items()};mask=x['response_mask'].bool()
                row,pos=mask.nonzero(as_tuple=True);ordinal=mask.long().cumsum(-1)-1
                def capture(_m,args):
                    if actor._adapter_head_weight is None:actor._adapter_head_weight=head.weight.detach().clone()
                    if head.bias is not None:raise ValueError('Adapter expects bias-free head')
                    items.append(dict(h=args[0][:,:-1][mask].detach().cpu().clone(),
                        ids=x['opd_top16_ids'][mask].cpu(),teacher=x['opd_teacher_top16_logits'][mask].cpu(),
                        row=(row+cursor).cpu(),pos=pos.cpu(),token=x['responses'][mask].cpu(),
                        length=mask.sum(1)[row].cpu(),first=(ordinal==0)[mask].cpu(),
                        relative=(ordinal.float()/(mask.sum(1,keepdim=True)-1).clamp_min(1))[mask].cpu()))
                handle=head.register_forward_pre_hook(capture)
                try:forward(actor,x,float(data.meta_info['temperature']),x['opd_top16_ids'],diagnostics=False,hidden_only=True)
                finally:handle.remove()
                cursor+=len(part)
    finally:adapter.suspended=False
    reshard_after_readout(actor)
    result=dict(items=items,replicated=replicated,local=already_local,count=len(data),row_keys=keys,
                tokens=int(data.batch['response_mask'].sum()))
    cache[key]=result
    actor._adapter_feature_seconds+=time.perf_counter()-started
    actor._adapter_feature_forwards+=len(items)
    return result


def chunks(actor,record):
    size=int(actor.config.policy_loss['fixed_u_adapter'].get('token_chunk',512))
    for item in record['items']:
        for start in range(0,len(item['h']),size):yield {k:v[start:start+size] for k,v in item.items()}


def cached_gradient(actor,data,basis,route,normalization,capture_hidden):
    record=features(actor,data);adapter=actor.fixed_u_adapter;device=adapter.A.device
    actor.actor_optimizer.zero_grad(set_to_none=True)
    moment=HiddenMoments(basis) if capture_hidden else None
    # A reused subset may have no rows on one rank. Still participate in the
    # same moment and gradient reductions, with genuine zero contributions.
    adapter.A.grad=torch.zeros_like(adapter.A)
    if moment is not None:moment.add(torch.empty((0,basis.shape[0]),device=device))
    sums=torch.zeros(len(data),dtype=torch.float64,device=device)
    for part in chunks(actor,record):
        h=adapter(part['h'].to(device));length=part['length'].to(device)
        denominator=record['tokens'] if normalization=='token' else len(data)*length
        def hook(g):
            if moment is not None:
                raw=g.float()*(denominator if isinstance(denominator,int) else denominator[:,None])
                moment.add(raw)
            if route=='direct':return torch.zeros_like(g)
            if route!='full':
                u=basis.to(g);p=(g@u)@u.T
                return p if route=='u' else g-p
            return g
        handle=h.register_hook(hook)
        z=selected_logits(h,actor._adapter_head_weight,part['ids'].to(device))
        kl=reverse_kl(z,part['teacher'].to(device))
        (kl/denominator).sum().backward();handle.remove()
        sums.index_add_(0,part['row'].to(device),(kl.detach()/length).double())
    if dp.size()>1:
        torch.distributed.all_reduce(adapter.A.grad)
        if record['replicated']:adapter.A.grad.div_(dp.size())
        else:torch.distributed.all_reduce(sums)
    grad=[adapter.A.grad.detach().cpu().float().clone().reshape(-1)]
    actor.actor_optimizer.zero_grad(set_to_none=True)
    return grad,sums.cpu().numpy(),moment.finish() if moment is not None else None


@torch.no_grad()
def cached_losses(actor,data):
    record=features(actor,data);a=actor.fixed_u_adapter;device=a.A.device
    sums=torch.zeros(len(data),device=device,dtype=torch.float64)
    for part in chunks(actor,record):
        z=selected_logits(a(part['h'].to(device)),actor._adapter_head_weight,part['ids'].to(device))
        kl=reverse_kl(z,part['teacher'].to(device))/part['length'].to(device)
        sums.index_add_(0,part['row'].to(device),kl.double())
    if dp.size()>1 and not record['replicated']:torch.distributed.all_reduce(sums)
    return sums.cpu().numpy()


class AdapterDiagnostics(StudyReplication):
    def begin(self,train,basis,step):
        super().begin(train,basis,step)
        offset=int(self.study_cfg.get('update_offset',0))
        self.completed+=offset
        self.measure=(self.completed+1)%self.interval==0 or step==1 or step==int(self.cfg['rounds'])
        self.reference.update(parameter_space='adapter A',backbone_frozen=True,head_frozen=True)

    def record(self,part,z,changed,total):
        if not self.measure:return
        if self.moments is None:self.moments=HiddenMoments(self.u)
        with torch.no_grad():
            values={k:v.detach().cpu().numpy() for k,v in output_scores(z,part['teacher'].to(z.device)).items()}
            teacher=part['teacher'].double().log_softmax(-1)
            values.update(teacher_entropy=(-(teacher.exp()*teacher).sum(-1)).numpy(),
                row=part['row'].numpy(),pos=part['pos'].numpy(),token=part['token'].numpy(),
                boundary=part['first'].numpy(),relative_position=part['relative'].numpy())
        def consume(gradient):
            g=gradient.detach().float()*total;self.moments.add(g)
            gu=g@self.u.float()
            values.update(g_u=gu.cpu().numpy(),raw_gradient_energy=g.square().sum(1).cpu().numpy(),
                          projected_gradient_energy=gu.square().sum(1).cpu().numpy())
            values['transient_full_gradient']=g.cpu().numpy()
            ix=torch.linspace(0,len(g)-1,min(64,len(g)),device=g.device).round().long()
            values.update(sample_full_gradient=g[ix].cpu().numpy(),sample_local_index=ix.cpu().numpy())
            return gradient
        changed.register_hook(consume);self.parts.append(values)

    def after_inner(self,inner):
        if self.study.payload is not None and inner==1:self.metrics.update(self.study.after_update())
        update=self.completed+inner
        source=self.study_cfg.get('motion_panel_path')
        if source and (update%self.interval==0 or self.step==int(self.cfg['rounds'])):
            started=time.perf_counter()
            if not hasattr(self,'motion_base'):
                saved=torch.load(source,map_location='cpu',weights_only=False)
                self.motion_h0=saved['h0'];record=features(self.actor,saved['panel']);pieces=[]
                for item in record['items']:
                    for row in item['row'].unique(sorted=True):
                        selected=item['h'][item['row']==row]
                        ix=torch.linspace(0,len(selected)-1,min(256,len(selected))).round().long()
                        pieces.append(selected[ix])
                self.motion_base=torch.cat(pieces)
            a=self.actor.fixed_u_adapter
            with torch.no_grad():
                current=torch.cat(dp.gather(a(self.motion_base.to(a.A.device)).float().cpu()))
            u,second,eigen,report=cumulative_basis(self.motion_h0,current,self.rank)
            report.update(step=update,raw_displacement_rms=float((current-self.motion_h0).square().sum(1).mean().sqrt()),
                observation_basis_energy=float(torch.trace(self.u.cpu().T@second@self.u.cpu())/eigen.sum()),
                fixed_backbone=True,changing_representation='post-Adapter hidden',basis_frozen=True)
            atomic_npz(self.output/'motion'/f'step_{update:03d}.npz',dict(second=second.numpy(),eigenvalues=eigen.numpy(),basis=u.numpy()))
            atomic_json(self.output/'motion'/f'step_{update:03d}.json',report)
            self.extra+=time.perf_counter()-started


def update_with_statistics(actor,data):
    a=actor.fixed_u_adapter
    if any(p.requires_grad for p in actor.actor_module.parameters()):raise RuntimeError('Backbone was not frozen')
    if int(actor.config.policy_loss.get('jump_reuse_updates',1))!=1:
        raise ValueError('Two-arm adaptive experiment uses one update per rollout')
    actor.actor_module.eval();device=next(actor.actor_module.parameters()).device;a.to(device)
    payload=data.meta_info.pop('gradient_study_payload',None)
    train=dp.global_batch(data)
    actor._adapter_feature_cache={};actor._adapter_head_weight=None
    actor._adapter_feature_seconds=0.;actor._adapter_feature_forwards=0;actor._adapter_subset_cache_hits=0
    if not hasattr(actor,'_adapter_diagnostics'):actor._adapter_diagnostics=AdapterDiagnostics(actor)
    obs=actor._adapter_diagnostics;obs.begin(train,a.U,int(data.meta_info['global_steps']))
    guard=actor.config.policy_loss['fixed_u_adapter'].get('guard',{}).get('enabled',False)
    if guard and not getattr(actor,'_adapter_continuity_checked',False):
        from .opd_adapter_guard import verify_continuity
        verify_continuity(actor,actor.config.policy_loss['fixed_u_adapter']['handoff_rng_dir'],obs.output)
        actor._adapter_continuity_checked=True
    initial_feature_seconds=actor._adapter_feature_seconds;initial_feature_forwards=actor._adapter_feature_forwards
    record=features(actor,train)
    train_cache_seconds=actor._adapter_feature_seconds-initial_feature_seconds
    train_cache_forwards=actor._adapter_feature_forwards-initial_feature_forwards
    if payload is not None:obs.start_study(pickle.loads(payload))
    actor.actor_optimizer.zero_grad(set_to_none=True);train_started=time.perf_counter();loss_sum=0.
    for part in chunks(actor,record):
        changed=a(part['h'].to(device));z=selected_logits(changed,actor._adapter_head_weight,part['ids'].to(device))
        obs.record(part,z,changed,record['tokens'])
        loss=reverse_kl(z,part['teacher'].to(device)).sum()/record['tokens']
        loss.backward();loss_sum+=float(loss.detach())
    if dp.size()>1:torch.distributed.all_reduce(a.A.grad)
    formal_backward_seconds=time.perf_counter()-train_started
    obs.before_update(0);optim_started=time.perf_counter()
    norm=torch.nn.utils.clip_grad_norm_([a.A],actor.config.grad_clip)
    if not torch.isfinite(norm):raise FloatingPointError('Nonfinite adapter gradient')
    guard_metrics={}
    if guard:
        from .opd_adapter_guard import guarded_step
        guard_metrics=guarded_step(actor,record,obs.step,obs.output)
        optim_seconds=guard_metrics['timing/adapter_guard_optimizer_seconds']
    else:
        actor.actor_optimizer.step();optim_seconds=time.perf_counter()-optim_started
    obs.after_inner(1);loss=dp.total(loss_sum);metrics=obs.finish([loss],[float(norm)])
    cache_bytes=sum(t.numel()*t.element_size() for c in actor._adapter_feature_cache.values() for part in c['items'] for t in part.values())
    metrics.update({'actor/opd_top16_kl':loss,'actor/grad_norm':float(norm),'actor/opd_optimizer_steps':1.,
        'adapter/local_update':float(obs.step),'adapter/total_update':float(obs.completed+1),
        'adapter/adam_step':float(actor.actor_optimizer.state[a.A]['step']),
        'adapter/trainable_parameters':float(a.A.numel()),'adapter/backbone_backwards':0.,
        'adapter/feature_cache_bytes':float(cache_bytes),'adapter/train_feature_forwards':float(train_cache_forwards),
        'adapter/diagnostic_feature_forwards':float(actor._adapter_feature_forwards-train_cache_forwards),
        'adapter/subset_feature_cache_hits':float(actor._adapter_subset_cache_hits),
        'timing/adapter_train_feature_seconds':train_cache_seconds,
        'timing/adapter_diagnostic_feature_seconds':actor._adapter_feature_seconds-train_cache_seconds,
        'timing/adapter_backward_seconds':formal_backward_seconds,'timing/opd_optimizer_seconds':optim_seconds,
        'monitor/opd_train_trajectories':float(len(train)),'monitor/opd_heldout_trajectories':0.})
    metrics.update(guard_metrics)
    actor.actor_optimizer.zero_grad(set_to_none=True)
    actor._adapter_feature_cache.clear();actor._adapter_head_weight=None
    if dp.primary():print(f'[ADAPTER] total_update={obs.completed+1} KL={loss:.6g} A_parameters={a.A.numel()} frozen_backbone=True',flush=True)
    return {k:[v] for k,v in metrics.items()}
