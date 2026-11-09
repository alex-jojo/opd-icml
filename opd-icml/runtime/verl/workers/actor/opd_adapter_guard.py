"""Zero-function handoff and bounded real AdamW steps for the frozen adapter."""
import copy,time
from pathlib import Path
import torch
import torch.nn.functional as F
from . import opd_distributed as dp
from .opd_plasticity import atomic_json


@torch.no_grad()
def continuity_forward(actor,panel):
    """Use the SAME full microbatch, autocast and compact head on both sides.

    Sampling hidden before a new, smaller GEMM is not an identity test of the
    loaded model: BF16 rounding and FP32 final-norm activations can change it.
    Keep this compatible with already saved step40 sentinels.
    """
    from .opd_top16 import forward,_final_norm_module
    from .opd_mode_absorption import observe_only
    from .opd_gradient_study import reshard_after_readout
    local,_=dp.shard(panel);hidden=[];logp=[]
    device=next(actor.actor_module.parameters()).device
    with observe_only(actor):
        for part in local.split(actor.config.ppo_micro_batch_size_per_gpu):
            x={k:v.to(device) for k,v in part.batch.items()};n=x['responses'].shape[1];mask=x['response_mask'].bool()
            ix=torch.linspace(0,int(mask.sum())-1,min(128,int(mask.sum())),device=device).round().long()
            def hook(_m,_a,h):hidden.append(h[:,-n-1:-1][mask][ix].float().cpu())
            handle=_final_norm_module(actor.actor_module).register_forward_hook(hook)
            try:_,z,_,_=forward(actor,x,float(panel.meta_info['temperature']),x['opd_top16_ids'],diagnostics=False)
            finally:handle.remove()
            logp.append(z[mask][ix].float().log_softmax(-1).cpu())
    reshard_after_readout(actor)
    return torch.cat(hidden),torch.cat(logp)


@torch.no_grad()
def save_continuity(worker,path):
    actor=worker.actor;runtime=actor._gradient_study_replication
    panel=runtime.train.select_idxs(list(range(min(2*dp.size(),len(runtime.train)))))
    hidden,logp=continuity_forward(actor,panel)
    torch.save(dict(panel=panel,hidden=hidden,logp=logp,world=dp.size()),
               Path(path)/f'handoff_continuity_rank_{dp.rank()}.pt')


@torch.no_grad()
def verify_continuity(actor,path,output):
    saved=torch.load(Path(path)/f'handoff_continuity_rank_{dp.rank()}.pt',map_location='cpu',weights_only=False)
    if saved['world']!=dp.size():raise ValueError('Continuity sentinel world size changed')
    a=actor.fixed_u_adapter
    h,p=continuity_forward(actor,saved['panel']);error=(p-saved['logp']).abs()
    row=dict(zero_initialized=bool(torch.count_nonzero(a.A)==0),
        hidden_relative_error=float((h-saved['hidden']).norm()/saved['hidden'].norm().clamp_min(1e-12)),
        top16_logp_mean_abs_error=float(error.mean()),top16_logp_max_abs_error=float(error.max()))
    rows=dp.gather(row)
    passed=all(r['zero_initialized'] and r['hidden_relative_error']<=1e-3 and
               r['top16_logp_mean_abs_error']<=.002 and r['top16_logp_max_abs_error']<=.05 for r in rows)
    atomic_json(Path(output)/'adapter_continuity.json',dict(passed=passed,ranks=rows,
        comparison='same inputs, microbatch, official forward, autocast, compact-head support and sampled positions',
        adapter_updates=0,
        tolerances=dict(hidden_relative=.001,top16_logp_mean=.002,top16_logp_max=.05)))
    if dp.primary():print(f'[ADAPTER CONTINUITY] passed={passed} ranks={rows}; report={Path(output)/"adapter_continuity.json"}',flush=True)
    if not passed:raise RuntimeError('Warmup-to-Adapter continuity failed; no Adapter optimizer update was performed; '
                                     f'same-forward errors={rows}')


def sample(record,size):
    # Uniform positions across the LOCAL training cache, without concatenating GB of h.
    total=sum(len(x['h']) for x in record['items'])
    indexes=torch.linspace(0,total-1,min(size,total)).round().long();parts=[];offset=0
    for item in record['items']:
        chosen=indexes[(indexes>=offset)&(indexes<offset+len(item['h']))]-offset
        if len(chosen):parts.append({k:item[k][chosen] for k in ('h','ids','teacher')})
        offset+=len(item['h'])
    return {k:torch.cat([p[k] for p in parts]) for k in parts[0]}


@torch.no_grad()
def distribution(adapter,head,probe):
    from .opd_top16 import reverse_kl
    h=adapter(probe['h'])
    with torch.autocast(device_type=h.device.type,dtype=torch.bfloat16,enabled=h.device.type=='cuda'):
        z=F.linear(h,head).float()
    logp=z.log_softmax(-1)
    kl=reverse_kl(z.gather(-1,probe['ids']),probe['teacher'])
    return logp,kl,h.float()


def guarded_step(actor,record,step,output):
    """Only a committed candidate advances Adam; rejected candidates restore all state."""
    cfg=dict(actor.config.policy_loss['fixed_u_adapter'].get('guard',{}));a=actor.fixed_u_adapter
    optimizer=actor.actor_optimizer;started=time.perf_counter()
    probe={k:v.to(a.A.device) for k,v in sample(record,int(cfg.get('tokens_per_rank',128))).items()}
    head=actor._adapter_head_weight.to(probe['h'].dtype)
    before,teacher_before,h_before=distribution(a,head,probe)
    base_A=a.A.detach().clone();base_state=copy.deepcopy(optimizer.state_dict())
    base_lrs=[g['lr'] for g in optimizer.param_groups]
    warm=min(step/max(1,int(cfg.get('warmup_updates',10))),1.)
    attempts=[];accepted=False;optimizer_seconds=0.
    for trial in range(int(cfg.get('backtracks',6))+1):
        with torch.no_grad():a.A.copy_(base_A)
        optimizer.load_state_dict(copy.deepcopy(base_state))
        scale=warm*(.5**trial)
        for group,lr in zip(optimizer.param_groups,base_lrs):group['lr']=lr*scale
        if a.A.is_cuda:torch.cuda.synchronize(a.A.device)
        tick=time.perf_counter();optimizer.step()
        if a.A.is_cuda:torch.cuda.synchronize(a.A.device)
        optimizer_seconds+=time.perf_counter()-tick
        after,teacher_after,h_after=distribution(a,head,probe)
        with torch.no_grad():
            policy=(before.exp()*(before-after)).sum(-1).clamp_min(0)
            relative=(h_after-h_before).norm(dim=-1)/h_before.norm(dim=-1).clamp_min(1e-8)
            local=dict(n=len(policy),policy_kl_mean=float(policy.mean()),policy_kl_max=float(policy.max()),
                hidden_step_max=float(relative.max()),teacher_before=float(teacher_before.mean()),teacher_after=float(teacher_after.mean()),
                finite=bool(torch.isfinite(after).all() and torch.isfinite(a.A).all()))
        rows=dp.gather(local);n=sum(r['n'] for r in rows)
        result={key:sum(r[key]*r['n'] for r in rows)/n for key in ('policy_kl_mean','teacher_before','teacher_after')}
        result.update(policy_kl_max=max(r['policy_kl_max'] for r in rows),hidden_step_max=max(r['hidden_step_max'] for r in rows),
            finite=all(r['finite'] for r in rows),trial=trial,lr_scale=scale)
        accepted=bool(result['finite'] and result['policy_kl_mean']<=cfg.get('max_policy_kl',.002) and
            result['policy_kl_max']<=cfg.get('max_token_policy_kl',.02) and result['hidden_step_max']<=cfg.get('max_hidden_step',.01) and
            result['teacher_after']<=result['teacher_before']*(1+cfg.get('max_teacher_relative_increase',.05))+1e-5)
        result['accepted']=accepted;attempts.append(result)
        if accepted:break
    if not accepted:
        with torch.no_grad():a.A.copy_(base_A)
        optimizer.load_state_dict(base_state)
    for group,lr in zip(optimizer.param_groups,base_lrs):group['lr']=lr
    atomic_json(Path(output)/'adapter_guard'/f'update_{step:03d}.json',dict(accepted=accepted,attempts=attempts,
        local_update=step,config=cfg,selection_data='deterministic training-token sample, NEVER validation',
        policy_metric='FULL vocabulary KL(before || candidate)',committed_updates=int(accepted)))
    if not accepted:
        if dp.primary():torch.save(dict(adapter=a.state_dict(),optimizer=optimizer.state_dict(),local_update=step-1),
            Path(output)/'adapter_guard/recovered_state.pt')
        raise RuntimeError('Adapter trust checks rejected every candidate; A and Adam restored, safe state saved; training stopped')
    return {'adapter/guard_trials':float(len(attempts)),'adapter/effective_lr':base_lrs[0]*scale,
        'adapter/guard_policy_kl':result['policy_kl_mean'],'adapter/guard_hidden_step_max':result['hidden_step_max'],
        'timing/adapter_guard_seconds':time.perf_counter()-started,'timing/adapter_guard_optimizer_seconds':optimizer_seconds}
