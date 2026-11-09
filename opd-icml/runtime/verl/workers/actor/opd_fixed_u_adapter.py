"""Scheme 1: frozen backbone/head, train A only, h' = h + (h A^T) U^T.

Separate experiment, NOT an equivalent implementation of projected-gradient r128.
Current-batch hidden cache is RAM-only and discarded after reuse.
"""
import hashlib
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from . import opd_distributed as dp


class FixedUAdapter(torch.nn.Module):
    def __init__(self,basis):
        super().__init__();u=basis.detach().float()
        if u.ndim!=2 or not torch.allclose(u.T@u,torch.eye(u.shape[1],device=u.device),atol=2e-3):
            raise ValueError('Adapter requires an orthonormal d x r basis')
        self.register_buffer('U',u);self.A=torch.nn.Parameter(torch.zeros(u.shape[1],u.shape[0],device=u.device))
        self.suspended=False

    def forward(self,h):
        if self.suspended:return h.detach()
        # Backbone is deliberately detached even if a caller enabled gradients.
        base=h.detach()
        with torch.autocast(device_type=h.device.type,enabled=False):
            correction=(base.float()@self.A.T)@self.U.T
        return (base.float()+correction).to(base.dtype)

    @torch.no_grad()
    def merged_head(self,weight,chunk=2048):
        result=torch.empty_like(weight)
        a=self.A.to(weight.device);u=self.U.to(weight.device)
        for start in range(0,len(weight),chunk):
            w=weight[start:start+chunk].float()
            result[start:start+chunk]=(w+(w@u)@a).to(weight.dtype)
        return result


def untie_output_head(model):
    """Copy the OUTPUT head only, preserving the exact pretrained function.

    Run after loading tied checkpoint weights, before FSDP/optimizer creation.
    Never merely set tie_word_embeddings=False before loading a checkpoint:
    tied safetensors often omit lm_head.weight, which would leave a random head.
    """
    head=model.lm_head
    embedding=model.get_input_embeddings() if hasattr(model,'get_input_embeddings') else getattr(model.model,'embed_tokens',None)
    shared=embedding is not None and head.weight is embedding.weight
    if getattr(model.config,'tie_word_embeddings',False) or shared:
        if head.weight.is_meta:raise ValueError('Load real warmup weights before untying the output head')
        if embedding is head:raise ValueError('Input and output modules are identical; unsupported architecture')
        head.weight=torch.nn.Parameter(head.weight.detach().clone(),requires_grad=False)
        model.config.tie_word_embeddings=False
    if embedding is not None and head.weight.data_ptr()==embedding.weight.data_ptr():
        raise RuntimeError('Output head still aliases the frozen input embedding')
    return model


def rollout_engine_kwargs(kwargs,model_hf_config):
    """Use a separate output head in vLLM too; original on-disk config stays tied."""
    result=dict(kwargs)
    if getattr(model_hf_config,'opd_fixed_u_adapter_untied_head',False):
        overrides=result.get('hf_overrides') or {}
        if not isinstance(overrides,dict):
            from collections.abc import Mapping
            if not isinstance(overrides,Mapping):raise TypeError('Adapter requires mapping hf_overrides')
        result['hf_overrides']={**dict(overrides),'tie_word_embeddings':False}
    return result


def create(model,cfg,device):
    untie_output_head(model)
    if cfg.get('basis_kind','learned')=='random':
        g=torch.Generator(device='cpu').manual_seed(int(cfg.get('seed',42)))
        basis=torch.linalg.qr(torch.randn(model.config.hidden_size,int(cfg.get('rank',128)),generator=g)).Q
    else:
        with np.load(cfg['basis_path'],allow_pickle=False) as f:basis=torch.from_numpy(f['basis'].copy())
    if basis.shape[0]!=model.config.hidden_size:raise ValueError('Wrong model-scale basis')
    model.requires_grad_(False)
    if hasattr(model,'gradient_checkpointing_disable'):model.gradient_checkpointing_disable()
    adapter=FixedUAdapter(basis).to(device)
    # External module: do NOT register A inside the already frozen FSDP model.
    object.__setattr__(model,'opd_external_adapter',adapter)
    model.model.norm.register_forward_hook(lambda m,a,h:adapter(h))
    return adapter


def merge_weights(weights,adapter):
    found=False
    for name,value in weights:
        if name=='lm_head.weight':
            if hasattr(value,'full_tensor'):value=value.full_tensor()
            if not isinstance(value,torch.Tensor):raise TypeError('Unsupported sharded LM-head export')
            value=adapter.merged_head(value);found=True
        yield name,value
    if not found:raise RuntimeError('Rollout sync did not contain lm_head.weight; refusing stale adapter rollout')


def selected_logits(h,weight,ids):
    unique,inverse=torch.unique(ids.reshape(-1),sorted=True,return_inverse=True)
    # FSDP holds FP32 master weights while cached model activations are BF16.
    # Match the ordinary autocast head without converting the whole vocabulary.
    selected=weight.index_select(0,unique).to(h.dtype)
    # Final RMSNorm may return FP32 even in a BF16 model. The official head
    # runs under CUDA BF16 autocast; cached training must use that same policy,
    # not silently switch to FP32 merely because h is FP32.
    with torch.autocast(device_type=h.device.type,dtype=torch.bfloat16,enabled=h.device.type=='cuda'):
        return F.linear(h,selected).gather(-1,inverse.reshape_as(ids))


def update(actor,data):
    if actor.config.policy_loss.get('gradient_study',{}).get('adapter_statistics',False):
        from .opd_adapter_diagnostics import update_with_statistics
        return update_with_statistics(actor,data)
    from .opd_top16 import forward,reverse_kl,_unwrap_actor_module
    from .opd_gradient_study import reshard_after_readout
    adapter=actor.fixed_u_adapter;cfg=actor.config.policy_loss['fixed_u_adapter']
    study_cfg=actor.config.policy_loss.get('gradient_study',{})
    if study_cfg.get('enabled',False) and not study_cfg.get('validation_only',False):
        raise ValueError('Adapter gradients live in A, not the full-model parameter space; run the full-model study separately')
    if any(p.requires_grad for p in actor.actor_module.parameters()):raise RuntimeError('Backbone/head not fully frozen')
    actor.actor_module.eval();device=next(actor.actor_module.parameters()).device
    adapter.to(device);cache=[];head_weight=None;started=time.perf_counter()
    head=_unwrap_actor_module(_unwrap_actor_module(actor.actor_module).lm_head)
    total=dp.total(float(data.batch['response_mask'].sum()))
    reuse=min(int(actor.config.policy_loss.get('jump_reuse_updates',1)),int(data.meta_info.get('opd_max_updates_this_batch',8)))
    if not 1<=reuse<=8:raise ValueError('invalid adapter reuse budget')
    adapter.suspended=True
    try:
        with torch.no_grad():
            for part in data.split(actor.config.ppo_micro_batch_size_per_gpu):
                x={k:v.to(device) for k,v in part.batch.items()};mask=x['response_mask'].bool()
                def capture(_m,args):
                    nonlocal head_weight
                    if head_weight is None:head_weight=head.weight.detach().clone()
                    if head.bias is not None:raise ValueError('This adapter implementation expects a bias-free output head')
                    h=args[0][:,:-1][mask].detach().to('cpu',copy=True)
                    cache.append((h,x['opd_top16_ids'][mask].cpu(),x['opd_teacher_top16_logits'][mask].cpu()))
                handle=head.register_forward_pre_hook(capture)
                try:forward(actor,x,float(data.meta_info['temperature']),x['opd_top16_ids'],diagnostics=False,hidden_only=True)
                finally:handle.remove()
    finally:adapter.suspended=False
    reshard_after_readout(actor);cache_seconds=time.perf_counter()-started
    if not cache:raise RuntimeError('Empty adapter feature cache')
    cache_bytes=sum(t.numel()*t.element_size() for entry in cache for t in entry)
    losses=[];norms=[];train_started=time.perf_counter()
    for inner in range(reuse):
        actor.actor_optimizer.zero_grad(set_to_none=True);loss_sum=0.
        for h,ids,teacher in cache:
            for offset in range(0,len(h),int(cfg.get('token_chunk',512))):
                sl=slice(offset,offset+int(cfg.get('token_chunk',512)))
                changed=adapter(h[sl].to(device))
                z=selected_logits(changed,head_weight,ids[sl].to(device))
                loss=reverse_kl(z,teacher[sl].to(device)).sum()/total
                loss.backward();loss_sum+=float(loss.detach())
        if dp.size()>1:torch.distributed.all_reduce(adapter.A.grad)
        norm=torch.nn.utils.clip_grad_norm_([adapter.A],actor.config.grad_clip)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite adapter gradient; update not committed')
        actor.actor_optimizer.step();losses.append(dp.total(loss_sum));norms.append(float(norm))
    del cache,head_weight
    step=int(data.meta_info.get('global_steps',0))
    if dp.primary():print(f'[FIXED-U ADAPTER] rollout={step} updates={reuse} KL={losses} backbone_backwards=0 feature_cache_reused={reuse-1}',flush=True)
    return {
        'actor/opd_optimizer_steps':[float(reuse)],'actor/opd_top16_kl':[losses[-1]],'actor/grad_norm':[norms[-1]],
        'adapter/trainable_parameters':[float(adapter.A.numel())],'adapter/backbone_backwards':[0.],
        'adapter/feature_cache_bytes':[float(cache_bytes)],'adapter/feature_cache_reuses':[float(reuse-1)],
        'timing/adapter_cache_seconds':[cache_seconds],'timing/adapter_updates_seconds':[time.perf_counter()-train_started],
        'monitor/opd_train_trajectories':[float(len(data)*dp.size())],
        **{f'adapter/inner_{i+1}_kl':[v] for i,v in enumerate(losses)}}


def save(worker,path,step):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    if dp.primary():
        tmp=path/'adapter.tmp'
        torch.save(dict(schema='fixed-u-adapter-v1',step=step,state={k:v.cpu() for k,v in worker.fixed_u_adapter.state_dict().items()},
                        optimizer=worker.actor_optimizer.state_dict(),scheduler=worker.actor_lr_scheduler.state_dict(),
                        warmup_model=str(worker.config.model.path),
                        continuity_verified=bool(getattr(worker.actor,'_adapter_continuity_checked',False)),
                        total_update=int(worker.config.actor.policy_loss['gradient_study'].get('update_offset',0))+step),tmp)
        tmp.replace(path/'adapter.pt')
    from .opd_adapter_handoff import save_rng
    save_rng(worker,path)
    dp.barrier()


def load(worker,path):
    saved=torch.load(Path(path)/'adapter.pt',map_location='cpu',weights_only=False)
    if saved['schema']!='fixed-u-adapter-v1' or saved['warmup_model']!=str(worker.config.model.path):
        raise ValueError('Adapter warmup model mismatch')
    worker.fixed_u_adapter.load_state_dict(saved['state']);worker.actor_optimizer.load_state_dict(saved['optimizer'])
    worker.actor_lr_scheduler.load_state_dict(saved['scheduler'])
    if not saved.get('continuity_verified',False):
        raise ValueError('Adapter resume checkpoint lacks verified original handoff')
    state=worker.actor_optimizer.state[worker.fixed_u_adapter.A]
    if int(state.get('step',-1))!=int(saved['step']):
        raise ValueError('Adapter Adam step does not match saved data/update cursor')
    worker.actor._adapter_continuity_checked=True
    from .opd_adapter_handoff import load_rng
    load_rng(worker,path)
    if dp.primary():print(f'[ADAPTER RESUME] A + Adam moments/step + scheduler + per-rank RNG restored; '
                          f'local_update={saved["step"]}, total_update={saved.get("total_update")}. No optimizer reset.',flush=True)
