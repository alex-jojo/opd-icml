"""Single-GPU student-support Top-16 reverse KL with hidden-space jump modes."""
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


_BASIS_CACHE={}
_BASIS_DEVICE_CACHE={}


def _first_update_memory(actor,stage):
    """Log the real first update, without an extra forward/backward/preflight."""
    from . import opd_distributed as dp
    device=next(actor.actor_module.parameters()).device
    if device.type!='cuda':return
    torch.cuda.synchronize(device)
    free,total=torch.cuda.mem_get_info(device)
    states=[float(s['step']) for s in actor.actor_optimizer.state.values() if 'step' in s]
    print(f'[FIRST UPDATE rank={dp.rank()}] {stage}: '
          f'allocated={torch.cuda.memory_allocated(device)/2**30:.2f} GiB '
          f'peak={torch.cuda.max_memory_allocated(device)/2**30:.2f} GiB '
          f'free={free/2**30:.2f}/{total/2**30:.2f} GiB '
          f'Adam_state_tensors={len(states)} step_range={([min(states),max(states)] if states else [])}',flush=True)


def hidden_projection_config(actor):
    cfg=getattr(actor.config,'policy_loss',{}).get('hidden_projection',None)
    return cfg if cfg and bool(cfg.get('enabled',False)) else None


def load_hidden_basis(actor):
    cfg=hidden_projection_config(actor)
    if cfg is None:return None
    path=Path(str(cfg.get('asset_path') or ''))
    rank=int(cfg.get('rank',64))
    if not path.is_file():raise FileNotFoundError(f'quoted hidden_projection.asset_path is required: {path}')
    key=(str(path.resolve()),rank)
    if key not in _BASIS_CACHE:
        with np.load(path,allow_pickle=False) as payload:basis=np.asarray(payload['basis'][:,:rank],dtype=np.float32)
        expected=int(_final_norm_module(actor.actor_module).weight.numel())
        if basis.ndim!=2 or basis.shape!=(expected,rank) or not np.isfinite(basis).all():
            raise ValueError(f'bad hidden projection basis: {basis.shape}, expected ({expected},{rank})')
        error=float(np.max(np.abs(basis.T@basis-np.eye(rank,dtype=np.float32))))
        if error>2e-3:raise ValueError(f'hidden projection basis is not orthonormal: {error}')
        _BASIS_CACHE[key]=torch.from_numpy(basis)
    device=next(actor.actor_module.parameters()).device
    dtype=getattr(actor,'param_dtype',torch.bfloat16)
    device_key=(key,str(device),dtype)
    if device_key not in _BASIS_DEVICE_CACHE:
        _BASIS_DEVICE_CACHE[device_key]=_BASIS_CACHE[key].to(device=device,dtype=dtype)
    return _BASIS_DEVICE_CACHE[device_key]


class _ProjectHiddenGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx,hidden,basis,state):
        ctx.save_for_backward(basis);ctx.state=state
        return hidden

    @staticmethod
    def backward(ctx,gradient):
        (basis,)=ctx.saved_tensors
        if ctx.state.get('raw_gradient_observer') is not None:
            ctx.state['raw_gradient_observer'](gradient)
        coefficients=gradient@basis
        projected=coefficients@basis.T
        with torch.no_grad():
            # Accumulate device scalars; converting each microbatch to Python
            # forced two GPU synchronizations per backward for diagnostics only.
            # FP32 reduction without materializing two full FP32 squared tensors.
            # The projected gradient itself is unchanged (including BF16 rounding).
            full=torch.linalg.vector_norm(gradient,dtype=torch.float32).square()
            kept=torch.linalg.vector_norm(projected,dtype=torch.float32).square()
            ctx.state['full']+=full;ctx.state['kept']+=kept
            if ctx.state.get('capture_direction',False):
                direction=coefficients.float().reshape(-1,basis.shape[1]).sum(dim=0).detach()
                previous=ctx.state.get('current_direction')
                ctx.state['current_direction']=direction if previous is None else previous+direction
        return projected,None,None


def _unwrap_actor_module(module):
    """Return the Hugging Face model underneath wrappers such as FSDP."""
    seen=set()
    while hasattr(module,'module') and id(module) not in seen:
        seen.add(id(module))
        module=module.module
    return module


def _final_norm_module(actor_module):
    """Locate the post-transformer norm whose output is consumed by lm_head."""
    model=_unwrap_actor_module(actor_module)
    backbone=getattr(model,'model',None)
    norm=getattr(backbone,'norm',None) if backbone is not None else None
    if norm is None:
        raise RuntimeError(
            'hidden-gradient projection requires a causal-LM final norm at '
            '<unwrapped_model>.model.norm; refusing to bypass the official model forward'
        )
    return norm


@contextmanager
def _project_official_final_hidden(actor_module,basis,state):
    """Project dL/dh inline while keeping the official FSDP-managed lm_head path."""
    norm=_final_norm_module(actor_module)

    def hook(_module,_args,output):
        if not torch.is_tensor(output):
            raise TypeError(f'expected final norm to return Tensor, got {type(output)!r}')
        if output.shape[-1]!=basis.shape[0]:
            raise ValueError(
                f'basis/hidden width mismatch: hidden={output.shape[-1]}, basis={basis.shape[0]}'
            )
        return _ProjectHiddenGradient.apply(output,basis,state)

    handle=norm.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def reverse_kl(student,teacher):
    lp=F.log_softmax(student.float(),dim=-1)
    lq=F.log_softmax(teacher.detach().float(),dim=-1)
    return (lp.exp()*(lp-lq)).sum(-1)


def adaptive_probe_indices(batch_size,probe_size):
    """Deterministic spread over the batch; never changes the training batch."""
    if not 1<=probe_size<=batch_size:raise ValueError('adaptive probe size must be 1..batch size')
    return np.linspace(0,batch_size-1,num=probe_size,dtype=np.int64).tolist()


@torch.no_grad()
def adaptive_probe(actor,data,indices):
    """One small student-only forward; no teacher call and no backward pass."""
    device=next(actor.actor_module.parameters()).device
    part=data.select_idxs(indices).to(device);inputs=dict(part.batch)
    response_len=inputs['responses'].shape[1]
    with torch.autocast(device_type=device.type,dtype=getattr(actor,'param_dtype',torch.bfloat16),
                        enabled=device.type=='cuda'):
        output=actor.actor_module(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'],
            position_ids=inputs['position_ids'],use_cache=False,return_dict=True,
            logits_to_keep=response_len+1)
        selected=output.logits[:,-response_len-1:-1,:].gather(-1,inputs['opd_top16_ids'])
    del output
    mask=inputs['response_mask'].float()
    denominator=mask.sum().clamp_min(1)
    policy_kl=float((reverse_kl(inputs['opd_student_top16_logits'],selected)*mask).sum()/denominator)
    teacher_kl=float((reverse_kl(selected,inputs['opd_teacher_top16_logits'])*mask).sum()/denominator)
    return policy_kl,teacher_kl


def split_heldout(data,probe_size):
    """Reserve entire trajectories, never tokens from trajectories being trained."""
    if not 1<=probe_size<len(data):
        raise ValueError('held-out probe size must be smaller than the rollout batch')
    probe_indices=adaptive_probe_indices(len(data),probe_size)
    excluded=set(probe_indices)
    train_indices=[i for i in range(len(data)) if i not in excluded]
    return data.select_idxs(train_indices),data.select_idxs(probe_indices)


def cached_teacher_kl(data):
    mask=data.batch['response_mask'].float()
    return float((reverse_kl(data.batch['opd_student_top16_logits'],
                             data.batch['opd_teacher_top16_logits'])*mask).sum()
                 /mask.sum().clamp_min(1))


def heldout_probe(actor,data,micro_batch_size):
    """Student-only no-grad forwards; teacher logits were cached at rollout time."""
    policy_sum=0.;teacher_sum=0.;tokens=0.
    for start in range(0,len(data),micro_batch_size):
        indices=list(range(start,min(start+micro_batch_size,len(data))))
        part=data.select_idxs(indices)
        weight=float(part.batch['response_mask'].sum())
        if weight<=0:continue
        policy,teacher=adaptive_probe(actor,part,list(range(len(part))))
        policy_sum+=policy*weight;teacher_sum+=teacher*weight;tokens+=weight
    if tokens<=0:raise ValueError('held-out set has no response tokens')
    return policy_sum/tokens,teacher_sum/tokens


def heldout_gradient_probe(actor,data,basis):
    """Server-side held-out loss and 128D hidden gradient; never optimizer.step()."""
    state={'full':0.,'kept':0.,'capture_direction':True,'current_direction':None}
    total=data.batch['response_mask'].sum().clamp_min(1)
    policy_sum=0.;teacher_sum=0.
    actor.actor_optimizer.zero_grad(set_to_none=True)
    try:
        for part in data.split(actor.config.ppo_micro_batch_size_per_gpu):
            part=part.to(next(actor.actor_module.parameters()).device)
            inputs=dict(part.batch)
            _,selected,_,_=forward(
                actor,inputs,float(data.meta_info['temperature']),inputs['opd_top16_ids'],
                gradient_basis=basis,projection_state=state)
            mask=inputs['response_mask'].float()
            denominator=total.to(mask.device)
            teacher_loss=(reverse_kl(selected,inputs['opd_teacher_top16_logits'])*mask).sum()/denominator
            with torch.no_grad():
                policy_loss=(reverse_kl(inputs['opd_student_top16_logits'],selected.detach())*mask
                             ).sum()/denominator
            teacher_loss.backward()
            teacher_sum+=float(teacher_loss.detach());policy_sum+=float(policy_loss)
        direction=state['current_direction']
        if direction is None or not bool(torch.isfinite(direction).all()):
            raise RuntimeError('held-out projected hidden gradient was not captured')
        return policy_sum,teacher_sum,direction.detach()
    finally:
        # The held-out trajectories must not contaminate the following train update.
        actor.actor_optimizer.zero_grad(set_to_none=True)


def gradient_novelty(direction,earlier):
    """Fraction of a held-out 128D direction outside earlier train directions."""
    if not earlier:return 1.
    matrix=torch.stack([value.float() for value in earlier],dim=1)
    basis,singular,_=torch.linalg.svd(matrix,full_matrices=False)
    if singular.numel()==0 or float(singular[0])<=0:return 1.
    rank=int((singular>singular[0]*1e-3).sum())
    unit=basis[:,:rank]
    vector=direction.float()
    residual=vector-unit@(unit.T@vector)
    return float(residual.norm()/vector.norm().clamp_min(1e-30))


@torch.no_grad()
def set_parameter_ray_scale(parameters,before,current_scale,target_scale):
    """Move on the actual post-AdamW parameter ray, not a fake 128D parameter basis."""
    if current_scale==target_scale:
        return  # The selected candidate is already installed; do not recopy the full model.
    ratio=target_scale/current_scale
    for parameter,base in zip(parameters,before):
        parameter.copy_((base.float()+ratio*(parameter.float()-base.float())).to(parameter.dtype))


def checked_super_jump(actor,heldout,before,initial_probe_loss,config):
    """One backward/AdamW step, then bounded no-grad held-out line search."""
    mid=float(config.get('super_jump_mid_scale',1.5))
    maximum=float(config.get('super_jump_max_scale',2.0))
    minimum_gain=float(config.get('super_jump_min_probe_gain',0.0001))
    maximum_policy_kl=float(config.get('super_jump_max_policy_kl',0.05))
    if not (1.<mid<maximum<=4. and minimum_gain>=0 and maximum_policy_kl>0):
        raise ValueError('invalid super-jump scale or trust-region configuration')
    parameters=[p for p in actor.actor_module.parameters() if p.requires_grad]
    if len(parameters)!=len(before):raise RuntimeError('parameter layout changed during super-jump')
    start=time.perf_counter();current_scale=1.;best_scale=1.
    policy,loss=heldout_probe(actor,heldout,actor.config.ppo_micro_batch_size_per_gpu)
    if not np.isfinite(policy) or not np.isfinite(loss):
        raise FloatingPointError('Super base-step held-out probe is nonfinite; abort and resume the last checkpoint')
    best_loss=loss;best_policy=policy
    try:
        if np.isfinite(loss) and loss<initial_probe_loss:
            for scale in (mid,maximum):
                set_parameter_ray_scale(parameters,before,current_scale,scale)
                current_scale=scale
                candidate_policy,candidate_loss=heldout_probe(
                    actor,heldout,actor.config.ppo_micro_batch_size_per_gpu)
                if (np.isfinite(candidate_loss) and np.isfinite(candidate_policy)
                        and candidate_policy<=maximum_policy_kl
                        and candidate_loss<best_loss-minimum_gain):
                    best_scale=scale;best_loss=candidate_loss;best_policy=candidate_policy
    except BaseException:
        set_parameter_ray_scale(parameters,before,current_scale,1.)
        raise
    set_parameter_ray_scale(parameters,before,current_scale,best_scale)
    return best_scale,best_policy,best_loss,time.perf_counter()-start


def _teacher_vocab_ids(actor,token_ids):
    """Swap student/native-teacher EOS labels for teacher output lookup."""
    student_eos=getattr(actor,'opd_student_eos_token_id',None)
    teacher_eos=getattr(actor,'opd_teacher_native_eos_token_id',None)
    if student_eos is None or teacher_eos is None or student_eos==teacher_eos:return token_ids
    mapped=token_ids.clone();student_mask=token_ids.eq(student_eos);teacher_mask=token_ids.eq(teacher_eos)
    mapped[student_mask]=teacher_eos;mapped[teacher_mask]=student_eos
    return mapped


def forward(actor,inputs,temperature,ids=None,teacher=False,gradient_basis=None,projection_state=None,diagnostics=True,hidden_only=False,response_position=None):
    if temperature!=1.:raise ValueError('This experiment fixes temperature=1')
    if actor.use_remove_padding or actor.use_ulysses_sp:raise ValueError('Top16 audit uses padded, single-GPU HF forwards')
    n=inputs['responses'].shape[1]
    if response_position is not None:
        if ids is None or diagnostics or hidden_only or gradient_basis is not None:
            raise ValueError('Single-position forward requires fixed IDs, diagnostics=False and no hidden projection')
        if int(response_position)!=response_position or not 0<=response_position<n:
            raise ValueError('Token position is outside the response')
        response_position=int(response_position)
    prefix='ref_' if teacher and 'ref_input_ids' in inputs else ''
    device=inputs[prefix+'input_ids'].device
    projection_context=(nullcontext() if gradient_basis is None else
        _project_official_final_hidden(actor.actor_module,gradient_basis,projection_state))
    from .opd_fast_head import compact_head, enabled
    use_compact=ids is not None and not diagnostics and enabled(actor)
    head=_unwrap_actor_module(_unwrap_actor_module(actor.actor_module).lm_head)
    gather_ids=_teacher_vocab_ids(actor,ids) if teacher and ids is not None else ids
    head_context=compact_head(head,gather_ids,hidden_only,response_position) if use_compact or hidden_only else nullcontext()
    input_ids=inputs[prefix+'input_ids'];attention_mask=inputs[prefix+'attention_mask'];position_ids=inputs[prefix+'position_ids']
    if getattr(actor.config,'policy_loss',{}).get('replication17',{}).get('trim_prompt_padding',False):
        # Only a COMMON all-zero leading prompt prefix; keep every response
        # position and at least one preceding token. Absolute RoPE IDs stay fixed.
        prompt_width=input_ids.shape[-1]-n
        if prompt_width>1:
            empty=(attention_mask[:,:prompt_width-1]==0).all(0)
            lead=int(empty.long().cumprod(0).sum())
            input_ids=input_ids[:,lead:];attention_mask=attention_mask[:,lead:];position_ids=position_ids[...,lead:]
    with projection_context,head_context:
        with torch.autocast(
            device_type=device.type,
            dtype=getattr(actor,'param_dtype',torch.bfloat16),
            enabled=device.type=='cuda',
        ):
            out=actor.actor_module(input_ids=input_ids,attention_mask=attention_mask,
                position_ids=position_ids,use_cache=False,return_dict=True,logits_to_keep=n+1)
            if response_position is not None and use_compact:
                # compact_head already selected the prediction BEFORE projection;
                # retain the official model's output transformations (e.g. softcap).
                logits=out.logits[:,:1,:]
            else:
                logits=out.logits[:,-n-1:-1,:]
                if response_position is not None:
                    logits=logits[:,response_position:response_position+1,:]
    if hidden_only:return None,None,None,None
    if response_position is not None:
        ids=ids[:,response_position:response_position+1,:]
    if gradient_basis is not None and ids is None:
        raise ValueError('hidden-gradient projection is actor-update only and requires fixed student Top-16 IDs')
    if ids is None:ids=logits.detach().topk(16,dim=-1).indices
    gather_ids=_teacher_vocab_ids(actor,ids) if teacher else ids
    selected=logits if use_compact else logits.gather(-1,gather_ids)
    if gradient_basis is not None or not diagnostics:
        with torch.no_grad():
            # The normalizer/sampled diagnostics were already measured before update.
            sampled=torch.zeros_like(selected[...,0],dtype=torch.float32)
            mass=torch.zeros_like(sampled)
        return ids,selected,sampled,mass
    # temperature fixed to 1; avoid a full-vocabulary FP32 copy
    # Chunked log normalizers avoid a second full FP32 vocabulary tensor.
    with torch.no_grad():
        lse=torch.cat([x.float().logsumexp(-1) for x in logits.split(64,dim=1)],dim=1)
        response_ids=_teacher_vocab_ids(actor,inputs['responses']) if teacher else inputs['responses']
        sampled=logits.detach().gather(-1,response_ids.unsqueeze(-1)).squeeze(-1).float()-lse
        mass=(selected.detach().float()-lse.unsqueeze(-1)).exp().sum(-1)
    return ids,selected,sampled,mass


@torch.no_grad()
def measure(actor,data):
    actor.actor_module.eval();out={};actor.last_top16_measurement={}
    teacher='opd_top16_ids' in data.batch
    for part in data.split(int(data.meta_info['micro_batch_size'])):
        device=next(actor.actor_module.parameters()).device
        part=part.to(device);inputs=dict(part.batch)
        ids,logits,lp,mass=forward(actor,inputs,float(data.meta_info['temperature']),inputs.get('opd_top16_ids'),teacher=teacher)
        fields=({'opd_teacher_top16_logits':logits,'opd_teacher_top16_mass':mass} if teacher else
                {'opd_top16_ids':ids,'opd_student_top16_logits':logits,'opd_student_top16_mass':mass})
        fields['log_prob']=lp
        for k,v in fields.items():out.setdefault(k,[]).append(v.detach().cpu())
    tensors={k:torch.cat(v) for k,v in out.items()}
    lp=tensors.pop('log_prob');actor.last_top16_measurement=tensors
    return lp,torch.zeros_like(lp)


def update(actor,data):
    from . import opd_distributed as dp
    if actor.config.policy_loss.get('fixed_u_adapter',{}).get('enabled',False):
        from .opd_fixed_u_adapter import update as adapter_update
        return adapter_update(actor,data)
    world=dp.size();rep_cfg=actor.config.policy_loss.get('replication17',{})
    study_payload=data.meta_info.pop('gradient_study_payload',None)
    if world!=1 and int(rep_cfg.get('distributed_world_size',1))!=world:
        raise ValueError('Multi-GPU OPD requires replication distributed_world_size to match the worker count')
    if world>1:
        if rep_cfg.get('heavy_steps') or not rep_cfg.get('enabled'):
            raise ValueError('Distributed OPD supports passive replication diagnostics, not disposable shadow updates')
        if len(data)!=actor.config.ppo_mini_batch_size:raise ValueError('Wrong rank-local rollout batch')
        data=dp.global_batch(data)
    if getattr(actor,'scaler',None) is not None:raise ValueError('Use BF16, not FP16 GradScaler, for this experiment')
    actor.actor_module.train()
    if actor.config.ppo_epochs!=1:raise ValueError('Top16 requires one epoch per rollout batch')
    if len(data)!=actor.config.ppo_mini_batch_size*world:
        raise ValueError('Top16 requires one full rollout batch per actor call')
    if actor.config.policy_loss.get('transfer_guided', {}).get('enabled', False):
        if actor.config.policy_loss.get('response_guided', {}).get('enabled', False):
            raise ValueError('select one V4/V5 runtime, not both')
        from .opd_transfer_guided import get_runtime
        if int(data.meta_info.get('opd_max_updates_this_batch', 1)) < 1:
            raise ValueError('no transfer-suite optimizer-update budget remains')
        return get_runtime(actor).update(data)
    if actor.config.policy_loss.get('response_guided', {}).get('enabled', False):
        from .opd_response_guided import get_runtime
        if int(data.meta_info.get('opd_max_updates_this_batch', 1)) < 1:
            raise ValueError('no V4 optimizer-update budget remains')
        return get_runtime(actor).update(data)
    reuse=int(getattr(actor.config,'policy_loss',{}).get('jump_reuse_updates',1))
    if not 1<=reuse<=8:raise ValueError('jump_reuse_updates must be in [1,8]')
    max_inner=min(reuse,int(data.meta_info.get('opd_max_updates_this_batch',reuse)))
    if not 1<=max_inner<=reuse:raise ValueError('invalid remaining OPD optimizer-update budget')
    total=data.batch['response_mask'].sum().clamp_min(1)
    basis=load_hidden_basis(actor)
    policy_cfg=getattr(actor.config,'policy_loss',{})
    bottleneck = None
    if policy_cfg.get('bottleneck', {}).get('enabled', False):
        from .opd_bottleneck import get_runtime
        bottleneck = get_runtime(actor)
    if bottleneck is not None:
        if (basis is not None or reuse != 1 or any(bool(policy_cfg.get(k, False)) for k in
                ('adaptive_reuse_enabled', 'marginal_reuse_enabled', 'super_jump_enabled'))
                or policy_cfg.get('plasticity_diagnostic', {}).get('enabled', False)
                or int(policy_cfg.get('fixed_heldout_probe_size', 0))):
            raise ValueError('adaptive bottleneck requires ordinary reuse=1 OPD without other projection/controllers/diagnostics')
        if any(isinstance(m, torch.nn.Dropout) and m.p > 0 for m in actor.actor_module.modules()):
            raise ValueError('bottleneck before/after response requires dropout-free forwards')
        model_cfg = getattr(actor.actor_module, 'config', None)
        if any(float(getattr(model_cfg, k, 0.) or 0.) > 0 for k in
               ('attention_dropout', 'hidden_dropout', 'hidden_dropout_prob', 'attention_probs_dropout_prob')):
            raise ValueError('bottleneck response requires dropout-free functional attention too')
        rates = {float(group['lr']) for group in actor.actor_optimizer.param_groups}
        if len(rates) != 1 or min(rates) <= 0:
            raise ValueError('bottleneck calibration expects one positive shared AdamW learning rate')
        bottleneck_lr = rates.pop()
    diagnostic=None
    if policy_cfg.get('plasticity_diagnostic',{}).get('enabled',False):
        from .opd_plasticity import get_diagnostic
        diagnostic=get_diagnostic(actor,data,basis,reuse)
    adaptive=bool(policy_cfg.get('adaptive_reuse_enabled',False))
    super_jump=bool(policy_cfg.get('super_jump_enabled',False))
    marginal=bool(policy_cfg.get('marginal_reuse_enabled',False))
    if sum((adaptive,super_jump,marginal))>1:
        raise ValueError('adaptive, super-jump and marginal-reuse modes are mutually exclusive')
    if marginal and basis is None:
        raise ValueError('marginal reuse requires the rank-128 final-hidden gradient basis')
    if super_jump and basis is None:
        raise ValueError('Super Jump requires the hidden-gradient projection basis')
    if super_jump and reuse!=1:
        raise ValueError('Super Jump uses exactly one actual AdamW update per fresh rollout')
    if marginal and reuse!=4:
        raise ValueError('marginal reuse requires jump_reuse_updates=4')
    if super_jump or marginal:
        probe_size=int(policy_cfg.get(
            'super_jump_probe_size' if super_jump else 'marginal_reuse_probe_size',8))
        train_data,heldout=split_heldout(data,probe_size)
        initial_probe_loss=cached_teacher_kl(heldout)
        if not np.isfinite(initial_probe_loss):
            raise FloatingPointError('cached held-out teacher KL is nonfinite')
        total=train_data.batch['response_mask'].sum().clamp_min(1)
    else:
        train_data=data;heldout=None;initial_probe_loss=float('nan')
        fixed_heldout=int(policy_cfg.get('fixed_heldout_probe_size',0))
        if fixed_heldout:
            train_data,heldout=split_heldout(data,fixed_heldout)
            total=train_data.batch['response_mask'].sum().clamp_min(1)
    if adaptive:
        if basis is None:raise ValueError('adaptive reuse requires a hidden-gradient projection basis')
        if reuse!=int(policy_cfg.get('adaptive_reuse_max_extra_updates',4))+1:
            raise ValueError('jump_reuse_updates must equal adaptive_reuse_max_extra_updates + 1')
        probe_indices=adaptive_probe_indices(len(data),int(policy_cfg.get('adaptive_reuse_probe_size',4)))
        max_policy_kl=float(policy_cfg.get('adaptive_reuse_max_policy_kl',0.005))
        loss_slack=float(policy_cfg.get('adaptive_reuse_teacher_loss_slack',0.02))
        min_cos=float(policy_cfg.get('adaptive_reuse_min_gradient_cosine',0.0))
        if not (0<max_policy_kl<1 and 0<=loss_slack<1 and -1<=min_cos<=1):
            raise ValueError('invalid adaptive reuse thresholds')
        with torch.no_grad():
            probe_data=data.select_idxs(probe_indices)
            probe_mask=probe_data.batch['response_mask'].float()
            probe_loss=float((reverse_kl(probe_data.batch['opd_student_top16_logits'],
                                         probe_data.batch['opd_teacher_top16_logits'])*probe_mask).sum()
                             /probe_mask.sum().clamp_min(1))
    response_observer = None
    replication = None
    if policy_cfg.get('replication17', {}).get('enabled', False):
        if bottleneck is not None or diagnostic is not None or adaptive or super_jump or marginal or policy_cfg.get('response_geometry', {}).get('enabled', False):
            raise ValueError('replication17 cannot be combined with other diagnostic/controllers')
        if policy_cfg.get('gradient_study',{}).get('enabled',False):
            if heldout is not None:raise ValueError('gradient study replaces the old held-out split; set size=0')
            from .opd_study_replication import StudyReplication
            if not hasattr(actor,'_gradient_study_replication'):
                actor._gradient_study_replication=StudyReplication(actor)
            replication=actor._gradient_study_replication
            replication.begin(train_data,basis,int(data.meta_info['global_steps']))
            if study_payload is not None:
                import pickle
                replication.start_study(pickle.loads(study_payload))
        else:
            if heldout is None:raise ValueError('replication17 requires the matched held-out split')
            from .opd_replication17 import get_runtime
            replication = get_runtime(actor)
            response_observer = replication.begin(train_data, heldout, basis, int(data.meta_info['global_steps']))
    if policy_cfg.get('response_geometry', {}).get('enabled', False):
        passive_full=bool(policy_cfg['response_geometry'].get('passive_full',False))
        if (basis is None and not passive_full) or (basis is not None and passive_full) or reuse != 1 or bottleneck is not None or diagnostic is not None or adaptive or super_jump or marginal:
            raise ValueError('response geometry requires ordinary fixed-basis r128, reuse=1, without other controllers')
        if heldout is None:
            raise ValueError('response geometry requires separate held-out trajectories')
        if policy_cfg['response_geometry'].get('token_audit',False):
            if not passive_full:raise ValueError('token audit requires passive FULL mode')
            from .opd_token_audit_runtime import get_observer
        elif passive_full:
            from .opd_mode_absorption import get_observer
        else:
            from .opd_response_geometry import get_observer
        response_observer = get_observer(actor)
        response_observer.begin(train_data, heldout, basis, int(data.meta_info['global_steps']))
    if marginal:
        minimum_updates=int(policy_cfg.get('marginal_reuse_min_updates',2))
        gain_ratio_threshold=float(policy_cfg.get('marginal_reuse_gain_ratio',0.15))
        min_gain=float(policy_cfg.get('marginal_reuse_min_probe_gain',0.0001))
        max_probe_drift=float(policy_cfg.get('marginal_reuse_max_policy_kl',0.05))
        if not (minimum_updates==2 and 0<=gain_ratio_threshold<=1 and
                min_gain>=0 and max_probe_drift>0):
            raise ValueError('invalid marginal-reuse controller configuration')
    projection_state={'full':0.,'kept':0.,'capture_direction':adaptive or marginal,'current_direction':None}
    if response_observer is not None:
        projection_state['raw_gradient_observer'] = response_observer.consume_raw
    losses=[];norms=[];policy_drifts=[]
    optimizer_seconds=0.;train_forward_backward_seconds=0.
    probe_seconds=0.;probe_policy_kl=0.;probe_teacher_kl=0.;direction_cos=0.
    cosine_available=False;inner_retained=[];probe_calls=0
    previous_direction=None;stop_reason='budget'
    marginal_gains=[];marginal_ratios=[];marginal_probe_loss=initial_probe_loss
    train_directions=[];heldout_novelties=[];heldout_train_cosines=[]
    super_scale=1.;super_policy=float('nan');super_loss=float('nan');super_seconds=0.
    # The teacher targets and token IDs are measured once before this call.
    # Every inner update below runs a new student forward and backward pass.
    local_train,train_offset=dp.shard(train_data)
    for inner in range(max_inner):
        actor.actor_optimizer.zero_grad(set_to_none=True)
        if replication is not None and hasattr(replication,'before_inner'):
            replication.before_inner(inner)
        diagnostic_step=int(data.meta_info.get('global_steps',0))
        observer=diagnostic.before(diagnostic_step,total) if diagnostic is not None else None
        if observer is not None:
            observer.projected = basis is not None
            projection_state['raw_gradient_observer'] = observer.consume if basis is not None else None
        projection_state['current_direction']=None
        previous_full=projection_state['full'];previous_kept=projection_state['kept']
        loss_sum=0.;drift_sum=0.
        if bottleneck is not None:
            bottleneck.begin(train_data, total)
            if bottleneck.geometry.step + 1 != diagnostic_step:
                raise ValueError('bottleneck state and trainer optimizer step disagree')
        train_cursor=train_offset
        if response_observer is not None:response_observer.cursor=train_offset
        train_fb_started=time.perf_counter()
        for part in local_train.split(actor.config.ppo_micro_batch_size_per_gpu):
            source_inputs = dict(part.batch) if bottleneck is not None else None
            part=part.to(next(actor.actor_module.parameters()).device);inputs=dict(part.batch)
            with (observer.watch(actor,inputs) if observer is not None else nullcontext()), \
                 (response_observer.watch(inputs) if response_observer is not None else nullcontext()), \
                 (replication.watch_gradient_scopes(inputs,inner,projection_state) if replication is not None else nullcontext()), \
                 (bottleneck.watch(inputs, source_inputs) if bottleneck is not None else nullcontext()), \
                 (replication.watch_training(inputs,inner,train_cursor) if replication is not None else nullcontext()):
                _,selected,_,_=forward(actor,inputs,float(data.meta_info['temperature']),inputs['opd_top16_ids'],
                    gradient_basis=basis,projection_state=projection_state,diagnostics=False)
                mask=inputs['response_mask']
                if reuse>1:
                    with torch.no_grad():
                        # Conditional policy drift on the cached 16-token support.
                        # It is diagnostic, not a full-vocabulary policy KL.
                        drift=reverse_kl(inputs['opd_student_top16_logits'],selected.detach())
                        drift_sum+=(drift*mask).sum()/total.to(drift.device)
                kl=reverse_kl(selected,inputs['opd_teacher_top16_logits'])
                if response_observer is not None:
                    response_observer.record_outputs(selected,inputs['opd_teacher_top16_logits'],mask,kl)
                if observer is not None:
                    observer.record_outputs(selected,inputs['opd_teacher_top16_logits'],mask,kl)
                # FSDP averages rank gradients: restore the GLOBAL token mean,
                # rather than averaging differently sized rank-local means.
                loss=(kl*mask).sum()*world/total.to(kl.device)
                loss.backward();loss_sum+=loss.detach()
            train_cursor+=len(part)
        loss_sum=dp.total(float(loss_sum),average=True);drift_sum=dp.total(float(drift_sum))
        train_forward_backward_seconds+=time.perf_counter()-train_fb_started
        first_study_update=(policy_cfg.get('gradient_study',{}).get('enabled',False)
                            and diagnostic_step==1 and inner==0)
        if first_study_update:_first_update_memory(actor,'backward completed')
        if not np.isfinite(loss_sum):
            actor.actor_optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError('nonfinite OPD loss; no optimizer update committed')
        if bottleneck is not None:
            bottleneck.validate_before_update()
        before=([p.detach().clone() for p in actor.actor_module.parameters() if p.requires_grad]
                if super_jump else None)
        transfer_observer = None
        if diagnostic is not None and diagnostic.transfer:
            from .opd_transfer import TransferObserver
            transfer_observer = TransferObserver(actor, diagnostic, diagnostic_step)
            transfer_observer.before_update(train_data, observer, basis)
        optimizer_started=time.perf_counter()
        if replication is not None:replication.before_update(inner)
        if response_observer is not None:
            if world>1:dp.gather_audit(response_observer)
            response_observer.before_update()
            optimizer_started=time.perf_counter()
        if first_study_update:_first_update_memory(actor,'before AdamW')
        optimizer_started=time.perf_counter()
        norm=actor._optimizer_step()
        if not np.isfinite(float(norm)):
            raise FloatingPointError('OPD optimizer skipped a nonfinite-gradient update; abort without counting it')
        optimizer_seconds+=time.perf_counter()-optimizer_started
        if first_study_update:_first_update_memory(actor,'AdamW completed')
        if response_observer is not None:
            response_observer.after_update()
        if replication is not None:
            replication.after_inner(inner + 1)
        if bottleneck is not None:
            bottleneck_metrics = bottleneck.after(bottleneck_lr)
        if transfer_observer is not None:
            transfer_observer.after_update()
        if diagnostic is not None:
            diagnostic_metrics=diagnostic.after(diagnostic_step,observer,data,loss_sum,norm)
            if transfer_observer is not None:
                diagnostic_metrics.update(transfer_observer.metrics(optimizer_seconds))
        if super_jump:
            super_scale,super_policy,super_loss,super_seconds=checked_super_jump(
                actor,heldout,before,initial_probe_loss,policy_cfg)
            del before
        losses.append(loss_sum);norms.append(float(norm));policy_drifts.append(drift_sum)
        if basis is not None:
            retained=((projection_state['kept']-previous_kept)/
                      max(projection_state['full']-previous_full,1e-30))
            inner_retained.append(min(1.,max(0.,retained)))
        if adaptive or marginal:
            direction=projection_state['current_direction']
            cosine_available=direction is not None and previous_direction is not None
            if cosine_available:
                direction_cos=float(F.cosine_similarity(direction,previous_direction,dim=0))
            previous_direction=direction
            if marginal:
                if direction is None:
                    raise RuntimeError('training projected hidden gradient was not captured')
                train_directions.append(direction.detach().clone())
            if adaptive and inner+1<max_inner:
                started=time.perf_counter()
                probe_policy_kl,probe_teacher_kl=adaptive_probe(actor,data,probe_indices)
                probe_calls+=1
                probe_seconds+=time.perf_counter()-started
                if not np.isfinite(probe_policy_kl) or not np.isfinite(probe_teacher_kl):
                    stop_reason='nonfinite_probe'
                elif probe_policy_kl>max_policy_kl:
                    stop_reason='policy_drift'
                elif probe_teacher_kl>probe_loss*(1+loss_slack)+1e-4:
                    stop_reason='teacher_loss'
                elif cosine_available and direction_cos<min_cos:
                    stop_reason='direction_flip'
                if stop_reason!='budget':break
                probe_loss=probe_teacher_kl
        if marginal:
            started=time.perf_counter()
            need_gradient=(bool(policy_cfg.get('marginal_reuse_heldout_gradient_enabled',False))
                           and inner+1>=minimum_updates and inner+1<max_inner)
            if need_gradient:
                probe_policy_kl,probe_teacher_kl,heldout_direction=heldout_gradient_probe(
                    actor,heldout,basis)
                heldout_novelties.append(gradient_novelty(heldout_direction,train_directions))
                heldout_train_cosines.append(float(F.cosine_similarity(
                    heldout_direction.float(),train_directions[-1].float(),dim=0)))
            else:
                probe_policy_kl,probe_teacher_kl=heldout_probe(
                    actor,heldout,actor.config.ppo_micro_batch_size_per_gpu)
            probe_calls+=1;probe_seconds+=time.perf_counter()-started
            gain=marginal_probe_loss-probe_teacher_kl
            marginal_gains.append(gain)
            first_gain=marginal_gains[0]
            ratio=(gain/first_gain if first_gain>0 else 0.)
            marginal_ratios.append(ratio)
            marginal_probe_loss=probe_teacher_kl
            if not np.isfinite(probe_teacher_kl) or not np.isfinite(probe_policy_kl):
                raise FloatingPointError('nonfinite adaptive held-out probe; abort and resume the last checkpoint')
            if inner+1>=minimum_updates and inner+1<max_inner:
                if probe_policy_kl>max_probe_drift:
                    stop_reason='policy_drift';break
                if first_gain<=min_gain or gain<=min_gain or ratio<gain_ratio_threshold:
                    stop_reason='marginal_gain';break
    actor.actor_optimizer.zero_grad(set_to_none=True)
    metrics={'actor/opd_top16_kl':[losses[-1]],'actor/grad_norm':[norms[-1]],'actor/top16_support':[16.],
            'monitor/top16_entropy_not_measured':[1.], 'monitor/teacher_present':[1.],
            'monitor/hidden_gradient_projection':[float(basis is not None)],
            'monitor/opd_train_trajectories':[float(len(train_data))],
            'monitor/opd_heldout_trajectories':[float(len(data)-len(train_data))],
            'timing/opd_optimizer_seconds':[optimizer_seconds],
            'timing/opd_train_forward_backward_seconds':[train_forward_backward_seconds],
            'actor/opd_optimizer_steps':[float(len(losses))]}
    if reuse>1:
        metrics.update({
            'actor/jump_kl_first_update':[losses[0]],
            'actor/jump_kl_last_update':[losses[-1]],
            'actor/jump_grad_norm_mean':[sum(norms)/len(norms)],
            'actor/jump_cached_top16_policy_kl_last_preupdate':[policy_drifts[-1]],
            'actor/jump_cached_top16_policy_kl_max_preupdate':[max(policy_drifts)],
            'monitor/jump_rollout_reuse_factor':[float(len(losses))],
            'monitor/jump_stale_trajectory_updates':[float(len(losses)-1)],
            'monitor/jump_teacher_target_measurements':[1.0],
        })
        for index,(value,drift) in enumerate(zip(losses,policy_drifts),start=1):
            metrics[f'actor/jump_inner_{index}_kl']=[value]
            metrics[f'actor/jump_inner_{index}_cached_top16_policy_kl']=[drift]
    if adaptive:
        metrics.update({
            'monitor/adaptive_reuse_enabled':[1.],
            'monitor/adaptive_reuse_configured_max_updates':[float(reuse)],
            'monitor/adaptive_reuse_actual_updates':[float(len(losses))],
            'actor/adaptive_probe_policy_kl':[probe_policy_kl],
            'actor/adaptive_probe_teacher_kl':[probe_teacher_kl],
            'monitor/adaptive_probe_calls':[float(probe_calls)],
            'actor/adaptive_projected_gradient_cosine':[direction_cos],
            'monitor/adaptive_gradient_cosine_available':[float(cosine_available)],
            'actor/adaptive_hidden_gradient_energy_retained_last':[inner_retained[-1]],
            'actor/adaptive_hidden_gradient_residual_energy_last':[1.-inner_retained[-1]],
            'timing/adaptive_probe_seconds':[probe_seconds],
            'monitor/adaptive_stop_policy_drift':[float(stop_reason=='policy_drift')],
            'monitor/adaptive_stop_teacher_loss':[float(stop_reason=='teacher_loss')],
            'monitor/adaptive_stop_direction_flip':[float(stop_reason=='direction_flip')],
            'monitor/adaptive_stop_nonfinite_probe':[float(stop_reason=='nonfinite_probe')],
            'monitor/adaptive_stop_budget':[float(stop_reason=='budget')],
        })
    if marginal:
        metrics.update({
            'monitor/marginal_reuse_enabled':[1.],
            'monitor/marginal_actual_updates':[float(len(losses))],
            'monitor/marginal_train_trajectories':[float(len(train_data))],
            'monitor/marginal_heldout_trajectories':[float(len(heldout))],
            'actor/marginal_heldout_kl_before':[initial_probe_loss],
            'actor/marginal_heldout_kl_after':[marginal_probe_loss],
            'actor/marginal_heldout_policy_kl':[probe_policy_kl],
            'actor/marginal_last_gain':[marginal_gains[-1]],
            'actor/marginal_last_gain_ratio':[marginal_ratios[-1]],
            'actor/marginal_projected_gradient_cosine':[direction_cos],
            'actor/marginal_heldout_gradient_novelty_last':[
                heldout_novelties[-1] if heldout_novelties else 0.],
            'actor/marginal_heldout_train_gradient_cosine_last':[
                heldout_train_cosines[-1] if heldout_train_cosines else 0.],
            'monitor/marginal_heldout_gradient_probe_count':[float(len(heldout_novelties))],
            'timing/marginal_probe_seconds':[probe_seconds],
            'monitor/marginal_stop_gain':[float(stop_reason=='marginal_gain')],
            'monitor/marginal_stop_policy_drift':[float(stop_reason=='policy_drift')],
            'monitor/marginal_stop_nonfinite':[float(stop_reason=='nonfinite_probe')],
            'monitor/marginal_stop_budget':[float(stop_reason=='budget')],
        })
        for index,(gain,ratio) in enumerate(zip(marginal_gains,marginal_ratios),start=1):
            metrics[f'actor/marginal_gain_{index}']=[gain]
            metrics[f'actor/marginal_gain_ratio_{index}']=[ratio]
    if super_jump:
        metrics.update({
            'monitor/super_jump_enabled':[1.],
            'monitor/super_jump_actual_adam_steps':[1.],
            'monitor/super_jump_train_trajectories':[float(len(train_data))],
            'monitor/super_jump_heldout_trajectories':[float(len(heldout))],
            'actor/super_jump_selected_ray_scale':[super_scale],
            'actor/super_jump_heldout_kl_before':[initial_probe_loss],
            'actor/super_jump_heldout_kl_after':[super_loss],
            'actor/super_jump_heldout_policy_kl':[super_policy],
            'timing/super_jump_probe_seconds':[super_seconds],
        })
    if basis is not None:
        if world>1:
            projection_state['full']=dp.total(float(projection_state['full']))
            projection_state['kept']=dp.total(float(projection_state['kept']))
        metrics['actor/hidden_gradient_energy_retained']=[float(projection_state['kept']/max(projection_state['full'],1e-30))]
        metrics['actor/hidden_projection_rank']=[float(basis.shape[1])]
    if diagnostic is not None:
        metrics.update({key:[value] for key,value in diagnostic_metrics.items()})
    if bottleneck is not None:
        metrics.update(bottleneck_metrics)
        metrics['monitor/hidden_gradient_projection'] = [float(bottleneck.u is not None)]
    if response_observer is not None:
        metrics.update({k: [v] for k, v in response_observer.metrics.items()})
    if replication is not None:
        metrics.update({k: [v] for k, v in replication.finish(losses, norms).items()})
    return metrics
