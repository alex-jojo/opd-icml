"""Passive, same-state interventions for fixed-U OPD. Single GPU NO_SHARD only.

No fitted optimizer, no parameter search, no benchmark evaluation. Candidate
weights are disposable exact AdamW *one-step* previews (including clipping,
momentum and weight decay); the real optimizer is never called in a probe.
All reported KLs are conditional on the cached student Top-16 support.
"""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from .opd_top16 import forward, reverse_kl, _final_norm_module, _unwrap_actor_module
from .opd_plasticity import atomic_npz, atomic_json
from .opd_focus_stats import concentration, allocation, distribution_pair
from .opd_priority_stats import analyze_priority


def select(data, indices):
    # DataProto selection owns no model state. Clone before .to(), which may
    # mutate the object in legacy DataProto implementations.
    result = data.select_idxs(list(map(int, indices)))
    result.batch = result.batch.clone().cpu() if hasattr(result.batch, 'clone') else {
        k: v.detach().cpu().clone() for k, v in result.batch.items()}
    return result


def fingerprint(data):
    h = hashlib.sha256()
    for key in ('input_ids', 'attention_mask', 'response_mask', 'opd_top16_ids', 'opd_teacher_top16_logits'):
        value = data.batch[key].detach().cpu().contiguous()
        h.update(key.encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def sqnorm(values):
    return sum(float(x.double().square().sum()) for x in values if x is not None)


def adam_preview(parameters, original, gradients, optimizer, clip):
    """Exact non-AMSGrad AdamW semantics, no state writes. CPU block streaming.

    The returned list is the *rounded parameter displacement*, not a raw
    gradient. No-gradient parameters (including their decay) remain untouched.
    """
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError('only torch.optim.AdamW is supported by this preview')
    groups = {id(p): g for g in optimizer.param_groups for p in g['params']}
    norm = sqnorm(gradients)**.5
    factor = min(1., float(clip)/(norm+1e-6))
    if not np.isfinite(norm):
        raise FloatingPointError('nonfinite diagnostic gradient')
    delta = []
    for p, w, grad in zip(parameters, original, gradients):
        group = groups[id(p)]
        if any(group.get(k, False) for k in ('amsgrad', 'maximize', 'differentiable', 'capturable')):
            raise ValueError('unsupported AdamW variant in diagnostic preview')
        if grad is None:
            delta.append(None)
            continue
        if w.dtype != torch.float32 or grad.dtype != torch.float32:
            raise ValueError('diagnostic preview requires FP32 master parameters/gradients with BF16 forward')
        state = optimizer.state.get(p, {})
        m = state['exp_avg'].detach().cpu().clone() if state else torch.zeros_like(w)
        v = state['exp_avg_sq'].detach().cpu().clone() if state else torch.zeros_like(w)
        step = float(state['step'])+1 if state else 1
        b1, b2 = group['betas']
        g = grad * factor
        m.lerp_(g, 1-b1)
        v.mul_(b2).addcmul_(g, g, value=1-b2)
        denom = (v.sqrt() / (1-b2**step)**.5).add_(group['eps'])
        updated = w.clone().mul_(1-float(group['lr'])*group['weight_decay'])
        updated.addcdiv_(m, denom, value=-float(group['lr'])/(1-b1**step))
        delta.append(updated-w)
    return delta, norm, factor


@contextmanager
def displaced(parameters, original, delta, scale=1.):
    """Always restore bitwise, including on forward/OOM/interrupt exceptions."""
    try:
        with torch.no_grad():
            for p, w, d in zip(parameters, original, delta):
                target = w if d is None else w + float(scale)*d
                p.copy_(target.to(device=p.device, dtype=p.dtype))
        yield
    finally:
        with torch.no_grad():
            for p, w in zip(parameters, original):
                p.copy_(w.to(device=p.device, dtype=p.dtype))


@contextmanager
def inert(actor):
    parameters = [p for p in actor.actor_module.parameters() if p.requires_grad]
    grads = [None if p.grad is None else p.grad.detach().cpu().clone() for p in parameters]
    modes = [(m, m.training) for m in actor.actor_module.modules()]
    device = parameters[0].device
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
        try:
            actor.actor_module.eval()
            yield parameters, grads
        finally:
            for p, g in zip(parameters, grads):
                p.grad = None if g is None else g.to(device=p.device, dtype=p.dtype)
            for m, flag in modes:
                m.training = flag


def field_projection(gradient, basis, mask, field, first_sequence=0, feature_split=32):
    """Disjoint feature/sample partitions; direct output-head gradient untouched.

    'sample_even/odd' partitions whole trajectories, not held-out positions.
    Prompt cotangents are preserved (zero for the response-only direct loss).
    """
    if field == 'full':
        return gradient
    n = mask.shape[1]
    result = gradient.clone()
    g = gradient[:, -n-1:-1]
    u = basis.to(device=g.device, dtype=g.dtype)
    coeff = g @ u
    if field == 'feature_head':
        coeff = coeff.clone(); coeff[..., feature_split:] = 0
    elif field == 'feature_tail':
        coeff = coeff.clone(); coeff[..., :feature_split] = 0
    elif field in ('sample_even', 'sample_odd'):
        parity = torch.arange(first_sequence, first_sequence+len(g), device=g.device) % 2
        coeff = coeff * (parity == (field == 'sample_odd'))[:, None, None]
    elif field != 'all':
        raise ValueError(field)
    projected = coeff @ u.T
    result[:, -n-1:-1] = torch.where(mask[..., None].bool(), projected, g)
    return result


def field_gradient(actor, data, basis, field, feature_split):
    actor.actor_optimizer.zero_grad(set_to_none=True)
    device = next(actor.actor_module.parameters()).device
    total = int(data.batch['response_mask'].sum())
    offset = 0
    for part in data.split(actor.config.ppo_micro_batch_size_per_gpu):
        inputs = {k: v.to(device) for k, v in part.batch.items()}
        def hook(_module, _args, h):
            h.register_hook(lambda g: field_projection(g, basis, inputs['response_mask'], field, offset, feature_split))
        handle = _final_norm_module(actor.actor_module).register_forward_hook(hook)
        try:
            _, z, _, _ = forward(actor, inputs, 1., inputs['opd_top16_ids'], diagnostics=False)
            loss = (reverse_kl(z, inputs['opd_teacher_top16_logits'])*inputs['response_mask']).sum()/total
            loss.backward()
        finally:
            handle.remove()
        offset += len(part)
    return [None if p.grad is None else p.grad.detach().cpu().clone()
            for p in actor.actor_module.parameters() if p.requires_grad]


@torch.no_grad()
def readout(actor, data, basis, samples=32, capture_gradient=False):
    device = next(actor.actor_module.parameters()).device
    result = {k: [] for k in ('z', 'q', 'h', 'row', 'pos', 'token', 'zrow', 'zpos', 'sample_kl','boundary')}
    if capture_gradient: result['g_u']=[]
    total = 0; loss_sum = 0.; cursor = 0; calls = 0
    for part in data.split(actor.config.ppo_micro_batch_size_per_gpu):
        x = {k: v.to(device) for k, v in part.batch.items()}
        mask = x['response_mask'].bool(); n = mask.shape[1]
        rows, positions = [], []
        for i in range(len(part)):
            valid = mask[i].nonzero().flatten()
            take = torch.linspace(0, len(valid)-1, min(samples, len(valid)), device=device).round().long()
            rows.extend([i]*len(take)); positions.extend(valid[take].tolist())
        ix = (torch.tensor(rows, device=device, dtype=torch.long), torch.tensor(positions, device=device, dtype=torch.long))
        captured = []
        head_handle=None
        if capture_gradient:
            # Compute while FSDP-managed head weights are live, never afterwards.
            from .opd_plasticity import kl_and_logit_gradient, head_gradient
            head=_unwrap_actor_module(_unwrap_actor_module(actor.actor_module).lm_head)
            def head_hook(module,_args,output):
                with torch.autocast(device_type=device.type,enabled=False):
                    ids=x['opd_top16_ids'][ix]
                    selected=output[:, -n-1:-1][ix].gather(-1,ids)
                    _,derivative=kl_and_logit_gradient(selected,x['opd_teacher_top16_logits'][ix])
                    g=head_gradient(derivative,ids,module.weight,forward_dtype=output.dtype)
                    result['g_u'].append((g @ basis.float()).cpu())
            head_handle=head.register_forward_hook(head_hook)
        handle = _final_norm_module(actor.actor_module).register_forward_hook(
            lambda _m, _a, h: captured.append(h[:, -n-1:-1][ix].float().cpu()))
        try:
            _, z, _, _ = forward(actor, x, 1., x['opd_top16_ids'], diagnostics=False)
        finally:
            handle.remove()
            if head_handle is not None: head_handle.remove()
        q = x['opd_teacher_top16_logits']
        loss_sum += float((reverse_kl(z, q)*mask).sum()); total += int(mask.sum()); calls += 1
        # Logits for ALL valid tokens allow exact token-weighted cached-support
        # teacher/policy KL. Hidden states only at fixed, recorded positions.
        result['z'].append(z[mask].float().cpu()); result['q'].append(q[mask].float().cpu())
        result['h'].append(captured[0]); result['row'].append(ix[0].cpu()+cursor)
        result['pos'].append(ix[1].cpu()); result['token'].append(x['responses'][ix].cpu())
        all_rows, all_positions = mask.nonzero(as_tuple=True)
        result['zrow'].append(all_rows.cpu()+cursor); result['zpos'].append(all_positions.cpu())
        result['sample_kl'].append(reverse_kl(z[ix],q[ix]).cpu())
        eos=getattr(getattr(_unwrap_actor_module(actor.actor_module),'config',None),'eos_token_id',None)
        eos_ids=[] if eos is None else ([eos] if isinstance(eos,int) else list(eos))
        boundary=(mask.long().cumsum(-1)==1)&mask
        for token in eos_ids:boundary |= (x['responses']==token)&mask
        result['boundary'].append(boundary[mask].cpu())
        cursor += len(part)
    if total == 0:
        raise ValueError('readout bank has no valid response tokens')
    return {**{k: torch.cat(v) for k, v in result.items()}, 'kl': loss_sum/total,
            'tokens': total, 'forwards': calls}


def response_metrics(before, after, basis):
    dz = after['z']-before['z']
    prob = before['z'].softmax(-1)
    centered = dz-(prob*dz).sum(-1, keepdim=True)
    dh = after['h']-before['h']
    du = F.normalize(after['h'], dim=-1)-F.normalize(before['h'], dim=-1)
    u = basis.float().cpu()
    def inside(x):
        return float((x @ u).square().sum()/x.square().sum().clamp_min(1e-30))
    kl_before=reverse_kl(before['z'],before['q']).numpy()
    kl_after=reverse_kl(after['z'],before['q']).numpy()
    gain=kl_before-kl_after
    fisher=(prob*centered.square()).sum(-1).numpy()
    raw=dh.square().sum(-1).numpy(); unit=du.square().sum(-1).numpy()
    result = dict(teacher_kl=after['kl'], teacher_kl_improvement=before['kl']-after['kl'],
                policy_kl_cached_support=float(reverse_kl(before['z'], after['z']).mean()),
                fisher_response_energy=float((prob*centered.square()).sum(-1).mean()),
                hidden_raw_rms=float(dh.square().mean().sqrt()), hidden_unit_rms=float(du.square().mean().sqrt()),
                hidden_raw_inside_u=inside(dh), hidden_unit_inside_u=inside(du),
                focus=dict(pre_kl=concentration(kl_before.clip(0)),
                    positive_kl_improvement=concentration(gain.clip(0)), negative_kl_change=concentration((-gain).clip(0)),
                    output_change=concentration(fisher), raw_hidden_change=concentration(raw), unit_hidden_change=concentration(unit),
                    pre_kl_to_signed_improvement=allocation(kl_before,gain),
                    pre_kl_to_output_change=allocation(kl_before,fisher),
                    pre_kl_to_unit_hidden_change=allocation(before['sample_kl'].numpy(),unit),
                    demand_output_distribution=distribution_pair(kl_before,fisher),
                    raw_u_direction_energy=concentration((dh@u).square().sum(0).numpy()),
                    unit_u_direction_energy=concentration((du@u).square().sum(0).numpy()),
                    interpretation='Concentration is not quality; signed gains and boundary effects must be considered.'))
    interior=~before['boundary'].numpy()
    if interior.any():
        result['focus']['excluding_first_and_eos']=dict(
            kl=concentration(kl_before[interior].clip(0)),output_change=concentration(fisher[interior]),
            kl_to_improvement=allocation(kl_before[interior],gain[interior]))
    if 'supervision_energy' in before:
        result['focus']['actual_training_gradient_to_improvement']=allocation(before['supervision_energy'].numpy(),gain)
        result['focus']['actual_training_gradient_to_output_change']=allocation(before['supervision_energy'].numpy(),fisher)
    for name in ('need', 'alignment', 'priority'):
        if name in before:
            signal=before[name].numpy()
            result['focus'][name+'_to_signed_improvement']=allocation(signal,gain)
            result['focus'][name+'_to_output_change']=allocation(signal,fisher)
    zrow=before['zrow'].numpy(); seq_ids=np.unique(zrow)
    result['focus']['equal_sequence_weight']=dict(
        pre_kl=concentration(np.asarray([kl_before[zrow==r].mean() for r in seq_ids]).clip(0)),
        output_change=concentration(np.asarray([fisher[zrow==r].mean() for r in seq_ids])),
        mean_signed_kl_improvement=float(np.mean([gain[zrow==r].mean() for r in seq_ids])))
    for label,movement in [('raw',dh),('unit',du)]:
        energy=movement.square().sum(-1).mean()
        mean=movement.mean(0)
        cosine=F.cosine_similarity(movement,mean[None,:],dim=-1)
        result['focus'][label+'_direction_coherence']=dict(
            shared_mean_energy_fraction=float(mean.square().sum()/energy.clamp_min(1e-30)),
            cosine_to_mean_quantiles=torch.quantile(cosine,torch.tensor([0.,.5,.9,1.])).tolist(),
            nonzero_response=bool(energy>1e-30))
    substantial=kl_before>=.01
    result['focus']['relative_correction_for_pre_kl_ge_001']=dict(count=int(substantial.sum()),
        signed_fraction_quantiles=np.quantile(gain[substantial]/kl_before[substantial],[0,.1,.5,.9,1]).tolist()
        if substantial.any() else None)
    return result


class Observer:
    def __init__(self, actor, config):
        self.actor, self.cfg = actor, dict(config)
        self.directory = Path(config['output_dir'])
        self.anchors = tuple(config.get('anchors', [11, 31, 51]))
        self.lags = tuple(config.get('lags', [1, 2]))
        if (any(x < 1 for x in self.lags) or any(b-a <= max(self.lags, default=0) for a,b in zip(self.anchors, self.anchors[1:]))):
            raise ValueError('anchors must be sorted and leave space for freshness checks')
        self.pending = None

    def begin(self, train, heldout, basis, step):
        if any(isinstance(m, torch.nn.Dropout) and m.p != 0 for m in self.actor.actor_module.modules()):
            raise ValueError('response diagnostics require dropout=0')
        config = getattr(self.actor.actor_module, 'config', None)
        if float(getattr(config, 'attention_dropout', 0.) or 0.) != 0:
            raise ValueError('response diagnostics require attention_dropout=0')
        from . import opd_distributed as dp
        if dp.size()!=1 and int(self.actor.config.policy_loss.get('replication17',{}).get('distributed_world_size',1))!=dp.size():
            raise ValueError('response diagnostics require explicit global-statistics DP2 support')
        self.training_projected=basis is not None
        # Read the already cached FP32 asset, not the BF16 projection twice.
        # Training continues to use its original basis and arithmetic unchanged.
        from .opd_top16 import _BASIS_CACHE
        cfg=self.actor.config.policy_loss['hidden_projection']
        key=(str(Path(cfg['asset_path']).resolve()),int(cfg['rank']))
        if key not in _BASIS_CACHE:
            with np.load(key[0],allow_pickle=False) as payload:
                u=np.asarray(payload['basis'][:,:key[1]],dtype=np.float32)
            expected=int(_final_norm_module(self.actor.actor_module).weight.numel())
            if u.shape!=(expected,key[1]) or not np.isfinite(u).all() or np.max(np.abs(u.T@u-np.eye(key[1])))>2e-3:
                raise ValueError('invalid observation-only basis')
            _BASIS_CACHE[key]=torch.from_numpy(u.copy())
        device=next(self.actor.actor_module.parameters()).device
        self.stats_basis=_BASIS_CACHE[key].to(device=device,dtype=torch.float32)
        basis=self.stats_basis if basis is None else basis
        self.train, self.heldout, self.basis, self.step = train, heldout, basis, step
        self.raw_z=[]
        self.metrics = {'response_geometry/measured': 0., 'response_geometry/freshness_measured': 0.}
        self.second = torch.zeros(basis.shape[1], basis.shape[1], device=basis.device)
        self.count = 0
        self.expected = None
        self.token_fields = {k:[] for k in ('kl','tv','student_entropy','teacher_entropy','raw_gradient_energy','projected_gradient_energy','row','pos','token','boundary','relative_position')}
        self.cursor = 0

    def record_outputs(self, student, teacher, mask, kl):
        with torch.no_grad():
            p=student.detach().float().softmax(-1);q=teacher.detach().float().softmax(-1)
            valid=mask.bool()
            for name,value in [('kl',kl.detach()),('tv',.5*(p-q).abs().sum(-1)),
                               ('student_entropy',-(p*p.clamp_min(1e-30).log()).sum(-1)),
                               ('teacher_entropy',-(q*q.clamp_min(1e-30).log()).sum(-1))]:
                self.token_fields[name].append(value[valid].cpu())

    def consume_raw(self, gradient):
        mask=self.active_mask; n=mask.shape[1]
        total=int(self.train.batch['response_mask'].sum())
        from .opd_distributed import size
        g=gradient[:, -n-1:-1][mask].float()*(total/size())
        self.token_fields['raw_gradient_energy'].append(g.square().sum(-1).cpu())
        z=g @ self.stats_basis
        self.raw_z.append(z.cpu())
        return z

    @contextmanager
    def watch(self, inputs):
        # Read-only observer attached BEFORE the projection forward hook: this
        # tensor's gradient already includes the projection's backward.
        n = inputs['responses'].shape[1]; mask = inputs['response_mask'].bool()
        self.active_mask = mask
        row,pos=mask.nonzero(as_tuple=True)
        self.token_fields['row'].append(row.cpu()+self.cursor)
        self.token_fields['pos'].append(pos.cpu()); self.token_fields['token'].append(inputs['responses'][mask].cpu())
        ordinal=mask.long().cumsum(-1)-1
        relative=ordinal.float()/(mask.sum(-1,keepdim=True)-1).clamp_min(1)
        boundary=(ordinal==0)&mask
        eos=getattr(getattr(_unwrap_actor_module(self.actor.actor_module),'config',None),'eos_token_id',None)
        for token in ([] if eos is None else [eos] if isinstance(eos,int) else eos):
            boundary |= (inputs['responses']==token)&mask
        self.token_fields['boundary'].append(boundary[mask].cpu())
        self.token_fields['relative_position'].append(relative[mask].cpu())
        self.cursor+=len(inputs['responses'])
        total = int(self.train.batch['response_mask'].sum())
        def hook(_m, _a, h):
            def consume(g):
                if not self.training_projected:
                    c=self.consume_raw(g)
                else:
                    from .opd_distributed import size
                    c=(g[:, -n-1:-1][mask].float() @ self.basis.float())*(total/size())
                if not self.cfg.get('cache_only_moments',False):self.second.add_(c.T @ c)
                self.count += len(c)
                self.token_fields['projected_gradient_energy'].append(c.square().sum(-1).cpu())
                return g
            h.register_hook(consume)
        handle = _final_norm_module(self.actor.actor_module).register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    def banks(self):
        indices = np.unique(np.linspace(0, len(self.train)-1, min(int(self.cfg.get('train_readout', 8)), len(self.train))).round().astype(int))
        return {'train': select(self.train, indices), 'heldout': select(self.heldout, range(len(self.heldout)))}, indices

    def before_update(self):
        start = time.perf_counter()
        self.fields={k:torch.cat(v).numpy() for k,v in self.token_fields.items()}
        if any(len(v)!=self.count for v in self.fields.values()):
            raise RuntimeError('unaligned priority token statistics')
        z=torch.cat(self.raw_z).numpy()
        self.priority,self.priority_arrays=analyze_priority(z,self.fields)
        self.raw_z.clear()
        del z
        self.metrics['timing/priority_statistics_seconds']=time.perf_counter()-start
        if self.step in self.anchors:
            self.anchor()
        elif self.pending is not None and self.step-self.pending['step'] in self.lags:
            self.freshness()
        self.metrics['timing/response_geometry_extra_seconds'] = time.perf_counter()-start

    def anchor(self):
        banks, indices = self.banks()
        with inert(self.actor) as (params, normal_gradient):
            original = [p.detach().cpu().clone() for p in params]
            before = {k: readout(self.actor, v, self.basis) for k,v in banks.items()}
            training_rows=torch.cat(self.token_fields['row'])
            training_energy=torch.cat(self.token_fields['projected_gradient_energy'])
            before['train']['supervision_energy']=torch.cat([training_energy[training_rows==int(i)] for i in indices])
            for name in ('need','alignment','priority'):
                before['train'][name]=torch.from_numpy(np.concatenate([
                    self.fields[name][self.fields['row']==int(i)] for i in indices]))
            base_delta, norm, factor = adam_preview(params, original, normal_gradient, self.actor.actor_optimizer, self.actor.config.grad_clip)
            records = []; responses = {}
            def measure(name, delta, scale=1.):
                begin = time.perf_counter()
                with displaced(params, original, delta, scale):
                    after = {k: readout(self.actor, v, self.basis) for k,v in banks.items()}
                actual_norm = (sqnorm(delta)**.5)*abs(scale)
                records.append(dict(name=name, parameter_update_norm=actual_norm,
                    seconds=time.perf_counter()-begin,
                    **{k: response_metrics(before[k], after[k], self.basis) for k in banks}))
                responses[name] = after
            for scale in (.5, 1., 2.):
                measure('ordinary_x'+str(scale), base_delta, scale)
            # Each field backward traverses ALL 128 actual training responses,
            # not the 8-row readout sample. LM head direct gradient stays intact.
            for field in ('full', 'feature_head', 'feature_tail', 'sample_even', 'sample_odd'):
                begin = time.perf_counter()
                gradients = field_gradient(self.actor, self.train, self.basis, field, int(self.cfg.get('feature_split', 32)))
                delta, gn, clip = adam_preview(params, original, gradients, self.actor.actor_optimizer, self.actor.config.grad_clip)
                measure(field, delta)
                records[-1].update(gradient_norm=gn, clip_factor=clip, total_probe_seconds=time.perf_counter()-begin)
                del gradients, delta
            # Zero backbone supervision; direct head + Adam memory/decay remain.
            gradients = field_gradient(self.actor, self.train, self.basis, 'feature_head', 0)
            delta, _, _ = adam_preview(params, original, gradients, self.actor.actor_optimizer, self.actor.config.grad_clip)
            measure('head_and_optimizer_background', delta)
            del gradients, delta
            arrays = {'basis': self.basis.float().cpu().numpy(), 'train_indices': indices,
                      'projected_gradient_second': (self.second/max(self.count,1)).cpu().numpy()}
            for bank, value in before.items():
                for key in ('z', 'q', 'row', 'pos', 'token', 'zrow', 'zpos','sample_kl','boundary'):
                    arrays[f'{bank}_before_{key}'] = value[key].numpy()
                if 'supervision_energy' in value:
                    arrays[f'{bank}_training_supervision_energy']=value['supervision_energy'].numpy()
                    for name in ('need','alignment','priority'):
                        arrays[f'{bank}_training_{name}']=value[name].numpy()
                arrays[f'{bank}_before_h_u'] = (value['h'] @ self.basis.float().cpu()).numpy()
                for name, after in responses.items():
                    arrays[f'{bank}_{name}_dz'] = (after[bank]['z']-value['z']).numpy()
                    dh = after[bank]['h']-value['h']
                    du = F.normalize(after[bank]['h'],dim=-1)-F.normalize(value['h'],dim=-1)
                    arrays[f'{bank}_{name}_dh_u'] = (dh @ self.basis.float().cpu()).numpy()
                    arrays[f'{bank}_{name}_du_u'] = (du @ self.basis.float().cpu()).numpy()
            directory = self.directory/f'anchor_{self.step:03d}'
            atomic_npz(directory/'responses.npz', arrays)
            atomic_json(directory/'results.json', dict(step=self.step, records=records,
                baseline_gradient_norm=norm, baseline_clip_factor=factor,
                train_sequences=len(self.train), heldout_sequences=len(self.heldout), train_readout_sequences=len(banks['train']),
                training_batch_sha256=fingerprint(self.train), bank_sha256={k:fingerprint(v) for k,v in banks.items()},
                teacher_calls_extra=0, gradient_sweeps_extra=6, new_optimizer_updates=0,
                scope='All KL is conditional cached-support KL; train readout is a named subset, heldout is never fitted.',
                coupling_limit='Component interventions include Adam nonlinearity; they do not identify a full token-by-token Jacobian.'))
            self.expected = [w if d is None else w+d for w,d in zip(original,base_delta)]
            self.pending = dict(step=self.step, delta=base_delta, banks=banks,
                                reference={k: response_metrics(before[k], responses['ordinary_x1.0'][k], self.basis) for k in banks})
            self.metrics.update({'response_geometry/measured': 1., 'response_geometry/shadow_candidates': len(records),
                                 'response_geometry/extra_gradient_sweeps': 6.,
                                 'response_geometry/train_kl_gain': records[1]['train']['teacher_kl_improvement'],
                                 'response_geometry/heldout_kl_gain': records[1]['heldout']['teacher_kl_improvement']})

    def freshness(self):
        # A frozen PARAMETER ACTION, not an assumed reusable 128x128 operator.
        # Read both the identical cached bank and new current-batch states.
        current, _ = self.banks()
        banks = {'fixed_'+k:v for k,v in self.pending['banks'].items()}
        banks.update({'new_'+k:v for k,v in current.items()})
        with inert(self.actor) as (params, _):
            original = [p.detach().cpu().clone() for p in params]
            before = {k:readout(self.actor,v,self.basis) for k,v in banks.items()}
            with displaced(params,original,self.pending['delta']):
                after = {k:readout(self.actor,v,self.basis) for k,v in banks.items()}
            record = dict(anchor=self.pending['step'], step=self.step, lag=self.step-self.pending['step'],
                kind='frozen_parameter_action_not_operator_identification',
                records={k:response_metrics(before[k],after[k],self.basis) for k in banks},
                anchor_reference=self.pending['reference'], bank_sha256={k:fingerprint(v) for k,v in banks.items()})
            atomic_json(self.directory/f"anchor_{self.pending['step']:03d}"/f'freshness_{self.step:03d}.json', record)
        if self.step-self.pending['step'] == max(self.lags):
            self.pending = None
        self.metrics['response_geometry/freshness_measured'] = 1.

    def after_update(self):
        if self.count != int(self.train.batch['response_mask'].sum()):
            raise RuntimeError('response observer saw wrong number of actual training tokens')
        if self.expected is not None:
            error = 0.; magnitude = 0.
            params = [p for p in self.actor.actor_module.parameters() if p.requires_grad]
            for p,w in zip(params,self.expected):
                error = max(error,float((p.detach().cpu()-w).abs().max()))
                magnitude = max(magnitude,float(w.abs().max()))
            self.expected = None
            self.metrics['response_geometry/adam_preview_max_absolute_error'] = error
            if error > 2e-6+2e-6*magnitude:
                raise RuntimeError('shadow AdamW does not match committed update; stop before trusting intervention results')
        fields=self.fields
        if any(len(v)!=self.count for v in fields.values()):
            raise RuntimeError('unaligned token pressure/gradient statistics')
        second=(self.second/self.count).cpu().numpy()
        focus={k:concentration(fields[k].clip(0)) for k in ('kl','tv','raw_gradient_energy','projected_gradient_energy')}
        focus['direction_energy_fixed_basis']=concentration(second.diagonal().clip(0))
        eigen=torch.linalg.eigvalsh(self.second/self.count).clamp_min(0).cpu().numpy()
        focus['direction_spectrum_basis_invariant']=concentration(eigen)
        focus['kl_to_gradient_allocation']=allocation(fields['kl'],fields['projected_gradient_energy'])
        # Quantify whether projection disproportionately removes pressure on
        # high-disagreement states. Ranking fixed by PRE-update KL.
        order=np.argsort(fields['kl'],kind='stable')
        focus['projection_retention_by_kl_decile']=[]
        for ix in np.array_split(order,10):
            raw=float(fields['raw_gradient_energy'][ix].sum())
            kept=float(fields['projected_gradient_energy'][ix].sum())
            focus['projection_retention_by_kl_decile'].append(dict(count=len(ix),raw_energy=raw,
                projected_energy=kept,ratio=kept/raw if raw>1e-30 else None))
        focus['boundary_first_token']={k:float(fields[k][fields['pos']==0].sum()/max(float(fields[k].sum()),1e-30))
                                       for k in ('kl','projected_gradient_energy')}
        nonboundary=~fields['boundary'].astype(bool)
        focus['excluding_first_and_eos']={k:concentration(fields[k][nonboundary].clip(0))
                                        for k in ('kl','projected_gradient_energy')}
        row=fields['row'];rows=np.arange(len(self.train))
        focus['sequence_mean_kl']=concentration(np.asarray([fields['kl'][row==r].mean() if (row==r).any() else 0. for r in rows]).clip(0))
        atomic_json(self.directory/f'focus_{self.step:03d}.json',focus)
        atomic_json(self.directory/f'priority_{self.step:03d}.json',self.priority)
        atomic_npz(self.directory/f'priority_{self.step:03d}.npz',self.priority_arrays)
        for name in ('need','alignment','priority'):
            stats=self.priority['score_concentration'][name]
            if stats['available']:
                self.metrics[f'priority/{name}_top10_share']=stats['top_shares']['0.1']
                self.metrics[f'priority/{name}_effective_fraction']=stats['effective_fraction']
        self.metrics['priority/mean_alignment']=float(fields['alignment'].mean())
        self.metrics['priority/joint_weight_signal_norm_ratio']=self.priority['sharpening']['0.0']['raw_signal_norm_ratio']
        rank=self.priority['weighted_geometry']['joint']['effective_rank']
        if rank is not None:self.metrics['priority/joint_effective_rank']=rank
        atomic_npz(self.directory/f'step_{self.step:03d}.npz', {
            'step': np.asarray(self.step), 'gradient_second': (self.second/self.count).cpu().numpy(),
            'valid_train_tokens': np.asarray(self.count), 'train_sequences': np.asarray(len(self.train)),
            'heldout_sequences': np.asarray(len(self.heldout)), **{'token_'+k:v for k,v in fields.items()}})
        for name in ('kl','projected_gradient_energy'):
            stats=focus[name]
            if stats['available']:
                self.metrics[f'focus/{name}_gini']=stats['gini']
                self.metrics[f'focus/{name}_effective_fraction']=stats['effective_fraction']
                self.metrics[f'focus/{name}_top10_share']=stats['top_shares']['0.1']


def get_observer(actor):
    if not hasattr(actor, '_response_geometry'):
        actor._response_geometry = Observer(actor, actor.config.policy_loss['response_geometry'])
    return actor._response_geometry
