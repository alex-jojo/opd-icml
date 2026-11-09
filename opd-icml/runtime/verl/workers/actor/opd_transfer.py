"""Actual-batch input/output transfer diagnostics, single-GPU FSDP NO_SHARD.

No optimizer intervention. Extra VJPs use official forward/backward, cached
teacher logits, and the SAME pre-update parameters. Never form a Jacobian.
"""
import time
from contextlib import contextmanager
import numpy as np
import torch

from .opd_plasticity import atomic_npz, head_gradient, kl_and_logit_gradient
from .opd_top16 import _final_norm_module, _unwrap_actor_module, forward


def scheduled(step):
    return step in (1, 2, 5, 10, 20, 30, 40, 50, 60, 61, 70, 80, 90, 100, 106)


def spectrum(second, basis):
    values, vectors = np.linalg.eigh(np.asarray(second, dtype=np.float64))
    values, vectors = values[::-1].clip(0), vectors[:, ::-1].copy()
    alpha = np.square(np.asarray(basis, dtype=np.float64).T @ vectors).sum(0)
    return values, vectors, alpha


def representative_modes(values, alpha):
    # Eight representatives across four occupied-spectrum strata; never pick nearly zero eigenvalues just
    # because their compatibility is extreme. Selection is fully recorded.
    pool = np.flatnonzero(values > max(values[0] * 1e-6, 1e-30))[:128]
    selected = []
    for group in np.array_split(pool, 4):
        if len(group):
            selected += [int(group[np.argmin(alpha[group])]), int(group[np.argmax(alpha[group])])]
    return list(dict.fromkeys(selected))


def dot_cpu(a, b):
    # Bounded temporaries, FP64 reduction, no model-sized flatten/cat.
    out = 0.
    for x, y in zip(a.reshape(-1).split(262144), b.reshape(-1).split(262144)):
        out += float((x.double() * y.double()).sum())
    return out


def gradient_snapshot(parameters):
    return [None if p.grad is None else p.grad.detach().cpu().clone() for p in parameters]


@contextmanager
def preserve_training_state(actor):
    parameters = [p for p in actor.actor_module.parameters() if p.requires_grad]
    saved = gradient_snapshot(parameters)
    modes = [(m, m.training) for m in actor.actor_module.modules()]
    device = next(actor.actor_module.parameters()).device
    devices = [device.index or 0] if device.type == 'cuda' else []
    # Qwen dropout=0 is required: same-state train/eval equivalence, no random
    # mask changes. Do not mutate optimizer state or parameters in these passes.
    with torch.random.fork_rng(devices=devices):
        try:
            actor.actor_module.eval()
            yield parameters, saved
        finally:
            for p, g in zip(parameters, saved):
                p.grad = None if g is None else g.to(device=p.device, dtype=p.dtype)
            for m, training in modes:
                m.training = training


def backbone_vjp(actor, data, basis, component, mode=None, sign_seed=1729):
    """Accumulate J_backbone^T cotangent over cached actual training rows.

    'parallel'/'perpendicular': true loss cotangents divided by full valid-token
    count. 'mode': q * fixed Rademacher signs / sqrt(N); a Hutchinson-style
    per-token Jacobian sensitivity estimate, NOT a coherent all-token push.
    The zero selected-logit term traverses the official FSDP head graph while
    giving it ZERO direct parameter gradient. Tied input embeddings still get
    their genuine backbone-path gradient.
    """
    actor.actor_optimizer.zero_grad(set_to_none=True)
    device = next(actor.actor_module.parameters()).device
    total = int(data.batch['response_mask'].sum())
    if total < 1:
        raise ValueError('empty VJP token field')
    generator = torch.Generator(device='cpu').manual_seed(sign_seed)
    for part in data.split(actor.config.ppo_micro_batch_size_per_gpu):
        part = part.to(device)
        inputs = dict(part.batch)
        n = inputs['responses'].shape[1]
        captured = {}
        def norm_hook(_m, _a, out):
            captured['hidden'] = out
        def head_hook(module, _args, out):
            if mode is not None:
                return
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                selected = out[:, -n-1:-1].gather(-1, inputs['opd_top16_ids'])
                _, derivative = kl_and_logit_gradient(selected, inputs['opd_teacher_top16_logits'])
                g = head_gradient(derivative.reshape(-1, 16),
                                  inputs['opd_top16_ids'].reshape(-1, 16),
                                  module.weight, forward_dtype=out.dtype).reshape(*selected.shape[:2], -1)
                parallel = (g @ basis) @ basis.T
                captured['g'] = parallel if component == 'parallel' else g - parallel
        h1 = _final_norm_module(actor.actor_module).register_forward_hook(norm_hook)
        h2 = _unwrap_actor_module(actor.actor_module).lm_head.register_forward_hook(head_hook)
        try:
            _, selected, _, _ = forward(actor, inputs, 1., inputs['opd_top16_ids'], diagnostics=False)
            hidden = captured['hidden'][:, -n-1:-1]
            valid = inputs['response_mask'].bool()
            if mode is None:
                cotangent = captured['g'] / total
            else:
                signs = torch.randint(0, 2, valid.shape, generator=generator).to(device).float()*2-1
                cotangent = signs[..., None] * mode[None, None, :] / total**.5
            loss = (hidden.float() * cotangent.detach() * valid[..., None]).sum() + selected.float().sum()*0
            loss.backward()
        finally:
            h1.remove()
            h2.remove()
        del captured, selected, hidden, loss


class TransferObserver:
    def __init__(self, actor, diagnostic, step):
        self.actor, self.diagnostic, self.step = actor, diagnostic, step
        self.parameters = [(n, p) for n, p in actor.actor_module.named_parameters() if p.requires_grad]
        self.arrays = {}
        self.started = time.perf_counter()

    def before_update(self, data, observer, projection_basis):
        basis = self.diagnostic.basis.float()
        cfg = getattr(_unwrap_actor_module(self.actor.actor_module),'config',None)
        if cfg is not None and float(getattr(cfg,'attention_dropout',0.)) != 0.:
            raise ValueError('same-state VJP diagnostics require zero attention dropout')
        if any(isinstance(m,torch.nn.Dropout) and m.p != 0 for m in self.actor.actor_module.modules()):
            raise ValueError('same-state VJP diagnostics require dropout=0')
        self.groups = {id(p):group for group in self.actor.actor_optimizer.param_groups for p in group['params']}
        state_summary=[]
        for _,p in self.parameters:
            state=self.actor.actor_optimizer.state.get(p,{})
            m,v=state.get('exp_avg'),state.get('exp_avg_sq')
            if m is None:
                state_summary.append([0.]*7)
            else:
                flat=v.detach().float().reshape(-1)
                sample=flat[::max(1,flat.numel()//4096)][:4096].sqrt()
                quantiles=torch.quantile(sample,torch.tensor([0.,.5,.9,1.],device=sample.device)).cpu().tolist()
                state_summary.append([float(m.float().square().sum()),
                    0. if p.grad is None else float((m.float()*p.grad.float()).sum()),
                    float(state['step']),*quantiles])
        self.arrays['adam_before_per_parameter']=np.asarray(state_summary)
        self.arrays['adam_before_columns']=np.asarray(['momentum_sq','momentum_dot_preclip_g','step',
            'sqrt_v_sample_min','sqrt_v_sample_median','sqrt_v_sample_p90','sqrt_v_sample_max'])
        self.arrays['optimizer_lr_wd_beta1_beta2_eps']=np.asarray([
            [self.groups[id(p)]['lr'],self.groups[id(p)]['weight_decay'],
             *self.groups[id(p)]['betas'],self.groups[id(p)]['eps']] for _,p in self.parameters])
        # Name rows may be FSDP flat blocks rather than layers. Retain exact names
        # and shapes, never label a mixed flat block as attention/MLP/head.
        self.arrays['parameter_names'] = np.asarray([n for n, _ in self.parameters])
        self.arrays['parameter_numel'] = np.asarray([p.numel() for _, p in self.parameters])
        self.arrays['gradient_preclip_sq'] = np.asarray([
            0. if p.grad is None else float(p.grad.detach().float().square().sum()) for _, p in self.parameters])
        self.arrays['vjp_measured'] = np.asarray(scheduled(self.step))
        self.arrays['input_field_u_energy'] = np.asarray(
            float(torch.trace(basis.T @ observer.full_second @ basis) / observer.moments.count))
        self.arrays['input_field_total_energy'] = np.asarray(float(torch.trace(observer.full_second) / observer.moments.count))
        if scheduled(self.step):
            values, vectors, alpha = spectrum(
                (observer.full_second / observer.moments.count).cpu().numpy(), basis.cpu().numpy())
            self.arrays.update(full_eigenvalues=values, full_eigenvectors=vectors.astype(np.float32),
                               u_compatibility=alpha, retained_eigen_energy=values*alpha)
            started = time.perf_counter()
            with preserve_training_state(self.actor) as (parameters, original):
                backbone_vjp(self.actor, data, basis, 'parallel')
                parallel = gradient_snapshot(parameters)
                backbone_vjp(self.actor, data, basis, 'perpendicular')
                terms = []
                for p, u, actual in zip(parameters, parallel, original):
                    # CPU copies only for this parameter block; no 3rd full-model snapshot.
                    v = torch.zeros_like(u) if p.grad is None and u is not None else (
                        p.grad.detach().cpu() if p.grad is not None else None)
                    if u is None and v is None:
                        terms.append([0.]*6)
                        continue
                    if u is None:
                        u = torch.zeros_like(v)
                    actual = torch.zeros_like(u) if actual is None else actual
                    direct = actual - (u if projection_basis is not None else u+v)
                    terms.append([dot_cpu(u,u), dot_cpu(v,v), dot_cpu(u,v),
                                  dot_cpu(direct,direct), dot_cpu(direct,u), dot_cpu(direct,v)])
                self.arrays['vjp_terms_per_parameter'] = np.asarray(terms)
                self.arrays['vjp_term_columns'] = np.asarray([
                    'parallel_sq','perpendicular_sq','parallel_dot_perpendicular',
                    'direct_head_residual_sq','head_dot_parallel','head_dot_perpendicular'])
                del parallel
                # Eight spread-out actual training sequences, identities stored below.
                # Two sign draws expose Monte Carlo variability; this is NOT a
                # claim to measure full-batch per-token leverage exactly.
                subset_indices=np.unique(np.linspace(0,len(data)-1,min(8,len(data))).round().astype(int))
                subset = data.select_idxs(subset_indices)
                ids = representative_modes(values, alpha)
                leverage = []
                for i in ids:
                    row = []
                    for seed in (1729, 1730):
                        q = torch.from_numpy(vectors[:, i].astype(np.float32)).to(basis.device)
                        backbone_vjp(self.actor, subset, basis, 'mode', q, seed)
                        row.append(sum(float(p.grad.detach().float().square().sum())
                                       for p in parameters if p.grad is not None))
                    leverage.append(row)
                self.arrays.update(mode_indices=np.asarray(ids), mode_leverage=np.asarray(leverage),
                    mode_sign_seeds=np.asarray([1729,1730]), mode_sequence_indices=subset_indices,
                    mode_token_count=np.asarray(int(subset.batch['response_mask'].sum())))
            self.arrays['vjp_seconds'] = np.asarray(time.perf_counter()-started)
        # Snapshot only after temporary VJP gradients have been released.
        self.before = [p.detach().cpu().clone() for _, p in self.parameters]
        self.arrays['before_update_diagnostic_seconds'] = np.asarray(time.perf_counter()-self.started)

    @torch.no_grad()
    def after_update(self):
        terms = []
        for (_, p), before in zip(self.parameters, self.before):
            delta = p.detach().cpu().float() - before.float()
            group=self.groups[id(p)]
            adaptive_delta=delta+float(group['lr'])*float(group['weight_decay'])*before.float()
            g = None if p.grad is None else p.grad.detach().cpu().float()
            terms.append([dot_cpu(before,before), dot_cpu(delta,delta),
                          0. if g is None else dot_cpu(g,g),
                          0. if g is None else dot_cpu(delta,g),dot_cpu(adaptive_delta,adaptive_delta)])
        self.arrays.update(update_terms_per_parameter=np.asarray(terms),
            update_term_columns=np.asarray(['weight_sq','actual_update_sq','postclip_gradient_sq',
                'update_dot_postclip_gradient','update_without_decoupled_weight_decay_sq']),
            step=np.asarray(self.step), hidden_projection=np.asarray(self.diagnostic.projection_enabled),
            diagnostic_seconds=np.asarray(time.perf_counter()-self.started))
        self.before = None
        self.diagnostic.last_update_norm = float(np.sqrt(self.arrays['update_terms_per_parameter'][:, 1].sum()))
        atomic_npz(self.diagnostic.directory / f'transfer_{self.step:03d}.npz', self.arrays)

    def metrics(self, optimizer_seconds):
        terms=self.arrays['update_terms_per_parameter'].sum(0)
        result={'transfer/actual_parameter_update_norm':float(np.sqrt(terms[1])),
                'transfer/vjp_measured':float(self.arrays['vjp_measured']),
                'timing/transfer_extra_seconds':max(0.,float(self.arrays['diagnostic_seconds'])-optimizer_seconds)}
        if bool(self.arrays['vjp_measured']):
            u,v,uv,*_=self.arrays['vjp_terms_per_parameter'].sum(0)
            result['transfer/backbone_parallel_perp_cosine']=float(uv/max(np.sqrt(u*v),1e-30))
            e_u = float(self.arrays['input_field_u_energy'])
            e_v = max(0., float(self.arrays['input_field_total_energy']) - e_u)
            if e_u > 1e-30 and e_v > 1e-30 and v > 1e-30:
                result['dynamics/rho_squared_gain'] = float((u / e_u) / (v / e_v))
            result['timing/transfer_vjp_seconds']=float(self.arrays['vjp_seconds'])
        return result
