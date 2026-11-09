"""V3: adaptive virtual coordinate refinement, one official model backward.

The displayed scalar remains the original KL. Its Transformer derivative is a
deliberate straight-through correction surrogate, NOT the derivative of the
unchanged numerical KL. The output head keeps its ordinary OPD derivative.
"""
from contextlib import contextmanager
import torch
import torch.nn.functional as F


def loss_derivative(logits, log_teacher):
    lp = F.log_softmax(logits.float(), dim=-1)
    p = lp.exp()
    loss = (p*(lp-log_teacher)).sum(-1)
    return loss, p*(lp-log_teacher-loss[..., None])


@torch.no_grad()
def refine(logits, teacher, b, hidden_norm, cfg):
    """Independent per-token trust-bounded steps, vectorized within a microbatch.

    Coordinates are offsets from c0. z(delta)=z0+B delta anchors the original
    (possibly BF16-rounded) logits EXACTLY. Frozen support/head/teacher throughout.
    Inner learning rates are unrelated to the outer Adam learning rate.
    """
    z0 = logits.float()
    lq = F.log_softmax(teacher.float(), dim=-1)
    current_loss, dz = loss_derivative(z0, lq)
    initial_loss = current_loss.clone()
    g0 = torch.einsum('tk,tkr->tr', dz, b)
    # Common logit offsets have no effect on softmax. Remove that null direction
    # when setting the step scale; monotone backtracking handles nonconvexity.
    centered = b-b.mean(1, keepdim=True)
    eta0 = cfg.inner_lr_factor/centered.square().sum((1, 2)).clamp_min(1e-8)
    radius = torch.minimum(hidden_norm.float()*cfg.inner_radius_relative,
                           torch.full_like(hidden_norm.float(), cfg.inner_radius_absolute))
    delta = torch.zeros_like(g0)
    eta = eta0.clone()
    first_eta = torch.ones_like(eta)
    first_gain = torch.zeros_like(eta)
    counts = torch.zeros_like(eta, dtype=torch.long)
    active = g0.square().sum(-1) > 1e-24
    # 0=no signal, 1=gain exhausted, 2=trust radius, 3=backtrack failed, 4=cap
    reason = torch.zeros_like(counts)
    factors = 0.5**torch.arange(cfg.inner_backtracks+1, device=z0.device)
    proposals = torch.zeros_like(counts)
    grad = g0
    for _ in range(cfg.inner_max_steps):
        # GPU masks avoid a host synchronization per token/inner iteration.
        # Finished tokens keep their accepted delta; bounded vectorized compute
        # may still include masked lanes up to the safety cap.
        steps = eta[:, None]*factors[None, :]
        candidates = delta[:, None, :]-steps[..., None]*grad[:, None, :]
        candidate_z = z0[:, None, :]+torch.einsum('tkr,tbr->tbk', b, candidates)
        losses, derivatives = loss_derivative(candidate_z, lq[:, None, :])
        reductions = current_loss[:, None]-losses
        inside = candidates.square().sum(-1) <= radius[:, None].square()
        finite = torch.isfinite(losses) & torch.isfinite(candidates).all(-1)
        decrease = reductions > cfg.inner_min_gain
        valid = inside & finite & decrease
        found = valid.any(-1)
        index = valid.to(torch.int64).argmax(-1)
        rows = torch.arange(len(delta), device=z0.device)
        candidate = candidates[rows, index]
        gain = reductions[rows, index]
        step_size = steps[rows, index]
        marginal = (counts == 0) | (gain >= cfg.inner_gain_ratio*first_gain)
        accepted = active & found & marginal
        first = accepted & (counts == 0)
        first_eta = torch.where(first, step_size, first_eta)
        first_gain = torch.where(first, gain, first_gain)
        no_valid = active & ~found
        reason = torch.where(no_valid, torch.where(~inside.any(-1), 2,
                             torch.where((inside & finite & ~decrease).any(-1), 1, 3)), reason)
        reason = torch.where(active & found & ~marginal, 1, reason)
        proposals += active.long()
        delta = torch.where(accepted[:, None], candidate, delta)
        current_loss = torch.where(accepted, losses[rows, index], current_loss)
        eta = torch.where(accepted, step_size, eta)
        counts += accepted.long()
        active = accepted
        # Reuse the accepted candidate's already computed derivative next time.
        grad = torch.einsum('tk,tkr->tr', derivatives[rows, index], b)
    reason = torch.where(active, 4, reason)
    # M=1 gives g0 exactly (up to arithmetic), not g0 / outer_Adam_lr.
    signal = -delta/first_eta[:, None]
    good = (counts > 0) & torch.isfinite(signal).all(-1) & ((signal*g0).sum(-1) > 0)
    signal = torch.where(good[:, None], signal, g0)
    ratio = signal.norm(dim=-1)/g0.norm(dim=-1).clamp_min(1e-20)
    scale = (cfg.inner_max_signal_gain/ratio.clamp_min(1e-20)).clamp(max=1.)
    signal *= scale[:, None]
    # This is the signal fed to the model, not an assertion that Adam realizes delta.
    stats = dict(tokens=torch.tensor(len(delta), device=z0.device),
                 accepted_steps=counts.sum(), proposed_steps=proposals.sum(),
                 initial_kl=initial_loss.sum(), virtual_final_kl=current_loss.sum(),
                 virtual_radius_fraction=(delta.norm(dim=-1)/radius.clamp_min(1e-20)).sum(),
                 signal_gain=(signal.norm(dim=-1)/g0.norm(dim=-1).clamp_min(1e-20)).sum(),
                 signal_clipped=(scale < 1).sum(), fallback=(~good).sum())
    for code, name in enumerate(('no_signal', 'gain', 'radius', 'backtrack', 'cap')):
        stats['stop_'+name] = (reason == code).sum()
    for count in range(cfg.inner_max_steps+1):
        stats['accepted_'+str(count)] = (counts == count).sum()
    return signal, delta, stats


class CorrectionCoordinates(torch.autograd.Function):
    """Explicit 128D coordinate node with a specified surrogate derivative."""
    @staticmethod
    def forward(ctx, coordinates, state):
        ctx.state = state
        return coordinates

    @staticmethod
    def backward(ctx, gradient):
        s = ctx.state
        if 'signal' not in s or s.get('consumed', False):
            raise RuntimeError('V3 coordinate signal missing or consumed twice')
        s['consumed'] = True
        return s['signal'].to(gradient.dtype), None


@contextmanager
def watch(runtime, inputs, source_inputs):
    """All weight reads happen INSIDE the official live lm_head forward hook."""
    from .opd_top16 import _unwrap_actor_module
    from .opd_bottleneck import token_indices
    model = _unwrap_actor_module(runtime.actor.actor_module)
    head = _unwrap_actor_module(model.lm_head)
    n = inputs['responses'].shape[1]
    mask = inputs['response_mask'].bool()
    valid_count = int(source_inputs['response_mask'].sum())
    state = {}
    u = runtime.u
    ix = tuple(v.to(u.device) for v in token_indices(source_inputs['response_mask'], runtime.cfg.samples_per_sequence))

    def norm_forward(_module, _args, hidden):
        if state:
            raise RuntimeError('V3 expected exactly one final norm forward')
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            h = hidden[:, -n-1:-1].float()
            c = h @ u
            state['norm'] = h.detach().norm(dim=-1)
            # The derivative enters Transformer only through this 128D node.
            c_surrogate = CorrectionCoordinates.apply(c, state)
            projection = c_surrogate @ u.T
            output = hidden.detach().clone()
            output[:, -n-1:-1] += (projection-projection.detach()).to(hidden.dtype)
        def audit(gradient):
            actual = gradient[:, -n-1:-1].float()
            sample = actual[ix]
            coords = sample @ u
            runtime.injected_energy.add_(actual[mask].square().sum())
            runtime.sampled_injected_energy.add_(sample.square().sum())
            runtime.sampled_outside_energy.add_((sample-coords@u.T).square().sum())
            runtime.audit_tokens += len(sample)
        hidden.register_hook(audit)
        return output

    def head_forward(module, _args, output):
        if 'norm' not in state or 'signal' in state:
            raise RuntimeError('V3 output-head ordering/call count is invalid')
        with torch.no_grad(), torch.autocast(device_type=output.device.type, enabled=False):
            selected = output[:, -n-1:-1].gather(-1, inputs['opd_top16_ids'])
            ids = inputs['opd_top16_ids'][mask]
            result = torch.zeros((*mask.shape, u.shape[1]), device=u.device)
            if valid_count:
                # Project only distinct cached support rows, once per microbatch.
                # Bfloat effective weights match the official head's compute dtype.
                unique = torch.unique(ids)
                if runtime.v3_head_cache is None:
                    runtime.v3_head_cache = torch.empty(module.weight.shape[0], u.shape[1], device=u.device)
                    runtime.v3_head_seen = torch.zeros(module.weight.shape[0], dtype=torch.bool, device=u.device)
                missing = unique[~runtime.v3_head_seen[unique]]
                for chunk in missing.split(1024):
                    rows = module.weight[chunk].to(output.dtype).float()
                    runtime.v3_head_cache[chunk] = rows @ u
                runtime.v3_head_seen[missing] = True
                b = runtime.v3_head_cache[ids]
                signal, _, stats = refine(selected[mask], inputs['opd_teacher_top16_logits'][mask],
                                          b, state['norm'][mask], runtime.cfg)
                result[mask] = signal/runtime.total
                runtime.second.add_(signal.square().sum(0))
                runtime.count += valid_count
                for key, value in stats.items():
                    runtime.v3_stats[key] = runtime.v3_stats.get(key, 0)+value.detach()
                runtime.v3_stats['head_unique_rows'] = runtime.v3_stats.get('head_unique_rows', 0)+len(unique)
                runtime.v3_stats['head_rows_projected'] = runtime.v3_stats.get('head_rows_projected', 0)+len(missing)
            state['signal'] = result

    norm_handle = runtime.norm.register_forward_hook(norm_forward)
    head_handle = head.register_forward_hook(head_forward)
    try:
        yield
        if 'signal' not in state or not state.get('consumed', False):
            raise RuntimeError('V3 graph was not forwarded/backpropagated exactly once')
    finally:
        head_handle.remove()
        norm_handle.remove()
