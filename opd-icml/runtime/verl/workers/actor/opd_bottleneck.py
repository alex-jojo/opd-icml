"""Adaptive *fixed-rank* hidden coordinates and two lagged preconditioners.

No teacher-free claim, parameter projection, first-moment replacement, or
latent weight decay. FULL discovery is followed by a frozen coordinate system.
All sufficient statistics are committed once per successful optimizer update.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
import time

import torch
import torch.nn.functional as F


@dataclass
class Settings:
    enabled: bool = False
    mode: str = 'second_moment'
    rank: int = 128
    min_steps: int = 1
    check_interval: int = 2
    patience: int = 3
    overlap: float = .85
    energy_tolerance: float = .02
    minimum_energy: float = .5
    ramp_steps: int = 5
    beta: float = .95
    damping: float = .05
    alpha: float = .5
    max_gain: float = 2.
    anchor_sequences: int = 32
    samples_per_sequence: int = 64
    output_dir: str = ''
    inner_max_steps: int = 8
    inner_backtracks: int = 3
    inner_lr_factor: float = 1.
    inner_gain_ratio: float = .1
    inner_min_gain: float = 1e-6
    inner_radius_relative: float = .02
    inner_radius_absolute: float = 1.
    inner_max_signal_gain: float = 2.

    def __post_init__(self):
        if self.mode not in ('second_moment', 'response', 'bottleneck_reuse'):
            raise ValueError('unknown bottleneck mode')
        if not (1 <= self.inner_max_steps <= 32 and 0 <= self.inner_backtracks <= 8
                and self.inner_lr_factor > 0 and 0 <= self.inner_gain_ratio <= 1
                and self.inner_min_gain >= 0 and self.inner_radius_relative > 0
                and self.inner_radius_absolute > 0 and self.inner_max_signal_gain >= 1):
            raise ValueError('invalid bounded inner-optimization settings')
        if min(self.rank, self.min_steps, self.check_interval, self.patience,
               self.ramp_steps, self.anchor_sequences, self.samples_per_sequence) < 1:
            raise ValueError('positive rank, history, ramp and sample counts required')
        if not (0 < self.overlap <= 1 and 0 < self.minimum_energy <= 1
                and 0 <= self.energy_tolerance < 1
                and 0 <= self.beta < 1 and 0 < self.damping <= 1
                and 0 < self.alpha <= 1 and self.max_gain >= 1):
            raise ValueError('invalid stability/preconditioner settings')


class Geometry:
    """CPU checkpointable sufficient statistics; GPU matrices only when used."""
    def __init__(self, cfg, width):
        self.cfg, self.width = cfg, width
        if cfg.rank >= width:
            raise ValueError('rank must be smaller than hidden width')
        self.step = 0
        self.freeze_step = 0
        self.stable_checks = 0
        self.covariance = torch.zeros(width, width)
        self.candidate = None
        self.basis = None
        self.energy = 0.
        self.similarity = 0.
        self.fresh_capture = 0.
        self.spectrum = torch.zeros(width)
        self.v = torch.zeros(cfg.rank)
        self.response = torch.zeros(cfg.rank)
        self.response_weight = torch.zeros(cfg.rank)
        self.moment_steps = 0
        self.last_response_positive = 0.

    def observe(self, delta):
        """Cumulative unit-hidden displacement from the SAME step-zero states.

        Never EMA interval outer products: those discard cross-time terms and
        do not reproduce the historical step10_unit_rank128 definition.
        """
        if self.basis is not None:
            raise RuntimeError('cannot rotate frozen bottleneck coordinates')
        delta = delta.detach().float()
        if delta.ndim != 2 or delta.shape[1] != self.width or not len(delta):
            raise ValueError('invalid fixed-panel drift')
        if not torch.isfinite(delta).all():
            raise FloatingPointError('nonfinite anchor drift')
        c = (delta.T @ delta / len(delta)).cpu()
        self.covariance.copy_(c)
        # Exact 1024-dimensional symmetric eigensolve, only at check intervals.
        eigenvalues, eigenvectors = torch.linalg.eigh(self.covariance.to(delta.device))
        eigenvalues = eigenvalues.clamp_min(0)
        self.spectrum = eigenvalues.flip(0).cpu()
        if float(eigenvalues.sum()) < 1e-16:
            self.stable_checks = 0
            return
        candidate = eigenvectors[:, -self.cfg.rank:].flip(1).cpu()
        energy = float(eigenvalues[-self.cfg.rank:].sum()/eigenvalues.sum())
        self.fresh_capture = (float((delta @ self.candidate.to(delta.device)).square().sum()
                                    /delta.square().sum().clamp_min(1e-30))
                              if self.candidate is not None else 0.)
        self.similarity = (float((candidate.T @ self.candidate).square().sum()/self.cfg.rank)
                           if self.candidate is not None else 0.)
        stable = (self.step >= self.cfg.min_steps and self.similarity >= self.cfg.overlap
                  and abs(energy-self.energy) <= self.cfg.energy_tolerance
                  and energy >= self.cfg.minimum_energy
                  and self.fresh_capture >= self.cfg.minimum_energy)
        self.stable_checks = self.stable_checks + 1 if stable else 0
        self.candidate, self.energy = candidate, energy
        if self.stable_checks >= self.cfg.patience:
            self.basis = candidate.clone()
            self.freeze_step = self.step

    def gains(self):
        """Uses only completed prior steps; same gains for every microbatch.

        Normalize the diagonal metric by its active-coordinate median before
        damping, so absolute loss reduction/LR units cannot explode the hook.
        Clips retain positive scaling and individual token signs. Not whitening.
        """
        if self.cfg.mode == 'bottleneck_reuse':
            return torch.ones_like(self.v)
        if self.cfg.mode == 'second_moment':
            if not self.moment_steps:
                return torch.ones_like(self.v)
            metric = self.v/(1-self.cfg.beta**self.moment_steps)
            valid = metric > max(float(metric.max())*1e-8, 1e-30)
            power = .5
        else:
            metric = self.response/self.response_weight.clamp_min(1e-12)
            valid = (self.response_weight > .1) & (metric > 0)
            power = self.cfg.alpha
        if not valid.any():
            return torch.ones_like(metric)
        scale = metric[valid].median().clamp_min(1e-30)
        gain = ((1+self.cfg.damping)/(metric.clamp_min(0)/scale+self.cfg.damping)).pow(power)
        gain = gain.clamp(1/self.cfg.max_gain, self.cfg.max_gain)
        # No amplification inferred from zero/negative/unobserved responses.
        return torch.where(valid, gain, torch.ones_like(gain))

    def commit_moment(self, second, count):
        if count <= 0 or not torch.isfinite(second).all():
            raise FloatingPointError('invalid bottleneck second moment')
        self.v.mul_(self.cfg.beta).add_(second.cpu()/count, alpha=1-self.cfg.beta)
        self.moment_steps += 1

    def commit_response(self, cross, energy, lr):
        # Regress against the ACTUALLY injected, batch-averaged hidden gradient.
        # Global Adam clipping, momentum and all shared-parameter effects remain
        # in this empirical slope; it is NOT a diagonal Jacobian estimate.
        if lr <= 0 or not torch.isfinite(cross).all() or not torch.isfinite(energy).all():
            raise FloatingPointError('invalid response sufficient statistics')
        measured = energy > max(float(energy.max())*1e-6, 1e-30)
        slope = -cross/(lr*energy.clamp_min(1e-30))
        positive = measured & (slope > 0)
        self.last_response_positive = float(positive.float().mean())
        # Bound noisy positive outliers before EMA. Negative evidence erases
        # stale positive state rather than being inverted into a large gain.
        if positive.any():
            slope = slope.clamp(max=float(slope[positive].median())*10)
        b = self.cfg.beta
        self.response.mul_(b).add_(torch.where(positive, slope, 0).cpu(), alpha=1-b)
        self.response_weight.mul_(b).add_(positive.float().cpu(), alpha=1-b)
        self.response[measured.cpu() & ~positive.cpu()] = 0
        self.response_weight[measured.cpu() & ~positive.cpu()] = 0


class ResponseMatrices:
    """Diagnostic regressions, NOT a causal cross-token response operator.

    Raw injected gradients are the input for both output geometries. Thus the
    unit-output fit is explicitly raw-gradient -> unit-hidden, not unit-gradient
    -> unit-hidden. Split by entire sequence, never by tokens of one sequence.
    All sequences still participate in the same training update; this split
    tests within-update generalization only, not independent interventions.
    """
    def __init__(self, rank, device):
        self.rank = rank
        self.parts = [dict(xx=torch.zeros(rank, rank, device=device),
                           xy=torch.zeros(rank, 2*rank, device=device),
                           yy=torch.zeros(2, device=device), count=0) for _ in range(2)]

    def add(self, signal, raw_delta, unit_delta, heldout):
        for index, selected in enumerate((~heldout, heldout)):
            x = signal[selected]
            y = torch.cat((raw_delta[selected], unit_delta[selected]), dim=1)
            p = self.parts[index]
            p['xx'].add_(x.T @ x)
            p['xy'].add_(x.T @ y)
            p['yy'].add_(y.reshape(-1, 2, self.rank).square().sum((0, 2)))
            p['count'] += len(x)

    def finish(self, lr):
        parts = [{k: v.cpu().double() if isinstance(v, torch.Tensor) else v
                  for k, v in p.items()} for p in self.parts]
        train, test = parts
        metrics = {'bottleneck/matrix_fit_tokens': train['count'],
                   'bottleneck/matrix_test_tokens': test['count']}
        result = dict(metrics=metrics, sufficient_statistics=parts,
                      input_geometry='actual_injected_raw_hidden_gradient',
                      split='even/odd training sequences; same shared parameter update',
                      used_for_optimization=False)
        scale = float(train['xx'].trace())/self.rank
        if not train['count'] or not test['count'] or scale <= 1e-30:
            metrics['bottleneck/matrix_measured'] = 0.
            return result
        # Solve a dimensionless, ridge-regularized system; never explicitly invert.
        ridge = .05
        slope = torch.linalg.solve(train['xx']/scale+ridge*torch.eye(self.rank, dtype=torch.float64),
                                   train['xy']/scale)
        result['ridge_relative'] = ridge
        metrics['bottleneck/matrix_measured'] = 1.
        for j, name in enumerate(('raw', 'unit')):
            b = slope[:, j*self.rank:(j+1)*self.rank]
            xy = test['xy'][:, j*self.rank:(j+1)*self.rank]
            yy = test['yy'][j]
            error = (yy-2*(b*xy).sum()+(b*(test['xx']@b)).sum()).clamp_min(0)
            diag_b = torch.diag(torch.diag(train['xy'][:, j*self.rank:(j+1)*self.rank]) /
                                (train['xx'].diag()+ridge*scale))
            diag_error = (yy-2*(diag_b*xy).sum()+(diag_b*(test['xx']@diag_b)).sum()).clamp_min(0)
            k = -b/lr
            result[name+'_K'] = k.float()
            metrics['bottleneck/'+name+'_matrix_zero_baseline_r2'] = float(1-error/yy) if yy > 1e-30 else 0.
            metrics['bottleneck/'+name+'_diagonal_zero_baseline_r2'] = float(1-diag_error/yy) if yy > 1e-30 else 0.
            metrics['bottleneck/'+name+'_response_target_measured'] = float(yy > 1e-30)
            metrics['bottleneck/'+name+'_K_offdiag_energy'] = float((k.square().sum()-k.diag().square().sum()).clamp_min(0)/k.square().sum().clamp_min(1e-30))
        return result


def cpu_inputs(inputs):
    return {k: inputs[k].detach().cpu().clone() for k in
            ('input_ids', 'attention_mask', 'position_ids', 'responses', 'response_mask')}


def token_indices(mask, maximum):
    rows, positions = [], []
    for row in range(mask.shape[0]):
        valid = mask[row].nonzero().flatten()
        if not len(valid):
            continue
        selected = torch.linspace(0, len(valid)-1, min(len(valid), maximum), device=mask.device).round().long()
        positions.append(valid[selected])
        rows.append(torch.full_like(selected, row))
    if not rows:
        empty = torch.empty(0, dtype=torch.long, device=mask.device)
        return empty, empty.clone()
    return torch.cat(rows), torch.cat(positions)


@contextmanager
def inert_evaluation(actor):
    module = actor.actor_module
    flags = [(m, m.training) for m in module.modules()]
    device = next(module.parameters()).device
    devices = [device.index or 0] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        module.eval()
        try:
            yield
        finally:
            for m, flag in flags:
                m.training = flag


class Runtime:
    def __init__(self, actor, cfg):
        self.actor, self.cfg = actor, cfg
        from .opd_top16 import _final_norm_module
        self.norm = _final_norm_module(actor.actor_module)
        self.geometry = None
        self.anchor = []
        self.anchor_initial = None
        self.baseline_parts = []
        self.anchor_cursor = 0
        self.device_basis = None
        self.records = []
        self.metrics = {}

    @torch.no_grad()
    def replay(self, inputs, indices):
        """Official FSDP forward; only one logit position, no teacher or backward."""
        device = next(self.actor.actor_module.parameters()).device
        x = {k: v.to(device) for k, v in inputs.items()}
        captured = []
        n = x['responses'].shape[1]
        rows, pos = [v.to(device) for v in indices]
        def read(_m, _a, h):
            captured.append(h[:, -n-1:-1][rows, pos].detach().float().clone())
        handle = self.norm.register_forward_hook(read)
        try:
            with inert_evaluation(self.actor), torch.autocast(device_type=device.type,
                    dtype=getattr(self.actor, 'param_dtype', torch.bfloat16), enabled=device.type == 'cuda'):
                self.actor.actor_module(input_ids=x['input_ids'], attention_mask=x['attention_mask'],
                    position_ids=x['position_ids'], use_cache=False, return_dict=True, logits_to_keep=1)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError('expected one official final-norm forward')
        return captured[0]

    def begin(self, data, total):
        started = time.perf_counter()
        if int(data.batch['response_mask'].sum()) == 0:
            raise ValueError('bottleneck training batch has no valid response tokens; no update committed')
        if not self.anchor:
            # First training batch supplies a frozen measurement bank. No extra
            # generation and no teacher cache pass. Never replaced by new text.
            tensors = {k:v[:self.cfg.anchor_sequences] for k,v in data.batch.items()}
            micro = self.actor.config.ppo_micro_batch_size_per_gpu
            for start in range(0, len(tensors['responses']), micro):
                x = cpu_inputs({k:v[start:start+micro] for k,v in tensors.items()})
                ix = token_indices(x['response_mask'], self.cfg.samples_per_sequence)
                self.anchor.append((x, ix))
            if not any(len(ix[0]) for _, ix in self.anchor):
                self.anchor.clear()
                raise ValueError('initial fixed bank has no valid tokens; increase anchor_sequences')
            # Dimension metadata only; do not bypass FSDP or do an extra forward.
            model_config = getattr(self.actor.actor_module, 'config', None)
            width = getattr(model_config, 'hidden_size', None)
            if width is None:
                shape = getattr(self.norm, 'normalized_shape', None)
                if not shape:
                    raise ValueError('final hidden width metadata is unavailable')
                width = shape[-1]
            self.geometry = Geometry(self.cfg, int(width))
        self.total = float(total)
        self.second = torch.zeros(self.cfg.rank, device=next(self.actor.actor_module.parameters()).device)
        self.count = 0
        self.records = []
        self.baseline_parts = []
        self.anchor_cursor = 0
        self.sequence_cursor = 0
        self.v3_stats = {}
        self.v3_head_cache = None
        self.v3_head_seen = None
        g = self.geometry
        if g.basis is not None and self.device_basis is None:
            self.device_basis = g.basis.to(self.second.device)
        self.u = self.device_basis if g.basis is not None else None
        self.gain = g.gains().to(self.second.device)
        self.ramp = min(1., (g.step+1-g.freeze_step)/self.cfg.ramp_steps) if self.u is not None else 0.
        if self.cfg.mode == 'bottleneck_reuse' and self.u is not None:
            self.ramp = 1.  # V3 has no A/B gain ramp or EMA preconditioner.
        # Transition ONLY the positive scaling INSIDE U, never mix in FULL g.
        self.applied_gain = (1-self.ramp)+self.ramp*self.gain
        self.full_energy = self.second.new_zeros(())
        self.injected_energy = self.second.new_zeros(())
        self.sampled_injected_energy = self.second.new_zeros(())
        self.sampled_outside_energy = self.second.new_zeros(())
        self.audit_tokens = 0
        self.metrics = {'bottleneck/frozen_before_update': float(self.u is not None),
                        'bottleneck/ramp': self.ramp,
                        'bottleneck/full_gradient_mix': 0. if self.u is not None else 1.,
                        'bottleneck/diagonal_gain_used': float(self.u is not None and self.cfg.mode != 'bottleneck_reuse'),
                        'bottleneck/response_measured': 0.,
                        'bottleneck/extra_student_forwards': 0.,
                        'bottleneck/discovery_measured_this_update': 0.}
        self.seconds = time.perf_counter()-started

    @contextmanager
    def watch(self, inputs, source_inputs=None):
        source_inputs = inputs if source_inputs is None else source_inputs
        if self.u is None:
            # Capture step-zero states in the actual first training forward.
            # No extra generation, teacher call, or initial student replay.
            handle = None
            if self.anchor_initial is None and self.anchor_cursor < len(self.anchor):
                x, ix_cpu = self.anchor[self.anchor_cursor]
                self.anchor_cursor += 1
                n = inputs['responses'].shape[1]
                ix = tuple(v.to(inputs['input_ids'].device) for v in ix_cpu)
                def baseline(_m, _a, h):
                    self.baseline_parts.append(F.normalize(h.detach()[:, -n-1:-1][ix].float(), dim=-1))
                handle = self.norm.register_forward_hook(baseline)
            try:
                yield
            finally:
                if handle is not None:
                    handle.remove()
            return
        if self.cfg.mode == 'bottleneck_reuse':
            from .opd_bottleneck_reuse import watch
            with watch(self, inputs, source_inputs):
                yield
            return
        n = inputs['responses'].shape[1]
        mask = inputs['response_mask'].bool()
        valid_count = int(source_inputs['response_mask'].sum())
        ix_cpu = token_indices(source_inputs['response_mask'], self.cfg.samples_per_sequence)
        ix = tuple(v.to(mask.device) for v in ix_cpu)
        record = None
        if self.cfg.mode == 'response' and valid_count:
            record = {'inputs': cpu_inputs(source_inputs), 'indices': ix_cpu,
                      'heldout': ((ix[0]+self.sequence_cursor) % 2 == 1)}
        self.sequence_cursor += len(source_inputs['responses'])
        def capture(_m, _a, hidden):
            if record is not None:
                with torch.autocast(device_type=hidden.device.type, enabled=False):
                    sampled = hidden.detach()[:, -n-1:-1][ix].float()
                    record['before'] = sampled @ self.u
                    record['before_unit'] = F.normalize(sampled, dim=-1) @ self.u
            def backward(gradient):
                # Final norm is outside checkpointed decoder layers. Freeze
                # gains for the whole batch; no EMA updates inside this hook.
                valid = gradient[:, -n-1:-1].float()
                z = valid @ self.u
                self.second.add_((z[mask]*self.total).square().sum(0))
                self.count += valid_count
                changed = (z*self.applied_gain) @ self.u.T
                changed = torch.where(mask[..., None], changed, valid)
                self.full_energy.add_(valid[mask].square().sum())
                result = gradient.clone()
                result[:, -n-1:-1] = changed.to(gradient.dtype)
                actual = result[:, -n-1:-1].float()
                self.injected_energy.add_(actual[mask].square().sum())
                # Audit the RETURNED gradient; BF16 rounding can produce tiny
                # out-of-span components even when the FP32 proposal is in U.
                sampled = actual[ix]
                sampled_coords = sampled @ self.u
                self.sampled_injected_energy.add_(sampled.square().sum())
                self.sampled_outside_energy.add_((sampled-sampled_coords@self.u.T).square().sum())
                self.audit_tokens += len(sampled)
                if record is not None:
                    # BF16 rounding occurs BEFORE the gradient reaches the model.
                    record['injected'] = sampled_coords
                    self.records.append(record)
                return result
            hidden.register_hook(backward)
        handle = self.norm.register_forward_hook(capture)
        try:
            yield
        finally:
            handle.remove()

    @torch.no_grad()
    def validate_before_update(self):
        """Fail before AdamW if a hook is missing/duplicated or calibration is bad."""
        if self.anchor_initial is None:
            if (len(self.baseline_parts) != len(self.anchor) or
                    sum(len(p) for p in self.baseline_parts) != sum(len(ix[0]) for _, ix in self.anchor)):
                raise RuntimeError('initial training forward did not capture the fixed anchor bank')
        if self.u is not None:
            if self.count != int(self.total):
                raise RuntimeError('bottleneck hook token count differs from batch; no update committed')
            if not torch.isfinite(self.second).all():
                raise FloatingPointError('nonfinite hidden-gradient statistics; no update committed')
            if self.cfg.mode == 'response' and not self.records:
                raise RuntimeError('missing response records; no update committed')

    @torch.no_grad()
    def after(self, lr):
        started = time.perf_counter()
        g = self.geometry
        if self.anchor_initial is None:
            if len(self.baseline_parts) != len(self.anchor):
                raise RuntimeError('initial training forward did not capture the fixed anchor bank')
            self.anchor_initial = torch.cat(self.baseline_parts).cpu()
        self.baseline_parts.clear()
        g.step += 1
        cross = energy = None
        response_matrix = None
        if self.u is not None:
            g.commit_moment(self.second, self.count)
            if self.cfg.mode == 'response':
                cross = torch.zeros(self.cfg.rank, device=self.u.device)
                energy = torch.zeros_like(cross)
                matrix_stats = ResponseMatrices(self.cfg.rank, self.u.device)
                for rec in self.records:
                    raw = self.replay(rec['inputs'], rec['indices'])
                    after = raw @ self.u
                    change = after-rec['before']
                    unit_change = F.normalize(raw, dim=-1) @ self.u-rec['before_unit']
                    signal = rec['injected']
                    cross += (signal*change).sum(0)
                    energy += signal.square().sum(0)
                    matrix_stats.add(signal, change, unit_change, rec['heldout'])
                g.commit_response(cross, energy, lr)
                response_matrix = matrix_stats.finish(lr)
                self.metrics.update(response_matrix['metrics'])
                self.metrics['bottleneck/response_measured'] = 1.
                self.metrics['bottleneck/extra_student_forwards'] = len(self.records)
            if self.cfg.mode == 'bottleneck_reuse':
                for key, value in self.v3_stats.items():
                    self.metrics['bottleneck_reuse/'+key] = float(value)
                tokens = max(float(self.v3_stats.get('tokens', 0)), 1.)
                for key in ('accepted_steps', 'initial_kl', 'virtual_final_kl', 'signal_gain', 'virtual_radius_fraction'):
                    self.metrics['bottleneck_reuse/'+key+'_mean'] = float(self.v3_stats.get(key, 0))/tokens
                self.metrics['bottleneck_reuse/full_backward_calls_per_microbatch'] = 1.
                self.metrics['bottleneck_reuse/extra_transformer_forwards'] = 0.
                self.metrics['bottleneck/raw_hidden_gradient_measured'] = 0.
            else:
                self.metrics['bottleneck/raw_hidden_gradient_energy'] = float(self.full_energy)
                self.metrics['bottleneck/raw_hidden_gradient_measured'] = 1.
            self.metrics.update({
                'bottleneck/injected_hidden_gradient_energy': float(self.injected_energy),
                'bottleneck/gain_min': float(self.applied_gain.min()), 'bottleneck/gain_max': float(self.applied_gain.max()),
                'bottleneck/projection_audit_tokens': self.audit_tokens,
                'bottleneck/sampled_outside_energy_fraction': float(self.sampled_outside_energy/self.sampled_injected_energy.clamp_min(1e-30)),
                'bottleneck/projection_audit_nonzero_signal': float(self.sampled_injected_energy > 1e-30),
                'bottleneck/response_positive_fraction': g.last_response_positive,
                'bottleneck/response_replay_microbatches': len(self.records)})
        elif g.step % self.cfg.check_interval == 0:
            self.metrics['bottleneck/discovery_measured_this_update'] = 1.
            nonempty = [(x, ix) for x, ix in self.anchor if len(ix[0])]
            after = F.normalize(torch.cat([self.replay(x, ix) for x, ix in nonempty]), dim=-1)
            g.observe(after-self.anchor_initial.to(after.device))
            self.metrics['bottleneck/extra_student_forwards'] = len(nonempty)
        self.metrics.update({'bottleneck/frozen': float(g.basis is not None),
            'bottleneck/freeze_step': g.freeze_step, 'bottleneck/overlap': g.similarity,
            'bottleneck/previous_basis_fresh_capture': g.fresh_capture,
            'bottleneck/top128_energy': g.energy, 'bottleneck/stable_checks': g.stable_checks,
            'bottleneck/moment_steps': g.moment_steps,
            'timing/bottleneck_measurement_seconds': self.seconds+time.perf_counter()-started})
        if self.cfg.output_dir:
            folder = Path(self.cfg.output_dir)
            folder.mkdir(parents=True, exist_ok=True)
            # Small per-update tensors enable auditing all 128 coordinates.
            target = folder/f'step_{g.step:04d}.pt'
            torch.save(dict(step=g.step, metrics=self.metrics, v=g.v, response=g.response,
                            response_weight=g.response_weight, gains=self.gain.cpu(),
                            applied_gains=self.applied_gain.cpu(),
                            current_second_sum=self.second.cpu(), valid_training_tokens=self.count,
                            response_cross=cross.cpu() if cross is not None else None,
                            response_signal_energy=energy.cpu() if energy is not None else None,
                            response_matrix=response_matrix,
                            v3_stats={k: v.cpu() if isinstance(v, torch.Tensor) else v for k,v in self.v3_stats.items()},
                            second_statistic=('unaveraged_coordinate_correction' if self.cfg.mode == 'bottleneck_reuse'
                                              else 'unaveraged_raw_projected_gradient'),
                            discovery_spectrum=g.spectrum), target.with_suffix('.tmp'))
            target.with_suffix('.tmp').replace(target)
            if g.freeze_step == g.step:
                torch.save(dict(basis=g.basis, covariance=g.covariance, step=g.step), folder/'frozen_basis.pt')
        self.records.clear()
        self.v3_head_cache = self.v3_head_seen = None
        return {k:[v] for k,v in self.metrics.items()}

    def state_dict(self):
        return dict(schema=3, discovery='cumulative_unit_from_step0',
                    projection=('coordinate_correction_surrogate' if self.cfg.mode == 'bottleneck_reuse'
                                else 'immediate_U_only_gain_ramp'), config=asdict(self.cfg),
                    geometry={k:v for k,v in vars(self.geometry).items() if k != 'cfg'},
                    anchor=self.anchor, anchor_initial=self.anchor_initial)

    def load_state_dict(self, state):
        expected = asdict(self.cfg)
        old = dict(state['config'])
        old.pop('output_dir'); expected.pop('output_dir')
        projection = 'coordinate_correction_surrogate' if self.cfg.mode == 'bottleneck_reuse' else 'immediate_U_only_gain_ramp'
        if (state['schema'] != 3 or state.get('discovery') != 'cumulative_unit_from_step0'
                or state.get('projection') != projection or old != expected):
            raise ValueError('bottleneck settings changed across resume')
        self.geometry = Geometry(self.cfg, state['geometry']['width'])
        self.geometry.__dict__.update(state['geometry'])
        self.anchor, self.anchor_initial = state['anchor'], state['anchor_initial']
        self.device_basis = None
        g = self.geometry
        if not (g.step >= g.freeze_step >= 0 and len(self.anchor) > 0
                and self.anchor_initial.ndim == 2 and self.anchor_initial.shape[1] == g.width):
            raise ValueError('invalid bottleneck checkpoint counters/anchors')
        for value in (g.covariance, g.v, g.response, g.response_weight, self.anchor_initial):
            if not torch.isfinite(value).all():
                raise ValueError('nonfinite bottleneck checkpoint')
        if g.basis is not None:
            if (g.basis.shape != (g.width,self.cfg.rank) or not torch.isfinite(g.basis).all()
                    or not torch.allclose(g.basis.T@g.basis,torch.eye(self.cfg.rank),atol=2e-3,rtol=0)):
                raise ValueError('invalid frozen basis in checkpoint')


def get_runtime(actor):
    config = actor.config.policy_loss.get('bottleneck', {})
    if not config.get('enabled', False):
        return None
    if not hasattr(actor, '_bottleneck'):
        actor._bottleneck = Runtime(actor, Settings(**dict(config)))
    return actor._bottleneck


def save_state(actor, directory):
    runtime = get_runtime(actor)
    if runtime is not None:
        if runtime.geometry is None:
            raise RuntimeError('cannot checkpoint uninitialized bottleneck training')
        target = Path(directory)/'bottleneck_state.pt'
        torch.save(runtime.state_dict(), target.with_suffix('.tmp'))
        target.with_suffix('.tmp').replace(target)


def load_state(actor, directory):
    runtime = get_runtime(actor)
    if runtime is not None:
        runtime.load_state_dict(torch.load(Path(directory)/'bottleneck_state.pt', map_location='cpu', weights_only=True))
