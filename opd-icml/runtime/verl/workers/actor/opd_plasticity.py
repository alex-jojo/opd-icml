"""Read-only hidden diagnostics for ONE ordinary, single-GPU OPD run.

No projection, parameter update, controller decision, or RNG consumption is
performed here. Gradients are reported per-token BEFORE batch averaging and
parameter-gradient clipping. Fixed-prefix diagnostics use an analytic logit
gradient, not a second Transformer backward pass.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F


def atomic_npz(path, arrays):
    if torch.distributed.is_initialized() and torch.distributed.get_rank()!=0:return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def atomic_json(path, value):
    if torch.distributed.is_initialized() and torch.distributed.get_rank()!=0:return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def sampled_positions(count, maximum=32):
    if count < 1:
        raise ValueError('empty response in diagnostic panel')
    return torch.linspace(0, count - 1, min(count, maximum)).round().long()


@torch.no_grad()
def response_channel_samples(hidden, gradient, lp, lq, ids, generated, index):
    """Reuse existing replay tensors: no model/head call or autograd operation.

    Entropies/top1/probabilities are CONDITIONAL ON CACHED SUPPORT, not full
    vocabulary. A generated token outside support has unknown probability (NaN),
    not zero teacher probability. Full sampled g allows future U-perp analysis.
    """
    sp, tq = lp[index].float(), lq[index].float()
    support = ids[index]
    present = support.eq(generated[index, None])
    probability = (tq.exp() * present).sum(-1)
    probability = probability.masked_fill(~present.any(-1), float('nan'))
    return dict(sample_full_gradient=gradient[index].float().cpu(),
                sample_student_entropy=(-(sp.exp()*sp).sum(-1)).cpu(),
                sample_teacher_entropy=(-(tq.exp()*tq).sum(-1)).cpu(),
                sample_support_top1_agree=(sp.argmax(-1)==tq.argmax(-1)).cpu(),
                sample_generated_in_support=present.any(-1).cpu(),
                sample_teacher_generated_conditional_probability=probability.cpu())


def kl_and_logit_gradient(student, teacher):
    """Conditional reverse KL on fixed support; derivative of EACH token loss."""
    lp = F.log_softmax(student.float(), dim=-1)
    lq = F.log_softmax(teacher.float(), dim=-1)
    p = lp.exp()
    kl = (p * (lp - lq)).sum(-1)
    derivative = p * (lp - lq - kl.unsqueeze(-1))
    return kl, derivative


def head_gradient(derivative, ids, weight, chunk=64, forward_dtype=None):
    # Bound the temporary [tokens,16,1024] gather, rather than materializing it
    # for an entire 2048-token response. This is NOT a model forward bypass.
    result = []
    for start in range(0, len(ids), chunk):
        rows = weight[ids[start:start + chunk]]
        if forward_dtype is not None:
            rows = rows.to(forward_dtype)
        rows = rows.float()
        result.append(torch.einsum('tk,tkd->td', derivative[start:start + chunk], rows))
    return torch.cat(result)


@contextmanager
def capture_fixed_head_gradient(actor_module, ids, teacher):
    """Compute dKL/dh while the official lm_head forward owns live weights.

    FSDP may release its BF16 parameter storage as soon as the root forward
    returns, even with NO_SHARD. Never retain/read a weight view afterwards.
    Only independently allocated results leave this hook; no extra model
    forward/backward, parameter copy, or summon_full_params is required.
    """
    from .opd_top16 import _unwrap_actor_module
    head = _unwrap_actor_module(_unwrap_actor_module(actor_module).lm_head)
    captured = {}

    def observe(module, _args, output):
        if captured:
            raise RuntimeError('fixed diagnostic expected exactly one output-head call')
        if not torch.is_tensor(output) or output.ndim != 3 or output.shape[0] != 1:
            raise ValueError('fixed diagnostic expects single-sequence causal LM logits')
        n = len(ids)
        if output.shape[1] < n + 1:
            raise ValueError('output-head logits missing causal predictor positions')
        # The enclosing model forward runs under autocast. Explicitly disable
        # it for analytic FP32 diagnostics, not for the model's computation.
        with torch.no_grad(), torch.autocast(device_type=output.device.type, enabled=False):
            selected = output[0, -n - 1:-1].gather(-1, ids)
            kl, derivative = kl_and_logit_gradient(selected, teacher)
            gradient = head_gradient(derivative, ids, module.weight,
                                     forward_dtype=output.dtype)
            captured.update(selected=selected, kl=kl, gradient=gradient)
        # Returning None leaves the model output unmodified.

    handle = head.register_forward_hook(observe)
    try:
        yield captured
        if not captured:
            raise RuntimeError('official output-head forward was not observed')
    finally:
        handle.remove()


class Moments:
    """Exact streaming token moments plus per-sequence aggregates (no cancellation)."""
    def __init__(self, basis, random_basis=None):
        self.basis = basis
        self.random_basis = random_basis
        self.random_energy = torch.zeros((), device=basis.device, dtype=torch.float64)
        rank = basis.shape[1]
        self.count = 0
        self.sum = torch.zeros(rank, device=basis.device, dtype=torch.float64)
        self.second = torch.zeros(rank, rank, device=basis.device, dtype=torch.float64)
        self.positive = torch.zeros_like(self.sum)
        self.absolute = torch.zeros_like(self.sum)
        self.full = torch.zeros((), device=basis.device, dtype=torch.float64)
        self.full_sum = torch.zeros(basis.shape[0], device=basis.device, dtype=torch.float64)
        self.sequences = []

    def add(self, values):
        values = values.detach().float()
        n = len(values)
        if n == 0:
            raise ValueError('no valid tokens')
        b = values @ self.basis
        self.count += n
        s = b.sum(0, dtype=torch.float64)
        gram = b.T @ b
        full = values.square().sum(dtype=torch.float64)
        self.sum += s
        self.second += gram.double()
        self.positive += (b > 0).sum(0)
        self.absolute += b.abs().sum(0, dtype=torch.float64)
        self.full += full
        self.full_sum += values.sum(0, dtype=torch.float64)
        if self.random_basis is not None:
            self.random_energy += (values @ self.random_basis).square().sum(dtype=torch.float64)
        # Kept on device until the step boundary: no scalar GPU synchronizations.
        self.sequences.append(torch.cat((s / n, b.square().mean(0).double(),
                                        full.reshape(1) / n,
                                        torch.tensor([n], device=values.device))))
        return b

    def arrays(self, prefix):
        if self.count == 0:
            raise ValueError('diagnostic hook did not observe any valid tokens')
        mean = self.sum / self.count
        second = self.second / self.count
        covariance = second - mean[:, None] * mean[None, :]
        full_mean = self.full_sum / self.count
        energy = self.full / self.count
        kept = second.diagonal().sum()
        values = {
            'count': np.asarray(self.count), 'mean': mean, 'second': second,
            'covariance': covariance, 'energy': second.diagonal(),
            'positive_fraction': self.positive / self.count,
            'mean_abs': self.absolute / self.count,
            'full_mean': full_mean, 'full_energy': energy,
            'parallel_fraction': kept / energy.clamp_min(1e-30),
            'perpendicular_energy': (energy - kept).clamp_min(0),
            'full_mean_energy_fraction': full_mean.square().sum() / energy.clamp_min(1e-30),
            'sequence_moments': torch.stack(self.sequences),
        }
        if self.random_basis is not None:
            values['random_rank128_fraction'] = self.random_energy / self.full.clamp_min(1e-30)
        return {prefix + '_' + key: value.detach().cpu().numpy() if torch.is_tensor(value)
                else value for key, value in values.items()}


class TrainingObserver:
    def __init__(self, basis, total, random_basis=None, transfer=False):
        self.moments = Moments(basis, random_basis)
        self.total = total.detach() if torch.is_tensor(total) else total
        self.samples = []
        self.sample_positions = []
        self.seconds = 0.
        self.output_statistics = []
        self.transfer = transfer
        self.full_second = torch.zeros((basis.shape[0], basis.shape[0]), device=basis.device) if transfer else None
        self.full_samples = []
        self.boundary_second = torch.zeros_like(self.full_second) if transfer else None
        self.boundary_count = 0
        self.projected = False

    def consume(self, gradient):
        started = time.perf_counter()
        rows = gradient[:, -self.active_mask.shape[1]-1:-1].detach().float()
        rows = rows * torch.as_tensor(self.total, device=rows.device)
        with torch.no_grad():
            for row, valid in zip(rows, self.active_mask):
                values = row[valid]
                coordinates = self.moments.add(values)
                index = sampled_positions(len(coordinates), 8).to(row.device)
                self.samples.append(coordinates[index].cpu())
                self.sample_positions.append(torch.nonzero(valid).flatten()[index].cpu())
                if self.transfer:
                    self.full_second.add_(values.T @ values)
                    # First valid predictor state is a response-start boundary.
                    # Save it separately so large shared boundary modes can be
                    # removed EXACTLY, rather than inferred from sparse samples.
                    self.boundary_second.add_(values[:1].T @ values[:1])
                    self.boundary_count += 1
                    self.full_samples.append(values[index].cpu())
        self.seconds += time.perf_counter() - started

    @torch.no_grad()
    def record_outputs(self, selected, teacher, mask, kl):
        lp = selected.detach().float().log_softmax(-1)
        lq = teacher.detach().float().log_softmax(-1)
        hs, ht = -(lp.exp() * lp).sum(-1), -(lq.exp() * lq).sum(-1)
        for losses, student_h, teacher_h, valid in zip(kl.detach(), hs, ht, mask.bool()):
            self.output_statistics.append(torch.stack((losses[valid].mean(), losses[valid].max(),
                                                       student_h[valid].mean(), teacher_h[valid].mean())))

    @contextmanager
    def watch(self, actor, inputs):
        from .opd_top16 import _final_norm_module
        mask = inputs['response_mask'].bool()
        self.active_mask = mask
        response_length = mask.shape[1]
        handles = []

        def capture(_module, _args, hidden):
            def backward(gradient):
                # KL was divided by the FULL train response-token count. Undo
                # that scalar ONLY for reporting; return the original gradient.
                if not self.projected:
                    self.consume(gradient)
                return gradient
            handles.append(hidden.register_hook(backward))

        forward_handle = _final_norm_module(actor.actor_module).register_forward_hook(capture)
        try:
            yield
        finally:
            forward_handle.remove()
            for handle in handles:
                handle.remove()

    def arrays(self):
        result = self.moments.arrays('gradient')
        result.update(sample_coefficients=torch.cat(self.samples).numpy(),
                      sample_positions=torch.cat(self.sample_positions).numpy(),
                      sample_lengths=np.asarray([len(x) for x in self.samples]),
                      train_token_denominator=np.asarray(float(self.total)),
                      observation_host_seconds=np.asarray(self.seconds))
        if self.output_statistics:
            result['sequence_kl_mean_max_student_entropy_teacher_entropy'] = torch.stack(self.output_statistics).cpu().numpy()
        if self.transfer:
            result['gradient_full_second'] = (self.full_second / self.moments.count).cpu().numpy()
            result['sample_full_gradient'] = torch.cat(self.full_samples).numpy()
            result['gradient_first_token_second_sum'] = self.boundary_second.cpu().numpy()
            result['gradient_first_token_count'] = np.asarray(self.boundary_count)
        return result


class PlasticityDiagnostic:
    def __init__(self, actor, config):
        self.actor = actor
        self.transfer = bool(config.get('transfer_enabled', False))
        self.projection_enabled = bool(actor.config.policy_loss.get('hidden_projection', {}).get('enabled', False))
        self.directory = Path(config['output_dir'])
        self.directory.mkdir(parents=True, exist_ok=True)
        basis_path = Path(config['basis_path'])
        panel_path = Path(config['panel_path'])
        self.device = next(actor.actor_module.parameters()).device
        with np.load(basis_path, allow_pickle=False) as asset:
            basis = asset['basis'].astype(np.float32)
            self.eigenvalues = asset['eigenvalues'].copy()
        if basis.shape != (1024, 128) or not np.isfinite(basis).all():
            raise ValueError('diagnostic requires the checked 1024x128 basis')
        if np.max(abs(basis.T @ basis - np.eye(128))) > 2e-3:
            raise ValueError('nonorthogonal diagnostic basis')
        self.basis = torch.from_numpy(basis).to(self.device)
        # Local generator: the random-reference measurement does not alter the
        # model/rollout RNG stream. Same dimensionality, no training intervention.
        random = np.random.default_rng(1729).standard_normal((1024, 128))
        random = np.linalg.qr(random, mode='reduced')[0].astype(np.float32)
        self.random_basis = torch.from_numpy(random).to(self.device)
        self.panel = torch.load(panel_path, map_location='cpu', weights_only=True)
        if len(self.panel['rows']) != 32 or self.panel['schema'] != 'opd-fixed-panel-v1':
            raise ValueError('expected 32 separately generated, non-training diagnostic responses')
        metadata = dict(schema='opd-plasticity-v1', basis_sha256=hashlib.sha256(basis_path.read_bytes()).hexdigest(),
                        panel_sha256=hashlib.sha256(panel_path.read_bytes()).hexdigest(),
                        projection_enabled=self.projection_enabled, per_token_gradient_unaveraged=True,
                        causal_positions='prompt-end + response[:-1], predicting response tokens',
                        reference_support='frozen step0 student top16, teacher logits cached once',
                        fixed_panel_training=False)
        marker = self.directory / 'manifest.json'
        if self.transfer:
            metadata['transfer_schema'] = 'actual-batch-vjp-v1'
        if marker.exists() and json.loads(marker.read_text()) != metadata:
            raise ValueError('diagnostic assets differ from the existing run')
        atomic_json(marker, metadata)
        atomic_npz(self.directory / 'basis.npz', dict(basis=basis, eigenvalues=self.eigenvalues, random_basis=random))
        self.base = None
        self.previous = None
        self.last_step = None
        self.initial_readout_seconds = 0.
        self.temporal_window = None
        if config.get('temporal_window_enabled', False):
            from .opd_temporal_window import TemporalWindow
            self.temporal_window = TemporalWindow()

    @contextmanager
    def evaluation_mode(self):
        module = self.actor.actor_module
        prior = [(child, child.training) for child in module.modules()]
        # Restoring RNG makes the observation inert even if a model unexpectedly
        # consumes random numbers in eval. CUDA-device list is explicit.
        devices = [self.device.index or 0] if self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            module.eval()
            try:
                yield
            finally:
                for child, training in prior:
                    child.training = training

    @torch.no_grad()
    def replay(self, step, persist=True):
        from .opd_top16 import _final_norm_module
        started = time.perf_counter()
        names = ('fixed_gradient', 'cumulative_unit', 'velocity_unit',
                 'cumulative_raw', 'velocity_raw')
        moments = {name: Moments(self.basis, self.random_basis if name == 'fixed_gradient' else None) for name in names}
        states, gradients, samples, sample_grad, sample_indices, lengths = [], [], [], [], [], []
        sequence_kl, sequence_mass, sequence_overlap, sequence_norm = [], [], [], []
        student_entropy, teacher_entropy, sampled_kl, sampled_mass = [], [], [], []
        channel_samples = []
        full_velocity = {name: torch.zeros((1024,1024),device=self.device) for name in
                         ('raw','unit')} if self.transfer else {}
        work, work_parallel, cosine, angular_work, radial_work = [], [], [], [], []
        with self.evaluation_mode():
            for i, row in enumerate(self.panel['rows']):
                inputs = row['input_ids'].to(self.device).reshape(1, -1)
                ids = row['top16_ids'].to(self.device).long()
                teacher = row['teacher_logits'].to(self.device)
                n = len(ids)
                captured = []
                handle = _final_norm_module(self.actor.actor_module).register_forward_hook(
                    lambda _m, _a, out: captured.append(out.detach()))
                try:
                    with capture_fixed_head_gradient(self.actor.actor_module, ids, teacher) as readout, \
                         torch.autocast(device_type=self.device.type,
                                        dtype=getattr(self.actor, 'param_dtype', torch.bfloat16),
                                        enabled=self.device.type == 'cuda'):
                        out = self.actor.actor_module(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                            position_ids=torch.arange(inputs.shape[1], device=self.device).reshape(1, -1),
                            use_cache=False, return_dict=True, logits_to_keep=n + 1)
                finally:
                    handle.remove()
                hidden = captured.pop()[0, -n - 1:-1].float()
                logits = out.logits[0, -n - 1:-1]
                selected, kl, g = readout['selected'], readout['kl'], readout['gradient']
                lp, lq = selected.float().log_softmax(-1), teacher.float().log_softmax(-1)
                student_entropy.append(float(-(lp.exp() * lp).sum(-1).mean()))
                teacher_entropy.append(float(-(lq.exp() * lq).sum(-1).mean()))
                coordinates = moments['fixed_gradient'].add(g)
                # Chunk vocabulary normalizers; never cast the entire vocabulary
                # logits for all tokens to FP32 simultaneously.
                mass = torch.cat([(s.float() - l.float().logsumexp(-1, keepdim=True)).exp().sum(-1)
                                  for s, l in zip(selected.split(64), logits.split(64))])
                current_ids = logits.topk(16, dim=-1).indices
                overlap = (current_ids[:, :, None] == ids[:, None, :]).any(-1).float().mean(-1)
                unit = F.normalize(hidden, dim=-1)
                base = hidden if self.base is None else self.base[i].to(self.device)
                prev = hidden if self.previous is None else self.previous['hidden'][i].to(self.device)
                delta_raw = hidden - prev
                delta_unit = unit - F.normalize(prev, dim=-1)
                for name, delta in (('raw',delta_raw),('unit',delta_unit)):
                    if name in full_velocity:
                        full_velocity[name].add_(delta.T @ delta)
                moments['cumulative_raw'].add(hidden - base)
                moments['cumulative_unit'].add(unit - F.normalize(base, dim=-1))
                moments['velocity_raw'].add(delta_raw)
                moments['velocity_unit'].add(delta_unit)
                if self.previous is not None:
                    old_g = self.previous['gradient'][i].to(self.device)
                    old_unit = F.normalize(prev, dim=-1)
                    old_norm = prev.norm(dim=-1)
                    radial = (old_g * old_unit).sum(-1)
                    tangent_g = old_norm[:, None] * (old_g - radial[:, None] * old_unit)
                    work.append((old_g * delta_raw).sum(-1).sum().cpu())
                    work_parallel.append(((old_g @ self.basis) * (delta_raw @ self.basis)).sum().cpu())
                    angular_work.append((tangent_g * delta_unit).sum().cpu())
                    radial_work.append((radial * (hidden.norm(dim=-1) - old_norm)).sum().cpu())
                    cosine.append(F.cosine_similarity(-old_g, delta_raw, dim=-1).sum().cpu())
                idx = sampled_positions(n).to(self.device)
                channel_samples.append(response_channel_samples(
                    hidden, g, lp, lq, ids, inputs[0, -n:], idx))
                samples.append(hidden[idx].cpu()); sample_grad.append(coordinates[idx].cpu())
                sample_indices.append(idx.cpu()); lengths.append(n)
                sampled_kl.append(kl[idx].cpu()); sampled_mass.append(mass[idx].cpu())
                states.append(hidden.cpu()); gradients.append(g.cpu())
                sequence_kl.append(float(kl.mean())); sequence_mass.append(float(mass.mean()))
                sequence_overlap.append(float(overlap.mean())); sequence_norm.append(float(hidden.norm(dim=-1).mean()))
                del out, logits, captured, hidden, readout, g
        arrays = {key: value for name, moment in moments.items() for key, value in moment.arrays(name).items()}
        arrays.update(step=np.asarray(step), sequence_lengths=np.asarray(lengths),
            sequence_kl=np.asarray(sequence_kl), sequence_support_mass=np.asarray(sequence_mass),
            sequence_support_overlap=np.asarray(sequence_overlap), sequence_hidden_norm=np.asarray(sequence_norm),
            sequence_student_entropy=np.asarray(student_entropy), sequence_teacher_entropy=np.asarray(teacher_entropy),
            sample_kl=torch.cat(sampled_kl).numpy(), sample_support_mass=torch.cat(sampled_mass).numpy(),
            sample_hidden=torch.cat(samples).numpy(), sample_gradient_coefficients=torch.cat(sample_grad).numpy(),
            sample_positions=torch.cat(sample_indices).numpy(), sample_lengths=np.asarray([len(x) for x in samples]),
            work_raw=np.asarray(sum(work).item() / sum(lengths) if work else 0.),
            work_parallel=np.asarray(sum(work_parallel).item() / sum(lengths) if work_parallel else 0.),
            work_angular=np.asarray(sum(angular_work).item() / sum(lengths) if angular_work else 0.),
            work_radial=np.asarray(sum(radial_work).item() / sum(lengths) if radial_work else 0.),
            descent_velocity_cosine=np.asarray(sum(cosine).item() / sum(lengths) if cosine else 0.),
            work_available=np.asarray(self.previous is not None),
            readout_seconds=np.asarray(time.perf_counter() - started))
        for key in channel_samples[0]:
            arrays[key] = torch.cat([row[key] for row in channel_samples]).numpy()
        for name, second in full_velocity.items():
            arrays['velocity_'+name+'_full_second'] = (second/sum(lengths)).cpu().numpy()
        if self.temporal_window is not None:
            arrays.update(self.temporal_window.observe(step, states))
        self.last_readout_arrays = arrays
        if persist:
            atomic_npz(self.directory / f'fixed_{step:03d}.npz', arrays)
        self.previous = dict(hidden=states, gradient=gradients)
        self.last_step = step
        if self.base is None:
            self.base = states
            path = self.directory / 'baseline_hidden.pt'
            temporary = path.with_suffix('.tmp')
            torch.save(states, temporary)
            temporary.replace(path)
        mean_kl = float(np.average(sequence_kl, weights=lengths))
        return {'diagnostic/fixed_teacher_kl': mean_kl,
                'diagnostic/fixed_support_mass': float(np.average(sequence_mass, weights=lengths)),
                'diagnostic/fixed_gradient_parallel_fraction': float(arrays['fixed_gradient_parallel_fraction']),
                'diagnostic/cumulative_unit_energy': float(arrays['cumulative_unit_full_energy']),
                'diagnostic/velocity_unit_energy': float(arrays['velocity_unit_full_energy']),
                'timing/diagnostic_fixed_seconds': float(arrays['readout_seconds'])}

    def before(self, step, total):
        if self.last_step is None:
            baseline = self.directory / 'baseline_hidden.pt'
            if step > 1:
                if not baseline.is_file():
                    raise ValueError('resume requires original step0 diagnostic baseline')
                self.base = torch.load(baseline, map_location='cpu', weights_only=True)
            # On resume reconstruct previous h and g at the LOADED checkpoint,
            # never use an ahead-of-checkpoint disk cache from an interrupted run.
            old_file = self.directory / f'fixed_{step-1:03d}.npz'
            initial = self.replay(step - 1, persist=not old_file.exists())
            self.initial_readout_seconds = initial['timing/diagnostic_fixed_seconds']
        if self.last_step != step - 1:
            raise ValueError('diagnostic step is not contiguous with actor updates')
        return TrainingObserver(self.basis, total, self.random_basis, self.transfer)

    def after(self, step, observer, data, train_loss, grad_norm):
        arrays = observer.arrays()
        arrays.update(step=np.asarray(step), train_loss=np.asarray(train_loss),
            parameter_grad_norm_preclip=np.asarray(float(grad_norm)),
            parameter_clip_threshold=np.asarray(float(self.actor.config.grad_clip)),
            lr=np.asarray(self.actor.actor_optimizer.param_groups[0]['lr']),
            response_tokens=data.batch['responses'].detach().cpu().numpy(),
            response_mask=data.batch['response_mask'].detach().cpu().numpy(),
            input_ids=data.batch['input_ids'].detach().cpu().numpy(),
            attention_mask=data.batch['attention_mask'].detach().cpu().numpy())
        for key in ('opd_student_top16_mass', 'opd_teacher_top16_mass', 'token_level_scores'):
            if key in data.batch:
                value = data.batch[key].detach()
                # NumPy cannot represent BF16; preserve numeric values as FP32.
                if value.dtype == torch.bfloat16:
                    value = value.float()
                arrays[key] = value.cpu().numpy()
        atomic_npz(self.directory / f'train_{step:03d}.npz', arrays)
        metrics = self.replay(step)
        if self.temporal_window is not None:
            # Reuse in-memory observations; no extra forward or archive reads.
            fixed = self.last_readout_arrays
            for name in ('raw', 'unit'):
                key = 'temporal_'+name+'_accumulation'
                if key in fixed and np.isfinite(fixed[key]):
                    metrics['dynamics/'+key] = float(fixed[key])
                metrics['dynamics/response_'+name+'_rms'] = float(np.sqrt(fixed['velocity_'+name+'_full_energy']))
            update_norm = getattr(self, 'last_update_norm', 0.)
            if update_norm > 0:
                for name in ('raw', 'unit'):
                    metrics['dynamics/'+name+'_rms_per_parameter_update'] = metrics['dynamics/response_'+name+'_rms']/update_norm
        metrics.update({'diagnostic/train_gradient_parallel_fraction': float(arrays['gradient_parallel_fraction']),
                        'diagnostic/train_gradient_full_energy': float(arrays['gradient_full_energy']),
                        'diagnostic/train_gradient_mean_energy_fraction': float(arrays['gradient_full_mean_energy_fraction']),
                        'timing/diagnostic_hook_host_seconds': observer.seconds,
                        'timing/diagnostic_initial_or_resume_seconds': self.initial_readout_seconds,
                        'monitor/plasticity_observation_only': 1.})
        self.initial_readout_seconds = 0.
        return metrics


def get_diagnostic(actor, data, basis, reuse):
    config = actor.config.policy_loss.get('plasticity_diagnostic', {})
    if not config or not config.get('enabled', False):
        return None
    policy = actor.config.policy_loss
    if (basis is not None and not config.get('transfer_enabled', False)) or reuse != 1 or any(policy.get(k, False) for k in
            ('adaptive_reuse_enabled', 'marginal_reuse_enabled', 'super_jump_enabled')):
        raise ValueError('plasticity run must be ordinary full OPD, reuse=1, NO projection/controller')
    if policy.get('fixed_heldout_probe_size', 0) or actor.use_remove_padding:
        raise ValueError('do not remove training rows; diagnostic panel is generated separately')
    if not data.meta_info.get('global_steps'):
        raise ValueError('global_steps missing for diagnostic step alignment')
    if not hasattr(actor, '_plasticity_diagnostic'):
        actor._plasticity_diagnostic = PlasticityDiagnostic(actor, config)
    if basis is not None:
        measured = actor._plasticity_diagnostic.basis.to(dtype=basis.dtype,device=basis.device)
        if basis.shape != measured.shape or not torch.equal(basis, measured):
            raise ValueError('training projector and diagnostic U must be the identical rank128 asset')
    return actor._plasticity_diagnostic
