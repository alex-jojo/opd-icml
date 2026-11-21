"""Portable RQ2 measurements of fixed-probe displacement and training hidden gradients."""
import numpy as np
import torch
import time
import torch.nn.functional as F

from .opd_plasticity import atomic_npz
from .opd_replication17 import _cumulative_basis_local, panel_positions


class MotionRecorder:
    reference_step = 10

    def __init__(self, directory, legacy_directory, rank):
        self.directory = directory
        self.legacy_directory = legacy_directory
        self.rank = rank
        self.reference = None
        self.early_hidden = None
        self.previous_hidden = None
        self.previous_step = None

    def incremental_change(self, update, initial, current, reference):
        """Summarize already computed probe states, without a model pass or SVD."""
        unit = F.normalize(current.float(), dim=-1)
        if self.previous_step is None and update == 1:
            self.previous_step = 0
            self.previous_hidden = F.normalize(initial.float(), dim=-1)
        early_path = self.directory / 'reference_010.npz'
        if update > self.reference_step and self.early_hidden is None and early_path.is_file():
            with np.load(early_path, allow_pickle=False) as saved:
                self.early_hidden = torch.from_numpy(saved['unit_hidden'].copy())
        if update == self.reference_step and reference is not None:
            self.early_hidden = unit
            atomic_npz(early_path, dict(unit_hidden=unit.numpy()))
        result = dict(increment_normalization='unit hidden per probe position',
                      since_step10_available=self.early_hidden is not None and update >= self.reference_step,
                      previous_observation_available=self.previous_step is not None,
                      previous_step_available=self.previous_step == update-1)

        def summarize(prefix, old):
            if old.shape != unit.shape:
                raise ValueError('Fixed probe shape changed across representation observations')
            energy = inside = 0.
            # Subtract first: subtracting two O(1) projected states erases tiny
            # late-training increments in FP32. Bound the CPU scratch buffers.
            for start in range(0, len(unit), 256):
                delta = unit[start:start+256] - old[start:start+256]
                energy += float(delta.double().square().sum())
                if reference is not None:
                    inside += float((delta @ reference).double().square().sum())
            result[prefix+'_energy'] = energy
            result[prefix+'_frobenius_norm'] = energy**.5
            result[prefix+'_rms'] = (energy / max(len(unit), 1))**.5
            if reference is not None and energy > 1e-20:
                result[prefix+'_fixed_u10_coverage'] = min(max(inside / energy, 0.), 1.)

        if result['since_step10_available']:
            summarize('since_step10', self.early_hidden)
        if result['previous_step_available']:
            result['previous_checkpoint'] = self.previous_step
            summarize('previous_step', self.previous_hidden)
        if result['previous_observation_available']:
            result['previous_observation_checkpoint'] = self.previous_step
            result['previous_observation_update_gap'] = update - self.previous_step
            if result['previous_step_available']:
                for key, value in list(result.items()):
                    if key.startswith('previous_step_') and key != 'previous_step_available':
                        result[key.replace('previous_step_', 'previous_observation_', 1)] = value
            else:
                summarize('previous_observation', self.previous_hidden)
        self.previous_step = update
        self.previous_hidden = unit
        return result

    def save_initial(self, panel, hidden, device):
        positions = panel_positions(panel.batch['response_mask'].to(device))
        rows = [torch.full_like(indices, i) for i, indices in enumerate(positions)]
        if sum(map(len, positions)) != len(hidden):
            raise ValueError('Initial hidden rows do not match the fixed probe positions')
        inputs = {key: panel.batch[key].detach().cpu().numpy() for key in
                  ('input_ids', 'attention_mask', 'position_ids', 'responses', 'response_mask')}
        atomic_npz(self.directory / 'probe_initial.npz', dict(
            **inputs, row=torch.cat(rows).cpu().numpy(), response_position=torch.cat(positions).cpu().numpy(),
            initial_hidden=hidden.numpy()))

    def load_reference(self):
        # Older FULL runs stored the step-10 basis only in replication/motion.
        for directory in (self.directory, self.legacy_directory):
            path = directory / f'step_{self.reference_step:03d}.npz'
            if path.is_file():
                with np.load(path, allow_pickle=False) as saved:
                    return torch.from_numpy(saved['basis'].copy())
        return None

    def reference_at(self, checkpoint):
        if checkpoint < self.reference_step:
            return None
        if self.reference is None:
            self.reference = self.load_reference()
        if self.reference is not None and self.reference.shape[1]:
            return self.reference
        return None

    def record_gradient(self, update, second, count, eigenvalues=None):
        # second = G.T @ G / count, collected during the actual training backward.
        matrix = torch.from_numpy(second).double()
        if eigenvalues is None:
            eigenvalues = torch.linalg.eigvalsh((matrix + matrix.T) * .5).flip(0).clamp_min(0).numpy()
        checkpoint = update - 1
        reference = self.reference_at(checkpoint)
        arrays = dict(optimizer_update=update, checkpoint_before=checkpoint, valid_tokens=count,
                      eigenvalues=eigenvalues, fixed_u10_available=reference is not None)
        metrics = {}
        total = eigenvalues.sum()
        if total > 0:
            own = float(eigenvalues[:128].sum() / total)
            arrays['own_top128_coverage'] = own
            metrics['rq2/gradient_own_top128_coverage'] = own
            if reference is not None:
                u10 = reference.double()
                fixed = float((((matrix @ u10) * u10).sum() / matrix.trace()).clamp(0, 1))
                arrays['fixed_u10_coverage'] = fixed
                metrics['rq2/gradient_fixed_u10_coverage'] = fixed
        atomic_npz(self.directory / f'gradient_{update:03d}.npz', arrays)
        return metrics

    def record(self, update, initial, current, *, legacy_due):
        basis, second, eigen, summary = _cumulative_basis_local(initial, current, self.rank, allow_zero=True)
        summary['raw_displacement_rms'] = float((current - initial).square().sum(1).mean().sqrt())
        eigenvalues = eigen.numpy()
        cumulative = np.cumsum(eigenvalues, dtype=np.float64)
        metrics = {f'rq2/{key}': summary[key] for key in ('raw_displacement_rms', 'unit_displacement_rms')}
        if basis.shape[1]:
            cumulative /= cumulative[-1]
            metrics['rq2/own_top_rank_coverage'] = float(cumulative[basis.shape[1]-1])
        else:
            cumulative = np.empty(0, dtype=np.float64)

        if update == self.reference_step:
            self.reference = basis
        reference = self.reference_at(update)
        arrays = dict(step=update, positions=len(current), eigenvalues=eigenvalues,
                      cumulative_energy_coverage=cumulative, basis=basis.numpy(),
                      raw_displacement_rms=summary['raw_displacement_rms'],
                      unit_displacement_rms=summary['unit_displacement_rms'])
        arrays['fixed_u10_available'] = reference is not None
        if basis.shape[1] and reference is not None:
            coverage = float((reference * (second @ reference)).sum() / eigen.sum())
            overlap = float((reference.T @ basis).square().sum() / reference.shape[1])
            arrays.update(fixed_u10_coverage=coverage, fixed_u10_overlap=overlap)
            metrics.update({'rq2/fixed_u10_coverage': coverage, 'rq2/fixed_u10_overlap': overlap})
        increment_started = time.perf_counter()
        increments = self.incremental_change(update, initial, current, reference)
        arrays.update(increments)
        metrics['timing/rq2_increment_summary_seconds'] = time.perf_counter() - increment_started
        for key in ('since_step10_fixed_u10_coverage', 'previous_step_fixed_u10_coverage',
                    'since_step10_rms', 'previous_step_rms'):
            if key in increments:
                metrics['rq2/'+key] = increments[key]
        atomic_npz(self.directory / f'step_{update:03d}.npz', arrays)
        # Only legacy observation steps need these dense matrices on other ranks.
        return metrics, (basis, second, eigen, summary) if legacy_due else None
