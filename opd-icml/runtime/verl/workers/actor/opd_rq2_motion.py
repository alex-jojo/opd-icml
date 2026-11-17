"""Portable fixed-probe RQ2 measurements after each actual FULL optimizer update."""
import numpy as np
import torch

from .opd_plasticity import atomic_npz
from .opd_replication17 import _cumulative_basis_local, panel_positions


class MotionRecorder:
    reference_step = 10

    def __init__(self, directory, legacy_directory, rank):
        self.directory = directory
        self.legacy_directory = legacy_directory
        self.rank = rank
        self.reference = None

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
        elif update > self.reference_step and self.reference is None:
            self.reference = self.load_reference()
        arrays = dict(step=update, positions=len(current), eigenvalues=eigenvalues,
                      cumulative_energy_coverage=cumulative, basis=basis.numpy(),
                      raw_displacement_rms=summary['raw_displacement_rms'],
                      unit_displacement_rms=summary['unit_displacement_rms'])
        arrays['fixed_u10_available'] = bool(update >= self.reference_step and
                                             self.reference is not None and self.reference.shape[1])
        if basis.shape[1] and arrays['fixed_u10_available']:
            reference = self.reference
            coverage = float((reference * (second @ reference)).sum() / eigen.sum())
            overlap = float((reference.T @ basis).square().sum() / reference.shape[1])
            arrays.update(fixed_u10_coverage=coverage, fixed_u10_overlap=overlap)
            metrics.update({'rq2/fixed_u10_coverage': coverage, 'rq2/fixed_u10_overlap': overlap})
        atomic_npz(self.directory / f'step_{update:03d}.npz', arrays)
        # Only legacy observation steps need these dense matrices on other ranks.
        return metrics, (basis, second, eigen, summary) if legacy_due else None
