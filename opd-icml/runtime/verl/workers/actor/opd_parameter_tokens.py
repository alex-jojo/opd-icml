"""Exact per-token FULL parameter-gradient energies before training backward.

Every rank replays the SAME token loss. FSDP reduces the same gradient, not an
average of unrelated token gradients. Only one sharded gradient is live at a
time; full parameter vectors are never serialized.
"""
import time
import numpy as np
import torch
from . import opd_distributed as dp


def causal_token_prefix(inputs, position):
    """Future suffix cannot affect a selected causal token's loss or gradient.

    Retain the one final input slot expected by forward()'s logits alignment,
    all prompt padding/position IDs, and the selected response token itself.
    """
    old_length = inputs['responses'].shape[1]
    length = int(position)+1
    if not 0 < length <= old_length:
        raise ValueError('Token position is outside the response')
    prefix = inputs['input_ids'].shape[1]-old_length
    trimmed = dict(inputs)
    for key in ('input_ids','attention_mask','position_ids'):
        trimmed[key] = inputs[key][..., :prefix+length]
    for key in ('responses','response_mask','opd_top16_ids','opd_teacher_top16_logits','opd_student_top16_logits'):
        if key in inputs:
            trimmed[key] = inputs[key][:, :length]
    return trimmed


class GradientEnergy:
    """Norm across unique trainable parameters, excluding flat padding."""
    def __init__(self, model):
        from .opd_gradient_study import ParameterLayout
        from torch.nn.parallel import DistributedDataParallel
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
        self.layout = ParameterLayout(model)
        if self.layout.replicated_adapter is not None:
            raise ValueError('Full parameter token audit requires full-model training, not an adapter')
        self.divisor = 1
        if dp.size() > 1:
            wrappers = [m for m in model.modules() if isinstance(m, FSDP)]
            strategies = {m.sharding_strategy for m in wrappers}
            if strategies and strategies <= {ShardingStrategy.FULL_SHARD, ShardingStrategy.SHARD_GRAD_OP}:
                if any(getattr(m, '_use_orig_params', False) for m in wrappers):
                    raise ValueError('Exact token audit currently requires FSDP use_orig_params=false')
            elif strategies == {ShardingStrategy.NO_SHARD} or isinstance(model, DistributedDataParallel):
                self.divisor = dp.size()
            else:
                raise ValueError('Exact token audit supports FSDP FULL_SHARD/SHARD_GRAD_OP/NO_SHARD or DDP')
        for p in self.layout.params:
            if hasattr(p, 'to_local'):
                raise ValueError('Exact token audit currently supports FSDP1, not DTensor/FSDP2')

    @torch.no_grad()
    def local_squared_norm(self, device):
        # Keep reductions on their device; scalar conversion per tiny block
        # otherwise forces thousands of CUDA/host synchronizations per token.
        subtotals = {}
        for index, start, count, _ in self.layout.slices:
            grad = self.layout.params[index].grad
            if grad is None:
                continue
            if grad.is_sparse:
                raise ValueError('Exact token audit requires dense parameter gradients')
            flat = grad.detach().reshape(-1)
            if flat.device not in subtotals:
                subtotals[flat.device] = torch.zeros((),device=flat.device,dtype=torch.float64)
            for offset in range(start, start + count, 262144):
                block = flat[offset:min(offset + 262144, start + count)].double()
                subtotals[flat.device].add_(block.square().sum())
        if not subtotals:
            return torch.zeros((), device=device, dtype=torch.float64)
        parts = iter(subtotals.values())
        value = next(parts).to(device)
        for subtotal in parts:
            value.add_(subtotal.to(device))
        return value if self.divisor == 1 else value / self.divisor

    def collect(self, values):
        # Row 0 is sharded energy; row 1 is a rank-local loss on the same token.
        if dp.size() > 1:
            device = torch.device('cuda', torch.cuda.current_device()) if torch.distributed.get_backend() == 'nccl' else 'cpu'
            values = values.to(device)
            torch.distributed.all_reduce(values[0])
        result = values.cpu().numpy()
        if not np.isfinite(result[0]).all():
            raise FloatingPointError('Nonfinite per-token parameter gradient energy')
        return result

    def squared_norm(self):
        energy = self.local_squared_norm(self.layout.params[0].device)
        return float(self.collect(energy.reshape(1, 1))[0, 0])


class FullParameterTokenAudit:
    def __init__(self, actor, cfg):
        self.actor = actor
        self.cfg = dict(cfg)
        # Bound the number of live token heads, without batching parameter
        # Jacobians. Setting 1 preserves the original low-memory replay path.
        self.group_size = self.cfg.get('graph_group_size', 4)
        if isinstance(self.group_size, bool) or int(self.group_size) != self.group_size or self.group_size < 1:
            raise ValueError('graph_group_size must be a positive integer')
        self.group_size = int(self.group_size)
        from .opd_graph_reuse import reuse_fallback_reason
        self.reuse_fallback = reuse_fallback_reason(actor.actor_module) if self.group_size > 1 else None
        if self.reuse_fallback:
            self.group_size = 1
        self.mode = self.cfg.get('mode', 'exact_all')
        if self.mode not in {'exact_all', 'kl_stratified'}:
            raise ValueError('Unsupported parameter token audit mode: ' + str(self.mode))
        if self.mode == 'kl_stratified':
            size, seed = self.cfg.get('sample_size', 256), self.cfg.get('seed', 42)
            if int(size) != size or int(seed) != seed or size < 2 or seed < 0:
                raise ValueError('sample_size must be an integer >= 2 and seed a nonnegative integer')
        from .opd_top16 import _unwrap_actor_module
        model = _unwrap_actor_module(actor.actor_module)
        config = getattr(model, 'config', None)
        self.prefix = bool(self.cfg.get('causal_prefix', True)) and getattr(config,'model_type',None) in {
            'llama','qwen3','qwen2','gemma3_text'}
        dropout = any(isinstance(m, torch.nn.modules.dropout._DropoutNd) and m.p > 0
                      for m in model.modules())
        dropout |= any(float(getattr(config, key, 0.) or 0.) > 0 for key in
                       ('attention_dropout', 'hidden_dropout', 'hidden_dropout_prob',
                        'resid_pdrop', 'attn_pdrop', 'embd_pdrop'))
        if dropout:
            raise ValueError('Exact paired token audit requires zero dropout for the same diagnostic/training objective')

    def measure(self, data, step, inner):
        from .opd_gradient_study import gradient_observation
        from .opd_top16 import forward, reverse_kl
        from .opd_graph_reuse import RetainedGraph
        started = time.perf_counter()
        # No training gradients need copying/restoring. No optimizer step.
        if any(p.grad is not None for p in self.actor.actor_module.parameters()):
            raise RuntimeError('Parameter token audit must start with cleared gradients')
        if hasattr(self.actor, 'fixed_u_adapter'):
            raise ValueError('Full parameter token audit does not support fixed-U adapters')
        layout = GradientEnergy(self.actor.actor_module)
        device = next(self.actor.actor_module.parameters()).device
        mask = data.batch['response_mask'].bool().cpu().numpy()
        rows, positions = np.nonzero(mask)
        population = len(rows)
        sampling = None
        if self.mode == 'kl_stratified':
            if 'opd_student_top16_logits' not in data.batch:
                raise ValueError('Stratified sampling requires cached pre-update student Top16 logits')
            from .opd_parameter_sampling import stratified_plan
            def choose():
                # Match output_scores' double-precision KL and stable ranking.
                lp = data.batch['opd_student_top16_logits'].detach().cpu().double().log_softmax(-1)
                lq = data.batch['opd_teacher_top16_logits'].detach().cpu().double().log_softmax(-1)
                kl = (lp.exp() * (lp-lq)).sum(-1).clamp_min(0).numpy()[mask]
                return stratified_plan(kl, int(self.cfg.get('sample_size', 256)),
                                       int(self.cfg.get('seed', 42)), step, inner)
            # Every FSDP rank differentiates the exact same sampled token loss.
            sampling = dp.primary_call(choose)
            rows, positions = rows[sampling['index']], positions[sampling['index']]
        measurements = np.empty((2, len(rows)), dtype=np.float64)
        if not len(rows):
            raise ValueError('Cannot audit an empty token population')
        if dp.primary():
            label = f'ALL {len(rows)}' if sampling is None else f'SAMPLED {len(rows)}/{population}'
            print(f'[parameter tokens] step={step} inner={inner+1}: exact full-model '
                  f'gradients for {label} tokens; {len(rows)} extra backwards', flush=True)
            print(f'[parameter tokens] graph group size={self.group_size}' +
                  (f'; fallback: {self.reuse_fallback}' if self.reuse_fallback else ''), flush=True)
        cursor = 0
        capacity = min(len(rows), 256)
        values = torch.empty((2, capacity), dtype=torch.float64, device=device)
        full_forward_tokens = actual_forward_tokens = forward_calls = 0
        with gradient_observation(self.actor):
            for row in np.unique(rows):
                selected = data.select_idxs([row])
                inputs = {k: v.to(device) for k, v in selected.batch.items()}
                row_positions = positions[rows == row]
                for start in range(0, len(row_positions), self.group_size):
                    group = row_positions[start:start+self.group_size].tolist()
                    self.actor.actor_optimizer.zero_grad(set_to_none=True)
                    token_inputs = causal_token_prefix(inputs,group[-1]) if self.prefix else inputs
                    full_forward_tokens += inputs['input_ids'].numel() * len(group)
                    actual_forward_tokens += token_inputs['input_ids'].numel()
                    forward_calls += 1
                    _, logits, _, _ = forward(self.actor, token_inputs, float(data.meta_info['temperature']),
                                              token_inputs['opd_top16_ids'], diagnostics=False,
                                              response_positions=group)
                    teacher_logits = token_inputs['opd_teacher_top16_logits'][:, group]
                    token_losses = reverse_kl(logits, teacher_logits)[0]
                    graph = RetainedGraph(self.actor.actor_module) if len(group) > 1 else None
                    for index in range(len(group)):
                        if index:
                            graph.prepare_next_backward()
                        token_loss = token_losses[index]
                        token_loss.backward(retain_graph=index+1 < len(group))
                        slot = cursor % capacity
                        values[0, slot] = layout.local_squared_norm(device)
                        values[1, slot] = token_loss.detach()
                        self.actor.actor_optimizer.zero_grad(set_to_none=True)
                        del token_loss
                        cursor += 1
                        if slot+1 == capacity or cursor == len(rows):
                            batch_start = cursor - (slot + 1)
                            measurements[:, batch_start:cursor] = layout.collect(values[:, :slot+1])
                        if dp.primary() and (cursor % (16 if sampling is not None else 256) == 0 or cursor == len(rows)):
                            print(f'[parameter tokens] {cursor}/{len(rows)}; '
                                  f'{time.perf_counter()-started:.1f}s', flush=True)
                    del logits, token_losses, graph
        if dp.primary():
            print(f'[parameter tokens] completed: {forward_calls} forwards, {len(rows)} backwards '
                  '(checkpoint recomputations excluded)', flush=True)
        result = dict(row=rows, pos=positions, energy=measurements[0], diagnostic_kl=measurements[1],
                      seconds=time.perf_counter()-started, backwards=len(rows),
                      causal_prefix=self.prefix, forward_input_tokens=actual_forward_tokens,
                      forward_calls=forward_calls, graph_group_size=self.group_size,
                      graph_reuse_fallback=self.reuse_fallback,
                      full_suffix_forward_input_tokens=full_forward_tokens)
        if sampling is not None:
            result['sampling'] = sampling
        return result


def attach_parameter_energy(fields, measurement):
    """Join by (global response row, response position), never by array order."""
    source = list(zip(measurement['row'].tolist(), measurement['pos'].tolist()))
    target = list(zip(fields['row'].tolist(), fields['pos'].tolist()))
    if len(set(source)) != len(source) or len(set(target)) != len(target) or set(source) != set(target):
        raise ValueError('Parameter and hidden audits must cover exactly the SAME valid tokens')
    lookup = {key: i for i, key in enumerate(source)}
    order = np.asarray([lookup[key] for key in target])
    fields['parameter_gradient_energy'] = measurement['energy'][order]
    return dict(mode='exact_all', exact=True, sampled=False, coverage_fraction=1.,
                token_count=len(target), extra_backwards=measurement['backwards'],
                diagnostic_optimizer_steps=0, extra_teacher_calls=0,
                parameter_vectors_saved=False, seconds=measurement['seconds'],
                causal_prefix=measurement.get('causal_prefix',False),
                forward_calls=measurement.get('forward_calls'),
                graph_group_size=measurement.get('graph_group_size',1),
                graph_reuse_fallback=measurement.get('graph_reuse_fallback'),
                forward_input_tokens=measurement.get('forward_input_tokens'),
                full_suffix_forward_input_tokens=measurement.get('full_suffix_forward_input_tokens'),
                definition='E_i = sum over unique trainable parameters theta of (d ell_i / d theta)^2',
                objective='unmodified conditional cached Top16 KL; individual token loss BEFORE batch mean and clipping',
                diagnostic_mode='dropout-free, checkpoint recomputation enabled; original module modes and RNG restored',
                includes='all trainable backbone, embedding, normalization and output-head parameters; tied weights counted once',
                groups='same actual training token KL rankings and step-1 threshold as hidden-state statistics',
                diagnostic_vs_training_kl_max_abs=float(np.max(np.abs(measurement['diagnostic_kl'][order]-fields['kl']))),
                warning='Sum of individual squared gradient norms is NOT squared norm of the summed gradient or AdamW update energy')
