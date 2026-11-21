"""Validate counted OPD resume before starting model workers; metadata only."""
import json
from pathlib import Path


def restore_paired_protocol(output, args):
    """Recover effective settings from the previous Hydra invocation before starting a new one."""
    path = output / 'hydra/.hydra/config.yaml'
    if not path.is_file():
        # An old checkpoint can also be moved without its Hydra log directory.
        if args.parameter_token_samples is not None and args.parameter_token_gradients is None:
            args.parameter_token_gradients = 'sampled'
        if (args.save_freq is None or args.parameter_token_gradients is None
                or (args.parameter_token_gradients == 'sampled' and args.parameter_token_samples is None)):
            raise ValueError(f'Missing saved training configuration {path}. Supply the original '
                             '--save-freq, --parameter-token-gradients and (for sampled mode) '
                             '--parameter-token-samples to resume this checkpoint.')
        return None

    import yaml
    try:
        cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
        tokens = cfg['actor_rollout_ref']['actor']['policy_loss']['gradient_study']['parameter_tokens']
        frequency = cfg['trainer']['save_freq']
        if type(frequency) is not int or frequency < 1 or type(tokens['enabled']) is not bool:
            raise ValueError('invalid save frequency or diagnostic setting')
        mode = {'kl_stratified': 'sampled', 'exact_all': 'exact-all'}[tokens['mode']] if tokens['enabled'] else 'off'
        samples = tokens['sample_size'] if mode == 'sampled' else None
        if mode == 'sampled' and (type(samples) is not int or samples < 2):
            raise ValueError('invalid diagnostic sample size')
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        raise ValueError(f'Invalid saved training protocol in {path}: {error}') from error

    for flag, saved in [('save_freq', frequency), ('parameter_token_gradients', mode),
                        ('parameter_token_samples', samples), ('parameter_token_group_size', tokens.get('graph_group_size', 4))]:
        requested = getattr(args, flag)
        if requested is not None and requested != saved:
            raise ValueError(f'--{flag.replace("_", "-")} conflicts with the saved training protocol '
                             f'(requested {requested}, saved {saved})')
    args.save_freq = frequency
    args.parameter_token_gradients = mode
    args.parameter_token_samples = samples
    return tokens


def validate_resume_checkpoint(directory, world_size, updates_per_round):
    root = Path(directory).resolve()
    tracker = root/'latest_checkpointed_iteration.txt'
    if not tracker.is_file():
        raise ValueError('--resume requires a completed resumable checkpoint; refusing to restart with old diagnostics')
    raw = tracker.read_text(encoding='utf-8').strip()
    if not raw.isdecimal() or int(raw) < 1:
        raise ValueError('Invalid latest checkpoint step')
    step = int(raw)
    current = root/f'global_step_{step}'
    if not current.is_dir() or not current.resolve().is_relative_to(root):
        raise ValueError('Invalid committed checkpoint directory')
    if (current/'final_model_ready.json').exists():
        raise ValueError('Final model-only export cannot resume training; use a checkpoint saved with --save-freq')
    config = current/'actor/fsdp_config.json'
    if not config.is_file():
        raise ValueError('Resume checkpoint lacks FSDP world-size metadata')
    saved_world = json.loads(config.read_text(encoding='utf-8')).get('world_size')
    if type(saved_world) is not int or saved_world != world_size:
        raise ValueError(f'Resume requires the same GPU count: checkpoint={saved_world}, requested={world_size}')
    required = ['data.pt', 'opd_optimizer_updates.json'] + [
        f'actor/{kind}_world_size_{world_size}_rank_{rank}.pt'
        for rank in range(world_size) for kind in ('model', 'optim', 'extra_state')]
    for relative in required:
        path = current/relative
        if not path.is_file() or not path.stat().st_size or not path.resolve().is_relative_to(current):
            raise ValueError(f'Incomplete resume checkpoint: {relative}')
    counter = json.loads((current/'opd_optimizer_updates.json').read_text(encoding='utf-8'))
    if (counter.get('rollout_round') != step or counter.get('optimizer_updates_done') != step*updates_per_round
            or counter.get('budget_unit') != 'rollout_rounds'):
        raise ValueError('Resume checkpoint rollout/update counters do not match this experiment')
    return current
