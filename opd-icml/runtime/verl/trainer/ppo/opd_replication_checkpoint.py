"""Commit a complete single-GPU replication checkpoint before rotating old ones.

Inventory comes from disk, not a worker's process-local previous_saved_paths.
Only generated global_step_<integer> directories beneath 'checkpoints' are eligible.
"""
import json
import os
from pathlib import Path
import re
import shutil


def publish_and_rotate(directory, step, reuse, keep=1, final_model_only=False):
    base = Path(directory)
    if base.is_symlink() or base.name != 'checkpoints' or not base.is_dir():
        raise ValueError('Expected a real replication checkpoints directory')
    base = base.resolve()
    current = base / f'global_step_{int(step)}'
    if current.is_symlink() or not current.is_dir() or current.resolve().parent != base:
        raise ValueError('Unsafe current checkpoint')
    config=current/'actor/fsdp_config.json'
    world=json.loads(config.read_text())['world_size'] if config.exists() else 1
    if not isinstance(world,int) or isinstance(world,bool) or world<1:
        raise ValueError('Invalid replication checkpoint world size')
    required = ('opd_optimizer_updates.json',)+tuple(f'actor/model_world_size_{world}_rank_{r}.pt' for r in range(world))
    if not final_model_only:
        required += ('data.pt',)+tuple(f'actor/{kind}_world_size_{world}_rank_{r}.pt'
            for kind in ('optim','extra_state') for r in range(world))
    for name in required:
        p = current / name
        if p.is_symlink() or not p.resolve().is_relative_to(current) or not p.is_file() or not p.stat().st_size:
            raise ValueError('Incomplete checkpoint; older checkpoints retained: ' + name)
    counter = json.loads((current / 'opd_optimizer_updates.json').read_text())
    if (counter.get('rollout_round') != step or counter.get('optimizer_updates_done') != step * reuse
            or counter.get('budget_unit') != 'rollout_rounds'):
        raise ValueError('Checkpoint budget mismatch; older checkpoints retained')
    if final_model_only:
        # This is an export/evaluation endpoint, NEVER an AdamW resume checkpoint.
        ready = current / 'final_model_ready.json'
        temp_ready = current / 'final_model_ready.tmp'
        if ready.is_symlink() or temp_ready.is_symlink():
            raise ValueError('Unsafe final-model marker')
        record = dict(rounds=int(step), optimizer_updates=int(step) * int(reuse),
                      resumable=False,world_size=world, model_bytes=sum((current / name).stat().st_size
                          for name in required if name.startswith('actor/model_')))
        with temp_ready.open('w') as stream:
            json.dump(record, stream); stream.flush(); os.fsync(stream.fileno())
        temp_ready.replace(ready)
    candidates = []
    for p in base.iterdir():
        match = re.fullmatch(r'global_step_(\d+)', p.name)
        if not match or int(match[1]) >= step:
            continue
        if p.is_symlink() or not p.is_dir() or p.resolve().parent != base:
            raise ValueError('Unsafe previous checkpoint; nothing removed')
        # Recognize only this pipeline's previous checkpoints, including folders
        # whose actor shards were already rotated by a pre-fix worker.
        metadata = p / 'opd_optimizer_updates.json'
        if metadata.is_symlink() or not metadata.is_file():
            continue
        old = json.loads(metadata.read_text())
        if (old.get('rollout_round') == int(match[1]) and old.get('budget_unit') == 'rollout_rounds'
                and old.get('optimizer_updates_done') == int(match[1]) * reuse):
            candidates.append(p)
    keep = max(int(keep or 1), 1)
    candidates.sort(key=lambda p: int(p.name.rsplit('_', 1)[1]), reverse=True)
    targets = candidates[keep - 1:]
    tracker = base / 'latest_checkpointed_iteration.txt'
    temporary = base / 'latest_checkpointed_iteration.tmp'
    if tracker.is_symlink() or temporary.is_symlink():
        raise ValueError('Unsafe checkpoint tracker')
    with temporary.open('w') as stream:
        stream.write(str(step)); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(tracker)
    removed = []
    for p in targets:
        shutil.rmtree(p)
        removed.append(p.name)
    print(f'[CHECKPOINT] round {step} complete; older generated checkpoints removed: {removed}', flush=True)
    return removed
