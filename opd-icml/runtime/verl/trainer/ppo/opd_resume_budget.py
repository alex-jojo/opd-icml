"""Explicit, narrow budget extension; never reset optimizer/data counters."""


def validate_resume_budget(counter, target, extension, scheduler_type, warmup_ratio):
    old = counter['target_optimizer_updates']
    if old == target:
        return
    if not (extension and old == 60 and target == 106
            and counter.get('optimizer_updates_done') == 60
            and counter.get('rollout_round') == 60
            and counter.get('budget_unit') == 'optimizer_updates'
            and scheduler_type == 'constant' and float(warmup_ratio) == 0.):
        raise ValueError('OPD optimizer-update target changed across checkpoint resume; '
                         'only explicit completed 60 -> 106 constant-LR extension is supported')
