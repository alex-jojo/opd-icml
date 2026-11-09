"""Exact optimizer-update accounting, independent of rollout-round numbering."""


def inner_update_limit(done, target, pause, configured_reuse):
    target, done, configured_reuse = int(target), int(done), int(configured_reuse)
    endpoint = target if pause is None else int(pause)
    if not 0 <= done < endpoint <= target or not 1 <= configured_reuse <= 8:
        raise ValueError(f"invalid OPD update budget: done={done}, pause={pause}, target={target}, reuse={configured_reuse}")
    return min(configured_reuse, endpoint - done)


def advance_updates(done, performed, limit, target):
    if int(performed) != performed or not 1 <= int(performed) <= int(limit):
        raise ValueError(f"actor performed {performed} updates with a budget of {limit}")
    result = int(done) + int(performed)
    if result > int(target):
        raise ValueError(f"OPD update budget exceeded: {result} > {target}")
    return result
