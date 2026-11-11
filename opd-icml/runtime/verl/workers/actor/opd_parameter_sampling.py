"""Stratified exact token gradients; population energies are survey estimates.

Sampling never selects training supervision. The two sampling strata use the
whole pre-update batch's cached conditional KL, not a sampled KL percentile.
"""
import numpy as np


def stratified_plan(kl, sample_size=256, seed=42, step=1, inner=0):
    kl = np.asarray(kl, dtype=np.float64)
    if kl.ndim != 1 or not len(kl) or not np.isfinite(kl).all():
        raise ValueError('Sampling requires a nonempty finite token KL vector')
    if int(sample_size) != sample_size or sample_size < 2 or int(seed) != seed or int(seed) < 0:
        raise ValueError('sample_size must be at least 2 and seed must be nonnegative')
    count = len(kl)
    high = np.zeros(count, bool)
    high[np.argsort(-kl, kind='stable')[:max(1, int(np.ceil(count * .1)))]] = True
    pools = [np.flatnonzero(high), np.flatnonzero(~high)]
    take = min(int(sample_size), count)
    if take == count:
        sizes = [len(p) for p in pools]
    else:
        first = min(len(pools[0]), max(1, int(round(take * .25))))
        second = min(len(pools[1]), take - first)
        first += take - first - second
        sizes = [first, second]
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(step), int(inner)]))
    chosen, weights, labels, strata = [], [], [], []
    for label, (pool, size) in enumerate(zip(pools, sizes)):
        strata.append(dict(name='kl_top10' if label == 0 else 'kl_remaining90',
                           population_count=len(pool), sample_count=size))
        if not size:
            continue
        indices = rng.choice(pool, size=size, replace=False)
        chosen.extend(indices.tolist())
        weights.extend([len(pool) / size] * size)
        labels.extend([label] * size)
    order = np.argsort(chosen)
    return dict(index=np.asarray(chosen, dtype=np.int64)[order],
                weight=np.asarray(weights, dtype=np.float64)[order],
                stratum=np.asarray(labels, dtype=np.int64)[order],
                population_count=count, strata=strata, seed=int(seed),
                seed_components=[int(seed), int(step), int(inner)], requested_size=int(sample_size))


def weighted_energy(values, membership, population_group_count, plan):
    """HT total and ratio-of-HT-totals; the ratio itself is not unbiased."""
    values = np.asarray(values, dtype=np.float64)
    membership = np.asarray(membership, dtype=bool)
    weights = plan['weight']
    total = float(weights @ values)
    selected = float(weights @ (values * membership))
    observed = int(membership.sum())
    available = population_group_count == 0 or observed > 0
    share = selected / total if available and total > 1e-30 else None
    variance = 0.
    if share is not None:
        residual = (membership.astype(float) - share) * values
        for label, stratum in enumerate(plan['strata']):
            size, population = stratum['sample_count'], stratum['population_count']
            if size == population:
                continue
            if size < 2:
                variance = None
                break
            variance += population ** 2 * (1 - size / population) * float(
                np.var(residual[plan['stratum'] == label], ddof=1)) / size
    error = float(np.sqrt(max(variance, 0.)) / total) if share is not None and variance is not None else None
    return dict(sum=selected if available else None,
                mean=selected / population_group_count if available and population_group_count else None,
                share=share, sample_count=observed,
                population_count=int(population_group_count),
                share_standard_error=error,
                share_ci95=[max(0., share - 1.96 * error), min(1., share + 1.96 * error)]
                    if error is not None else None,
                estimated=len(values) != plan['population_count'],
                interval_method='stratified finite-population delta method; approximate',
                missing_reason='no sampled tokens in nonempty group' if not available else None)


def add_sampled_report(report, fields, measurement, thresholds, *, grouping=None):
    """Keep full hidden fields intact and add weighted paired sample statistics."""
    from .opd_token_audit import prepare_groups
    target = list(zip(fields['row'].tolist(), fields['pos'].tolist()))
    source = list(zip(measurement['row'].tolist(), measurement['pos'].tolist()))
    if len(set(target)) != len(target) or len(set(source)) != len(source):
        raise ValueError('Duplicate paired sample token identifiers')
    lookup = {key: i for i, key in enumerate(target)}
    if any(key not in lookup for key in source):
        raise ValueError('Sampled parameter tokens must belong to the SAME hidden token population')
    indices = np.asarray([lookup[key] for key in source], dtype=np.int64)
    plan = measurement['sampling']
    if plan['population_count'] != len(target):
        raise ValueError('Parameter and hidden population sizes differ')
    weights = plan['weight']
    energy = measurement['energy']
    total = float(weights @ energy)
    sampled = len(source) != len(target)
    report.update(parameter_gradient_rms=float(np.sqrt(total / len(target))),
                  parameter_gradient_frobenius=float(np.sqrt(total)),
                  parameter_gradient_l2_mean=float(weights @ np.sqrt(energy) / len(target)))
    if grouping is None:
        grouping = prepare_groups(fields, thresholds)
    masks = grouping['masks']
    for label, chosen in grouping['kl_parts'].items():
        if label not in report.get('kl_high_middle_low', {}):
            continue
        mask = np.zeros(len(target), bool)
        mask[chosen] = True
        quantity = weighted_energy(energy, mask[indices], int(mask.sum()), plan)
        report['kl_high_middle_low'][label].update(
            parameter_gradient_energy=quantity['sum'],
            parameter_gradient_rms=float(np.sqrt(quantity['mean'])) if quantity['mean'] is not None else None,
            parameter_gradient_energy_share=quantity['share'],
            parameter_gradient_estimated=sampled,
            parameter_gradient_sample_count=quantity['sample_count'])
    paired = {}
    quantities = dict(parameter_gradient_energy=energy,
                      raw_gradient_energy=fields['raw_gradient_energy'][indices],
                      projected_gradient_energy=fields['projected_gradient_energy'][indices])
    for name, mask in masks.items():
        count = int(mask.sum())
        paired[name] = {key: weighted_energy(value, mask[indices], count, plan)
                        for key, value in quantities.items()}
        report['groups'][name]['quantities']['parameter_gradient_energy'] = paired[name]['parameter_gradient_energy']
    metadata = dict(mode='kl_stratified', sampled=sampled, exact=not sampled,
                    individual_token_gradients_exact=True, token_count=len(source),
                    population_token_count=len(target), coverage_fraction=len(source) / len(target),
                    requested_token_count=plan['requested_size'], strata=plan['strata'],
                    seed=plan['seed'], seed_components=plan['seed_components'],
                    sampling='uniform without replacement within each cached-KL stratum',
                    ranking='all valid tokens, pre-update cached conditional student/teacher KL; stable ties',
                    estimator='Horvitz-Thompson totals; ratios of estimated totals (not unbiased ratios)',
                    paired_hidden_population='identical sampled (response row, token position) with identical weights',
                    full_hidden_statistics_preserved=True,
                    extra_backwards=measurement['backwards'], seconds=measurement['seconds'],
                    causal_prefix=measurement.get('causal_prefix',False),
                    forward_calls=measurement.get('forward_calls'),
                    graph_group_size=measurement.get('graph_group_size',1),
                    graph_reuse_fallback=measurement.get('graph_reuse_fallback'),
                    forward_input_tokens=measurement.get('forward_input_tokens'),
                    full_suffix_forward_input_tokens=measurement.get('full_suffix_forward_input_tokens'),
                    diagnostic_optimizer_steps=0, extra_teacher_calls=0, parameter_vectors_saved=False,
                    definition='E_i = squared L2 norm of individual token loss gradient over all trainable parameters',
                    diagnostic_vs_training_kl_max_abs=float(np.max(np.abs(
                        measurement['diagnostic_kl'] - fields['kl'][indices]))),
                    parameter_energy_ranked_concentration_available=False,
                    warning='Population parameter energies are estimates; paired hidden estimates are separate from full hidden statistics')
    report['parameter_gradient_measurement'] = metadata
    report['paired_parameter_hidden_sample'] = dict(groups=paired, estimated=sampled,
        population_token_count=len(target), sample_count=len(source))
    arrays = {key: np.asarray(value)[indices] for key, value in fields.items()}
    arrays.update(parameter_gradient_energy=energy, diagnostic_kl=measurement['diagnostic_kl'],
                  sampling_weight=weights, sampling_stratum=plan['stratum'])
    return metadata, arrays
