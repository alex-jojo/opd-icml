"""Bounded token diagnostics; RQ2's all-token Gram is collected separately.

The hidden sample extends the unchanged parameter sample within each KL stratum.
Only current KL is read over the population, to retain exact KL group boundaries.
All other population summaries use inverse inclusion weights and are estimates.
"""
import numpy as np
import torch
from . import opd_distributed as dp
from .opd_parameter_sampling import stratified_plan, weighted_energy, add_sampled_report
from .opd_token_audit import output_scores, enriched, prepare_groups, top_mask
from .opd_priority_stats import spectrum
from .opd_gradient_scopes import summarize as spectral_summary


def extend_plan(kl, base, size, step, inner):
    target = stratified_plan(kl, size, base['seed'], step, inner)
    if any(a['sample_count'] > b['sample_count'] for a, b in zip(base['strata'], target['strata'])):
        raise ValueError('Token statistics sample must contain the parameter sample in each stratum')
    high = top_mask(kl, .1)
    rng = np.random.default_rng(np.random.SeedSequence([base['seed'], step, inner, 128]))
    chosen, weights, labels = [], [], []
    for label, mask in enumerate((high, ~high)):
        old = base['index'][base['stratum'] == label]
        pool = np.setdiff1d(np.flatnonzero(mask), old, assume_unique=True)
        count = target['strata'][label]['sample_count']
        extra = rng.choice(pool, count-len(old), replace=False)
        chosen.extend(np.concatenate((old, extra)).tolist())
        labels.extend([label]*count)
        weights.extend([int(mask.sum())/count]*count if count else [])
    order = np.argsort(chosen)
    target.update(index=np.asarray(chosen, dtype=np.int64)[order],
                  weight=np.asarray(weights)[order], stratum=np.asarray(labels)[order],
                  seed_components=[base['seed'], step, inner, 128])
    return target


class TokenSample:
    def __init__(self, data, measurement, parameter_cfg, size, step, inner):
        mask = data.batch['response_mask'].bool().cpu().numpy()
        self.offsets = np.r_[0, np.cumsum(mask.sum(1))]
        self.response_counts = mask.sum(1)
        def choose():
            cached = data.batch['opd_student_top16_logits'].detach().cpu()
            teacher = data.batch['opd_teacher_top16_logits'].detach().cpu()
            # Mask before FP64 conversion: padded positions do not need scoring.
            kl = output_scores(cached[mask], teacher[mask], kl_only=True)['kl'].numpy()
            base = measurement['sampling'] if measurement is not None else stratified_plan(
                kl, parameter_cfg.get('sample_size', 64), parameter_cfg.get('seed', 42), step, inner)
            return extend_plan(kl, base, size, step, inner)
        self.plan = dp.primary_call(choose)

    def select(self, mask, cursor):
        local = mask.detach().cpu().numpy()
        row, pos = np.nonzero(local)
        ordinal = local.cumsum(1)[row, pos]-1
        global_index = self.offsets[row+cursor]+ordinal
        selected = np.flatnonzero(np.isin(global_index, self.plan['index']))
        return global_index, selected

    def merge(self, parts):
        population = {key: np.concatenate([p['_population_'+key] for p in parts])
                      for key in ('index', 'kl')}
        fields = {key: np.concatenate([p[key] for p in parts])
                  for key in parts[0] if not key.startswith('_')}
        population = dp.merge_fields(dp.gather(population))
        fields = dp.merge_fields(dp.gather(fields))
        def order():
            order = np.argsort(population['index'])
            if not np.array_equal(population['index'][order], np.arange(self.plan['population_count'])):
                raise ValueError('Missing or duplicated population KL positions')
            order_sample = np.argsort(fields['index'])
            result = {k: v[order_sample] for k, v in fields.items()}
            if not np.array_equal(result['index'], self.plan['index']):
                raise ValueError('Missing or duplicated sampled hidden gradients')
            result['response_token_count'] = self.response_counts[result['row']]
            return result, population['kl'][order]
        return dp.primary_call(order)


def weighted_top(values, weights, fraction, valid=None):
    valid = np.ones(len(values), bool) if valid is None else valid
    ids = np.flatnonzero(valid)
    ids = ids[np.argsort(-values[ids], kind='stable')]
    mask = np.zeros(len(values), bool)
    if len(ids):
        count = np.searchsorted(np.cumsum(weights[ids]), fraction*weights[ids].sum(), side='left')+1
        mask[ids[:count]] = True
    return mask


def grouping_for(fields, population_kl, thresholds, plan, detailed):
    ix, w = plan['index'], plan['weight']
    if detailed:
        grouping = prepare_groups(fields, thresholds)
        masks = grouping['masks']
        for score in ('tv', 'entropy', 'tip', 'logratio_variance', 'ier', 'alignment', 'priority'):
            for fraction in (.01, .05, .1, .2):
                masks[f'{score}_top{int(fraction*100)}'] = weighted_top(
                    fields[score], w, fraction, fields['ier_valid'].astype(bool) if score == 'ier' else None)
    else:
        grouping = dict(masks={}, kl_parts={})
        masks = grouping['masks']
    exact = {'kl_above_step1_p90': population_kl >= thresholds['kl_p90']}
    for fraction in ((.01, .05, .1, .2) if detailed else (.1,)):
        exact[f'kl_top{int(fraction*100)}'] = top_mask(population_kl, fraction)
    if detailed:
        for lo, hi in zip((0, .01, .1, 1), (.01, .1, 1, np.inf)):
            exact[f'kl_range_{lo}_{hi}'] = (population_kl >= lo) & (population_kl < hi)
        exact['all'] = np.ones(len(population_kl), bool)
        from .opd_group_scopes import partitions
        grouping['kl_parts'] = {}
        grouping['kl_ranges'] = {}
        for label, ids in partitions(population_kl).items():
            member = np.zeros(len(population_kl), bool)
            member[ids] = True
            grouping['kl_parts'][label] = np.flatnonzero(member[ix])
            grouping['kl_ranges'][label] = (float(population_kl[ids].min()), float(population_kl[ids].max()))
            exact['partition_'+label] = member
    counts = {key: float(w @ mask) for key, mask in masks.items()}
    for key, mask in exact.items():
        counts[key] = int(mask.sum())
        if not key.startswith('partition_'):
            masks[key] = mask[ix]
    grouping.update(population_counts=counts, exact_count_groups=list(exact))
    return grouping


def concentration(values, weights):
    """Plug-in population distribution, with inverse inclusion masses."""
    x = np.maximum(np.asarray(values, float), 0)
    w = np.asarray(weights, float)
    n, total = w.sum(), float(w @ x)
    result = dict(count=float(n), sample_count=len(x), total=total, estimated=True, available=total>1e-30)
    if total <= 1e-30:
        return {**result, 'gini': None, 'effective_fraction': None,
                'entropy_effective_fraction': None, 'relative_hhi': None, 'top_shares': {}}
    order = np.argsort(x, kind='stable')
    a, b = x[order], w[order]
    p = x/total
    hhi = float(w @ (p*p))
    positive = p > 0
    entropy = float(-(w[positive]*p[positive]*np.log(p[positive])).sum())
    shares = {}
    cum = np.r_[0, np.cumsum(b[::-1])]
    for f in (.01, .05, .1, .2):
        mass = np.maximum(0, np.minimum(b[::-1], f*n-cum[:-1]))
        shares[str(f)] = float(mass @ a[::-1]/total)
    result.update(gini=float(np.sum(b*a*(2*np.cumsum(b)-b-n))/(n*total)),
                  effective_fraction=1/(n*hhi), entropy_effective_fraction=float(np.exp(entropy)/n),
                  relative_hhi=n*hhi, top_shares=shares)
    return result


def sampled_scopes(g, u, weights, population_count):
    """Thin row Gram, at most 128 squared, never a grouped hidden-dimension Gram."""
    g = np.asarray(g, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    z = g @ u
    stats, arrays = {}, {}
    for name, values, dimension in (('full', g, g.shape[1]), ('u', z, u.shape[1]),
                                    ('outside', g-z@u.T, g.shape[1]-u.shape[1])):
        if not len(g):
            stats[name] = dict(available=False, sample_count=0)
            arrays[name+'_eigenvalues'] = np.empty(0)
            continue
        matrix = torch.from_numpy(values*np.sqrt(weights/population_count)[:, None])
        e = torch.linalg.eigvalsh(matrix @ matrix.T).flip(0).clamp_min(0).numpy()[:dimension]
        # Zero-padding states the hidden dimension, not extra observed directions.
        e = np.pad(e, (0, max(0, dimension-len(e))))
        stats[name] = spectral_summary(e, population_count)
        stats[name].update(available=True, estimated=True, sample_count=len(g),
                           sample_rank_limit=min(len(g), dimension))
        arrays[name+'_eigenvalues'] = e
    return stats, arrays


def projected_summary(e):
    """Preserve the projected-spectrum schema without repeating an eigensolve."""
    from .opd_focus_stats import concentration as spectral_concentration
    stats = spectral_concentration(e)
    summary = spectral_summary(e, 1)
    total = float(e.sum())
    stats.update({key: summary[key] for key in ('effective_rank','stable_rank','participation_rank','singular_entropy_rank')})
    stats['top_modes'] = {str(k): float(e[:k].sum()/total) if total>1e-30 else None
                          for k in (1,4,8,16,32,64,128) if k<=len(e)}
    stats['relative_eigengaps'] = {str(k): float((e[k-1]-e[k])/max(float(e[0]),1e-30))
                                  for k in (4,16,32) if k<len(e)}
    stats['effective_rank_definition'] = 'entropy rank of energy eigenvalues; singular_entropy_rank uses singular values'
    stats.update(estimated=True, sample_limited=True)
    return stats


def make_report(fields, population_kl, thresholds, plan, basis, measurement, detailed, response_count):
    w, n = plan['weight'], plan['population_count']
    grouping = grouping_for(fields, population_kl, thresholds, plan, detailed)
    quantities = ('kl', 'tv', 'raw_gradient_energy', 'projected_gradient_energy', 'logratio_variance') if detailed else ('raw_gradient_energy',)
    groups = {}
    for name, mask in grouping['masks'].items():
        count = grouping['population_counts'][name]
        values = {key: weighted_energy(fields[key], mask, count, plan) for key in quantities}
        # Sample-selected boundaries do not have design-based CI coverage.
        for value in values.values():
            value['population_count'] = count
            if name not in grouping['exact_count_groups']:
                value.update(share_standard_error=None, share_ci95=None,
                             interval_method='unavailable for sample-defined groups')
        for key, value in values.items():
            x = np.asarray(fields[key], dtype=float)
            value.update(equal_response_contribution=float(np.sum(w[mask]*x[mask]/fields['response_token_count'][mask])/response_count),
                         positive=float(w[mask] @ np.maximum(x[mask], 0)),
                         negative=float(w[mask] @ np.maximum(-x[mask], 0)))
        groups[name] = dict(count=count, fraction=count/n, sample_count=int(mask.sum()),
                            count_estimated=name not in grouping['exact_count_groups'], quantities=values)
    metadata = dict(sample_count=len(w), population_token_count=n, requested_token_count=plan['requested_size'],
        sampling='nested uniform without replacement within cached-KL strata', strata=plan['strata'],
        seed_components=plan['seed_components'], estimated=len(w)!=n,
        estimator='inverse inclusion weighted totals and ratios; nonlinear summaries are plug-in estimates',
        current_kl_ranking='ALL valid tokens; exact group sizes',
        other_score_ranking='weighted sample quantiles; TIP min/max from sample',
        rq2_overall_hidden_spectrum='ALL valid tokens, collected separately without subsampling',
        grouped_spectrum='sampled weighted spectra; rank limited by sampled group size',
        extra_forwards=0, extra_backwards=0, extra_teacher_calls=0)
    total = float(w @ fields['raw_gradient_energy'])
    report = dict(groups=groups, thresholds=thresholds, token_statistics_measurement=metadata,
        scope='conditional Top16; sampled token loss gradients BEFORE batch mean/clipping',
        notes='Scores never select training tokens; full RQ2 moments are saved separately.',
        gradient_rms_full=float(np.sqrt(total/n)), gradient_frobenius_full=float(np.sqrt(total)),
        gradient_l2_mean_full=float(w @ np.sqrt(fields['raw_gradient_energy'])/n))
    arrays = dict(fields, sampling_weight=w, sampling_stratum=plan['stratum'],
        in_current_kl_top10=grouping['masks']['kl_top10'],
        above_initial_kl_threshold=grouping['masks']['kl_above_step1_p90'])
    group_report, group_arrays = None, {}
    if detailed:
        z = fields['g_u'].astype(np.float64)
        matrix = z.T @ (w[:, None]*z)/n
        e, q, stats = spectrum(matrix)
        mean = w @ z/n
        kept = float(w @ fields['projected_gradient_energy'])
        report.update(spectrum=stats, nuclear_norm_token_normalized=float(np.sqrt(e).sum()),
            gradient_rms_u=float(np.sqrt(kept/n)), gradient_frobenius_u=float(np.sqrt(kept)),
            gradient_l2_mean_u=float(w @ np.sqrt(fields['projected_gradient_energy'])/n),
            signed_mean_energy=float(mean@mean), signed_mean_fraction=float(mean@mean/max(e.sum(), 1e-30)),
            concentration={k: concentration(fields[k], w) for k in ('kl','tv','entropy','tip','logratio_variance',
                'ier','alignment','priority','raw_gradient_energy','projected_gradient_energy')},
            ier_valid_fraction=float(w @ fields['ier_valid']/n), teachability_available=False,
            teachability_reason='Full-vocabulary teacher normalization / student support mass not recorded',
            kl_high_middle_low={})
        arrays.update(gradient_second=matrix, eigenvalues=e, Q=q, gradient_mean=mean)
        entries = {}
        matched = min(map(len, grouping['kl_parts'].values()))
        for label, ix in grouping['kl_parts'].items():
            count = grouping['population_counts']['partition_'+label]
            full, spectra = sampled_scopes(fields['full_gradient'][ix], basis, w[ix], count)
            ordered = ix[np.argsort(fields['kl'][ix], kind='stable')]
            take = ordered[np.linspace(0, len(ix)-1, matched).round().astype(int)] if matched else ix[:0]
            equal, equal_spectra = sampled_scopes(fields['full_gradient'][take], basis, np.ones(len(take)), max(1, len(take)))
            lo, hi = grouping['kl_ranges'][label]
            entries[label] = dict(count=count, fraction=count/n, sample_count=len(ix), matched_count=len(take),
                                  kl_min=lo, kl_max=hi, all=full, matched=equal)
            group_arrays.update({f'{label}_all_{k}': v for k,v in spectra.items()})
            group_arrays.update({f'{label}_matched_{k}': v for k,v in equal_spectra.items()})
            su, sm = full['u'], equal['u']
            report['kl_high_middle_low'][label] = dict(entries[label],
                gradient_energy_u=float(w[ix] @ fields['projected_gradient_energy'][ix]) if len(ix) else None,
                gradient_rms_full=full['full'].get('rms'), gradient_rms_u=su.get('rms'),
                spectrum=projected_summary(spectra['u_eigenvalues']), nuclear_norm_token_normalized=su.get('nuclear_norm_token_normalized'),
                matched_spectrum=projected_summary(equal_spectra['u_eigenvalues']), matched_nuclear_norm=sm.get('nuclear_norm_token_normalized'))
        group_report = dict(exact_all_tokens=False, sampled=True, groups=entries,
            definition='Exact global KL low10/middle80/high10 membership; weighted sampled hidden spectra',
            matched_definition='Equal observed sample counts, evenly spaced sample indices in KL order; descriptive only',
            transformer_replays=0, sample_count=len(w))
        priority = w*fields['priority']
        weighted = z.T @ (priority[:, None]*z)/priority.sum() if priority.sum()>1e-30 else np.zeros_like(matrix)
        we, wq, ws = spectrum(weighted)
        report['need_alignment_geometry'] = dict(available=bool(priority.sum()>1e-30), spectrum=ws,
            need='TIP entropy/KL soft-OR using sample extrema', alignment='U retained energy fraction',
            priority='need * alignment', used_for_training=False, estimated=True)
        arrays.update(need_alignment_second=weighted, need_alignment_eigenvalues=we, need_alignment_Q=wq)
    parameter_metadata, paired = None, {}
    if measurement is not None:
        parameter_metadata, paired = add_sampled_report(report, fields, measurement, thresholds,
            grouping=grouping, observation_plan=plan)
    return report, arrays, group_report, group_arrays, parameter_metadata, paired
