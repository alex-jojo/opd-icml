"""Read-only token heterogeneity statistics on CONDITIONAL cached support.

No score below changes the training loss. Only KL discrepancy, log-ratio variance and geometric retention
statistics are observed.
"""
import numpy as np
import torch
from .opd_focus_stats import concentration
from .opd_priority_stats import spectrum


def output_scores(student, teacher, *, kl_only=False):
    with torch.no_grad():
        lp = student.detach().double().log_softmax(-1)
        lq = teacher.detach().double().log_softmax(-1)
        p = lp.exp()
        ratio = lp-lq
        kl = (p*ratio).sum(-1)
        if kl_only:
            return {'kl': kl.clamp_min(0)}
        variance = (p*(ratio-kl[..., None]).square()).sum(-1)
        result = dict(kl=kl.clamp_min(0), logratio_variance=variance,
                      top1_disagree=lp.argmax(-1)!=lq.argmax(-1))
        return result


def enriched(fields):
    f = dict(fields)
    f['alignment'] = np.divide(f['projected_gradient_energy'], f['raw_gradient_energy'],
                              out=np.zeros(len(f['kl'])), where=f['raw_gradient_energy']>1e-30)
    return f


def top_mask(x, fraction):
    out = np.zeros(len(x), dtype=bool)
    out[np.argsort(-np.asarray(x), kind='stable')[:max(1, int(np.ceil(len(x)*fraction)))]] = True
    return out


def groups(f, thresholds):
    n = len(f['kl'])
    result = {'all': np.ones(n, bool), 'interior': ~f['boundary'].astype(bool)}
    for score in ('kl', 'logratio_variance', 'alignment'):
        valid = np.ones(n, bool)
        indices = np.flatnonzero(valid)
        order = indices[np.argsort(-np.asarray(f[score])[valid], kind='stable')] if len(indices) else indices
        for fraction in (.01, .05, .1, .2):
            mask = np.zeros(n, bool)
            mask[order[:max(1, int(np.ceil(len(order)*fraction)))]] = True
            result[f'{score}_top{int(fraction*100)}'] = mask
    edges = (0, .01, .1, 1, np.inf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        result[f'kl_range_{lo}_{hi}'] = (f['kl']>=lo)&(f['kl']<hi)
    result['kl_above_step1_p90'] = f['kl']>=thresholds['kl_p90']
    for j in range(5): result[f'position_{j}'] = (f['relative_position']>=j/5)&(f['relative_position']<(j+1)/5 if j<4 else f['relative_position']<=1)
    for k in ('first', 'eos', 'numeric', 'operator', 'latex', 'other'):
        if k in f: result['class_'+k] = f[k].astype(bool)
    return result


def prepare_groups(f, thresholds):
    """Update-local group membership shared by full and sampled reports.

    KL partitions keep their original ascending stable tie order. Top-percent
    groups use descending stable order; reversing that order would change ties.
    """
    order = np.argsort(f['kl'], kind='stable')
    count = max(1, int(np.ceil(len(order)*.1)))
    return dict(masks=groups(f, thresholds),
                kl_parts=dict(low=order[:count], middle=order[count:-count], high=order[-count:]))


def summarize(f, masks, quantities):
    """Both token and equal-response means. Empty groups remain missing."""
    out = {}
    rows,inverse = np.unique(f['row'],return_inverse=True)
    denominators=np.bincount(inverse,minlength=len(rows))
    prepared={k:(np.asarray(v,float),float(np.asarray(v,float).sum()),bool((np.asarray(v)>=0).all())) for k,v in quantities.items()}
    for name, mask in masks.items():
        count = int(mask.sum())
        row = dict(count=count, fraction=count/len(mask), quantities={})
        selected_rows=inverse[mask]
        for key, (values,total,nonnegative) in prepared.items():
            selected=values[mask];part=float(selected.sum())
            # Includes zero contribution from responses with no selected token.
            equal=np.bincount(selected_rows,weights=selected,minlength=len(rows))/np.maximum(denominators,1)
            row['quantities'][key] = dict(sum=part, mean=part/count if count else None,
                share=part/total if total>1e-30 and nonnegative else None,
                equal_response_contribution=float(np.mean(equal)),
                positive=float(selected.clip(0).sum()), negative=float((-selected).clip(0).sum()))
        out[name] = row
    return out


def training_summary(f, z, thresholds, *, grouping=None):
    if grouping is None:
        grouping = prepare_groups(f, thresholds)
    masks = grouping['masks']
    quantities=['kl','raw_gradient_energy','projected_gradient_energy','logratio_variance']
    concentration_fields=['kl','logratio_variance',
                          'alignment','raw_gradient_energy','projected_gradient_energy']
    if 'parameter_gradient_energy' in f:
        quantities.append('parameter_gradient_energy')
        concentration_fields.append('parameter_gradient_energy')
    matrix = z.T.astype(np.float64)@z/len(z)
    e, q, stats = spectrum(matrix)
    mean = z.mean(0, dtype=np.float64)
    report = dict(scope='conditional Top16; token loss gradients BEFORE batch mean/clipping',
        thresholds=thresholds, groups=summarize(f, masks, {k:f[k] for k in quantities}),
        spectrum=stats, nuclear_norm_token_normalized=float(np.sqrt(e).sum()),
        gradient_rms_full=float(np.sqrt(f['raw_gradient_energy'].mean())),
        gradient_rms_u=float(np.sqrt(f['projected_gradient_energy'].mean())),
        gradient_frobenius_full=float(np.sqrt(f['raw_gradient_energy'].sum())),
        gradient_frobenius_u=float(np.sqrt(f['projected_gradient_energy'].sum())),
        gradient_l2_mean_full=float(np.sqrt(f['raw_gradient_energy']).mean()),
        gradient_l2_mean_u=float(np.sqrt(f['projected_gradient_energy']).mean()),
        signed_mean_energy=float(mean@mean), signed_mean_fraction=float(mean@mean/max(e.sum(),1e-30)),
        concentration={k:concentration(f[k]) for k in concentration_fields},
        teachability_available=False,
        teachability_reason='Full-vocabulary teacher normalization / student support mass not recorded',
        notes='Hidden mean cancellation is NOT parameter-gradient cancellation; scores never select training tokens.')
    if 'parameter_gradient_energy' in f:
        energy=np.asarray(f['parameter_gradient_energy'],dtype=np.float64)
        report.update(parameter_gradient_rms=float(np.sqrt(energy.mean())),
                      parameter_gradient_frobenius=float(np.sqrt(energy.sum())),
                      parameter_gradient_l2_mean=float(np.sqrt(energy).mean()))
    # A direct high/middle/low comparison, alongside equal-count subsampling.
    # Do not confuse more positions with more independent correction directions.
    k=max(1,int(np.ceil(len(f['kl'])*.1)))
    parts=grouping['kl_parts']
    report['kl_high_middle_low']={}
    for label,ix in parts.items():
        if not len(ix):continue
        aligned=z[ix].astype(np.float64)
        e0,_,s0=spectrum(aligned.T@aligned/len(ix))
        take=ix[np.linspace(0,len(ix)-1,min(k,len(ix))).round().astype(int)]
        if np.array_equal(take,ix):
            # Same rows, ordering and normalization: reuse the exact spectrum.
            e1,s1=e0,s0.copy()
        else:
            sample=z[take].astype(np.float64);e1,_,s1=spectrum(sample.T@sample/len(take))
        report['kl_high_middle_low'][label]=dict(count=len(ix),fraction=len(ix)/len(z),
            kl_min=float(f['kl'][ix].min()),kl_max=float(f['kl'][ix].max()),
            gradient_energy_u=float(np.square(aligned).sum()),spectrum=s0,
            gradient_rms_full=float(np.sqrt(f['raw_gradient_energy'][ix].mean())),
            gradient_rms_u=float(np.sqrt(f['projected_gradient_energy'][ix].mean())),
            nuclear_norm_token_normalized=float(np.sqrt(e0).sum()),
            matched_count=len(take),matched_spectrum=s1,
            matched_nuclear_norm=float(np.sqrt(e1).sum()))
        if 'parameter_gradient_energy' in f:
            energy=np.asarray(f['parameter_gradient_energy'][ix],dtype=np.float64)
            report['kl_high_middle_low'][label].update(
                parameter_gradient_energy=float(energy.sum()),
                parameter_gradient_rms=float(np.sqrt(energy.mean())),
                parameter_gradient_energy_share=float(energy.sum()/f['parameter_gradient_energy'].sum())
                    if f['parameter_gradient_energy'].sum()>1e-30 else None)
    return report, dict(gradient_second=matrix, eigenvalues=e, Q=q, gradient_mean=mean)



def response_summary(before, after, thresholds):
    f = enriched(before)
    dh, du = after['h_u']-before['h_u'], after['unit_h_u']-before['unit_h_u']
    gain = before['kl']-after['kl']
    quantities = dict(raw_u_change=(dh*dh).sum(-1), unit_u_change=(du*du).sum(-1),
        raw_hidden_change=after['raw_delta_energy'], unit_hidden_change=after['unit_delta_energy'],
        kl_gain=gain,
        signed_work_u=-(before['g_u']*dh).sum(-1),
        remaining_gradient_u=after['projected_gradient_energy'])
    lp=torch.from_numpy(before['z']).double().log_softmax(-1)
    la=torch.from_numpy(after['z']).double().log_softmax(-1)
    policy=(lp.exp()*(lp-la)).sum(-1).numpy()
    answer=[dict(row=int(r),kl_gain=float(gain[f['row']==r].mean()),
        policy_kl=float(policy[f['row']==r].mean())) for r in np.unique(f['row'])]
    return dict(groups=summarize(f, groups(f, thresholds), quantities),
        mean_kl_before=float(before['kl'].mean()), mean_kl_after=float(after['kl'].mean()),
        per_response=answer,mean_policy_kl=float(policy.mean()),
        response_coverage=float(np.mean([r['kl_gain']>1e-4 for r in answer])),
        response_negative_transfer=float(np.mean([r['kl_gain']< -1e-4 for r in answer])),
        improved_fraction=float((gain>1e-4).mean()), worsened_fraction=float((gain< -1e-4).mean()),
        note='Receiver locations and signed local work are associations, NOT additive causal attribution.')
