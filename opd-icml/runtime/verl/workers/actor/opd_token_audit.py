"""Read-only token heterogeneity statistics on CONDITIONAL cached support.

No score below changes the training loss. IER is the hypothetical one-sample
logit estimator reliability, NOT noise in the exact Top-K training gradient.
"""
import numpy as np
import torch
from .opd_focus_stats import concentration
from .opd_priority_stats import minmax, spectrum


def output_scores(student, teacher):
    with torch.no_grad():
        lp = student.detach().double().log_softmax(-1)
        lq = teacher.detach().double().log_softmax(-1)
        p, q = lp.exp(), lq.exp()
        ratio = lp-lq
        kl = (p*ratio).sum(-1)
        variance = (p*(ratio-kl[..., None]).square()).sum(-1)
        # p_a L(a) = 1-p_a; no reciprocal of tiny probabilities is needed.
        leverage = 1-p
        baseline = (leverage*ratio).sum(-1)/leverage.sum(-1)
        noise = ((ratio-baseline[..., None]).square()*leverage).sum(-1)-variance
        valid = (variance > 1e-15) & (noise > 1e-15)
        ier = torch.where(valid, variance/noise.clamp_min(1e-15), torch.zeros_like(variance))
        return dict(kl=kl.clamp_min(0), tv=.5*(p-q).abs().sum(-1),
                    entropy=-(p*lp).sum(-1), logratio_variance=variance,
                    ier=ier, ier_valid=valid, ier_noise=noise.clamp_min(0),
                    top1_disagree=p.argmax(-1)!=q.argmax(-1))


def enriched(fields):
    f = dict(fields)
    f['tip'] = minmax(f['entropy'])+minmax(f['kl'])-minmax(f['entropy'])*minmax(f['kl'])
    f['alignment'] = np.divide(f['projected_gradient_energy'], f['raw_gradient_energy'],
                              out=np.zeros(len(f['kl'])), where=f['raw_gradient_energy']>1e-30)
    f['priority']=f['tip']*f['alignment']
    return f


def top_mask(x, fraction):
    out = np.zeros(len(x), dtype=bool)
    out[np.argsort(-np.asarray(x), kind='stable')[:max(1, int(np.ceil(len(x)*fraction)))]] = True
    return out


def groups(f, thresholds):
    n = len(f['kl'])
    result = {'all': np.ones(n, bool), 'interior': ~f['boundary'].astype(bool)}
    for score in ('kl', 'tv', 'entropy', 'tip', 'logratio_variance', 'ier', 'alignment', 'priority'):
        valid = f['ier_valid'].astype(bool) if score == 'ier' else np.ones(n, bool)
        for fraction in (.01, .05, .1, .2):
            mask = np.zeros(n, bool)
            if valid.any(): mask[valid] = top_mask(f[score][valid], fraction)
            result[f'{score}_top{int(fraction*100)}'] = mask
    edges = (0, .01, .1, 1, np.inf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        result[f'kl_range_{lo}_{hi}'] = (f['kl']>=lo)&(f['kl']<hi)
    result['kl_above_step1_p90'] = f['kl']>=thresholds['kl_p90']
    he, hd = f['entropy']>=thresholds['entropy_median'], f['kl']>=thresholds['kl_median']
    for label, mask in [('uncertain_disagree', he&hd), ('uncertain_agree', he&~hd),
                        ('confident_disagree', ~he&hd), ('confident_agree', ~he&~hd)]:
        result['tip_quadrant_'+label] = mask
    for j in range(5): result[f'position_{j}'] = (f['relative_position']>=j/5)&(f['relative_position']<(j+1)/5 if j<4 else f['relative_position']<=1)
    for k in ('first', 'eos', 'numeric', 'operator', 'latex', 'other'):
        if k in f: result['class_'+k] = f[k].astype(bool)
    return result


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


def training_summary(f, z, thresholds):
    masks = groups(f, thresholds)
    matrix = z.T.astype(np.float64)@z/len(z)
    e, q, stats = spectrum(matrix)
    mean = z.mean(0, dtype=np.float64)
    report = dict(scope='conditional Top16; token loss gradients BEFORE batch mean/clipping',
        thresholds=thresholds, groups=summarize(f, masks, {k:f[k] for k in
            ('kl','tv','raw_gradient_energy','projected_gradient_energy','logratio_variance')}),
        spectrum=stats, nuclear_norm_token_normalized=float(np.sqrt(e).sum()),
        gradient_rms_full=float(np.sqrt(f['raw_gradient_energy'].mean())),
        gradient_rms_u=float(np.sqrt(f['projected_gradient_energy'].mean())),
        gradient_frobenius_full=float(np.sqrt(f['raw_gradient_energy'].sum())),
        gradient_frobenius_u=float(np.sqrt(f['projected_gradient_energy'].sum())),
        gradient_l2_mean_full=float(np.sqrt(f['raw_gradient_energy']).mean()),
        gradient_l2_mean_u=float(np.sqrt(f['projected_gradient_energy']).mean()),
        signed_mean_energy=float(mean@mean), signed_mean_fraction=float(mean@mean/max(e.sum(),1e-30)),
        concentration={k:concentration(f[k]) for k in ('kl','tv','entropy','tip','logratio_variance','ier',
            'alignment','priority','raw_gradient_energy','projected_gradient_energy')},
        ier_valid_fraction=float(f['ier_valid'].mean()), teachability_available=False,
        teachability_reason='Full-vocabulary teacher normalization / student support mass not recorded',
        notes='Hidden mean cancellation is NOT parameter-gradient cancellation; scores never select training tokens.')
    # A direct high/middle/low comparison, alongside equal-count subsampling.
    # Do not confuse more positions with more independent correction directions.
    order=np.argsort(f['kl'],kind='stable');k=max(1,int(np.ceil(len(order)*.1)))
    parts={'low':order[:k],'middle':order[k:-k],'high':order[-k:]}
    report['kl_high_middle_low']={}
    for label,ix in parts.items():
        if not len(ix):continue
        aligned=z[ix].astype(np.float64)
        e0,_,s0=spectrum(aligned.T@aligned/len(ix))
        take=ix[np.linspace(0,len(ix)-1,min(k,len(ix))).round().astype(int)]
        sample=z[take].astype(np.float64);e1,_,s1=spectrum(sample.T@sample/len(take))
        report['kl_high_middle_low'][label]=dict(count=len(ix),fraction=len(ix)/len(z),
            kl_min=float(f['kl'][ix].min()),kl_max=float(f['kl'][ix].max()),
            gradient_energy_u=float(np.square(aligned).sum()),spectrum=s0,
            gradient_rms_full=float(np.sqrt(f['raw_gradient_energy'][ix].mean())),
            gradient_rms_u=float(np.sqrt(f['projected_gradient_energy'][ix].mean())),
            nuclear_norm_token_normalized=float(np.sqrt(e0).sum()),
            matched_count=len(take),matched_spectrum=s1,
            matched_nuclear_norm=float(np.sqrt(e1).sum()))
    # Requested Need x U-retention and its weighted 128D geometry: READ ONLY.
    weights=f['priority'].astype(np.float64)
    available=float(weights.sum())>1e-30
    weighted=(z.T.astype(np.float64)@(z*weights[:,None]))/weights.sum() if available else np.zeros_like(matrix)
    we,wq,ws=spectrum(weighted)
    report['need_alignment_geometry']=dict(available=available,need='TIP entropy/KL soft-OR',
        alignment='squared norm retained in U / full hidden gradient squared norm',
        priority='need * alignment',spectrum=ws,used_for_training=False)
    return report, dict(gradient_second=matrix, eigenvalues=e, Q=q, gradient_mean=mean,
        need_alignment_second=weighted,need_alignment_eigenvalues=we,need_alignment_Q=wq)


def response_summary(before, after, thresholds):
    f = enriched(before)
    dh, du = after['h_u']-before['h_u'], after['unit_h_u']-before['unit_h_u']
    gain = before['kl']-after['kl']
    quantities = dict(raw_u_change=(dh*dh).sum(-1), unit_u_change=(du*du).sum(-1),
        raw_hidden_change=after['raw_delta_energy'], unit_hidden_change=after['unit_delta_energy'],
        kl_gain=gain, tv_gain=before['tv']-after['tv'],
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
