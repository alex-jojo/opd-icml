"""Read-only token priority and spectral sharpening diagnostics. No optimizer.

TIP Eq.7 Soft-OR, adapted to conditional cached Top-16 distributions; min-max
normalization is over ALL valid training tokens in the optimizer batch.
Alignment is geometric retention, not demonstrated teachability.
"""
import numpy as np
from .opd_focus_stats import concentration, allocation, distribution_pair


def minmax(x):
    x = np.asarray(x, dtype=np.float64)
    spread = float(np.ptp(x))
    # Constant input has no ranking information. Do not amplify roundoff.
    return (x-x.min())/spread if spread > 1e-12 else np.zeros_like(x)


def scores(entropy, kl, raw_energy, projected_energy):
    h = minmax(np.maximum(entropy, 0))
    divergence = minmax(np.maximum(kl, 0))
    need = h + divergence - h*divergence
    raw = np.asarray(raw_energy, dtype=np.float64)
    energy = np.asarray(projected_energy, dtype=np.float64)
    valid = raw > 1e-30
    alignment = np.divide(energy, raw, out=np.zeros_like(raw), where=valid)
    if np.any(alignment > 1.005) or not np.isfinite(alignment).all():
        raise ValueError('invalid projection retention: verify raw gradient and orthonormal U')
    return dict(need=need, alignment=alignment.clip(0, 1),
                priority=need*alignment.clip(0, 1), alignment_valid=valid,
                entropy_minmax=h, divergence_minmax=divergence)


def normalized_weights(score):
    score = np.asarray(score, dtype=np.float64)
    fallback = float(score.sum()) <= 1e-30
    return (np.ones_like(score) if fallback else score/score.mean()), fallback


def spectrum(matrix):
    eigen, q = np.linalg.eigh((matrix+matrix.T)*.5)
    eigen = eigen[::-1].clip(0); q = q[:, ::-1]
    total = float(eigen.sum())
    stats = concentration(eigen)
    stats['top_modes'] = {str(k): float(eigen[:k].sum()/total) if total > 1e-30 else None
                          for k in (1, 4, 8, 16, 32, 64, 128) if k <= len(eigen)}
    stats['effective_rank'] = (stats['entropy_effective_fraction']*len(eigen)
                               if stats['available'] else None)
    stats['stable_rank']=float(total/eigen[0]) if total>1e-30 else None
    stats['participation_rank']=float(total**2/(eigen@eigen)) if total>1e-30 else None
    singular=np.sqrt(eigen)
    ps=singular/singular.sum() if singular.sum()>1e-30 else singular
    stats['singular_entropy_rank']=float(np.exp(-(ps[ps>0]*np.log(ps[ps>0])).sum())) if total>1e-30 else None
    stats['effective_rank_definition']='entropy rank of energy eigenvalues; singular_entropy_rank uses singular values'
    stats['relative_eigengaps'] = {str(k):float((eigen[k-1]-eigen[k])/max(float(eigen[0]),1e-30))
                                  for k in (4,16,32) if k<len(eigen)}
    return eigen, q, stats


def analyze_priority(z, fields):
    """One transient N x r raw-gradient projection; only small matrices saved.

    All hypothetical signals are detached numpy arrays. In particular they
    never replace a training gradient. Norm controls are hidden-signal controls,
    not claims of equal parameter movement or equal effective Adam learning rate.
    """
    z = np.asarray(z, dtype=np.float32)
    n, rank = z.shape
    energy = np.einsum('ij,ij->i', z, z).astype(np.float64)
    s = scores(fields['student_entropy'], fields['kl'], fields['raw_gradient_energy'], energy)
    fields.update(s)
    fields['raw_projected_energy'] = energy
    w, fallback = normalized_weights(s['priority'])
    fields['priority_weight'] = w
    result = dict(schema='priority-spectrum-v1', training_changed=False,
        scope='conditional Top-16; batch min-max Soft-OR; raw gradient projected once using FP32 asset U',
        zero_score_uniform_fallback=fallback, zero_gradient_count=int((~s['alignment_valid']).sum()),
        score_concentration={k: concentration(s[k]) for k in ('need', 'alignment', 'priority')},
        weight_concentration=concentration(w), weight_mean=float(w.mean()),
        caveat='Alignment is not absorption; mean weight 1 and equal hidden norm do not fix the Adam step.',
        weighted_geometry={}, sharpening={})
    arrays = {}
    moments = {}
    for name, score in [('uniform', np.ones(n)), ('joint', s['priority'])]:
        weight, empty = normalized_weights(score)
        matrix = z.T @ (z*weight.astype(np.float32)[:, None])/n
        matrix = matrix.astype(np.float64)
        eigen, q, stats = spectrum(matrix)
        mean = (z*weight[:, None]).mean(0)
        stats.update(shared_mean_energy_fraction=float(mean@mean/max(float(np.trace(matrix)), 1e-30)),
                     uniform_fallback=empty)
        result['weighted_geometry'][name] = stats
        arrays[name+'_M'] = matrix; arrays[name+'_eigenvalues'] = eigen
        arrays[name+'_Q'] = q
        moments[name] = (matrix, eigen, q)
    # These modes describe UNcentered supervision energy, not Fisher/curvature.
    matrix, eigen, q = moments['joint']
    coef = z @ q.astype(np.float32)
    coef2 = np.square(coef)
    mean = (coef*w[:, None]).mean(0)
    coherence = np.divide(mean**2, eigen, out=np.zeros_like(eigen), where=eigen > 1e-30)
    result['joint_mode_signed_coherence'] = coherence.clip(0, 1).tolist()
    result['coherence_scope']='Common signed hidden coordinates only; NOT parameter-gradient agreement or conflict.'
    centered = matrix - np.outer((z*w[:, None]).mean(0), (z*w[:, None]).mean(0))
    result['centered_joint_spectrum'] = spectrum(centered)[2]
    boundary = fields['boundary'].astype(bool)
    # Attach need to JOINT FEATURE MODES, not a sweep of token selectors.
    # An energetic mode can be dominated by already-aligned or boundary states.
    mode_mass=coef2.sum(0,dtype=np.float64)
    result['mode_need_mean']=np.divide(s['need']@coef2,mode_mass,
        out=np.zeros(rank),where=mode_mass>1e-30).tolist()
    result['mode_alignment_mean']=np.divide(s['alignment']@coef2,mode_mass,
        out=np.zeros(rank),where=mode_mass>1e-30).tolist()
    result['mode_boundary_energy_share']=np.divide(coef2[boundary].sum(0,dtype=np.float64),mode_mass,
        out=np.zeros(rank),where=mode_mass>1e-30).tolist()
    result['need_vs_retained_energy'] = distribution_pair(s['need'], energy)
    result['need_to_alignment'] = allocation(s['need'], s['alignment'])
    result['priority_to_projected_energy'] = allocation(s['priority'], energy)
    # A large fraction can coexist with essentially no absolute correction.
    positive = energy[energy > 1e-30]
    small = np.median(positive)*.01 if len(positive) else 0.
    result['high_alignment_tiny_signal'] = dict(energy_threshold=float(small),
        fraction=float(((s['alignment'] >= .8) & (energy <= small)).mean()))
    result['boundary_shares'] = {k: float(v[boundary].sum()/max(float(v.sum()),1e-30))
        for k,v in [('need',s['need']),('priority',s['priority']),('weighted_signal_energy',w*w*energy)]}
    result['interior_priority'] = concentration(s['priority'][~boundary])
    row = fields['row']; rows = np.unique(row)
    sequence_mode_energy=np.stack([(coef2[row==r]*w[row==r,None]).sum(0) for r in rows])
    mode_total=sequence_mode_energy.sum(0)
    mode_squares=np.square(sequence_mode_energy).sum(0)
    result['mode_effective_sequences']=np.divide(mode_total**2,mode_squares,
        out=np.zeros(rank),where=mode_squares>1e-30).tolist()
    result['sequence_priority_mass'] = concentration([s['priority'][row==r].sum() for r in rows])
    result['sequence_priority_mean'] = concentration([s['priority'][row==r].mean() for r in rows])
    result['position_priority_shares'] = [float(s['priority'][(fields['relative_position']>=j/5)&
        (fields['relative_position']<((j+1)/5 if j<4 else 1.01))].sum()/max(float(s['priority'].sum()),1e-30)) for j in range(5)]
    # Use one eigen-coordinate multiplication for all gamma values. No repeated
    # Nx128 transforms, no model forward, and no per-token dense matrix storage.
    base_total = float(energy.sum())
    weighted_mode_mass=(w*w)@coef2
    for gamma in (0., .25, .5, 1.):
        gains = np.ones(rank) if gamma == 0 else (eigen/max(float(eigen[0]),1e-30))**gamma
        candidate_energy = (coef2 @ (gains*gains))*(w*w)
        total = float(candidate_energy.sum())
        calibration = (base_total/total)**.5 if total > 1e-30 else 0.
        cross = float(np.sum(coef2 @ gains*w))*calibration
        calibrated = candidate_energy*calibration**2
        per_dot = (coef2 @ gains)*w*calibration
        cosine = np.divide(per_dot, np.sqrt(energy*calibrated), out=np.zeros(n), where=energy*calibrated > 1e-30)
        mode_energy = weighted_mode_mass*(gains*calibration)**2
        result['sharpening'][str(gamma)] = dict(
            raw_signal_norm_ratio=(total/max(base_total,1e-30))**.5,
            norm_calibration=calibration, calibrated_norm_ratio=(calibrated.sum()/max(base_total,1e-30))**.5,
            field_cosine_to_ordinary=cross/max(base_total,1e-30) if base_total > 1e-30 else None,
            token_cosine_quantiles=np.quantile(cosine[energy>1e-30],[0,.1,.5,.9,1]).tolist() if base_total>1e-30 else None,
            token_energy_concentration=concentration(calibrated), mode_energy_concentration=concentration(mode_energy),
            boundary_energy_share=float(calibrated[boundary].sum()/max(float(calibrated.sum()),1e-30)))
    # Cross-trajectory replication: do dominant joint modes also appear in an
    # independent half of this batch? Descriptive, not a downstream guarantee.
    halves = []
    for parity in (0, 1):
        keep = row % 2 == parity
        if not keep.any() or w[keep].sum() <= 1e-30: break
        m = z[keep].T @ (z[keep]*w[keep,None].astype(np.float32))/w[keep].sum()
        if np.trace(m)<=1e-30: break
        halves.append(spectrum(m)[1])
    result['split_sequence_mode_overlap'] = ({str(k):float(np.square(halves[0][:,:k].T@halves[1][:,:k]).sum()/k)
        for k in (4,16,32) if k <= rank} if len(halves)==2 else None)
    arrays['joint_mode_signed_coherence'] = coherence
    return result, arrays
