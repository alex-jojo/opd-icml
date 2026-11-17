"""Recomputable gradient geometry. No optimizer or model mutations."""
import numpy as np
import torch


def gram_summary(gram):
    k=np.asarray(gram,dtype=np.float64)
    if k.ndim!=2 or k.shape[0]!=k.shape[1] or not np.isfinite(k).all():
        raise ValueError('invalid gradient Gram')
    n=len(k);norm2=np.maximum(np.diag(k),0);den=np.sqrt(norm2[:,None]*norm2[None,:])
    cosine=np.divide(k,den,out=np.full_like(k,np.nan),where=den>1e-30)
    cosine=np.clip(cosine,-1,1);values=cosine[np.triu_indices(n,1)];values=values[np.isfinite(values)]
    mean2=float(k.sum()/n**2)
    variance=float(max((np.trace(k)-k.sum()/n)/(n-1),0)) if n>1 else None
    signal=mean2-variance/n if variance is not None else None
    return dict(gram=k.tolist(),norms=np.sqrt(norm2).tolist(),
        cosine=[[None if not np.isfinite(x) else float(x) for x in row] for row in cosine],
        valid_pairs=len(values),mean_cosine=float(values.mean()) if len(values) else None,
        negative_fraction=float((values<0).mean()) if len(values) else None,
        quantiles=np.quantile(values,[0,.1,.25,.5,.75,.9,1]).tolist() if len(values) else [],
        mean_gradient_energy=mean2,variance=variance,corrected_signal_energy=signal,
        noise_signal_ratio=variance/mean2 if variance is not None and mean2>1e-30 else None,
        corrected_noise_signal_ratio=variance/signal if signal is not None and signal>1e-30 else None,
        signal_resolved=bool(signal is not None and signal>1e-30),
        definition='variance / mean-gradient energy at the recorded batch scale; not batch-size-invariant GNS')


def scope_geometry(second,mean,basis):
    c=torch.as_tensor(second,dtype=torch.float64);mu=torch.as_tensor(mean,dtype=torch.float64)
    u=torch.linalg.qr(torch.as_tensor(basis,dtype=torch.float64),mode='reduced').Q
    cu=c@u;inside=u.T@cu;outside=c-cu@u.T-u@cu.T+u@inside@u.T
    result={}
    for name,m,v,d in [('full',c,mu,len(c)),('u',inside,u.T@mu,u.shape[1]),
                       ('outside',outside,mu-u@(u.T@mu),len(c)-u.shape[1])]:
        entry=dict(mean=v.numpy(),dimension=d,mean_energy=float(v@v),energy=float(torch.trace(m)))
        entry['coherence']=entry['mean_energy']/entry['energy'] if entry['energy']>1e-30 else None
        for label,matrix in [('uncentered',m),('centered',m-v[:,None]*v[None,:])]:
            e,q=torch.linalg.eigh((matrix+matrix.T)*.5);e=e.flip(0)[:d].clamp_min(0);q=q.flip(1)[:,:min(128,d)]
            entry[label]=dict(eigenvalues=e.numpy(),vectors=q.numpy())
        result[name]=entry
    return result


def compare_hidden(items):
    result={}
    for scope in ('full','u','outside'):
        means=np.stack([x[scope]['mean'] for x in items]);out=gram_summary(means@means.T)
        out['coherence']=[x[scope]['coherence'] for x in items];out['overlap']={}
        d=items[0][scope]['dimension']
        for label in ('uncentered','centered'):
            for k in (1,4,8,16,32,64,128):
                if k>d:continue
                matrix=np.full((len(items),len(items)),np.nan)
                for i,a in enumerate(items):
                    for j,b in enumerate(items):
                        # Do not manufacture directions from a zero/rank-deficient moment.
                        ea=a[scope][label]['eigenvalues'];eb=b[scope][label]['eigenvalues']
                        if ea[k-1]<=max(ea[0]*1e-9,1e-30) or eb[k-1]<=max(eb[0]*1e-9,1e-30):continue
                        qa=a[scope][label]['vectors'][:,:k];qb=b[scope][label]['vectors'][:,:k]
                        matrix[i,j]=np.square(qa.T@qb).sum()/k
                clean=lambda m:[[None if not np.isfinite(v) else float(v) for v in r] for r in m]
                out['overlap'][f'{label}_k{k}']=dict(raw=clean(matrix),random_baseline=k/d,
                    adjusted=clean((matrix-k/d)/(1-k/d)) if k<d else None)
        result[scope]=out
    return result


def subset_alignment(train_rows, parameter, full_hidden):
    """Figure 7(b,c): the same unordered subset pairs for all three measurements."""
    matrices = dict(parameter_gradient_cosine=parameter['cosine'],
                    mean_hidden_gradient_cosine=full_hidden['cosine'],
                    centered_covariance_top4_overlap=full_hidden['overlap'].get('centered_k4', {}).get('raw'))
    pairs = []
    for a in range(len(train_rows)):
        for b in range(a+1, len(train_rows)):
            row = {key: matrix[a][b] if matrix is not None else None for key, matrix in matrices.items()}
            pairs.append(dict(subset_a=a, subset_b=b, **row))
    means, valid_pairs = {}, {}
    for key in matrices:
        values = [row[key] for row in pairs if row[key] is not None]
        valid_pairs[key] = len(values)
        means[key] = float(np.mean(values)) if values else None
    return dict(subset_train_rows=train_rows, pairs=pairs, means=means, valid_pairs=valid_pairs)


def transfer_report(before,after,validation_norm2,update_norm2,dot,epsilon=1e-6):
    b=np.asarray(before,float);a=np.asarray(after,float)
    if b.shape!=a.shape or not np.isfinite(b).all() or not np.isfinite(a).all():raise ValueError('invalid paired validation')
    gains=b-a;pred=-float(dot);den=np.sqrt(max(validation_norm2,0)*max(update_norm2,0))
    actual=float(gains.mean())
    return dict(kl_before=float(b.mean()),kl_after=float(a.mean()),gain=actual,
        predicted_gain=pred,prediction_residual=actual-pred,
        update_alignment=pred/den if den>1e-30 else None,
        validation_gradient_norm=np.sqrt(max(validation_norm2,0)),update_norm=np.sqrt(max(update_norm2,0)),
        gain_per_prompt=gains.tolist(),kl_before_per_prompt=b.tolist(),kl_after_per_prompt=a.tolist(),
        relative_gain_per_prompt=[float(g/x) if x>1e-12 else None for g,x in zip(gains,b)],
        positive_fraction=float((gains>epsilon).mean()),negative_fraction=float((gains<-epsilon).mean()),
        neutral_fraction=float((np.abs(gains)<=epsilon).mean()),threshold=epsilon,
        lower20_mean=float(np.sort(gains)[:max(1,int(np.ceil(.2*len(gains))))].mean()),
        gain_quantiles=np.quantile(gains,[0,.1,.25,.5,.75,.9,1]).tolist())


def nested_variance(within,between_sample,k):
    # Sample variance of K-rollout prompt means includes within-prompt noise / K.
    corrected=between_sample-within/k
    return dict(within_prompt=within,between_prompt_sample=between_sample,
        between_prompt_corrected=corrected,between_resolved=corrected>=0,
        rollouts_per_prompt=k,definition='equal-prompt, equal-rollout; corrected between = sample between - within/K')
