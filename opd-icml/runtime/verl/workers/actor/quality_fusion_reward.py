"""Experimental frozen geometry + token-likelihood quality readout.

One frozen decoder replay is reused. The LM head is applied in chunks; no
online teacher, KL, second decoder replay, actor-hidden backward or baseline.
"""
import json
from pathlib import Path
import torch
from torch import nn
from .quality_readout_reward import unit,joint_features


def spectral_statistics(h):
    mean=h.mean(0);c=h-mean;n=max(len(h)-1,1)
    trace=c.square().sum()/n
    # Exact eigenvalues, in contrast with offline randomized SVD approximation.
    lam=torch.linalg.eigvalsh(c.T@c/n).flip(0)[:16].clamp_min(0)
    cutoff=torch.maximum(lam[0]*1e-7,lam.new_tensor(1e-10))
    lam=torch.where(lam>cutoff,lam,torch.zeros_like(lam))
    share=lam/trace.clamp_min(1e-9);prob=lam/lam.sum().clamp_min(1e-9)
    entropy=-(prob*prob.clamp_min(1e-9).log()).sum()
    erank=1/prob.square().sum().clamp_min(1e-9)
    return torch.cat((mean.norm().view(1),share,prob,entropy.view(1),erank.view(1)))


def likelihood_statistics(nll,numeric):
    q=torch.quantile(nll,torch.tensor([.5,.9,.99],device=nll.device))
    window=min(16,len(nll));rolling=nll.unfold(0,window,1).mean(-1).max()
    nv=nll[numeric];zero=nll.new_zeros(())
    return torch.stack((nll.mean(),nll.std(unbiased=False),*q.unbind(),nll.max(),
                        (nll>5).float().mean(),(nll>10).float().mean(),rolling,
                        nv.mean() if len(nv) else zero,nv.max() if len(nv) else zero,
                        numeric.float().mean(),nll.new_tensor(float(len(nll))).log1p()))


class FrozenQualityFusion(nn.Module):
    requires_output_head=True
    def __init__(self,manifest_path,*,minimum_tokens=1):
        super().__init__();p=Path(manifest_path);self.manifest=json.loads(p.read_text(encoding='utf-8'))
        if self.manifest['method']!='geometry_likelihood_quality_v8':raise ValueError('wrong quality fusion method')
        a=torch.load(p.parent/self.manifest['weights_file'],map_location='cpu',weights_only=True)
        self.hidden_size=int(a.pop('hidden_size'));self.minimum_tokens=max(1,int(minimum_tokens))
        for k,v in a.items():self.register_buffer(k,v)
        self.eval();self.requires_grad_(False)

    def validate_runtime(self,*,hidden_size,model_path):
        if hidden_size!=self.hidden_size:raise ValueError('quality fusion hidden dimension mismatch')
        if not Path(model_path).exists() and model_path!=self.manifest['model_id']:raise ValueError('wrong frozen backbone')

    def score_features(self,joint,small):
        margin,reward,_,_=self._score_details(joint,small)
        return margin,reward

    def _score_details(self,joint,small):
        pc=(((joint-self.pca_input_mean)/self.pca_input_scale)-self.pca_mean)@self.pca_components.T
        features=torch.cat((small,pc))
        x=(features-self.readout_mean)/self.readout_scale
        distance=(self.support_vectors-x).square().sum(-1)
        kernels=torch.exp(-self.gamma*distance)
        margin=(kernels*self.dual_coef).sum()+self.intercept
        return margin,torch.tanh(margin/self.reward_scale),features,kernels.max()

    @torch.no_grad()
    def forward(self,*,full_hidden,responses,response_mask,output_head):
        if full_hidden.ndim!=3 or full_hidden.shape[-1]!=self.hidden_size:raise ValueError('expected [B,S,D] hidden')
        if responses.shape!=response_mask.shape or responses.shape[0]!=full_hidden.shape[0]:raise ValueError('mask mismatch')
        n=responses.shape[1]
        if n==0 or full_hidden.shape[1]<=n:raise ValueError('missing prompt-end hidden')
        if output_head.shape[1]!=self.hidden_size:raise ValueError('LM head dimension mismatch')
        valid=response_mask.bool()&~torch.isin(responses,self.special_token_ids);count=valid.sum(-1)
        margins=[];means=[];numbers=[];features=[];similarities=[]
        nll_trace=full_hidden.new_zeros(responses.shape,dtype=torch.float32)
        with torch.autocast(device_type=full_hidden.device.type,enabled=False):
            for i in range(len(full_hidden)):
                zero=full_hidden.new_zeros((),dtype=torch.float32)
                if count[i]<self.minimum_tokens:
                    margins.append(zero);means.append(zero);numbers.append(zero)
                    features.append(torch.zeros_like(self.readout_mean));similarities.append(zero);continue
                mask=valid[i];h=full_hidden[i,-n:][mask].float();p=full_hidden[i,-n-1].float()
                # Each target uses its preceding state, never its own state.
                prev=full_hidden[i,-n-1:-1][mask];target=responses[i][mask]
                values=[]
                for j in range(0,len(target),32):
                    logits=(prev[j:j+32].to(output_head.dtype)@output_head.T).float()
                    values.append(logits.logsumexp(-1)-logits.gather(1,target[j:j+32,None])[:,0])
                nll=torch.cat(values);numeric=self.numeric_token[target]
                nll_trace[i,mask]=nll
                ls=likelihood_statistics(nll,numeric)
                small=torch.cat((ls,spectral_statistics(unit(h)),spectral_statistics(unit(h-p))))
                margin,_,feature,similarity=self._score_details(joint_features(h,p)[:-1],small)
                margins.append(margin);means.append(ls[0]);numbers.append(ls[9])
                features.append(feature);similarities.append(similarity)
        margin=torch.stack(margins);ok=count>=self.minimum_tokens
        reward=torch.where(ok,torch.tanh(margin/self.reward_scale),torch.zeros_like(margin));zero=torch.zeros_like(margin)
        return dict(reward=reward,raw_score=margin,ridge_contrast=zero,observed_contrast=zero,
                    support=zero,valid=ok.float(),valid_tokens=count.float(),
                    quality_readout_enabled=torch.ones_like(margin),quality_fusion_enabled=torch.ones_like(margin),
                    quality_token_nll=torch.stack(means),quality_numeric_nll=torch.stack(numbers),
                    quality_features=torch.stack(features),quality_rbf_max_similarity=torch.stack(similarities),
                    quality_token_nll_trace=nll_trace,quality_token_valid_mask=valid)
