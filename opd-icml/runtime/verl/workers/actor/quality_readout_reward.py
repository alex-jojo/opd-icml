"""Frozen, full-dimensional readout. Experimental, NOT a correctness oracle.

The continuous margin is not a probability or a per-rubric score. It is used
as a detached on-policy reward; no gradient moves actor hidden toward a
fabricated high-quality trajectory. All valid response tokens enter moments.
"""
import json
from pathlib import Path
import torch
from torch import nn


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def joint_features(h, prompt):
    h, prompt = h.float(), prompt.float()
    u, r = unit(h), unit(h-prompt)
    return torch.cat((u.mean(0), u.std(0, unbiased=False),
                      r.mean(0), r.std(0, unbiased=False),
                      unit(prompt), u[-1], u[0], u.mean(0),
                      h.new_tensor([len(h)]).log1p()))


class FrozenQualityReadout(nn.Module):
    def __init__(self, manifest_path, *, minimum_tokens=1):
        super().__init__()
        path=Path(manifest_path)
        self.manifest=json.loads(path.read_text(encoding='utf-8'))
        if self.manifest['method']!='full_hidden_quality_readout_v5':
            raise ValueError('unsupported quality readout asset')
        # Tensor-only artifact; no NumPy pickle or arbitrary class deserialization.
        a=torch.load(path.parent/self.manifest['weights_file'],map_location='cpu',weights_only=True)
        self.hidden_size=int(a['hidden_size'])
        self.minimum_tokens=max(1,int(minimum_tokens))
        for k in ('mean','scale','coef','intercept','reward_scale','special_token_ids'):
            self.register_buffer(k,a[k])
        self.eval()
        self.requires_grad_(False)

    def validate_runtime(self, *, hidden_size, model_path):
        if hidden_size!=self.hidden_size:
            raise ValueError('frozen readout hidden dimension mismatch')
        # Identity is additionally recorded in the exported manifest. Local model
        # directories are allowed; dimension alone is not provenance verification.
        if not Path(model_path).exists() and model_path!=self.manifest['model_id']:
            raise ValueError('frozen measurement backbone differs from calibration')

    def score_features(self, features):
        with torch.autocast(device_type=features.device.type, enabled=False):
            margin=((features.float()-self.mean)/self.scale)@self.coef+self.intercept
        return margin,torch.tanh(margin/self.reward_scale)

    @torch.no_grad()
    def forward(self, *, full_hidden, responses, response_mask):
        if full_hidden.ndim!=3 or full_hidden.shape[-1]!=self.hidden_size:
            raise ValueError('expected full final-layer hidden [B,S,D]')
        if responses.shape!=response_mask.shape or full_hidden.shape[0]!=responses.shape[0]:
            raise ValueError('response/mask batch shape mismatch')
        n=responses.shape[1]
        if n==0 or full_hidden.shape[1]<=n:
            raise ValueError('prompt-end position must precede response columns')
        h,p=full_hidden[:,-n:],full_hidden[:,-n-1]
        mask=response_mask.bool() & ~torch.isin(responses,self.special_token_ids)
        count=mask.sum(-1);scores=[]
        for i in range(len(h)):
            if count[i]>=self.minimum_tokens:
                scores.append(self.score_features(joint_features(h[i][mask[i]],p[i]))[0])
            else:
                scores.append(h.new_zeros((),dtype=torch.float32))
        margin=torch.stack(scores);valid=count>=self.minimum_tokens
        reward=torch.where(valid,torch.tanh(margin/self.reward_scale),torch.zeros_like(margin))
        zeros=torch.zeros_like(margin)
        # Legacy monitoring fields deliberately zero; explicit readout flag says
        # no ridge transition field / nearest-neighbour direction was used.
        return dict(reward=reward,raw_score=margin,ridge_contrast=zeros,
                    observed_contrast=zeros,support=zeros,valid=valid.float(),
                    valid_tokens=count.float(),quality_readout_enabled=torch.ones_like(margin))
