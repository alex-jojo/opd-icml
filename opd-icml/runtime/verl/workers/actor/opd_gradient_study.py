"""Fixed-prompt validation + exact sharded parameter-gradient diagnostics.

No trial optimizer steps. Parameter vectors live in bounded CPU buffers, never
in result files. Exact Gram entries reuse cached gradients when the host-memory
budget permits; otherwise the original bounded replay path remains in use.
"""
import hashlib
import json
import os
import pickle
import re
import time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from . import opd_distributed as dp
from .opd_top16 import forward,reverse_kl,_final_norm_module
from .opd_mode_absorption import observe_only
from .opd_plasticity import atomic_json,atomic_npz
from .opd_gradient_study_math import gram_summary,scope_geometry,compare_hidden,subset_alignment,transfer_report,nested_variance


def parameter_groups(name):
    groups=['all']
    if 'lm_head' in name:groups.append('head')
    else:groups.append('backbone')
    match=re.search(r'layers\.(\d+)\.',name)
    if match:groups.append('layer_'+match.group(1))
    if 'embed_tokens' in name:groups.append('embedding')
    return groups


def reshard_after_readout(actor):
    # Same root-handle lifecycle as the existing compute_log_prob worker.
    handle=getattr(actor.actor_module,'_handle',None)
    if dp.size()>1 and handle is not None:handle.reshard(True)


class ParameterLayout:
    def __init__(self,model):
        self.params=[];self.slices=[];seen=set()
        self.replicated_adapter=getattr(model,'opd_external_adapter',None)
        if self.replicated_adapter is not None:
            self.params=[self.replicated_adapter.A]
            self.slices=[(0,0,self.params[0].numel(),['all','adapter'])]
            return
        for name,p in model.named_parameters():
            if not p.requires_grad or id(p) in seen:continue
            seen.add(id(p));index=len(self.params);self.params.append(p)
            fqns=getattr(p,'_fqns',None);infos=getattr(p,'_shard_param_infos',None)
            if fqns is not None and infos is not None:
                prefix=name.rsplit('_flat_param',1)[0]
                for fqn,info in zip(fqns,infos):
                    if info.in_shard:
                        self.slices.append((index,int(info.offset_in_shard),int(info.numel_in_shard),parameter_groups(prefix+fqn)))
            elif fqns is not None:
                raise RuntimeError('FSDP flat parameter lacks shard layout; refusing incorrect parameter-group statistics')
            else:self.slices.append((index,0,p.numel(),parameter_groups(name)))
        if not self.params:raise ValueError('no trainable parameters')
        # Shared embeddings/output weights need a single non-overlapping label.
        from .opd_top16 import _unwrap_actor_module
        raw=_unwrap_actor_module(model)
        if getattr(getattr(raw,'config',None),'tie_word_embeddings',False):
            for _,_,_,groups in self.slices:
                if 'embedding' in groups or 'head' in groups:
                    groups[:]=['all','shared_embedding_head']

    def capture(self,gradient=True):
        return [(torch.zeros(p.numel(),dtype=torch.float32) if p.grad is None else
                 p.grad.detach().reshape(-1).to(device='cpu',dtype=torch.float32,copy=True))
                if gradient else p.detach().reshape(-1).cpu().clone() for p in self.params]

    def dot(self,a,b):
        values={}
        for i,start,n,groups in self.slices:
            value=0.
            for offset in range(start,start+n,262144):
                end=min(offset+262144,start+n)
                value+=float(torch.dot(a[i][offset:end].double(),b[i][offset:end].double()))
            for group in groups:values[group]=values.get(group,0.)+value
        all_values=dp.gather(values)
        divisor=dp.size() if self.replicated_adapter is not None else 1
        return {key:sum(v.get(key,0.) for v in all_values)/divisor for key in set().union(*(v.keys() for v in all_values))}

    def delta(self,before):
        return [p.detach().reshape(-1).cpu().float()-b.float() for p,b in zip(self.params,before)]

    @torch.no_grad()
    def restore_weights(self,values):
        for p,value in zip(self.params,values):p.copy_(value.reshape(p.shape).to(p.device,p.dtype))


def _gradient_cache_slots(layout, count):
    """Choose a common number of retained tail gradients within the CPU budget.

    Account for cached vectors, the working anchor/other vectors, and one
    capture scratch vector. The last gradients are reused most often. Missing
    memory information disables caching; every rank uses the smallest capacity.
    """
    local_slots = 0
    try:
        import psutil
        cap = float(os.environ.get('OPD_GRADIENT_CACHE_MAX_GIB', '64')) * (1024 ** 3)
        vector_bytes = sum(p.numel() * 4 for p in layout.params)
        available = int(psutil.virtual_memory().available)
        # Respect container limits in addition to the host's free memory.
        for limit_path, used_path in (
                ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.current'),
                ('/sys/fs/cgroup/memory/memory.limit_in_bytes',
                 '/sys/fs/cgroup/memory/memory.usage_in_bytes')):
            try:
                limit = Path(limit_path).read_text().strip()
                if limit != 'max':
                    available = min(available, max(0, int(limit) - int(Path(used_path).read_text())))
            except (OSError, ValueError):
                pass
        budget = min(cap, available // 4) / dp.size()
        if vector_bytes > 0:
            for slots in range(1, count):
                # If at most two indices are uncached, no uncached pair is
                # revisited after the tail cache fills. Only the anchor is extra.
                working = 1 if count - slots <= 2 else 2
                required = (slots + working + 1) * vector_bytes
                if required <= budget:
                    local_slots = slots
    except (ImportError, OSError, ValueError, AttributeError, OverflowError):
        pass
    return min(dp.gather(local_slots))


def exact_gram(oracle,layout,count):
    """Reuse as many exact FP32 gradients as fit; retain original pair ordering."""
    # Keep the same dot product implementation, ordering and parameter groups.
    # The cache lives only for this invocation at one fixed checkpoint.
    if getattr(layout,'replicated_adapter',None) is not None:
        vectors=[oracle(i) for i in range(count)];matrices={}
        for i in range(count):
            for j in range(i,count):
                for group,value in layout.dot(vectors[i],vectors[j]).items():
                    matrices.setdefault(group,np.zeros((count,count),np.float64))[i,j]=value
                    matrices[group][j,i]=value
        return matrices,count
    slots=_gradient_cache_slots(layout,count)
    if dp.primary():
        print(f'[gradient Gram] retaining {slots}/{max(0,count-1)} reusable gradients',flush=True)
    matrices={};calls=0;cache={}
    cache_start=count-slots
    for i in range(count):
        if i in cache:
            anchor=cache.pop(i)
        else:
            anchor=oracle(i);calls+=1
        for j in range(i,count):
            if i==j:
                other=anchor
            elif j in cache:
                other=cache[j]
            else:
                other=oracle(j);calls+=1
                if j>=cache_start:
                    cache[j]=other
            values=layout.dot(anchor,other)
            for group,value in values.items():
                matrices.setdefault(group,np.zeros((count,count),np.float64))[i,j]=value
                matrices[group][j,i]=value
            if i!=j:del other
        del anchor
    return matrices,calls


class HiddenMoments:
    chunk_tokens=1024
    def __init__(self,basis=None):
        # Spectrum-only observations need the global uncentered Gram on rank zero.
        self.basis = basis.detach().cpu().float().numpy() if basis is not None else None
        self.gram=None;self.sum=None;self.count=0
    @torch.no_grad()
    def add(self,g):
        g=g.detach()
        if self.gram is None:
            self.gram=torch.zeros((g.shape[-1],g.shape[-1]),device=g.device,dtype=torch.float32)
            if self.basis is not None:
                self.sum=torch.zeros(g.shape[-1],device=g.device,dtype=torch.float64)
        with torch.autocast(device_type=g.device.type,enabled=False):
            for block in g.split(self.chunk_tokens):
                part=block.float()
                self.gram.addmm_(part.T,part)
                # Reduction accumulates in FP64 without an N x hidden FP64 copy.
                if self.sum is not None:
                    self.sum.add_(part.sum(0,dtype=torch.float64))
        self.count+=len(g)
    def finish(self):
        gram,n=dp.sum_gram(self.gram,self.count)
        if n<=0:raise ValueError('empty hidden observation')
        result = dict(count=np.asarray(n))
        if self.basis is not None or dp.primary():
            result['second'] = gram.cpu().double().div_(n).float().numpy()
        if self.basis is not None:
            if dp.size()>1:torch.distributed.all_reduce(self.sum)
            result.update(mean=(self.sum.cpu()/n).numpy(), basis=self.basis)
        return result


@contextmanager
def gradient_observation(actor):
    """Deterministic readout WITH backward and checkpoint recomputation.

    eval() alone keeps autograd enabled but disables HF checkpoint layers.
    Enable training only on checkpoint owners, not their attention/dropout
    children; this preserves the previous dropout-free diagnostic objective.
    No optimizer step, RNG change, or lasting module-mode change is allowed.
    """
    with observe_only(actor), torch.enable_grad():
        for module in actor.actor_module.modules():
            if getattr(module,'gradient_checkpointing',False):
                module.training=True
        try:
            yield
        finally:
            actor.actor_optimizer.zero_grad(set_to_none=True)


@contextmanager
def hidden_route(actor,basis,route):
    if route=='full':yield;return
    def hook(module,args,h):
        def backward(g):
            if route=='direct':return torch.zeros_like(g)
            projected=(g@basis.to(g))@basis.to(g).T
            return projected if route=='u' else g-projected
        h.register_hook(backward)
    handle=_final_norm_module(actor.actor_module).register_forward_hook(hook)
    try:yield
    finally:handle.remove()


def gradient(actor,data,basis,route='full',normalization='token',capture_hidden=False):
    """All ranks see the same diagnostic batch; gradients remain sharded."""
    if hasattr(actor,'fixed_u_adapter'):
        from .opd_adapter_diagnostics import cached_gradient
        return cached_gradient(actor,data,basis,route,normalization,capture_hidden)
    layout=ParameterLayout(actor.actor_module);actor.actor_optimizer.zero_grad(set_to_none=True)
    device=next(actor.actor_module.parameters()).device
    replicated=len(data)%dp.size()!=0
    local,offset=(data,0) if replicated else dp.shard(data)
    factor=1 if replicated else dp.size();total=int(data.batch['response_mask'].sum())
    moments=HiddenMoments(basis) if capture_hidden else None;losses=[]
    with gradient_observation(actor),hidden_route(actor,basis,route):
        for part in local.split(actor.config.ppo_micro_batch_size_per_gpu):
            x={k:v.to(device) for k,v in part.batch.items()};mask=x['response_mask'].bool();n=mask.shape[1]
            handle=None
            if moments is not None:
                if replicated:raise ValueError('hidden moments require a rank-divisible batch')
                def hook(_m,_a,h):
                    # Scope gradients are per-token d ell / dh, before mean/projection.
                    def backward(g):
                        row,pos=mask.nonzero(as_tuple=True);start=g.shape[1]-n-1
                        lengths=mask.sum(1)
                        for lo in range(0,len(row),HiddenMoments.chunk_tokens):
                            rr=row[lo:lo+HiddenMoments.chunk_tokens]
                            pp=pos[lo:lo+HiddenMoments.chunk_tokens]+start
                            raw=g[rr,pp].float()
                            scale=total/factor if normalization=='token' else len(data)*lengths[rr,None]/factor
                            moments.add(raw*scale)
                        return g
                    h.register_hook(backward)
                handle=_final_norm_module(actor.actor_module).register_forward_hook(hook)
            try:
                _,z,_,_=forward(actor,x,float(data.meta_info['temperature']),x['opd_top16_ids'],diagnostics=False)
                kl=reverse_kl(z,x['opd_teacher_top16_logits']);per=(kl*mask).sum(1)/mask.sum(1).clamp_min(1)
                loss=(kl*mask).sum()*factor/total if normalization=='token' else per.sum()*factor/len(data)
                loss.backward();losses.extend(per.detach().cpu().tolist())
                # Release the previous microbatch's outputs before the next
                # forward, including autograd references held by loss/per/z.
                del z,kl,per,loss,x
            finally:
                if handle is not None:handle.remove()
        vector=layout.capture()
    loss=np.asarray(losses) if replicated else np.concatenate(dp.gather(np.asarray(losses)))
    return vector,loss,moments.finish() if moments is not None else None


@torch.no_grad()
def losses(actor,data):
    if hasattr(actor,'fixed_u_adapter'):
        from .opd_adapter_diagnostics import cached_losses
        return cached_losses(actor,data)
    device=next(actor.actor_module.parameters()).device;local,_=dp.shard(data);out=[]
    with observe_only(actor):
        for part in local.split(actor.config.ppo_micro_batch_size_per_gpu):
            x={k:v.to(device) for k,v in part.batch.items()}
            _,z,_,_=forward(actor,x,float(data.meta_info['temperature']),x['opd_top16_ids'],diagnostics=False)
            mask=x['response_mask'];kl=reverse_kl(z,x['opd_teacher_top16_logits'])
            out.extend(((kl*mask).sum(1)/mask.sum(1).clamp_min(1)).cpu().tolist())
    reshard_after_readout(actor)
    return np.concatenate(dp.gather(np.asarray(out)))


class GradientStudy:
    def __init__(self,actor,cfg,output):
        self.actor=actor;self.cfg=dict(cfg);self.output=Path(output)/'gradient_study';self.payload=None
        self.rq3_directory = Path(output).parent / 'training_records' / 'rq3'
        self.calls=0

    def start(self,train,basis,payload,checkpoint):
        self.started=time.perf_counter();self.diagnostic_seconds=0.;self.checkpoint=checkpoint;self.train=train;self.basis=basis.float()
        self.payload=payload;self.directory=self.output/f'checkpoint_{checkpoint:03d}'
        self.layout=ParameterLayout(self.actor.actor_module);self.calls=0;self.report={}
        self.subset_alignment = self.cross_batch() if payload.get('geometry',False) else None
        if payload.get('paths') is not None:self.path_variance(payload['paths'])
        self.validation=payload['validation']
        self.val_grad,self.before,val_moments=self.grad(self.validation,normalization='response',capture_hidden=True)
        atomic_npz(self.directory/'validation_hidden_moments.npz',val_moments)
        self.val_norm=self.layout.dot(self.val_grad,self.val_grad)
        self.report.update(checkpoint_optimizer_updates=checkpoint,validation_prompts=len(self.validation),
            validation_normalization='mean of per-response token-mean KL',
            parameter_gradient_exact=True,parameter_gradient_disk_bytes=0,
            diagnostic_optimizer_steps=0,validation_used_for_training=False,
            basis_reference=payload.get('basis_reference','see saved basis'),
            parameter_groups=sorted(self.val_norm))
        if hasattr(self.actor,'fixed_u_adapter'):
            self.report.update(parameter_space='trainable adapter A',backbone_frozen=True,head_frozen=True,
                               frozen_parameter_gradients='zero by design; cosine undefined, not omitted measurements')
        self.diagnostic_seconds+=time.perf_counter()-self.started

    def grad(self,*args,**kwargs):
        self.calls+=1
        return gradient(self.actor,*args,basis=self.basis,**kwargs)

    def cross_batch(self):
        m=int(self.cfg.get('batches',4))
        if len(self.train)%m or (len(self.train)//m)%dp.size():raise ValueError('diagnostic batch split must divide DP ranks')
        size=len(self.train)//m
        # Stable random partition avoids reward/length sorting defining the batches.
        indices=np.random.default_rng(int(self.cfg.get('seed',42))+self.checkpoint).permutation(len(self.train))
        batches=[self.train.select_idxs(indices[i*size:(i+1)*size].tolist()) for i in range(m)]
        hidden={}
        def oracle(i):
            v,_,mom=self.grad(batches[i],capture_hidden=i not in hidden)
            if mom is not None:
                hidden[i]=mom
                atomic_npz(self.directory/f'train_batch_{i}_hidden_moments.npz',mom)
            return v
        grams,replays=exact_gram(oracle,self.layout,m)
        effective=None
        if self.actor.config.policy_loss.get('hidden_projection',{}).get('enabled',False):
            eg,_=exact_gram(lambda i:self.grad(batches[i],route='u')[0],self.layout,m)
            effective={key:gram_summary(value) for key,value in eg.items()}
        groups = {key:gram_summary(k) for key,k in grams.items()}
        train_rows = indices.reshape(m,size).tolist()
        atomic_json(self.directory/'parameter_cross_batch.json',dict(
            groups=groups,
            raw_opd_groups=True,effective_r128_groups=effective,
            batches=m,prompts_per_batch=size,normalization='token mean per diagnostic batch',
            train_indices=train_rows,exact_gradient_evaluations=replays,
            comparison='same checkpoint, different disjoint subsets of current on-policy training rollout'))
        # CPU decompositions once, not once per GPU.
        result=dp.primary_call(lambda:compare_hidden([scope_geometry(hidden[i]['second'],hidden[i]['mean'],hidden[i]['basis']) for i in range(m)]))
        atomic_json(self.directory/'hidden_cross_batch.json',result)
        self.report['cross_batch_gradient_evaluations']=replays
        self.report['cross_batch_complete']=True
        return subset_alignment(train_rows, groups['all'], result['full'])

    def save_rq3(self, result):
        tokens = self.validation.batch['responses'].detach().cpu().numpy()
        mask = self.validation.batch['response_mask'].detach().cpu().bool().numpy()
        # The row number joins to validation128/checkpoint_XXX.json. The digest
        # also identifies the exact sampled response if the same prompt is regenerated.
        rows = zip(self.payload['validation_source_indices'], tokens, mask,
                   result['kl_before_per_prompt'], result['kl_after_per_prompt'],
                   result['gain_per_prompt'], strict=True)
        responses = []
        for row, (source, ids, valid, before, after, gain) in enumerate(rows):
            answer = np.asarray(ids[valid], dtype='<i8')
            responses.append(dict(response_row=row, source_index=int(source), response_tokens=len(answer),
                                  response_sha256=hashlib.sha256(answer.tobytes()).hexdigest(),
                                  kl_before=before, kl_after=after, gain=gain))
        record = dict(checkpoint_before=self.checkpoint, optimizer_update=self.checkpoint+1,
                      gain_epsilon=result['threshold'], kl_before=result['kl_before'], kl_after=result['kl_after'],
                      gain=result['gain'], positive_fraction=result['positive_fraction'], responses=responses)
        if self.subset_alignment is not None:
            record['subset_alignment'] = self.subset_alignment
        atomic_json(self.rq3_directory / f'checkpoint_{self.checkpoint:03d}.json', record)

    def path_variance(self,data):
        k=int(self.cfg.get('path_repeats',4));p=len(data)//k
        if p*k!=len(data) or p<2 or k<2:raise ValueError('invalid prompt/rollout panel')
        overall=None;means_norms=[];within=[];same=[];adjacent=[];previous=None;evaluations=0
        for prompt in range(p):
            items=[data.select_idxs([prompt*k+r]) for r in range(k)]
            mean=None;seen=set()
            def oracle(i):
                nonlocal mean
                v,_,_=self.grad(items[i],normalization='response')
                if i not in seen:
                    if mean is None:mean=[x/k for x in v]
                    else:
                        for a,b in zip(mean,v):a.add_(b,alpha=1/k)
                    seen.add(i)
                return v
            gram,calls=exact_gram(oracle,self.layout,k);evaluations+=calls
            stats=gram_summary(gram['all']);within.append(stats['variance']);same.append(stats)
            # The prompt mean was accumulated during the Gram evaluation: no extra backward.
            norm=self.layout.dot(mean,mean)['all'];means_norms.append(norm)
            if overall is None:overall=[x.clone()/p for x in mean]
            else:
                for a,b in zip(overall,mean):a.add_(b,alpha=1/p)
            if previous is not None:
                dot=self.layout.dot(previous,mean)['all'];old=self.layout.dot(previous,previous)['all']
                adjacent.append(dict(prompt_a=prompt-1,prompt_b=prompt,
                    mean_gradient_cosine=dot/np.sqrt(old*norm) if old*norm>1e-30 else None))
                previous=None
            else:previous=mean
        mean2=self.layout.dot(overall,overall)['all']
        between=max((sum(means_norms)-p*mean2)/(p-1),0)
        result=nested_variance(float(np.mean(within)),between,k)
        result.update(prompts=p,within_prompt_gram=same,between_prompt_pairs=adjacent,exact_gradient_evaluations=evaluations,
            training_pool_indices=self.payload.get('path_training_indices'),
            between_pair_definition='disjoint adjacent prompt MEANS; exact, sampled pairs, not all trajectory pairs',
            full_128_by_128_gram=False,parameter_vectors_saved=False)
        atomic_json(self.directory/'path_variance.json',result);self.report['path_variance_complete']=True

    def before_update(self):
        started=time.perf_counter()
        train=self.layout.capture();self.train_norm=self.layout.dot(train,train)
        self.train_val=self.layout.dot(train,self.val_grad);self.train_gradient=train
        self.weights=self.layout.capture(gradient=False)
        self.diagnostic_seconds+=time.perf_counter()-started

    def after_update(self):
        started=time.perf_counter()
        delta=self.layout.delta(self.weights);norm=self.layout.dot(delta,delta);dot=self.layout.dot(self.val_grad,delta)
        train_delta=self.layout.dot(self.train_gradient,delta);self.train_gradient=None
        after=losses(self.actor,self.validation)
        result=transfer_report(self.before,after,self.val_norm['all'],norm['all'],dot['all'],
            float(self.cfg.get('gain_epsilon',1e-6)))
        result['groups']={g:dict(validation_norm=np.sqrt(self.val_norm[g]),update_norm=np.sqrt(norm[g]),
            predicted_gain=-dot[g],train_norm=np.sqrt(self.train_norm[g]),
            train_descent_update_cosine=-train_delta[g]/np.sqrt(self.train_norm[g]*norm[g])
            if self.train_norm[g]*norm[g]>1e-30 else None,
            train_validation_cosine=self.train_val[g]/np.sqrt(self.train_norm[g]*self.val_norm[g])
            if self.train_norm[g]*self.val_norm[g]>1e-30 else None) for g in self.val_norm}
        if self.payload.get('geometry',False) and hasattr(self.actor,'fixed_u_adapter'):
            # dL/dA = U^T g h^T. Projecting g into the SAME fixed U leaves
            # this gradient unchanged; outside and frozen-head routes are zero.
            # No reason to replay two extra Adapter backwards to obtain zeros.
            result['parameter_path_gain']=dict(u=-dot['all'],outside=0.,direct=0.,
                definition='exact chain-rule identity for h+UA h with frozen Backbone/head and the same orthonormal U')
        elif self.payload.get('geometry',False) and self.cfg.get('parameter_path_gain',False):
            # Diagnostics at BEFORE parameters, with the ACTUAL committed displacement.
            current=self.layout.capture(gradient=False)
            try:
                self.layout.restore_weights(self.weights)
                routed,_,_=self.grad(self.validation,route='u',normalization='response')
                u_plus_direct=self.layout.dot(routed,delta)['all'];del routed
                routed,_,_=self.grad(self.validation,route='direct',normalization='response')
                direct=self.layout.dot(routed,delta)['all'];del routed
                result['parameter_path_gain']=dict(u=-(u_plus_direct-direct),
                    outside=-(dot['all']-u_plus_direct),direct=-direct,
                    definition='U/outside final-hidden routes plus direct head path; outside by exact gradient linearity; AdamW not decomposed')
            finally:self.layout.restore_weights(current)
        self.actor.actor_optimizer.zero_grad(set_to_none=True)
        self.diagnostic_seconds+=time.perf_counter()-started
        result.update(self.report,gradient_evaluations=self.calls,diagnostic_seconds=self.diagnostic_seconds,
            actual_optimizer_updates=1,status='complete',schema='fixed128-gradient-study-v1')
        atomic_json(self.directory/'validation_transfer.json',result)
        dp.primary_call(lambda: self.save_rq3(result))
        self.val_grad=None;self.weights=None;self.payload=None;self.validation=None
        metrics={'validation128/gain':result['gain'],'validation128/predicted_gain':result['predicted_gain'],
                'validation128/kl_before':result['kl_before'],'validation128/kl_after':result['kl_after'],
                'validation128/positive_fraction':result['positive_fraction'],
                'validation128/negative_fraction':result['negative_fraction'],
                'validation128/alignment_available':float(result['update_alignment'] is not None),
                'timing/gradient_study_seconds':result['diagnostic_seconds']}
        if result['update_alignment'] is not None:metrics['validation128/update_alignment']=result['update_alignment']
        cosine=result['groups']['all']['train_validation_cosine']
        if cosine is not None:metrics['validation128/train_validation_gradient_cosine']=cosine
        return metrics
