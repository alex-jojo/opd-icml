"""One FULL trajectory; disposable real-optimizer token-source interventions.

Single GPU NO_SHARD, dropout=0, no scaler. Never approximates AdamW with a
handwritten preview. The ordinary branch is checked against the real commit.
"""
import copy
import hashlib
import json
import re
import random
import time
from contextlib import contextmanager
import numpy as np
import torch
import torch.nn.functional as F
from .opd_response_geometry import Observer, select, fingerprint
from .opd_mode_absorption import observe_only
from .opd_top16 import forward, reverse_kl, _final_norm_module, _unwrap_actor_module
from .opd_plasticity import atomic_json, atomic_npz, kl_and_logit_gradient, head_gradient
from .opd_token_audit import output_scores, enriched, top_mask, training_summary, response_summary


def clone_state(value):
    if torch.is_tensor(value): return (value.detach().cpu().clone(), value.device)
    if isinstance(value, dict): return {k:clone_state(v) for k,v in value.items()}
    return copy.deepcopy(value)


def restore_state(value):
    if isinstance(value, tuple) and len(value)==2 and torch.is_tensor(value[0]):
        return value[0].to(value[1], copy=True)
    if isinstance(value, dict): return {k:restore_state(v) for k,v in value.items()}
    return copy.deepcopy(value)


def model_optimizer_digest(actor):
    digest = hashlib.sha256()
    for p in actor.actor_module.parameters():
        digest.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        for k, v in sorted(actor.actor_optimizer.state.get(p, {}).items()):
            digest.update(k.encode())
            digest.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                          if torch.is_tensor(v) else repr(v).encode())
    return digest.hexdigest()


class Snapshot:
    def __init__(self, actor):
        if not isinstance(actor.actor_optimizer, torch.optim.AdamW) or actor.scaler is not None:
            raise ValueError('audit requires real AdamW without GradScaler')
        self.actor = actor
        self.params = list(actor.actor_module.parameters())
        self.weights = [p.detach().cpu().clone() for p in self.params]
        self.grads = [None if p.grad is None else p.grad.detach().cpu().clone() for p in self.params]
        self.state = {p:clone_state(s) for p,s in actor.actor_optimizer.state.items()}
        self.buffers = [(b,b.detach().cpu().clone()) for b in actor.actor_module.buffers()]

    def restore(self, scale=1.):
        with torch.no_grad():
            for p,w,g in zip(self.params,self.weights,self.grads):
                p.copy_(w.to(p.device))
                p.grad = None if g is None else (g.to(p.device,copy=True)*scale)
            for b,v in self.buffers: b.copy_(v.to(b.device))
        self.actor.actor_optimizer.state.clear()
        self.actor.actor_optimizer.state.update({p:restore_state(s) for p,s in self.state.items()})


@contextmanager
def disposable(actor):
    snap = Snapshot(actor)
    device = snap.params[0].device
    modes = [(m,m.training) for m in actor.actor_module.modules()]
    numpy_rng=np.random.get_state();python_rng=random.getstate()
    with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
        try: yield snap
        finally:
            snap.restore()
            np.random.set_state(numpy_rng);random.setstate(python_rng)
            for m,flag in modes: m.training=flag


def replay_gradient(actor, data, remove, basis=None):
    """Drop source losses, including head/embedding paths; original denominator."""
    actor.actor_optimizer.zero_grad(set_to_none=True)
    total = int(data.batch['response_mask'].sum())
    device = next(actor.actor_module.parameters()).device
    cursor = 0
    for part in data.split(actor.config.ppo_micro_batch_size_per_gpu):
        x = {k:v.to(device) for k,v in part.batch.items()}
        _,z,_,_ = forward(actor,x,float(data.meta_info['temperature']),x['opd_top16_ids'],diagnostics=False,
            gradient_basis=basis,projection_state={'full':0.,'kept':0.})
        mask = x['response_mask'].float()
        keep = ~remove[cursor:cursor+len(part)].to(device)
        loss = (reverse_kl(z,x['opd_teacher_top16_logits'])*mask*keep).sum()/total
        loss.backward()
        cursor += len(part)


@contextmanager
def capture_bank(actor,x,u,row_labels=None,cost=None,scope_accumulator=None):
    """Capture selected rows in an existing forward; hooks never build graphs."""
    device=x['responses'].device;mask=x['response_mask'].bool().clone();n=mask.shape[1]
    if row_labels is not None:mask &= row_labels[:,None]>=0
    row,pos=mask.nonzero(as_tuple=True);out={}
    def add(k,v):out[k]=v.detach().cpu().numpy()
    @torch.no_grad()
    def norm_hook(m,a,h):
        started=time.perf_counter()
        hidden=h[:,-n-1:-1][mask].float()
        with torch.autocast(device_type=device.type,enabled=False):
            add('h',hidden);add('h_u',hidden@u);add('unit_h_u',F.normalize(hidden,dim=-1)@u)
        if cost is not None:cost['seconds']=cost.get('seconds',0.)+time.perf_counter()-started
    @torch.no_grad()
    def head_hook(m,a,logits):
        started=time.perf_counter()
        with torch.autocast(device_type=device.type,enabled=False):
            selected=logits[:,-n-1:-1][mask]
            z=selected if getattr(m,'_opd_compact_active',False) else selected.gather(-1,x['opd_top16_ids'][mask])
            teacher=x['opd_teacher_top16_logits'][mask]
            _,derivative=kl_and_logit_gradient(z,teacher)
            g=head_gradient(derivative,x['opd_top16_ids'][mask],m.weight,forward_dtype=logits.dtype)
            if scope_accumulator is not None:scope_accumulator.add(g)
            gu=g@u
            add('g_u',gu);add('raw_gradient_energy',g.square().sum(-1));add('projected_gradient_energy',gu.square().sum(-1))
            for k,v in output_scores(z,teacher).items():add(k,v)
            add('z',z.float());add('q',teacher.float())
        if cost is not None:cost['seconds']=cost.get('seconds',0.)+time.perf_counter()-started
    head=_unwrap_actor_module(_unwrap_actor_module(actor.actor_module).lm_head)
    a=_final_norm_module(actor.actor_module).register_forward_hook(norm_hook)
    b=head.register_forward_hook(head_hook)
    try:yield out
    finally:a.remove();b.remove()
    add('row',row if row_labels is None else row_labels[row]);add('pos',pos);add('token',x['responses'][mask])
    ordinal=mask.long().cumsum(-1)-1
    add('relative_position',(ordinal.float()/(mask.sum(-1,keepdim=True)-1).clamp_min(1))[mask])
    add('boundary',(ordinal==0)[mask])


@torch.no_grad()
def read_bank(actor, data, u, scope_label=None):
    """Same capture as formal-forward reuse; separate forwards only when needed."""
    device=next(actor.actor_module.parameters()).device;parts=[];cursor=0
    replication=getattr(actor,'_replication17',None);accumulator=None
    if (scope_label is not None and replication is not None and replication.measure
            and replication.cfg.get('all_gradient_scopes',False)):
        from .opd_gradient_scopes import GradientScopes
        accumulator=GradientScopes(u)
    from . import opd_distributed as dp
    local,cursor=dp.shard(data)
    for part in local.split(actor.config.ppo_micro_batch_size_per_gpu):
        x={k:v.to(device) for k,v in part.batch.items()}
        labels=torch.arange(cursor,cursor+len(part),device=device)
        with capture_bank(actor,x,u,labels,scope_accumulator=accumulator) as values:
            forward(actor,x,float(data.meta_info['temperature']),x['opd_top16_ids'],diagnostics=False)
        parts.append(values);cursor+=len(part)
    if accumulator is not None:
        from .opd_gradient_scopes import save
        report,arrays=accumulator.finish(int(data.batch['response_mask'].sum()))
        bank,phase,inner=scope_label
        report.update(bank=bank,phase=phase,
            definition='Analytic per-token conditional Top16 final-hidden loss gradient on the existing diagnostic forward')
        save(report,arrays,replication.output/'bank_gradient_scopes'/bank/phase,
            replication.step,inner,int(replication.cfg.get('reuse',1)),
            replication.audit.reference if replication.full else 'FULL frozen step10 U128')
    values={k:np.concatenate([v[k] for v in parts]) for k in parts[0]}
    return dp.merge_fields(dp.gather(values))


def differences(before, after):
    for k in ('row','pos','token'): np.testing.assert_array_equal(before[k],after[k])
    dh=after['h']-before['h']
    bh=before['h']/np.maximum(np.linalg.norm(before['h'],axis=-1,keepdims=True),1e-12)
    ah=after['h']/np.maximum(np.linalg.norm(after['h'],axis=-1,keepdims=True),1e-12)
    after['raw_delta_energy']=np.square(dh).sum(-1)
    after['unit_delta_energy']=np.square(ah-bh).sum(-1)
    return after


def compact_bank(value):
    # Exact summaries use ALL tokens. Only vector payloads are subsampled on
    # disk; scalar outputs/identifiers remain complete for offline regrouping.
    ix=[]
    for r in np.unique(value['row']):
        rows=np.flatnonzero(value['row']==r)
        ix.extend(rows[np.linspace(0,len(rows)-1,min(256,len(rows))).round().astype(int)])
    ix=np.asarray(ix,dtype=np.int64)
    return {**{k:v[ix] if k in ('h_u','unit_h_u','g_u') else v for k,v in value.items() if k!='h'},
            'hidden_vector_index':ix}


class TokenAudit(Observer):
    def begin(self,train,heldout,basis,step):
        super().begin(train,heldout,basis,step)
        self.scores=[]
        if basis is not None: raise ValueError('token audit is FULL OPD only')
        if not hasattr(self,'thresholds'):
            state=self.directory/'thresholds.json'
            self.thresholds=json.loads(state.read_text()) if state.exists() else None
            if self.thresholds is None and step!=1: raise ValueError('missing step-1 thresholds on resume')
            self.fixed=None;self.labels={}
            self.tokenizer=None
            path=self.cfg.get('tokenizer_path')
            if path:
                from transformers import AutoTokenizer
                self.tokenizer=AutoTokenizer.from_pretrained(path)
            fixed=self.directory/'fixed_bank.pt'
            from . import opd_distributed as dp
            if dp.exists(fixed):
                # Own locally written diagnostic cache, never arbitrary input pickle.
                self.fixed=torch.load(fixed,map_location='cpu',weights_only=False)
            elif step>1: raise ValueError('missing frozen diagnostic bank on resume')
        if self.fixed is None:
            self.fixed=select(heldout,range(len(heldout)))
            self.directory.mkdir(parents=True,exist_ok=True)
            from . import opd_distributed as dp
            if dp.primary():
                tmp=self.directory/'fixed_bank.tmp';torch.save(self.fixed,tmp);tmp.replace(self.directory/'fixed_bank.pt')
            dp.barrier()

    def decorate(self,f):
        f=dict(f);f['first']=f['pos']==0
        eos=getattr(getattr(_unwrap_actor_module(self.actor.actor_module),'config',None),'eos_token_id',None)
        f['eos']=np.isin(f['token'],[] if eos is None else [eos] if isinstance(eos,int) else eos)
        f['boundary']=f['first']|f['eos']
        for token in np.unique(f['token']):
            if int(token) not in self.labels:
                text=self.tokenizer.decode([int(token)]) if self.tokenizer is not None else ''
                self.labels[int(token)]=(bool(re.search(r'\d',text)),bool(re.search(r'[+\-*/=<>×÷]',text)),
                    bool(re.search(r'[\\{}_^$]',text)),text)
        for j,k in enumerate(('numeric','operator','latex')):f[k]=np.array([self.labels[int(t)][j] for t in f['token']])
        f['other']=~(f['boundary']|f['numeric']|f['operator']|f['latex'])
        return enriched(f)

    def record_outputs(self,student,teacher,mask,kl):
        values=output_scores(student,teacher);valid=mask.bool()
        # Compute these once, retaining the same FP64 score definitions.
        with torch.no_grad():
            lq=teacher.detach().double().log_softmax(-1)
            teacher_entropy=-(lq.exp()*lq).sum(-1)
            for name,value in [('kl',values['kl']),('tv',values['tv']),
                               ('student_entropy',values['entropy']),('teacher_entropy',teacher_entropy)]:
                self.token_fields[name].append(value[valid].cpu())
        self.scores.append({k:v[valid].cpu().numpy() for k,v in values.items()})

    def before_update(self):
        start=time.perf_counter()
        f={k:torch.cat(v).numpy() for k,v in self.token_fields.items()}
        f.update({k:np.concatenate([s[k] for s in self.scores]) for k in self.scores[0]})
        self.scores.clear();z=torch.cat(self.raw_z).numpy();self.raw_z.clear()
        if len(z)!=int(self.train.batch['response_mask'].sum()):raise RuntimeError('token count mismatch')
        f=self.decorate(f)
        if self.thresholds is None:
            self.thresholds=dict(kl_p90=float(np.quantile(f['kl'],.9)),kl_median=float(np.median(f['kl'])),
                                 entropy_median=float(np.median(f['entropy'])))
            atomic_json(self.directory/'thresholds.json',self.thresholds)
        from .opd_distributed import primary_call
        report,arrays=primary_call(lambda:training_summary(f,z,self.thresholds))
        report.update(step=self.step,lexical_labels_available=self.tokenizer is not None)
        atomic_json(self.directory/f'step_{self.step:03d}.json',report)
        atomic_npz(self.directory/f'step_{self.step:03d}.npz',dict(**arrays,**{'token_'+k:v for k,v in f.items()}))
        self.metrics.update({'token_audit/gradient_rms_full':report['gradient_rms_full'],
            'token_audit/gradient_rms_u':report['gradient_rms_u'],
            'token_audit/u_effective_rank':report['spectrum']['effective_rank'] or 0.,
            'token_audit/high_kl_fraction':report['groups']['kl_above_step1_p90']['fraction']})
        if self.step in self.anchors:self.probe(f)
        self.metrics['timing/token_audit_before_seconds']=time.perf_counter()-start

    def probe(self,fields):
        print(f'[token-audit] update {self.step}: measuring disposable source interventions',flush=True)
        banks,indices=self.banks();banks['fixed']=self.fixed
        directory=self.directory/f'probe_{self.step:03d}'
        high=top_mask(fields['kl'],.1)
        random=np.zeros(len(high),bool);rng=np.random.default_rng(1729+self.step)
        # Exact selected count per response x position quintile x boundary.
        for row in np.unique(fields['row']):
            for quintile in range(5):
                for boundary in (False,True):
                    mask=(fields['row']==row)&(np.minimum((fields['relative_position']*5).astype(int),4)==quintile)&(fields['boundary']==boundary)
                    ix=np.flatnonzero(mask);k=int(high[mask].sum())
                    if k:random[rng.choice(ix,k,replace=False)]=True
        remove={}
        for name,chosen in [('drop_high10',high),('drop_random10',random)]:
            mask=torch.zeros_like(self.train.batch['response_mask'],device='cpu',dtype=torch.bool)
            mask[torch.as_tensor(fields['row'][chosen]),torch.as_tensor(fields['pos'][chosen])]=True
            remove[name]=mask
        with disposable(self.actor) as snap:
            with observe_only(self.actor):before={k:self.decorate(read_bank(self.actor,v,self.stats_basis)) for k,v in banks.items()}
            for name,v in before.items():atomic_npz(directory/f'{name}_before.npz',compact_bank(v))
            records={};base_response=None
            candidates=['ordinary','drop_high10','drop_random10','uniform_0.9']
            if self.cfg.get('probe_r128',False):candidates.append('r128_only')
            for candidate in candidates:
                print(f'[token-audit] update {self.step}: {candidate}',flush=True)
                snap.restore(scale=.9 if candidate=='uniform_0.9' else 1.)
                if candidate in remove:replay_gradient(self.actor,self.train,remove[candidate])
                if candidate=='r128_only':
                    replay_gradient(self.actor,self.train,torch.zeros_like(self.train.batch['response_mask'],dtype=torch.bool),
                        self.stats_basis.to(dtype=getattr(self.actor,'param_dtype',torch.bfloat16)))
                norm=self.actor._optimizer_step()
                if not np.isfinite(float(norm)):raise FloatingPointError('nonfinite disposable update')
                if candidate=='ordinary':self.expected_digest=model_optimizer_digest(self.actor)
                record={};response={}
                with observe_only(self.actor):
                    for name,data in banks.items():
                        after=self.decorate(read_bank(self.actor,data,self.stats_basis))
                        if hasattr(self,'observed_bank'):self.observed_bank(candidate,name,after)
                        differences(before[name],after)
                        record[name]=response_summary(before[name],after,self.thresholds)
                        du=after['unit_h_u']-before[name]['unit_h_u']
                        response[name]=du
                        if base_response is not None:
                            b=base_response[name]
                            record[name]['difference_from_ordinary_unit_u_energy']=float(np.square(du-b).sum())
                            record[name]['cosine_to_ordinary_unit_u']=float((du*b).sum()/max(np.linalg.norm(du)*np.linalg.norm(b),1e-30))
                        atomic_npz(directory/f'{name}_{candidate}.npz',compact_bank(after))
                if candidate=='ordinary':base_response=response
                records[candidate]=dict(gradient_norm=float(norm),banks=record)
            atomic_npz(directory/'source_masks.npz',dict(high=high,random=random,row=fields['row'],pos=fields['pos']))
            atomic_json(directory/'results.json',dict(step=self.step,records=records,train_indices=indices.tolist(),
                bank_sha256={k:fingerprint(v) for k,v in banks.items()},extra_backwards=2+int('r128_only' in candidates),
                temporary_optimizer_steps=len(candidates),teacher_calls_extra=0,committed_verified=False,
                random_high_overlap=float((high&random).sum()/max(high.sum(),1)),
                definition='Source loss removal includes direct head path; denominator is original valid-token count. Marginal effects are NOT additive contributions.'))
        self.probe_directory=directory

    def after_update(self):
        if self.step not in self.anchors:return
        actual=model_optimizer_digest(self.actor)
        if actual!=self.expected_digest:raise RuntimeError('REAL disposable vs committed model/optimizer mismatch')
        path=self.probe_directory/'results.json';result=json.loads(path.read_text())
        result.update(committed_verified=True,committed_model_optimizer_sha256=actual)
        atomic_json(path,result)
        print(f'[token-audit] update {self.step}: formal model + AdamW state verified identical',flush=True)
        self.metrics['token_audit/commit_verified']=1.
        # Context examples are diagnostic only; no model-based semantic labels.
        atomic_json(self.directory/'token_dictionary.json',{str(k):v[-1] for k,v in self.labels.items()})


def get_observer(actor):
    if not hasattr(actor,'_response_geometry'):
        actor._response_geometry=TokenAudit(actor,actor.config.policy_loss['response_geometry'])
    if not isinstance(actor._response_geometry,TokenAudit):raise ValueError('observer protocol changed')
    return actor._response_geometry
