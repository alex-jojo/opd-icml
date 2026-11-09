"""V4: three shared hidden-supervision channels, calibrated by real AdamW previews.

Single GPU, FP32 master parameters, dropout-free Qwen final-norm interface.
No token selection, virtual hidden optimization, or additional teacher query.
Preview gradients are decomposed BEFORE clipping/Adam, never after Adam.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
from contextlib import contextmanager
import hashlib
import itertools
import json
import math
import time

import numpy as np
import torch

from .opd_top16 import forward, reverse_kl, _final_norm_module, _unwrap_actor_module
from .opd_response_geometry import adam_preview, displaced, inert, select
from .opd_transfer_stats import row_losses, statistics


@dataclass
class Settings:
    enabled: bool = False
    interval: int = 10
    head_rank: int = 16
    diagnostic_size: int = 32
    control_size: int = 32
    difference: float = .25
    coefficient_max: float = 2.
    max_change: float = .5
    smoothing: float = .5
    fisher_penalty: float = 1.
    ridge_relative: float = .1
    response_budget: float = .001
    policy_ratio: float = 1.25
    policy_floor: float = .0001
    policy_ceiling: float = .01
    min_gain: float = .00001
    min_prediction_ratio: float = .1
    paired_controls: bool = True
    channel_count: int = 3
    output_dir: str = ''

    def validate(self):
        if self.channel_count not in (2,3):
            raise ValueError('V4 channel_count must be 2 or 3')
        if not (self.interval >= 1 and 0 < self.head_rank < 128 and
                self.diagnostic_size > 0 and self.control_size > 0):
            raise ValueError('invalid V4 schedule/rank/split')
        positive = ('difference','coefficient_max','max_change','smoothing',
                    'fisher_penalty','ridge_relative','response_budget','policy_ratio',
                    'policy_floor','policy_ceiling')
        if any(not math.isfinite(getattr(self,k)) or getattr(self,k) <= 0 for k in positive):
            raise ValueError('V4 finite positive settings required')
        if (self.smoothing > 1 or self.coefficient_max < 1 or
                self.difference > self.coefficient_max/2 or self.policy_ratio < 1 or
                self.policy_floor > self.policy_ceiling or not 0 <= self.min_prediction_ratio <= 1 or
                not math.isfinite(self.min_gain) or self.min_gain < 0):
            raise ValueError('invalid V4 trust/acceptance settings')


def transform(g, u, head, coefficients):
    """Same linear operator for EVERY position; output-head derivative unchanged."""
    a,b,c = map(float, coefficients)
    x = g.float()
    result = c*x
    if b != c:
        result = result + (b-c)*((x@u)@u.T)
    if a != b:
        result = result + (a-b)*((x@head)@head.T)
    return result.to(g.dtype)


@contextmanager
def watch(actor, u, head, coefficients, covariance=None, mask=None, total=1):
    def hook(_module, _args, h):
        if not h.requires_grad:
            return
        def backward(g):
            if covariance is not None:
                # Undo the global loss denominator. Never average microbatch
                # means: short/padded answers must not receive extra weight.
                n=mask.shape[1]
                z=(g[:, -n-1:-1].float()[mask.bool()]@u)*total
                covariance.add_(z.T@z)
            return transform(g,u,head,coefficients)
        h.register_hook(backward)
    handle=_final_norm_module(actor.actor_module).register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def gradients(actor, data, u, head, coefficients, covariance=None, capture=True):
    actor.actor_optimizer.zero_grad(set_to_none=True)
    total=int(data.batch['response_mask'].sum())
    if total <= 0:
        raise ValueError('empty V4 training response mask')
    loss_sum=0.
    for part in data.split(actor.config.ppo_micro_batch_size_per_gpu):
        inputs={k:v.to(u.device) for k,v in part.batch.items()}
        with watch(actor,u,head,coefficients,covariance,inputs['response_mask'],total):
            _,z,_,_=forward(actor,inputs,1.,inputs['opd_top16_ids'],diagnostics=False)
            loss=(reverse_kl(z,inputs['opd_teacher_top16_logits'])*inputs['response_mask']).sum()/total
            loss.backward()
        loss_sum+=float(loss.detach())
    if not math.isfinite(loss_sum):
        raise FloatingPointError('nonfinite V4 training loss')
    saved=([None if p.grad is None else p.grad.detach().cpu().clone()
            for p in actor.actor_module.parameters() if p.requires_grad] if capture else None)
    return saved,loss_sum


def combine(direct, full, inside, leading, coefficients):
    """D+a(A-D)+b(U-A)+c(F-U). D includes tied output-head path ONCE.

    The same tied parameter may also have an embedding-input contribution;
    subtracting the zero-hidden pass separates paths, not parameter names.
    """
    a,b,c=map(float,coefficients)
    result=[]
    for ds,fs,us,hs in zip(direct,full,inside,leading):
        terms=[((1-a),ds),(c,fs),(b-c,us),(a-b,hs)]
        prototype=next((x for _,x in terms if x is not None),None)
        if prototype is None:
            result.append(None)
        else:
            value=torch.zeros_like(prototype)
            for weight,x in terms:
                if x is not None and weight != 0:
                    value.add_(x,alpha=weight)
            result.append(value)
    return result


@torch.no_grad()
def readout(actor,data):
    device=next(actor.actor_module.parameters()).device
    zs=[]; qs=[]
    for part in data.split(actor.config.ppo_micro_batch_size_per_gpu):
        inputs={k:v.to(device) for k,v in part.batch.items()}
        _,z,_,_=forward(actor,inputs,1.,inputs['opd_top16_ids'],diagnostics=False)
        mask=inputs['response_mask'].bool()
        zs.append(z[mask].float().cpu()); qs.append(inputs['opd_teacher_top16_logits'][mask].float().cpu())
    z=torch.cat(zs); q=torch.cat(qs)
    if not len(z) or not torch.isfinite(z).all() or not torch.isfinite(q).all():
        raise FloatingPointError('invalid V4 readout')
    return z,q


def response_model(base, teacher, responses):
    """Signed reverse-KL gain and PSD output Fisher metric, NOT its Hessian."""
    lp=base.double().log_softmax(-1); p=lp.exp(); lq=teacher.double().log_softmax(-1)
    residual=lp-lq
    derivative=p*(residual-(p*residual).sum(-1,keepdim=True))
    r=torch.stack(responses,-1).double()
    r=r-(p[...,None]*r).sum(1,keepdim=True)
    s=-(derivative[...,None]*r).sum(1).mean(0)
    fisher=torch.einsum('nk,nkj,nkl->jl',p,r,r)/len(p)
    return s,(fisher+fisher.T)/2


def propose(base,s,fisher,cfg):
    """Exact 3D box-QP active-set enumeration, followed by trust contraction."""
    base=base.double(); s=s.double(); fisher=fisher.double()
    if not torch.isfinite(s).all() or not torch.isfinite(fisher).all():
        raise FloatingPointError('nonfinite V4 local response model')
    dimension=len(base)
    ridge=cfg.ridge_relative*max(float(fisher.trace())/dimension,1e-12)
    matrix=cfg.fisher_penalty*fisher+2*ridge*torch.eye(dimension,dtype=torch.float64)
    lower=torch.maximum(-base,torch.full_like(base,-cfg.max_change))
    upper=torch.minimum(cfg.coefficient_max-base,torch.full_like(base,cfg.max_change))
    best=torch.zeros_like(base); score=0.
    for active in itertools.product((-1,0,1),repeat=dimension):
        fixed=[i for i,v in enumerate(active) if v]; free=[i for i,v in enumerate(active) if not v]
        x=torch.zeros_like(base)
        for i in fixed:
            x[i]=lower[i] if active[i]<0 else upper[i]
        if free:
            rhs=s[free]-matrix[free][:,fixed]@x[fixed]
            x[free]=torch.linalg.solve(matrix[free][:,free],rhs)
        if bool((x<lower-1e-10).any() or (x>upper+1e-10).any()):
            continue
        value=float(s@x-.5*x@matrix@x)
        if value>score:
            score=value; best=x
    best*=cfg.smoothing
    movement=float(.5*best@fisher@best)
    if movement>cfg.response_budget:
        best*=math.sqrt(cfg.response_budget/movement)
    return (base+best).clamp(0,cfg.coefficient_max),float(s@best)


def acceptable(base_loss,candidate_loss,prediction,candidate_policy,incumbent_loss,incumbent_policy,cfg):
    values=(base_loss,candidate_loss,prediction,candidate_policy,incumbent_loss,incumbent_policy)
    if not all(math.isfinite(v) for v in values):
        return False
    cap=min(cfg.policy_ceiling,max(cfg.policy_floor,cfg.policy_ratio*incumbent_policy))
    return (prediction>cfg.min_gain and base_loss-candidate_loss>=cfg.min_gain and
            incumbent_loss-candidate_loss>=cfg.min_gain and
            base_loss-candidate_loss>=cfg.min_prediction_ratio*prediction and candidate_policy<=cap)


def split(data,count):
    if len(data)<=count:
        raise ValueError('V4 requires nonempty disjoint gradient/control/diagnostic sets')
    indices=set(np.linspace(0,len(data)-1,count,dtype=int).tolist())
    return select(data,[i for i in range(len(data)) if i not in indices]),select(data,sorted(indices))


class Runtime:
    def __init__(self,actor,settings=None):
        self.actor=actor; self.cfg=Settings(**dict(settings if settings is not None else actor.config.policy_loss['response_guided']))
        self.cfg.validate(); self.validate_actor()
        source=actor.config.policy_loss['hidden_projection']
        path=Path(source['asset_path'])
        self.basis_hash=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path,allow_pickle=False) as f:
            self.u_cpu=torch.from_numpy(np.array(f['basis'][:,:128],dtype=np.float32,copy=True))
        if self.u_cpu.shape!=(1024,128) or not torch.isfinite(self.u_cpu).all() or not torch.allclose(
                self.u_cpu.T@self.u_cpu,torch.eye(128),atol=2e-3,rtol=0):
            raise ValueError('V4 requires the verified orthonormal historical U128')
        self.q=torch.eye(128); self.coefficients=torch.tensor([1.,1.,0.],dtype=torch.float64)
        self.step=0

    def validate_actor(self):
        a=self.actor; p=a.config.policy_loss
        if getattr(a.config,'strategy','fsdp')!='fsdp':
            raise ValueError('V4 first version supports FSDP1 NO_SHARD only')
        if (int(p.get('jump_reuse_updates',1))!=1 or any(p.get(k,False) for k in
                ('adaptive_reuse_enabled','marginal_reuse_enabled','super_jump_enabled')) or
                any(p.get(k,{}).get('enabled',False) for k in ('bottleneck','response_geometry','plasticity_diagnostic'))):
            raise ValueError('V4 cannot be combined with reuse/bottleneck/passive controllers')
        if not p.get('hidden_projection',{}).get('enabled',False) or int(p['hidden_projection']['rank'])!=128:
            raise ValueError('V4 requires historical hidden_projection rank128 asset')
        if int(p.get('fixed_heldout_probe_size',0))!=self.cfg.diagnostic_size:
            raise ValueError('V4 diagnostic split disagrees with fixed_heldout_probe_size')
        if not isinstance(a.actor_optimizer,torch.optim.AdamW):
            raise TypeError('V4 requires AdamW')
        for group in a.actor_optimizer.param_groups:
            if any(group.get(k,False) for k in ('amsgrad','maximize','differentiable','capturable','fused')):
                raise ValueError('V4 preview requires non-fused standard AdamW')
        if any(p.dtype!=torch.float32 for p in a.actor_module.parameters() if p.requires_grad):
            raise ValueError('V4 requires FP32 master weights')
        if any(isinstance(m,(torch.nn.Dropout,torch.nn.modules.batchnorm._BatchNorm)) and
               (not isinstance(m,torch.nn.Dropout) or m.p>0) for m in a.actor_module.modules()):
            raise ValueError('V4 requires deterministic dropout-free, stateless forwards')
        model=_unwrap_actor_module(a.actor_module)
        config=getattr(model,'config',None)
        if any(float(getattr(config,k,0) or 0)>0 for k in
               ('attention_dropout','hidden_dropout','hidden_dropout_prob','attention_probs_dropout_prob')):
            raise ValueError('V4 requires attention dropout=0')
        if getattr(model,'is_gradient_checkpointing',False):
            raise ValueError('V4 first version requires gradient checkpointing disabled')
        if not math.isfinite(float(a.config.grad_clip)) or float(a.config.grad_clip)<=0:
            raise ValueError('V4 requires positive gradient clipping threshold')

    def calibrate(self,fit,control,diagnostic,u):
        a=self.actor; cfg=self.cfg; counts={'backwards':0,'control_forwards':0,'diagnostic_forwards':0}
        old_head=u@self.q[:,:cfg.head_rank].to(u.device)
        with inert(a) as (parameters,_):
            original=[p.detach().cpu().clone() for p in parameters]
            covariance=torch.zeros((128,128),device=u.device)
            def grad(coef,head=old_head,cov=None):
                counts['backwards']+=1
                return gradients(a,fit,u,head,coef,cov)
            full,loss=grad((1,1,1),cov=covariance)
            _,q=torch.linalg.eigh(covariance.cpu().double())
            q=q.flip(1).float(); head=u@q[:,:cfg.head_rank].to(u.device)
            direct,_=grad((0,0,0))
            inside,_=grad((1,1,0))
            leading=inside if cfg.channel_count==2 and not cfg.paired_controls else grad((1,0,0),head)[0]
            if float(self.coefficients[0])==float(self.coefficients[1]):
                old_leading=leading
            else:
                old_leading,_=grad((1,0,0),old_head)
            # Every preview begins at these same weights and optimizer state.
            a.actor_optimizer.zero_grad(set_to_none=True)
            z0,teacher=readout(a,control); counts['control_forwards']+=1
            control_counts=control.batch['response_mask'].sum(-1).int().tolist()
            def preview(coef,lead=leading):
                g=combine(direct,full,inside,lead,coef)
                delta,norm,clip=adam_preview(parameters,original,g,a.actor_optimizer,a.config.grad_clip)
                del g
                return delta,norm,clip
            def evaluate(coef,lead=leading):
                delta,norm,clip=preview(coef,lead)
                with displaced(parameters,original,delta):
                    z,_=readout(a,control)
                counts['control_forwards']+=1
                record={'teacher_kl':float(reverse_kl(z,teacher).mean()),
                        'answer_loss':row_losses(z,teacher,control_counts).tolist(),
                        'policy_kl':float(reverse_kl(z0,z).mean()),'grad_norm':norm,'clip_factor':clip}
                return z,record
            base_z,base=evaluate(self.coefficients)
            if old_leading is leading:
                incumbent_z,incumbent=base_z,base
            else:
                incumbent_z,incumbent=evaluate(self.coefficients,old_leading)
            responses=[]
            axes=(0,2) if cfg.channel_count==2 else (0,1,2)
            for j in axes:
                coef=self.coefficients.clone()
                eps=cfg.difference if coef[j]+cfg.difference<=cfg.coefficient_max else -cfg.difference
                coef[j]+=eps
                if cfg.channel_count==2 and j==0:
                    coef[1]=coef[0]
                zj,_=evaluate(coef)
                responses.append((zj-base_z)/eps)
            s,fisher=response_model(base_z,teacher,responses)
            candidate,prediction=propose(self.coefficients[list(axes)],s,fisher,cfg)
            if cfg.channel_count==2:
                candidate=candidate[[0,0,1]]
            if torch.equal(candidate,self.coefficients):
                candidate_z,candidate_record=base_z,base
            else:
                candidate_z,candidate_record=evaluate(candidate)
            accepted=acceptable(base['teacher_kl'],candidate_record['teacher_kl'],prediction,
                candidate_record['policy_kl'],incumbent['teacher_kl'],incumbent['policy_kl'],cfg)
            chosen=candidate if accepted else self.coefficients.clone()
            chosen_q=q if accepted or old_leading is leading else self.q.clone()
            chosen_lead=leading if accepted else old_leading
            # Independent diagnostic is read ONLY AFTER selection; never an
            # input to propose/acceptable, and never receives a gradient.
            diag0,diagt=readout(a,diagnostic); counts['diagnostic_forwards']+=1
            diag_counts=diagnostic.batch['response_mask'].sum(-1).int().tolist()
            diag_before=row_losses(diag0,diagt,diag_counts)
            controls={}
            candidates=[('selected',chosen,chosen_lead,1.)]
            if cfg.paired_controls:
                candidates += [('r128',(1,1,0),leading,1.),
                               ('uniform_half',(1,1,0),leading,.5),('uniform_double',(1,1,0),leading,2.),
                               ('full',(1,1,1),leading,1.),
                               ('fixed_nonuniform',(1.25,.75,0),leading,1.)]
            # Keep at most ONE preview displacement; reuse it for ray scales.
            # Reuse equal selected/r128 scores rather than rereading diagnostic.
            last_key=None; evaluated={}
            for name,coef,lead,scale in candidates:
                key=(tuple(map(float,coef)),id(lead))
                score_key=(key,scale)
                if score_key in evaluated:
                    controls[name]=dict(evaluated[score_key]);continue
                if key!=last_key:
                    delta,_,_=preview(coef,lead);last_key=key
                with displaced(parameters,original,delta,scale):
                    zd,_=readout(a,diagnostic); counts['diagnostic_forwards']+=1
                controls[name]={'teacher_kl':float(reverse_kl(zd,diagt).mean()),
                                'policy_kl':float(reverse_kl(diag0,zd).mean()),
                                'gain':float((reverse_kl(diag0,diagt)-reverse_kl(zd,diagt)).mean())}
                controls[name]['answers']=statistics(diag_before,row_losses(zd,diagt,diag_counts))
                evaluated[score_key]=controls[name]
            record={'step':self.step+1,'accepted':accepted,'previous_coefficients':self.coefficients.tolist(),
                'candidate_coefficients':candidate.tolist(),'coefficients':chosen.tolist(),
                'predicted_gain':prediction,'actual_candidate_gain':base['teacher_kl']-candidate_record['teacher_kl'],
                'control_before_kl':float(reverse_kl(z0,teacher).mean()),'control_incumbent':incumbent,
                'control_new_basis_base':base,'control_candidate':candidate_record,
                'diagnostic_before_kl':float(reverse_kl(diag0,diagt).mean()),'diagnostic_controls':controls,
                'signed_gain':s.tolist(),'fisher':fisher.tolist(),**counts}
            # No second full backward for committing the chosen candidate.
            commit=combine(direct,full,inside,chosen_lead,chosen)
        return commit,loss,chosen,chosen_q,record

    def update(self,data):
        step=int(data.meta_info['global_steps'])
        if step!=self.step+1:
            raise ValueError('V4 checkpoint state/update counter mismatch; refusing silent restart')
        if float(data.meta_info['temperature'])!=1.:
            raise ValueError('V4 fixes temperature=1')
        cfg=self.cfg; a=self.actor; started=time.perf_counter()
        u=self.u_cpu.to(next(a.actor_module.parameters()).device)
        train,diagnostic=split(data,cfg.diagnostic_size)
        calibration=step%cfg.interval==0
        coefficients=self.coefficients.clone(); q=self.q.clone(); record={}
        try:
            if calibration:
                fit,control=split(train,cfg.control_size)
                print(f'V4 calibration {step}: gradient={len(fit)}, control={len(control)}, '
                      f'independent={len(diagnostic)}; temporary AdamW previews, one final update.',flush=True)
                commit,loss,coefficients,q,record=self.calibrate(fit,control,diagnostic,u)
                for p,g in zip((p for p in a.actor_module.parameters() if p.requires_grad),commit):
                    p.grad=None if g is None else g.to(p.device)
                del commit
            else:
                fit=train
                head=u@q[:,:cfg.head_rank].to(u.device)
                _,loss=gradients(a,fit,u,head,coefficients,capture=False)
            # Fail BEFORE the irreversible optimizer call on bad gradients.
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in a.actor_module.parameters()):
                raise FloatingPointError('nonfinite V4 gradient; no optimizer update committed')
            optimizer_started=time.perf_counter()
            norm=float(a._optimizer_step())
            if not math.isfinite(norm):
                raise FloatingPointError('V4 optimizer skipped nonfinite update')
            optimizer_seconds=time.perf_counter()-optimizer_started
            self.step=step; self.coefficients=coefficients; self.q=q
        finally:
            a.actor_optimizer.zero_grad(set_to_none=True)
        elapsed=time.perf_counter()-started
        metrics={'actor/opd_top16_kl':loss,'actor/grad_norm':norm,'actor/opd_optimizer_steps':1.,
            'actor/top16_support':16.,'actor/entropy':0.,'monitor/top16_entropy_not_measured':1.,
            'monitor/top16_direct_kl_no_advantages':1.,'monitor/teacher_present':1.,
            'monitor/hidden_gradient_projection':1.,'actor/hidden_projection_rank':128.,
            'monitor/jump_rollout_reuse_factor':1.,'monitor/jump_stale_trajectory_updates':0.,
            'v4/calibration':float(calibration),'v4/train_responses':len(fit),
            'v4/control_responses':cfg.control_size if calibration else 0,
            'v4/diagnostic_responses':len(diagnostic),'v4/rollout_responses':len(data),
            'v4/coefficient_a':float(coefficients[0]),'v4/coefficient_b':float(coefficients[1]),
            'v4/coefficient_c':float(coefficients[2]),
            'v4/complement_enabled':float(coefficients[2]>0),
            'monitor/opd_train_trajectories':float(len(fit)),
            'monitor/opd_heldout_trajectories':float(len(data)-len(fit)),
            'v4/backward_passes':record.get('backwards',1),
            'v4/control_forward_passes':record.get('control_forwards',0),
            'v4/diagnostic_forward_passes':record.get('diagnostic_forwards',0),
            'timing/v4_update_seconds':elapsed,'timing/opd_optimizer_seconds':optimizer_seconds}
        if calibration:
            print(f'V4 calibration {step}: accepted={record["accepted"]}, '
                  f'coefficients={coefficients.tolist()}, total_seconds={elapsed:.2f}',flush=True)
            metrics.update({'v4/accepted':float(record['accepted']),'v4/predicted_gain':record['predicted_gain'],
                'v4/actual_candidate_gain':record['actual_candidate_gain'],
                'v4/diagnostic_gain':record['diagnostic_controls']['selected']['gain'],
                'v4/diagnostic_policy_kl':record['diagnostic_controls']['selected']['policy_kl']})
            record['elapsed_seconds']=elapsed
            if cfg.output_dir:
                folder=Path(cfg.output_dir); folder.mkdir(parents=True,exist_ok=True)
                target=folder/f'calibration_{step:06d}.json'
                temporary=target.with_suffix('.tmp')
                temporary.write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')
                temporary.replace(target)
        return {key:[value] for key,value in metrics.items()}

    def state_dict(self):
        return dict(schema='response-guided-v4-1',settings=asdict(self.cfg),basis_hash=self.basis_hash,
                    step=self.step,q=self.q,coefficients=self.coefficients)

    def load_state_dict(self,state):
        if (state['schema']!='response-guided-v4-1' or state['settings']!=asdict(self.cfg) or
                state['basis_hash']!=self.basis_hash):
            raise ValueError('V4 checkpoint protocol/basis mismatch')
        q=state['q']; c=state['coefficients']; step=state['step']
        if (q.shape!=(128,128) or c.shape!=(3,) or not torch.isfinite(q).all() or
                not torch.isfinite(c).all() or not torch.allclose(q.T@q,torch.eye(128),atol=2e-4,rtol=0) or
                bool((c<0).any() or (c>self.cfg.coefficient_max).any()) or not isinstance(step,int) or step<0):
            raise ValueError('invalid V4 checkpoint geometry/counter')
        self.q=q.clone(); self.coefficients=c.clone().double(); self.step=step


def get_runtime(actor):
    if not actor.config.policy_loss.get('response_guided',{}).get('enabled',False):
        return None
    if not hasattr(actor,'_response_guided'):
        actor._response_guided=Runtime(actor)
    return actor._response_guided


def save_state(actor,directory):
    runtime=get_runtime(actor)
    if runtime is not None:
        target=Path(directory)/'response_guided_state.pt'
        temporary=target.with_suffix('.tmp'); torch.save(runtime.state_dict(),temporary); temporary.replace(target)


def load_state(actor,directory):
    runtime=get_runtime(actor)
    if runtime is not None:
        runtime.load_state_dict(torch.load(Path(directory)/'response_guided_state.pt',map_location='cpu',weights_only=True))
