"""V5 cross-state transfer prototype and matched short-run controls.

Optimizer VJP is analytic (including global gradient clipping). Hidden JVP is
a two-resolution central finite difference through the official FSDP forward.
This is a checked approximation, NOT an exact differentiable BF16 optimizer.
"""
from contextlib import contextmanager
from dataclasses import dataclass,asdict
from pathlib import Path
import json
import math
import time

import numpy as np
import torch

from . import opd_response_guided as v4
from .opd_top16 import forward,reverse_kl,_final_norm_module
from .opd_response_geometry import inert,displaced,adam_preview,sqnorm
from .opd_transfer_stats import answer_means,row_losses,statistics,lower_tail


@dataclass
class Settings:
    enabled: bool = False
    mode: str = 'v5'
    calibration_steps: tuple = (5,15,25)
    diagnostic_size: int = 32
    control_size: int = 32
    discovered_modes: int = 4
    operator_gain: float = .25
    history_weight: float = .25
    jvp_step_ratio: float = 1.
    jvp_relative_error: float = .5
    jvp_min_cosine: float = .8
    min_gain: float = .00001
    min_prediction_ratio: float = .1
    tail_weight: float = .5
    worst_extra_loss: float = .005
    policy_ratio: float = 1.25
    policy_floor: float = .0001
    policy_ceiling: float = .01
    max_calibration_seconds: float = 240.
    paired_v4_steps: tuple = (5,25)
    output_dir: str = ''

    def validate(self):
        if self.mode not in ('full_matched','r128_matched','r128_standard','v4','v4_two','v5'):
            raise ValueError('unknown transfer-suite arm')
        for key in ('calibration_steps','paired_v4_steps'):
            values=list(getattr(self,key))
            if values!=sorted(set(values)) or any(not isinstance(x,int) or x<1 for x in values):
                raise ValueError('invalid calibration schedule')
        if not 1<=self.discovered_modes<=4 or min(self.control_size,self.diagnostic_size)<1:
            raise ValueError('V5 first prototype caps operator adjustment rank at eight')
        if not 0<=self.history_weight<1 or not 0<self.operator_gain<=.5:
            raise ValueError('invalid V5 history/gain')
        if not 0<=self.jvp_min_cosine<=1 or not 0<=self.min_prediction_ratio<=1:
            raise ValueError('invalid V5 reliability thresholds')
        for key in ('jvp_step_ratio','jvp_relative_error','min_gain','tail_weight','worst_extra_loss',
                    'policy_ratio','policy_floor','policy_ceiling','max_calibration_seconds'):
            if not math.isfinite(getattr(self,key)) or getattr(self,key)<=0:
                raise ValueError('V5 finite positive settings required: '+key)


def optimizer_vjp(parameters,gradients,optimizer,clip,output_cotangent):
    """(d AdamWStep / d raw_grad)^T v, frozen weights/history, CPU streamed.

    Includes the non-diagonal rank-one derivative of GLOBAL clipping. The
    derivative treats the real-valued Adam formula, not FP32/BF16 rounding.
    No optimizer history is modified. Weight decay has zero derivative here.
    """
    norm=sqnorm(gradients)**.5
    factor=min(1.,float(clip)/(norm+1e-6))
    groups={id(p):group for group in optimizer.param_groups for p in group['params']}
    raw=[]; dot=0.
    for p,g,h in zip(parameters,gradients,output_cotangent):
        if g is None:
            raw.append(None);continue
        group=groups[id(p)];state=optimizer.state.get(p,{})
        b1,b2=group['betas'];t=float(state['step'])+1 if state else 1
        x=g.double()*factor
        m=state['exp_avg'].detach().cpu().double()*b1 if state else torch.zeros_like(x)
        v=state['exp_avg_sq'].detach().cpu().double()*b2 if state else torch.zeros_like(x)
        m.add_(x,alpha=1-b1);v.addcmul_(x,x,value=1-b2)
        root=v.sqrt();bias2=math.sqrt(1-b2**t)
        denominator=root/bias2+float(group['eps'])
        derivative=(1-b1)/denominator
        derivative-=m*(1-b2)*x/(bias2*root.clamp_min(1e-300)*denominator.square())
        derivative*=-float(group['lr'])/(1-b1**t)
        value=derivative*(torch.zeros_like(x) if h is None else h.double())
        dot+=float((value*g.double()).sum())
        raw.append(value.float())
    cross=float(clip)*dot/(norm*(norm+1e-6)**2) if factor<1 and norm>0 else 0.
    result=[None if v is None else factor*v-cross*g for v,g in zip(raw,gradients)]
    if any(x is not None and not torch.isfinite(x).all() for x in result):
        raise FloatingPointError('nonfinite optimizer VJP')
    return result


def operator(g,u,v=None,matrix=None,full=False):
    if full:return g
    x=g.float();out=(x@u)@u.T
    if v is not None and v.shape[1]:out+=((x@v)@matrix)@v.T
    return out.to(g.dtype)


@contextmanager
def watch(actor,mask,u,v,matrix,raw=None,full=False):
    def hook(_m,_a,h):
        def back(g):
            if raw is not None:
                n=mask.shape[1]
                raw.append(g[:, -n-1:-1][mask.bool()].detach().float().cpu())
            return operator(g,u,v,matrix,full)
        h.register_hook(back)
    handle=_final_norm_module(actor.actor_module).register_forward_hook(hook)
    try:yield
    finally:handle.remove()


def gradient(actor,data,u,v=None,matrix=None,raw=None,full=False,answer_equal=False,capture=True,logits=None):
    actor.actor_optimizer.zero_grad(set_to_none=True)
    total=int(data.batch['response_mask'].sum());value=0.
    if total<1 or bool((data.batch['response_mask'].sum(-1)<=0).any()):
        raise ValueError('V5 requires nonempty answer masks')
    for part in data.split(actor.config.ppo_micro_batch_size_per_gpu):
        inputs={k:x.to(u.device) for k,x in part.batch.items()};mask=inputs['response_mask']
        with watch(actor,mask,u,v,matrix,raw,full):
            _,z,_,_=forward(actor,inputs,1.,inputs['opd_top16_ids'],diagnostics=False)
            if logits is not None:logits.append(z[mask.bool()].detach().float().cpu())
            losses=reverse_kl(z,inputs['opd_teacher_top16_logits'])*mask
            loss=((losses.sum(-1)/mask.sum(-1)).sum()/len(data) if answer_equal else losses.sum()/total)
            loss.backward();value+=float(loss.detach())
    if not math.isfinite(value):raise FloatingPointError('nonfinite V5 loss')
    grads=([None if p.grad is None else p.grad.detach().cpu().clone()
            for p in actor.actor_module.parameters() if p.requires_grad] if capture else None)
    return grads,value


@torch.no_grad()
def hidden(actor,data):
    values=[];device=next(actor.actor_module.parameters()).device
    for part in data.split(actor.config.ppo_micro_batch_size_per_gpu):
        inputs={k:x.to(device) for k,x in part.batch.items()};mask=inputs['response_mask'].bool();n=mask.shape[1]
        def capture(_m,_a,h):values.append(h[:, -n-1:-1][mask].float().cpu())
        handle=_final_norm_module(actor.actor_module).register_forward_hook(capture)
        try:forward(actor,inputs,1.,inputs['opd_top16_ids'],diagnostics=False)
        finally:handle.remove()
    return values


def signed_matrix(raw,plus,minus,factor,device):
    if not raw or len(raw)!=len(plus) or len(raw)!=len(minus):raise ValueError('JVP microbatch alignment changed')
    s=torch.zeros((raw[0].shape[-1],raw[0].shape[-1]),device=device)
    for g,hp,hm in zip(raw,plus,minus):
        if g.shape!=hp.shape or g.shape!=hm.shape:raise ValueError('JVP token alignment changed')
        for gg,dp,dm in zip(g.split(1024),hp.split(1024),hm.split(1024)):
            q=(dp.to(device)-dm.to(device))*factor
            product=gg.to(device).T@q
            s.add_(product+product.T,alpha=-.5)
    return s.cpu()


def transfer_matrix(actor,data,u,parameters,original,base_delta,base_grad,control_grad,raw,cfg):
    w=optimizer_vjp(parameters,base_grad,actor.actor_optimizer,actor.config.grad_clip,control_grad)
    length=sqnorm(w)**.5
    if not math.isfinite(length) or length<=1e-30:
        return torch.zeros((u.shape[0],u.shape[0])),dict(valid=False,reason='zero_optimizer_response',hidden_forwards=0)
    direction=[None if x is None else x/length for x in w];del w
    eps=max(sqnorm(base_delta)**.5*cfg.jvp_step_ratio,1e-4)
    matrices=[]
    for distance in (eps,eps/2):
        with displaced(parameters,original,direction,distance):plus=hidden(actor,data)
        with displaced(parameters,original,direction,-distance):minus=hidden(actor,data)
        matrices.append(signed_matrix(raw,plus,minus,length/(2*distance),u.device))
        del plus,minus
    coarse,fine=matrices;den=float(fine.norm())
    if not bool(torch.isfinite(coarse).all() and torch.isfinite(fine).all()):
        return torch.zeros_like(fine),dict(valid=False,reason='nonfinite_hidden_finite_difference',epsilon=eps,
                                           hidden_forwards=4)
    error=float((fine-coarse).norm())/max(den,1e-30)
    cosine=float((fine*coarse).sum())/max(float(fine.norm()*coarse.norm()),1e-30)
    valid=den>1e-20 and math.isfinite(error) and error<=cfg.jvp_relative_error and cosine>=cfg.jvp_min_cosine
    return fine,dict(valid=valid,reason='passed' if valid else 'finite_difference_resolution_failed',
        epsilon=eps,relative_error=error,cosine=cosine,optimizer_cotangent_norm=length,hidden_forwards=4)


def candidate_operators(s,u,modes=4,gain=.25):
    """Discover signed directions, not gradient-PCA bins. Adjustment rank <=8.

    Split discovered vectors into U and U-perp components; their joint span is
    invariant under P_U. PSD clipping in this small span therefore guarantees
    0<=T<=2 without a dense 1024D optimizer or arbitrary negative supervision.
    """
    s=s.double();u=u.double();values,vectors=torch.linalg.eigh((s+s.T)/2)
    chosen=vectors[:,values.abs().argsort(descending=True)[:modes]]
    inside=u@(u.T@chosen);outside=chosen-inside
    results=[]
    for name,z in (('inside',inside),('outside',outside),('mixed',torch.cat((inside,outside),1))):
        left,singular,_=torch.linalg.svd(z,full_matrices=False)
        z=left[:,singular>1e-6]
        if not z.shape[1]:continue
        base=(u.T@z).T@(u.T@z);local=z.T@s@z
        scale=max(float(values.abs().max()),1e-30)
        proposed=(base+gain*local/scale);lam,q=torch.linalg.eigh((proposed+proposed.T)/2)
        adjustment=(q*lam.clamp(0,2))@q.T-base
        if float(adjustment.norm())<1e-8:continue
        results.append((name,z.float(),adjustment.float()))
    return results,values.float()


def block_statistics(s,u):
    u=u.double();s=s.double();inside=u.T@s@u
    cross=s@u-u@inside
    outside=s-u@(u.T@s)-(s@u)@u.T+u@inside@u.T
    total=float(s.square().sum())
    return dict(inside_energy=float(inside.square().sum()),outside_energy=float(outside.square().sum()),
                cross_energy=2*float(cross.square().sum()),total_energy=total)


class Panel:
    def __init__(self,batch):self.batch=batch
    def __len__(self):return len(self.batch['responses'])
    def split(self,count):return [Panel({k:v[i:i+count] for k,v in self.batch.items()}) for i in range(0,len(self),count)]


class Runtime:
    def __init__(self,actor):
        self.actor=actor;self.cfg=Settings(**dict(actor.config.policy_loss['transfer_guided']));self.cfg.validate()
        self.cfg.calibration_steps=tuple(self.cfg.calibration_steps);self.cfg.paired_v4_steps=tuple(self.cfg.paired_v4_steps)
        self.controller=v4.Runtime(actor,dict(enabled=True,diagnostic_size=self.cfg.diagnostic_size,
            control_size=self.cfg.control_size,paired_controls=True,channel_count=2 if self.cfg.mode=='v4_two' else 3))
        self.u_cpu=self.controller.u_cpu;self.step=0;self.v=torch.empty(1024,0);self.matrix=torch.empty(0,0)
        self.previous_s=None;self.pending=[]

    def write(self,name,record):
        if not self.cfg.output_dir:return
        folder=Path(self.cfg.output_dir);folder.mkdir(parents=True,exist_ok=True)
        target=folder/name;temporary=target.with_suffix('.tmp')
        temporary.write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8');temporary.replace(target)

    def calibrate(self,fit,control,diagnostic,u):
        a=self.actor;cfg=self.cfg;started=time.perf_counter();backwards=0;control_forwards=0;diag_forwards=0
        with inert(a) as (parameters,_):
            original=[p.detach().cpu().clone() for p in parameters];raw=[]
            base_grad,loss=gradient(a,fit,u,raw=raw);backwards+=1
            base_delta,_,_=adam_preview(parameters,original,base_grad,a.actor_optimizer,a.config.grad_clip)
            c0,teacher=v4.readout(a,control);control_forwards+=1
            counts=control.batch['response_mask'].sum(-1).int().tolist()
            initial=row_losses(c0,teacher,counts)
            with displaced(parameters,original,base_delta):
                cached=[]
                control_grad,_=gradient(a,control,u,full=True,answer_equal=True,logits=cached);backwards+=1
                cb=torch.cat(cached);del cached
            a.actor_optimizer.zero_grad(set_to_none=True)
            base_loss=row_losses(cb,teacher,counts)
            base_policy=float(answer_means(reverse_kl(c0,cb),counts).mean())
            s,gate=transfer_matrix(a,fit,u,parameters,original,base_delta,base_grad,control_grad,raw,cfg)
            del control_grad,raw
            block=block_statistics(s,self.u_cpu)
            discovery=s.clone();history_cosine=None
            if self.previous_s is not None:
                history_cosine=float((s*self.previous_s).sum())/max(float(s.norm()*self.previous_s.norm()),1e-30)
                discovery=(1-cfg.history_weight)*s+cfg.history_weight*self.previous_s*(s.norm()/self.previous_s.norm().clamp_min(1e-30))
            if gate['valid']:
                candidates,spectrum=candidate_operators(discovery,self.u_cpu,cfg.discovered_modes,cfg.operator_gain)
            else:candidates,spectrum=[],torch.linalg.eigvalsh(discovery.double()).float()
            best_grad=base_grad;best_v=torch.empty(1024,0);best_matrix=torch.empty(0,0);best_score=0.;best_name='r128'
            records=[];responses=[]
            for name,vc,mc in candidates:
                if not gate['valid'] or time.perf_counter()-started>cfg.max_calibration_seconds:break
                prediction=float(((vc.T@s@vc)*mc).sum())
                if prediction<=cfg.min_gain:
                    records.append(dict(name=name,accepted=False,reason='nonpositive_predicted_mean_gain',prediction=prediction));continue
                cg,_=gradient(a,fit,u,vc.to(u.device),mc.to(u.device));backwards+=1
                delta,_,_=adam_preview(parameters,original,cg,a.actor_optimizer,a.config.grad_clip)
                with displaced(parameters,original,delta):cz,_=v4.readout(a,control)
                control_forwards+=1;cl=row_losses(cz,teacher,counts)
                stats=statistics(initial,cl,base_loss);policy=float(answer_means(reverse_kl(c0,cz),counts).mean())
                score=stats['mean_extra_gain']+cfg.tail_weight*stats['lower_tail_extra']
                cap=min(cfg.policy_ceiling,max(cfg.policy_floor,cfg.policy_ratio*base_policy))
                accepted=(score>cfg.min_gain and stats['mean_extra_gain']>=cfg.min_prediction_ratio*prediction and
                          min(stats['extra_gain'])>=-cfg.worst_extra_loss and policy<=cap)
                records.append(dict(name=name,accepted=accepted,prediction=prediction,policy_kl=policy,robust_score=score,answers=stats))
                responses.append((cz-cb).float())
                if accepted and score>best_score:
                    best_grad=cg;best_v=vc;best_matrix=mc;best_score=score;best_name=name
                del delta
            # Selection is finished. Independent data cannot change best_*.
            dz0,dteacher=v4.readout(a,diagnostic);diag_forwards+=1
            dcounts=diagnostic.batch['response_mask'].sum(-1).int().tolist();dbefore=row_losses(dz0,dteacher,dcounts)
            independent={}
            def measure(name,g,scale=1.):
                nonlocal diag_forwards
                dd,_,_=adam_preview(parameters,original,g,a.actor_optimizer,a.config.grad_clip)
                with displaced(parameters,original,dd,scale):zz,_=v4.readout(a,diagnostic)
                diag_forwards+=1
                independent[name]=dict(after=row_losses(zz,dteacher,dcounts).tolist(),
                    policy_kl=float(answer_means(reverse_kl(dz0,zz),dcounts).mean()))
            measure('r128',base_grad)
            if best_name=='r128':independent['selected']=dict(independent['r128'])
            else:measure('selected',best_grad)
            measure('uniform_half',base_grad,.5);measure('uniform_double',base_grad,2.)
            fg,_=gradient(a,fit,u,full=True);backwards+=1;measure('full',fg);del fg
            for name,item in independent.items():
                item['answers']=statistics(dbefore,item['after'],independent['r128']['after'])
            paired=None
            if self.step+1 in cfg.paired_v4_steps:
                # Local V4 probe at exactly these same weights/state/data, NOT
                # the trajectory of an independently trained V4 model.
                local=v4.Runtime(a,dict(enabled=True,diagnostic_size=cfg.diagnostic_size,
                    control_size=cfg.control_size,paired_controls=False))
                local.step=self.step
                _,_,_,_,paired=local.calibrate(fit,control,diagnostic,u)
                backwards+=paired['backwards'];control_forwards+=paired['control_forwards'];diag_forwards+=paired['diagnostic_forwards']
            overlap=None
            if responses:
                _,gram=v4.response_model(cb,teacher,responses)
                diagonal=gram.diag().clamp_min(1e-30).sqrt()
                overlap=(gram/(diagonal[:,None]*diagonal[None,:])).tolist()
            record=dict(schema='v5-calibration-1',step=self.step+1,selected=best_name,gate=gate,
                signed_transfer_eigenvalues=torch.linalg.eigvalsh(s.double()).tolist(),
                discovery_eigenvalues=spectrum.tolist(),blocks=block,history_cosine=history_cosine,
                candidates=records,independent=independent,paired_v4=paired,response_overlap=overlap,
                response_overlap_names=[x['name'] for x in records if 'answers' in x],
                control_before=initial.tolist(),control_r128=base_loss.tolist(),diagnostic_before=dbefore.tolist(),
                backward_passes=backwards,control_forward_passes=control_forwards,
                diagnostic_forward_passes=diag_forwards,hidden_jvp_forward_passes=gate['hidden_forwards'],
                operator_adjustment_rank=best_v.shape[1],calibration_seconds=time.perf_counter()-started,
                budget_exhausted=time.perf_counter()-started>cfg.max_calibration_seconds)
            if cfg.output_dir:
                folder=Path(cfg.output_dir);folder.mkdir(parents=True,exist_ok=True)
                path=folder/f'transfer_{self.step+1:06d}.npz'
                with path.with_suffix('.tmp').open('wb') as f:
                    np.savez_compressed(f,signed_transfer=s.numpy(),directions=best_v.numpy(),adjustment=best_matrix.numpy())
                path.with_suffix('.tmp').replace(path)
        return best_grad,loss,best_v,best_matrix,s if gate['valid'] else self.previous_s,record

    def remember(self,diagnostic,before,after):
        self.pending.append(dict(anchor=self.step,batch={k:v.detach().cpu().clone() for k,v in diagnostic.batch.items()},
                                 before=list(before),after=list(after),remaining=[1,2,4]))

    def retention(self):
        kept=[];forwards=0
        for item in self.pending:
            lag=self.step-item['anchor']
            if lag in item['remaining']:
                with inert(self.actor):z,q=v4.readout(self.actor,Panel(item['batch']))
                forwards+=1
                counts=item['batch']['response_mask'].sum(-1).int().tolist();after=row_losses(z,q,counts)
                self.write(f'retention_{item["anchor"]:06d}_{lag}.json',dict(anchor=item['anchor'],lag=lag,
                    note='Observed subsequent trajectory with new batches, not causal attribution to anchor alone',
                    from_before=statistics(item['before'],after),from_immediate_after=statistics(item['after'],after)))
                item['remaining'].remove(lag)
            if item['remaining']:kept.append(item)
        self.pending=kept
        return forwards

    def update(self,data):
        a=self.actor;cfg=self.cfg;step=int(data.meta_info['global_steps']);started=time.perf_counter()
        if step!=self.step+1:raise ValueError('transfer-suite resume counter mismatch')
        train,diag=v4.split(data,cfg.diagnostic_size);calibration=step in cfg.calibration_steps
        fit,control=(v4.split(train,cfg.control_size) if calibration and cfg.mode!='r128_standard' else (train,None))
        u=self.u_cpu.to(next(a.actor_module.parameters()).device);record={};calibration_before=None;calibration_after=None
        # New on-policy state distribution, before update; no extra forward.
        fresh={}
        if 'opd_student_top16_logits' in diag.batch:
            counts=diag.batch['response_mask'].sum(-1).int().tolist();mask=diag.batch['response_mask'].bool()
            per=row_losses(diag.batch['opd_student_top16_logits'][mask],diag.batch['opd_teacher_top16_logits'][mask],counts)
            fresh=dict(answer_kl=per.tolist(),mean=float(per.mean()),note='Fresh on-policy distribution, not paired counterfactual')
        next_v=self.v;next_m=self.matrix;next_s=self.previous_s;next_c=self.controller.coefficients;next_q=self.controller.q
        try:
            if calibration:
                print(f'Transfer suite {cfg.mode} step {step}: fit={len(fit)}, control={0 if control is None else len(control)}, independent={len(diag)}',flush=True)
            if cfg.mode=='v5' and calibration:
                commit,loss,next_v,next_m,next_s,record=self.calibrate(fit,control,diag,u)
                calibration_before=record['diagnostic_before'];calibration_after=record['independent']['selected']['after']
            elif cfg.mode in ('v4','v4_two') and calibration:
                self.controller.step=self.step
                commit,loss,next_c,next_q,record=self.controller.calibrate(fit,control,diag,u)
                answers=record['diagnostic_controls']['selected']['answers']
                calibration_before=answers['before'];calibration_after=answers['after']
            else:
                if calibration:
                    with inert(a):zd,qt=v4.readout(a,diag)
                    counts=diag.batch['response_mask'].sum(-1).int().tolist()
                    calibration_before=row_losses(zd,qt,counts).tolist()
                if cfg.mode in ('v4','v4_two'):
                    head=u@self.controller.q[:,:16].to(u.device)
                    _,loss=v4.gradients(a,fit,u,head,self.controller.coefficients,capture=False)
                else:
                    _,loss=gradient(a,fit,u,self.v.to(u.device),self.matrix.to(u.device),
                        full=cfg.mode=='full_matched',capture=False)
                commit=None
            if commit is not None:
                for p,g in zip((p for p in a.actor_module.parameters() if p.requires_grad),commit):
                    p.grad=None if g is None else g.to(p.device)
                del commit
            finite=[torch.isfinite(p.grad).all() for p in a.actor_module.parameters() if p.grad is not None]
            if not finite or not bool(torch.stack(finite).all()):raise FloatingPointError('nonfinite transfer-suite gradients')
            norm=float(a._optimizer_step())
            if not math.isfinite(norm):raise FloatingPointError('transfer-suite optimizer skipped update')
            self.step=step;self.v=next_v;self.matrix=next_m;self.previous_s=next_s
            self.controller.coefficients=next_c;self.controller.q=next_q;self.controller.step=step
        finally:a.actor_optimizer.zero_grad(set_to_none=True)
        if calibration:
            # Verify the ACTUAL committed output, not just preview output.
            with inert(a):actual,qt=v4.readout(a,diag)
            counts=diag.batch['response_mask'].sum(-1).int().tolist();actual_loss=row_losses(actual,qt,counts)
            if calibration_after is not None:
                record['preview_commit_max_answer_kl_error']=float((actual_loss-torch.tensor(calibration_after)).abs().max())
                record['preview_commit_verified']=record['preview_commit_max_answer_kl_error']<=1e-4
                if not record['preview_commit_verified']:
                    self.write(f'FAILED_COMMIT_{step:06d}.json',record)
                    raise RuntimeError('actual commit differs from candidate preview; stop instead of silently continuing')
            calibration_after=actual_loss.tolist()
            record.update(step=step,mode=cfg.mode,train_answers=len(fit),control_answers=0 if control is None else len(control),
                actual_committed_answers=statistics(calibration_before,calibration_after))
            self.remember(diag,calibration_before,calibration_after)
        retention_forwards=self.retention()
        elapsed=time.perf_counter()-started
        if calibration:
            record['total_update_seconds']=elapsed
            record['committed_diagnostic_forwards']=1
            record['diagnostic_gain_per_update_second']=record['actual_committed_answers']['mean_gain']/max(elapsed,1e-9)
            self.write(f'calibration_{step:06d}.json',record)
        self.write(f'fresh_{step:06d}.json',dict(step=step,mode=cfg.mode,**fresh))
        metrics={'actor/opd_optimizer_steps':1.,'actor/opd_top16_kl':loss,'actor/grad_norm':norm,
            'actor/top16_support':16.,'monitor/top16_entropy_not_measured':1.,'monitor/teacher_present':1.,
            'monitor/opd_train_trajectories':len(fit),'monitor/opd_heldout_trajectories':len(data)-len(fit),
            'monitor/hidden_gradient_projection':float(cfg.mode!='full_matched'),
            'transfer/calibration':float(calibration),'transfer/update_seconds':elapsed,
            'transfer/retention_forwards':retention_forwards,
            'transfer/backward_passes':record.get('backward_passes',record.get('backwards',1)),
            'transfer/operator_adjustment_rank':self.v.shape[1],
            'transfer/v4_a':float(next_c[0]),'transfer/v4_b':float(next_c[1]),'transfer/v4_c':float(next_c[2])}
        if calibration:
            metrics.update({'transfer/coverage':record['actual_committed_answers']['coverage'],
                            'transfer/negative_transfer':record['actual_committed_answers']['negative_transfer'],
                            'transfer/independent_gain':record['actual_committed_answers']['mean_gain']})
        if cfg.mode=='v5' and calibration:
            metrics['transfer/jvp_gate_passed']=float(record['gate']['valid'])
            metrics['transfer/selected_nonbaseline']=float(record['selected']!='r128')
        return {k:[v] for k,v in metrics.items()}

    def state_dict(self):
        return dict(schema='transfer-guided-v5-1',settings=asdict(self.cfg),step=self.step,v=self.v,matrix=self.matrix,
                    previous_s=self.previous_s,pending=self.pending,controller=self.controller.state_dict())

    def load_state_dict(self,state):
        if state['schema']!='transfer-guided-v5-1' or state['settings']!=asdict(self.cfg):raise ValueError('V5 checkpoint protocol mismatch')
        self.controller.load_state_dict(state['controller'])
        v=state['v'];m=state['matrix']
        if v.ndim!=2 or v.shape[0]!=1024 or v.shape[1]>8 or m.shape!=(v.shape[1],v.shape[1]) or not torch.isfinite(v).all() or not torch.isfinite(m).all():
            raise ValueError('invalid V5 saved operator')
        if state['step']!=self.controller.step:raise ValueError('V5/controller counter mismatch')
        if v.shape[1]:
            t=self.u_cpu@self.u_cpu.T+v@m@v.T;eig=torch.linalg.eigvalsh(t.double())
            if float(eig.min()) < -1e-5 or float(eig.max())>2.00001:raise ValueError('saved V5 operator is not bounded PSD')
        self.step=state['step'];self.v=v;self.matrix=m;self.previous_s=state['previous_s'];self.pending=state['pending']


def get_runtime(actor):
    if not actor.config.policy_loss.get('transfer_guided',{}).get('enabled',False):return None
    if not hasattr(actor,'_transfer_guided'):actor._transfer_guided=Runtime(actor)
    return actor._transfer_guided


def save_state(actor,directory):
    rt=get_runtime(actor)
    if rt is not None:
        target=Path(directory)/'transfer_guided_state.pt';tmp=target.with_suffix('.tmp')
        torch.save(rt.state_dict(),tmp);tmp.replace(target)


def load_state(actor,directory):
    rt=get_runtime(actor)
    if rt is not None:rt.load_state_dict(torch.load(Path(directory)/'transfer_guided_state.pt',map_location='cpu',weights_only=True))
