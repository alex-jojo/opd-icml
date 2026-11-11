"""Replacement replication runtime: no current-batch holdout or first-round validation bank."""
import hashlib
import json
import time
import re
from pathlib import Path
from contextlib import contextmanager
import numpy as np
import torch
from . import opd_distributed as dp
from .opd_replication17 import cumulative_basis,panel_hidden,install_basis
from .opd_top16 import _final_norm_module,_unwrap_actor_module
from .opd_plasticity import atomic_json,atomic_npz
from .opd_token_audit import output_scores,enriched,training_summary,prepare_groups
from .opd_gradient_scopes import summarize,metric_values
from .opd_gradient_study import HiddenMoments,GradientStudy,reshard_after_readout


class StudyReplication:
    def __init__(self,actor):
        self.actor=actor;self.cfg=dict(actor.config.policy_loss['replication17'])
        self.study_cfg=dict(actor.config.policy_loss['gradient_study'])
        self.output=Path(self.cfg['output_dir']);self.asset=Path(actor.config.policy_loss['hidden_projection']['asset_path'])
        self.dimension=int(_final_norm_module(actor.actor_module).weight.numel());self.rank=int(self.cfg.get('rank',128))
        self.full=self.cfg['mode']=='full';self.freeze=int(self.cfg.get('freeze_step',10));self.interval=int(self.cfg.get('interval',5))
        self.study=GradientStudy(actor,self.study_cfg,self.output);self.stability=None
        self.parameter_tokens=None;self.parameter_measurement=None
        parameter_cfg=self.study_cfg.get('parameter_tokens',{})
        if parameter_cfg.get('enabled',False):
            if not self.full:raise ValueError('Full parameter token audit requires ordinary FULL mode')
            from .opd_parameter_tokens import FullParameterTokenAudit
            self.parameter_tokens=FullParameterTokenAudit(actor,parameter_cfg)
        self.tokenizer=None;self.labels={}
        if self.cfg.get('tokenizer_path'):
            from transformers import AutoTokenizer
            self.tokenizer=AutoTokenizer.from_pretrained(self.cfg['tokenizer_path'])
        if self.cfg.get('adaptive_stability',{}).get('enabled',False):
            from .opd_adaptive_stability import Stability
            self.stability=Stability(self.output,self.cfg['adaptive_stability'])

    def begin(self,train,basis,step):
        self.step=step;self.train=train;self.extra=0.;self.metrics={};self.parts=[];self.moments=None
        reuse=int(self.cfg.get('reuse',1));self.completed=(step-1)*reuse
        self.measure=(step==1 or any((self.completed+i)%self.interval==0 for i in range(1,reuse+1)) or step==int(self.cfg['rounds']))
        device=next(self.actor.actor_module.parameters()).device
        if self.full:
            cache=self.output/'training_motion_panel.pt'
            if not hasattr(self,'panel'):
                if dp.exists(cache):
                    saved=torch.load(cache,map_location='cpu',weights_only=False);self.panel=saved['panel'];self.h0=saved['h0']
                else:
                    if step!=1:raise ValueError('Missing training-side motion panel on resume')
                    self.panel=train.select_idxs(list(range(min(32,len(train)))))
                    self.h0=panel_hidden(self.actor,self.panel)
                    reshard_after_readout(self.actor)
                    if dp.primary():
                        cache.parent.mkdir(parents=True,exist_ok=True);tmp=cache.with_suffix('.tmp')
                        torch.save(dict(panel=self.panel,h0=self.h0,role='training_motion_only_not_validation'),tmp);tmp.replace(cache)
                    dp.barrier()
            if not hasattr(self,'u'):
                if self.asset.is_file() and step>1:
                    with np.load(self.asset,allow_pickle=False) as f:self.u=torch.from_numpy(f['basis'].copy()).to(device)
                else:self.u=torch.eye(self.dimension,device=device)[:,:self.rank]
            self.reference=dict(valid_u128=self.completed>=min(self.interval,self.freeze),role='observation_basis',source='training_motion_panel',
                frozen=self.completed>=self.freeze,cumulative_to=min((self.completed//self.interval)*self.interval,self.freeze))
        else:
            with np.load(self.asset,allow_pickle=False) as f:self.u=torch.from_numpy(f['basis'].copy()).to(device)
            self.reference=dict(valid_u128=True,role='frozen_full_training_motion_basis',sha256=hashlib.sha256(self.asset.read_bytes()).hexdigest())
        self.threshold_file=self.output/'training_token_thresholds.json'
        if not hasattr(self,'thresholds') and dp.exists(self.threshold_file):self.thresholds=json.loads(self.threshold_file.read_text())

    def start_study(self,payload):
        payload['basis_reference']=self.reference
        self.study.start(self.train,self.u,payload,self.completed)

    def before_inner(self,inner):
        self.parameter_measurement=None
        if self.measure and self.parameter_tokens is not None:
            self.parameter_measurement=self.parameter_tokens.measure(self.train,self.step,inner)

    @contextmanager
    def watch_training(self,inputs,inner,cursor):
        if not self.measure:yield;return
        n=inputs['responses'].shape[1];mask=inputs['response_mask'].bool()
        row,pos=mask.nonzero(as_tuple=True);part={}
        def hook(_m,_a,out):
            with torch.no_grad():
                z=out[:,-n-1:-1]
                if z.shape[-1]!=inputs['opd_top16_ids'].shape[-1]:z=z.gather(-1,inputs['opd_top16_ids'])
                part.update({k:v.detach().cpu().numpy() for k,v in output_scores(
                    z[mask],inputs['opd_teacher_top16_logits'][mask],include_teacher_entropy=True).items()})
                ordinal=mask.long().cumsum(-1)-1
                part.update(row=(row+cursor).cpu().numpy(),pos=pos.cpu().numpy(),token=inputs['responses'][mask].cpu().numpy(),
                    boundary=(ordinal==0)[mask].cpu().numpy(),
                    relative_position=(ordinal.float()/(mask.sum(1,keepdim=True)-1).clamp_min(1))[mask].cpu().numpy())
        head=_unwrap_actor_module(_unwrap_actor_module(self.actor.actor_module).lm_head)
        handle=head.register_forward_hook(hook)
        self.current_part=part
        try:yield
        finally:handle.remove()
        self.parts.append(part)

    @contextmanager
    def watch_gradient_scopes(self,inputs,inner,state):
        if not self.measure:yield;return
        if self.moments is None:self.moments=HiddenMoments(self.u)
        n=inputs['responses'].shape[1];mask=inputs['response_mask'].bool();total=int(self.train.batch['response_mask'].sum())
        def consume(gradient):
            row,pos=mask.nonzero(as_tuple=True);count=len(row)
            full=np.empty((count,gradient.shape[-1]),dtype=np.float32)
            inside=np.empty((count,self.u.shape[1]),dtype=np.float32)
            energy=np.empty(count,dtype=np.float32);kept=np.empty(count,dtype=np.float32)
            start=gradient.shape[1]-n-1
            # No full N x hidden CUDA copy at the backward activation peak.
            for lo in range(0,count,HiddenMoments.chunk_tokens):
                hi=min(count,lo+HiddenMoments.chunk_tokens)
                g=gradient[row[lo:hi],pos[lo:hi]+start].detach().float()
                g.mul_(total/dp.size())
                self.moments.add(g)
                with torch.autocast(device_type=g.device.type,enabled=False):gu=g@self.u.float()
                full[lo:hi]=g.cpu().numpy();inside[lo:hi]=gu.cpu().numpy()
                energy[lo:hi]=g.square().sum(1).cpu().numpy()
                kept[lo:hi]=gu.square().sum(1).cpu().numpy()
                del g,gu
            self.current_part.update(g_u=inside,raw_gradient_energy=energy,projected_gradient_energy=kept)
            self.current_part['transient_full_gradient']=full
            # Deterministic bounded full-vector sample, with identifiers retained.
            ix=torch.linspace(0,count-1,min(64,count),device=gradient.device).round().long()
            self.current_part['sample_full_gradient']=full[ix.cpu().numpy()]
            self.current_part['sample_local_index']=ix.cpu().numpy()
        old=state.get('raw_gradient_observer');handle=None
        if self.cfg['mode']=='r128':state['raw_gradient_observer']=consume
        else:
            def hook(_m,_a,h):
                def backward(g):consume(g);return g
                h.register_hook(backward)
            handle=_final_norm_module(self.actor.actor_module).register_forward_hook(hook)
        try:yield
        finally:
            if handle is not None:handle.remove()
            state['raw_gradient_observer']=old

    def before_update(self,inner):
        if self.study.payload is not None and inner==0:self.study.before_update()
        if self.moments is None:return
        started=time.perf_counter();moment=self.moments.finish();self.moments=None
        atomic_npz(self.output/'gradient_moments'/f'round_{self.step:03d}_inner_{inner+1}.npz',moment)
        # Reuse the established CPU eigensolver on moments; no extra transformer passes.
        def build():
            c=torch.from_numpy(moment['second']).double();u=torch.linalg.qr(torch.from_numpy(moment['basis']).double()).Q
            cu=c@u;inside=u.T@cu;outside=c-cu@u.T-u@cu.T+u@inside@u.T;arrays={};stats={}
            for name,m in [('full',c),('u',inside),('outside',outside)]:
                e=torch.linalg.eigvalsh((m+m.T)/2).flip(0).clamp_min(0)
                if name=='outside':e=e[:self.dimension-self.rank]
                arrays[name+'_eigenvalues']=e.numpy();stats[name]=summarize(e.numpy(),int(moment['count']))
            return dict(scopes=stats,basis_reference=self.reference,round=self.step,inner=inner+1,optimizer_update=self.completed+inner+1,
                        hidden_moment_saved=True,mean_saved=True),arrays
        report,arrays=dp.primary_call(build)
        atomic_json(self.output/'gradient_scopes'/f'round_{self.step:03d}_inner_{inner+1}.json',report)
        atomic_npz(self.output/'gradient_scopes'/f'round_{self.step:03d}_inner_{inner+1}.npz',arrays)
        self.metrics.update(metric_values(report))
        samples=[];sample_rows=[];sample_pos=[];full_parts=[]
        for part in self.parts:
            full_parts.append(part.pop('transient_full_gradient'))
            ix=part.pop('sample_local_index');samples.append(part.pop('sample_full_gradient'))
            sample_rows.append(part['row'][ix]);sample_pos.append(part['pos'][ix])
        merged=dp.merge_fields(dp.gather({k:np.concatenate([p[k] for p in self.parts]) for k in self.parts[0]}));self.parts=[]
        from .opd_group_scopes import compute
        group_started=time.perf_counter()
        group_report,group_arrays=compute(full_parts,merged['kl'],self.u,self.u.device)
        del full_parts
        group_report.update(basis_reference=self.reference,optimizer_update=self.completed+inner+1)
        group_report['seconds']=time.perf_counter()-group_started
        atomic_json(self.output/'token_group_scopes'/f'round_{self.step:03d}_inner_{inner+1}.json',group_report)
        atomic_npz(self.output/'token_group_scopes'/f'round_{self.step:03d}_inner_{inner+1}.npz',group_arrays)
        self.metrics['timing/token_group_scopes_seconds']=group_report['seconds']
        sample=dp.merge_fields(dp.gather(dict(g=np.concatenate(samples),row=np.concatenate(sample_rows),pos=np.concatenate(sample_pos))))
        atomic_npz(self.output/'gradient_moments'/f'token_sample_{self.step:03d}_{inner+1}.npz',sample)
        if not hasattr(self,'thresholds'):
            self.thresholds=dict(kl_p90=float(np.quantile(merged['kl'],.9)),kl_median=float(np.median(merged['kl'])),
                                 entropy_median=float(np.median(merged['entropy'])))
            atomic_json(self.threshold_file,self.thresholds)
        z=merged.pop('g_u')
        model_config=getattr(_unwrap_actor_module(self.actor.actor_module),'config',None)
        eos=getattr(model_config,'opd_response_eos_token_ids',getattr(model_config,'eos_token_id',None))
        merged['first']=merged['boundary'].astype(bool)
        merged['eos']=np.isin(merged['token'],[] if eos is None else [eos] if isinstance(eos,int) else eos)
        merged['boundary']=merged['first']|merged['eos']
        if self.tokenizer is not None:
            for token in np.unique(merged['token']):
                if int(token) not in self.labels:
                    text=self.tokenizer.decode([int(token)])
                    self.labels[int(token)]=(bool(re.search(r'\d',text)),bool(re.search(r'[+\-*/=<>×÷]',text)),bool(re.search(r'[\\{}_^$]',text)))
            for j,key in enumerate(('numeric','operator','latex')):merged[key]=np.array([self.labels[int(t)][j] for t in merged['token']])
            merged['other']=~(merged['boundary']|merged['numeric']|merged['operator']|merged['latex'])
        merged=enriched(merged)
        parameter_metadata=None
        sampled_measurement=self.parameter_measurement if self.parameter_measurement is not None and 'sampling' in self.parameter_measurement else None
        if self.parameter_measurement is not None and sampled_measurement is None:
            from .opd_parameter_tokens import attach_parameter_energy
            parameter_metadata=attach_parameter_energy(merged,self.parameter_measurement)
            self.parameter_measurement=None
        def build_token_report():
            # Keep masks/rankings on the primary rank, shared only for this update.
            grouping=prepare_groups(merged,self.thresholds)
            report,arrays=training_summary(merged,z,self.thresholds,grouping=grouping)
            if sampled_measurement is None:
                return report,arrays,None,None
            from .opd_parameter_sampling import add_sampled_report
            metadata,paired=add_sampled_report(report,merged,sampled_measurement,self.thresholds,grouping=grouping)
            return report,arrays,metadata,paired
        token_report,token_arrays,sampled_metadata,paired_arrays=dp.primary_call(build_token_report)
        if sampled_measurement is not None:
            parameter_metadata=sampled_metadata
            atomic_npz(self.output/'token_audit'/f'parameter_sample_{self.step:03d}_inner_{inner+1}.npz',paired_arrays)
            self.parameter_measurement=None
        if parameter_metadata is not None:
            token_report['parameter_gradient_measurement']=parameter_metadata
            self.metrics['timing/parameter_token_gradient_seconds']=parameter_metadata['seconds']
            self.metrics['token_audit/parameter_gradient_rms']=token_report['parameter_gradient_rms']
            self.metrics['token_audit/parameter_gradient_frobenius']=token_report['parameter_gradient_frobenius']
            self.metrics['token_audit/kl_top10_parameter_energy_share']=token_report['groups']['kl_top10']['quantities']['parameter_gradient_energy']['share'] or 0.
            self.metrics['token_audit/parameter_token_extra_backwards']=parameter_metadata['extra_backwards']
            self.metrics['token_audit/parameter_token_sample_count']=parameter_metadata['token_count']
            self.metrics['token_audit/parameter_token_coverage_fraction']=parameter_metadata['coverage_fraction']
        token_report.update(basis_reference=self.reference,old_validation_banks_removed=True)
        token_report['full_u_outside_kl_groups']=group_report
        self.metrics.update({'token_audit/high_kl_fraction':token_report['groups']['kl_above_step1_p90']['fraction'],
            'token_audit/student_entropy_mean':float(merged['entropy'].mean()),
            'token_audit/teacher_entropy_mean':float(merged['teacher_entropy'].mean()),
            'token_audit/logratio_variance_mean':float(merged['logratio_variance'].mean()),
            'token_audit/valid_learned_u':float(self.reference['valid_u128'])})
        for label,entry in group_report['groups'].items():
            for scope,values in entry['all'].items():
                for key in ('effective_rank','nuclear_norm_token_normalized','rms'):
                    if values[key] is not None:self.metrics[f'token_groups/{label}/{scope}/{key}']=values[key]
        atomic_json(self.output/'token_audit'/f'step_{self.step:03d}_inner_{inner+1}.json',token_report)
        atomic_npz(self.output/'token_audit'/f'step_{self.step:03d}_inner_{inner+1}.npz',{**token_arrays,**merged})
        self.extra+=time.perf_counter()-started

    def after_inner(self,inner):
        if self.study.payload is not None and inner==1:self.metrics.update(self.study.after_update())
        update=self.completed+inner
        if self.full and (update%self.interval==0 or update==self.freeze or self.step==int(self.cfg['rounds'])):
            started=time.perf_counter();current=panel_hidden(self.actor,self.panel)
            reshard_after_readout(self.actor)
            u,second,eigen,record=cumulative_basis(self.h0,current,self.rank)
            if self.stability is not None:
                decision=self.stability.observe(update,u,second);self.u=u.to(next(self.actor.actor_module.parameters()).device)
                install_basis(self.asset,u)
                meta=dict(freeze_step=update if decision['stable'] else None,stable=decision['stable'])
            else:
                meta=dict(freeze_step=self.freeze)
                if update<=self.freeze:
                    self.u=u.to(next(self.actor.actor_module.parameters()).device);install_basis(self.asset,u)
            if update==self.freeze or self.stability is not None:
                atomic_json(self.asset.with_suffix('.json'),dict(**meta,rank=self.rank,hidden_size=self.dimension,
                    basis_sha256=hashlib.sha256(self.asset.read_bytes()).hexdigest(),source='training_motion_panel',**record))
            record.update(step=update,raw_displacement_rms=float((current-self.h0).square().sum(1).mean().sqrt()),
                observation_basis_energy=float(torch.trace(self.u.cpu().T@second@self.u.cpu())/eigen.sum()))
            atomic_npz(self.output/'motion'/f'step_{update:03d}.npz',dict(second=second.numpy(),eigenvalues=eigen.numpy(),basis=u.numpy()))
            atomic_json(self.output/'motion'/f'step_{update:03d}.json',record);self.extra+=time.perf_counter()-started

    def finish(self,losses,norms):
        atomic_json(self.output/f'round_{self.step:03d}.json',dict(round=self.step,reuse=len(losses),
            optimizer_updates=self.completed+len(losses),preupdate_train_kl=losses,gradient_norms=norms,
            diagnostic_revision='fixed128-gradient-study-v1',old_heldout_fixed_enabled=False))
        self.metrics['timing/replication_readout_seconds']=self.extra
        if self.stability is not None:self.metrics['replication/adaptive_stop']=float(self.stability.stable)
        return self.metrics
