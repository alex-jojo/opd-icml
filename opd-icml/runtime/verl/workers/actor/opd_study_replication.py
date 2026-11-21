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
from .opd_replication17 import panel_hidden,install_basis
from .opd_rq2_motion import MotionRecorder
from .opd_top16 import _final_norm_module,_unwrap_actor_module
from .opd_plasticity import atomic_json,atomic_npz,atomic_moment
from .opd_token_audit import output_scores,enriched,training_summary,prepare_groups
from .opd_gradient_scopes import summarize,metric_values
from .opd_gradient_study import HiddenMoments,GradientStudy,losses as response_losses,reshard_after_readout


class StudyReplication:
    def __init__(self,actor):
        self.actor=actor;self.cfg=dict(actor.config.policy_loss['replication17'])
        self.study_cfg=dict(actor.config.policy_loss['gradient_study'])
        self.output=Path(self.cfg['output_dir']);self.asset=Path(actor.config.policy_loss['hidden_projection']['asset_path'])
        self.dimension=int(_final_norm_module(actor.actor_module).weight.numel());self.rank=int(self.cfg.get('rank',128))
        self.full=self.cfg['mode']=='full';self.freeze=int(self.cfg.get('freeze_step',10));self.interval=int(self.cfg.get('interval',5))
        self.diagnostic_interval = int(self.cfg.get('diagnostic_interval', 1))
        if self.diagnostic_interval < 1:
            raise ValueError('diagnostic_interval must be positive')
        self.study=GradientStudy(actor,self.study_cfg,self.output);self.stability=None
        self.snapshot_only = self.cfg.get('adaptive_stability', {}).get('snapshot_only', False)
        self.parameter_tokens=None;self.parameter_measurement=None
        parameter_cfg=self.study_cfg.get('parameter_tokens',{})
        if parameter_cfg.get('enabled',False):
            if not self.full:raise ValueError('Full parameter token audit requires ordinary FULL mode')
            from .opd_parameter_tokens import FullParameterTokenAudit
            self.parameter_tokens=FullParameterTokenAudit(actor,parameter_cfg)
        self.sample_rq1 = self.parameter_tokens is not None and self.parameter_tokens.mode == 'kl_stratified'
        self.save_hidden_samples = bool(self.cfg.get('save_hidden_gradient_samples', False))
        self.records = self.output.parent / 'training_records'
        if self.full:
            self.motion = MotionRecorder(self.records / 'rq2', self.output / 'motion', self.rank)
        self.tokenizer=None;self.labels={}
        if self.cfg.get('tokenizer_path'):
            from transformers import AutoTokenizer
            self.tokenizer=AutoTokenizer.from_pretrained(self.cfg['tokenizer_path'])
        if self.cfg.get('adaptive_stability',{}).get('enabled',False):
            from .opd_adaptive_stability import Stability
            self.stability=Stability(self.output,self.cfg['adaptive_stability'])

    def begin(self,train,basis,step):
        self.step=step;self.train=train;self.extra=0.;self.metrics={};self.parts=[];self.moments=None
        if self.snapshot_only and train.meta_info.get('opd_early_stop_selected', False):
            # A resumed FULL run already has its committed early checkpoint.
            self.stability.stable = True
        self.pending_gradient = None
        self.pending_subset_alignment = None
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
                dp.primary_call(lambda: self.motion.save_initial(self.panel, self.h0, device))
            if not hasattr(self,'u'):
                self.u = dp.primary_call(self.resume_basis).to(device)
            self.reference=dict(valid_u128=self.completed>=min(self.interval,self.freeze),role='observation_basis',source='training_motion_panel',
                frozen=self.completed>=self.freeze,cumulative_to=min((self.completed//self.interval)*self.interval,self.freeze))
            if self.stability is not None and not self.snapshot_only:
                observed = self.completed // self.stability.cfg['interval'] * self.stability.cfg['interval']
                self.reference.update(valid_u128=observed > 0, frozen=self.stability.stable, cumulative_to=observed)
        else:
            with np.load(self.asset,allow_pickle=False) as f:self.u=torch.from_numpy(f['basis'].copy()).to(device)
            self.reference=dict(valid_u128=True,role='frozen_full_training_motion_basis',sha256=hashlib.sha256(self.asset.read_bytes()).hexdigest())
        self.threshold_file=self.output/'training_token_thresholds.json'
        if not hasattr(self,'thresholds') and dp.exists(self.threshold_file):self.thresholds=json.loads(self.threshold_file.read_text())
        if self.sample_rq1 and step > 1 and not hasattr(self, 'thresholds'):
            first = self.records / 'rq1_step_0001.json'
            if not dp.exists(first):
                raise ValueError('Cannot resume RQ1 without the original step-1 KL thresholds')
            self.thresholds = dp.primary_call(lambda: json.loads(first.read_text())['thresholds'])

    def resume_basis(self):
        """Restore the observation basis at the committed update, not a future asset."""
        expected = (self.completed // self.interval) * self.interval
        if self.stability is None or self.snapshot_only:
            expected = self.freeze if self.completed >= self.freeze else expected
        if expected == 0:
            return torch.eye(self.dimension)[:, :self.rank]
        for checkpoint in sorted({expected, *range(self.interval, expected+1, self.interval)}, reverse=True):
            candidates = [self.records/'rq2'/f'step_{checkpoint:03d}.npz',
                          self.output/'motion'/f'step_{checkpoint:03d}.npz',
                          self.output/'stability'/f'basis_{checkpoint:03d}.npz']
            found = False
            for path in candidates:
                if path.is_file():
                    with np.load(path, allow_pickle=False) as saved:
                        basis = torch.from_numpy(saved['basis'].copy())
                    if basis.shape[1]:
                        return basis
                    found = True
                    break  # zero motion kept the previous observation basis
            if found:
                continue
            metadata = self.asset.with_suffix('.json')
            if self.asset.is_file() and metadata.is_file():
                info = json.loads(metadata.read_text(encoding='utf-8'))
                if info.get('cumulative_to', info.get('freeze_step')) == checkpoint:
                    with np.load(self.asset, allow_pickle=False) as saved:
                        return torch.from_numpy(saved['basis'].copy())
            raise ValueError(f'Missing observation basis for committed update {checkpoint}; refusing a later asset')
        return torch.eye(self.dimension)[:, :self.rank]

    def start_study(self,payload):
        payload['basis_reference']=self.reference
        self.study.start(self.train,self.u,payload,self.completed)

    def diagnostic_due(self, update):
        final = int(self.cfg['rounds']) * int(self.cfg.get('reuse', 1))
        return (update == 1 or update == final or update == self.freeze or
                update % self.diagnostic_interval == 0 or update % self.interval == 0)

    def before_inner(self,inner):
        update = self.completed + inner + 1
        # A reuse round has multiple updates: never carry a measured value into
        # an unmeasured inner update's metrics.
        for key in list(self.metrics):
            if key.startswith(('token_audit/parameter_', 'token_audit/kl_top10_parameter_', 'timing/parameter_token_', 'rq2/', 'timing/rq2_')):
                self.metrics.pop(key)
        self.pending_subset_alignment = None
        final = int(self.cfg['rounds']) * int(self.cfg.get('reuse', 1))
        subset_due = self.study.subset_due(update, final)
        if subset_due or (self.study.payload is not None and self.study.subset_alignment is not None):
            if self.study.payload is not None:
                self.pending_subset_alignment = self.study.subset_alignment
            else:
                started = time.perf_counter()
                self.pending_subset_alignment = self.study.cross_batch(self.train, self.u, self.completed + inner)
                self.extra += time.perf_counter() - started
        if int(self.cfg.get('reuse', 1)) > 1:
            update = self.completed + inner + 1
            self.measure = (update == 1 or update % self.interval == 0 or
                            update == int(self.cfg['rounds']) * int(self.cfg['reuse']))
        self.parameter_measurement=None
        if self.parameter_tokens is not None and (self.measure or (self.sample_rq1 and self.diagnostic_due(update))):
            self.parameter_measurement=self.parameter_tokens.measure(self.train,self.step,inner)

    @contextmanager
    def watch_training(self,inputs,inner,cursor):
        if not (self.measure or self.sample_rq1):yield;return
        n=inputs['responses'].shape[1];mask=inputs['response_mask'].bool()
        row,pos=mask.nonzero(as_tuple=True);part={}
        def hook(_m,_a,out):
            with torch.no_grad():
                z=out[:,-n-1:-1]
                if z.shape[-1]!=inputs['opd_top16_ids'].shape[-1]:z=z.gather(-1,inputs['opd_top16_ids'])
                part.update({k:v.detach().cpu().numpy() for k,v in output_scores(
                    z[mask],inputs['opd_teacher_top16_logits'][mask],include_teacher_entropy=True,
                    kl_only=not self.measure).items()})
                part.update(row=(row+cursor).cpu().numpy(),pos=pos.cpu().numpy(),token=inputs['responses'][mask].cpu().numpy())
                if self.measure:
                    ordinal=mask.long().cumsum(-1)-1
                    part.update(boundary=(ordinal==0)[mask].cpu().numpy(),
                        relative_position=(ordinal.float()/(mask.sum(1,keepdim=True)-1).clamp_min(1))[mask].cpu().numpy())
        head=_unwrap_actor_module(_unwrap_actor_module(self.actor.actor_module).lm_head)
        handle=head.register_forward_hook(hook)
        self.current_part=part
        try:yield
        finally:handle.remove()
        self.parts.append(part)

    @contextmanager
    def watch_gradient_scopes(self,inputs,inner,state):
        if not (self.full or self.measure):yield;return
        collect_moments = self.measure or (self.full and self.diagnostic_due(self.completed + inner + 1))
        if collect_moments and self.moments is None:
            self.moments = HiddenMoments(self.u if self.measure else None)
        n=inputs['responses'].shape[1];mask=inputs['response_mask'].bool();total=int(self.train.batch['response_mask'].sum())
        def consume(gradient):
            row,pos=mask.nonzero(as_tuple=True);count=len(row)
            energy = np.empty(count, dtype=np.float32) if self.measure or self.sample_rq1 else None
            if count == 0 and self.moments is not None:
                # An all-padding local shard contributes a zero Gram to the global reduction.
                self.moments.add(gradient.new_empty((0, gradient.shape[-1])))
            if self.measure:
                full=np.empty((count,gradient.shape[-1]),dtype=np.float32)
                inside=np.empty((count,self.u.shape[1]),dtype=np.float32)
                kept=np.empty(count,dtype=np.float32)
            start=gradient.shape[1]-n-1
            # No full N x hidden CUDA copy at the backward activation peak.
            for lo in range(0,count,HiddenMoments.chunk_tokens):
                hi=min(count,lo+HiddenMoments.chunk_tokens)
                g=gradient[row[lo:hi],pos[lo:hi]+start].detach().float()
                g.mul_(total/dp.size())
                if self.moments is not None:
                    self.moments.add(g)
                if energy is not None:
                    energy[lo:hi]=g.square().sum(1).cpu().numpy()
                if not self.measure:
                    continue
                with torch.autocast(device_type=g.device.type,enabled=False):gu=g@self.u.float()
                full[lo:hi]=g.cpu().numpy();inside[lo:hi]=gu.cpu().numpy()
                kept[lo:hi]=gu.square().sum(1).cpu().numpy()
                del g,gu
            if energy is not None:
                self.current_part['raw_gradient_energy']=energy
            if not self.measure:return
            self.current_part.update(g_u=inside,projected_gradient_energy=kept)
            self.current_part['transient_full_gradient']=full
            # Deterministic bounded full-vector sample, with identifiers retained.
            if self.save_hidden_samples:
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

    def gather_token_fields(self):
        local = {key: np.concatenate([part[key] for part in self.parts]) for key in self.parts[0]}
        self.parts = []
        return dp.merge_fields(dp.gather(local))

    def save_rq1(self, inner, report, arrays):
        groups = {}
        for name in ('kl_top10', 'kl_above_step1_p90'):
            group = report['groups'][name]
            quantities = {key: group['quantities'][key] for key in ('raw_gradient_energy', 'parameter_gradient_energy') if key in group['quantities']}
            groups[name] = dict(count=group['count'], fraction=group['fraction'], quantities=quantities)
        step = self.completed + inner + 1
        metadata = report.get('parameter_gradient_measurement')
        hidden = groups['kl_top10']['quantities']['raw_gradient_energy']
        self.metrics.update({'token_audit/step1_kl_p90': self.thresholds['kl_p90'],
                             'token_audit/kl_top10_hidden_energy_share': hidden['share'] or 0.})
        if not dp.primary():return
        sample_keys = ('row', 'pos', 'token', 'kl', 'in_current_kl_top10', 'sampling_stratum',
                       'sampling_weight', 'raw_gradient_energy', 'parameter_gradient_energy', 'diagnostic_kl')
        record = dict(step=step, round=self.step, inner=inner+1, thresholds=self.thresholds,
                       groups=groups, parameter_gradient_measurement=metadata or {'measured': False},
                      samples={key: arrays[key].tolist() for key in sample_keys if key in arrays})
        for key in ('gradient_rms_full', 'gradient_frobenius_full', 'parameter_gradient_rms',
                    'parameter_gradient_frobenius'):
            record[key] = report.get(key)
        atomic_json(self.records / f'rq1_step_{step:04d}.json', record)

    def record_parameter_metrics(self, report, metadata):
        self.metrics['timing/parameter_token_gradient_seconds']=metadata['seconds']
        self.metrics['token_audit/parameter_gradient_rms']=report['parameter_gradient_rms']
        self.metrics['token_audit/parameter_gradient_frobenius']=report['parameter_gradient_frobenius']
        self.metrics['token_audit/kl_top10_parameter_energy_share']=report['groups']['kl_top10']['quantities']['parameter_gradient_energy']['share'] or 0.
        self.metrics['token_audit/parameter_token_extra_backwards']=metadata['extra_backwards']
        self.metrics['token_audit/parameter_token_sample_count']=metadata['token_count']
        self.metrics['token_audit/parameter_token_coverage_fraction']=metadata['coverage_fraction']

    def before_update(self,inner):
        if self.study.payload is not None and inner==0:self.study.before_update()
        started = time.perf_counter()
        moment = None
        if self.full and self.diagnostic_due(self.completed + inner + 1) and self.moments is None:
            raise ValueError('Missing training hidden gradients for this FULL update')
        if self.moments is not None:
            moment = self.moments.finish()
            self.moments = None
            if self.full:
                count = int(moment['count'])
                if count != int(self.train.batch['response_mask'].sum()):
                    raise ValueError('Incomplete training hidden-gradient spectrum')
                if dp.primary():
                    self.pending_gradient = dict(second=moment['second'], count=count)
        if self.sample_rq1 and not self.measure:
            from .opd_parameter_sampling import rq1_only_report
            fields = self.gather_token_fields()
            report, metadata, arrays = dp.primary_call(
                lambda: rq1_only_report(fields, self.parameter_measurement, self.thresholds))
            self.parameter_measurement = None
            self.save_rq1(inner, report, arrays)
            if metadata is not None:
                self.record_parameter_metrics(report, metadata)
            self.metrics['token_audit/high_kl_fraction'] = report['groups']['kl_above_step1_p90']['fraction']
        if not self.measure or moment is None:
            self.extra += time.perf_counter() - started
            return
        atomic_moment(self.output/'gradient_moments'/f'round_{self.step:03d}_inner_{inner+1}.npz',moment)
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
        if self.full and dp.primary():
            self.pending_gradient['eigenvalues'] = arrays['full_eigenvalues']
        atomic_json(self.output/'gradient_scopes'/f'round_{self.step:03d}_inner_{inner+1}.json',report)
        atomic_npz(self.output/'gradient_scopes'/f'round_{self.step:03d}_inner_{inner+1}.npz',arrays)
        self.metrics.update(metric_values(report))
        samples=[];sample_rows=[];sample_pos=[];full_parts=[]
        for part in self.parts:
            full_parts.append(part.pop('transient_full_gradient'))
            if self.save_hidden_samples:
                ix=part.pop('sample_local_index');samples.append(part.pop('sample_full_gradient'))
                sample_rows.append(part['row'][ix]);sample_pos.append(part['pos'][ix])
        merged=self.gather_token_fields()
        from .opd_group_scopes import compute
        group_started=time.perf_counter()
        group_report,group_arrays=compute(full_parts,merged['kl'],self.u,self.u.device)
        del full_parts
        group_report.update(basis_reference=self.reference,optimizer_update=self.completed+inner+1)
        group_report['seconds']=time.perf_counter()-group_started
        atomic_json(self.output/'token_group_scopes'/f'round_{self.step:03d}_inner_{inner+1}.json',group_report)
        atomic_npz(self.output/'token_group_scopes'/f'round_{self.step:03d}_inner_{inner+1}.npz',group_arrays)
        self.metrics['timing/token_group_scopes_seconds']=group_report['seconds']
        if self.save_hidden_samples:
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
            self.save_rq1(inner, token_report, paired_arrays)
            self.parameter_measurement=None
        if parameter_metadata is not None:
            token_report['parameter_gradient_measurement']=parameter_metadata
            self.record_parameter_metrics(token_report, parameter_metadata)
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
        update = self.completed + inner
        if self.pending_subset_alignment is not None:
            record = dict(checkpoint_before=update-1, optimizer_update=update, rollout_round=self.step,
                          inner=inner, **self.pending_subset_alignment)
            dp.primary_call(lambda: atomic_json(self.study.rq3_directory / f'subset_alignment_{update:03d}.json', record))
            self.pending_subset_alignment = None
        if self.study.payload is not None and inner==1:self.metrics.update(self.study.after_update())
        if not self.full:return
        if not self.diagnostic_due(update):
            return
        started = time.perf_counter()
        current = panel_hidden(self.actor, self.panel)
        reshard_after_readout(self.actor)
        stability_due = (self.stability is not None and not (self.snapshot_only and self.stability.stable)
                         and update % self.stability.cfg['interval'] == 0)
        legacy_due = (stability_due or update % self.interval == 0 or
                      update == self.freeze or self.step == int(self.cfg['rounds']))
        def record():
            gradient_metrics = {}
            if self.pending_gradient is not None:
                started_gradient = time.perf_counter()
                gradient_metrics = self.motion.record_gradient(update, **self.pending_gradient)
                gradient_metrics['timing/rq2_gradient_archive_seconds'] = time.perf_counter() - started_gradient
                self.pending_gradient = None
            metrics, legacy = self.motion.record(update, self.h0, current, legacy_due=legacy_due)
            metrics.update(gradient_metrics)
            return metrics, legacy
        metrics, legacy = dp.primary_call(record)
        for key in ('rq2/gradient_own_top128_coverage', 'rq2/gradient_fixed_u10_coverage'):
            self.metrics.pop(key, None)
        self.metrics.update(metrics)
        if legacy_due:
            self.save_legacy_motion(update, stability_due, *legacy)
        elapsed = time.perf_counter() - started
        self.metrics['timing/rq2_motion_seconds'] = elapsed - metrics.get('timing/rq2_gradient_archive_seconds', 0.)
        self.extra += elapsed

    def save_legacy_motion(self, update, stability_due, u, second, eigen, record):
        meta = None
        if u.shape[1] and stability_due:
            decision = self.stability.observe(update, u, second)
            atomic_json(self.records / 'rq4' / f'early_stop_{update:03d}.json', decision)
            if not self.snapshot_only:
                self.u = u.to(next(self.actor.actor_module.parameters()).device)
                install_basis(self.asset, u)
                meta = dict(freeze_step=update if decision['stable'] else None, stable=decision['stable'])
                self.reference.update(valid_u128=True, cumulative_to=update, frozen=decision['stable'])
        elif stability_due:
            self.stability.stable = False
            atomic_json(self.records / 'rq4' / f'early_stop_{update:03d}.json', dict(
                step=update, stable=False, decision='continue', reason='zero_probe_displacement',
                config=self.stability.cfg, ready=False))
        if u.shape[1] and (self.stability is None or self.snapshot_only) and update <= self.freeze:
            self.u = u.to(next(self.actor.actor_module.parameters()).device)
            install_basis(self.asset, u)
            self.reference.update(valid_u128=True, cumulative_to=update, frozen=update == self.freeze)
            if update == self.freeze:
                meta = dict(freeze_step=self.freeze)
        if meta is not None:
            atomic_json(self.asset.with_suffix('.json'), dict(**meta, rank=self.rank, hidden_size=self.dimension,
                basis_sha256=hashlib.sha256(self.asset.read_bytes()).hexdigest(), source='training_motion_panel', **record))
        record.update(step=update, observation_basis_energy=(
            float(torch.trace(self.u.cpu().T @ second @ self.u.cpu()) / eigen.sum()) if u.shape[1] else None))
        atomic_moment(self.output/'motion'/f'step_{update:03d}.npz', dict(second=second.numpy(),eigenvalues=eigen.numpy(),basis=u.numpy()))
        atomic_json(self.output/'motion'/f'step_{update:03d}.json', record)

    def finish(self,losses,norms):
        if len(losses) == 2 and int(self.cfg.get('reuse', 1)) == 2 and (
                self.step % self.interval == 0 or self.step == int(self.cfg['rounds'])):
            # Both training losses are pre-update. Re-evaluate the identical cached
            # responses and teacher targets at the parameters committed by update 2.
            started = time.perf_counter()
            counts = self.train.batch['response_mask'].sum(1).cpu().numpy()
            after = float(np.average(response_losses(self.actor, self.train), weights=counts))
            second_gain = losses[1] - after
            atomic_json(self.records / 'rq4' / f'reuse_round_{self.step:03d}.json', dict(
                rollout_round=self.step, optimizer_updates=[self.completed+1, self.completed+2],
                responses=len(counts), valid_tokens=int(sum(counts)), normalization='mean_over_valid_training_tokens',
                kl_before_first=losses[0], kl_after_first=losses[1], kl_after_second=after,
                first_gain=losses[0]-losses[1], second_gain=second_gain))
            self.metrics['rq4/second_gain'] = second_gain
            self.extra += time.perf_counter() - started
        atomic_json(self.output/f'round_{self.step:03d}.json',dict(round=self.step,reuse=len(losses),
            optimizer_updates=self.completed+len(losses),preupdate_train_kl=losses,gradient_norms=norms,
            diagnostic_revision='fixed128-gradient-study-v1',old_heldout_fixed_enabled=False))
        self.metrics['timing/replication_readout_seconds']=self.extra
        if self.stability is not None:self.metrics['replication/adaptive_stop']=float(self.stability.stable)
        return self.metrics
