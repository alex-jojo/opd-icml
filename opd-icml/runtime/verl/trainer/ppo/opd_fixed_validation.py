"""Fixed disjoint prompt split; fresh on-policy responses, one set per checkpoint."""
import hashlib
import json
import pickle
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Subset


def split_indices(count,size=128,seed=42):
    if count<=size:raise ValueError('Not enough examples for a disjoint validation split')
    val=np.sort(np.random.default_rng(seed).choice(count,size,replace=False))
    train=np.setdiff1d(np.arange(count),val)
    return train.tolist(),val.tolist()


def install(trainer,dataset):
    cfg=trainer.config.actor_rollout_ref.actor.policy_loss.get('gradient_study',{})
    if not cfg.get('enabled',False):return dataset
    # Dataloaders are constructed before init_workers assigns async_rollout_mode.
    if trainer.config.actor_rollout_ref.rollout.get('mode','sync')=='async':
        raise ValueError('Fixed128 study currently requires synchronous rollout')
    train,val=split_indices(len(dataset),128,int(cfg.get('seed',42)))
    # Also exclude duplicate prompt text, not just equal row numbers.
    key=trainer.config.data.get('prompt_key','prompt')
    frame=getattr(dataset,'dataframe',None)
    if frame is None:raise ValueError('Fixed128 split requires the filtered RLHF dataframe')
    hashes=[hashlib.sha256(json.dumps(row[key],ensure_ascii=False,sort_keys=True,default=str).encode()).hexdigest() for row in frame]
    chosen=set(hashes[i] for i in val)
    if len(chosen)!=128:raise ValueError('Validation split contains duplicate prompts; deduplicate the input dataset first')
    train=[i for i in train if hashes[i] not in chosen]
    manifest=dict(seed=int(cfg.get('seed',42)),validation_indices=val,validation_prompt_sha256=[hashes[i] for i in val],
                  train_indices=train,source_prompt_sha256=hashlib.sha256(''.join(hashes).encode()).hexdigest(),
                  validation_count=128,train_count=len(train),old_heldout_fixed_enabled=False)
    out=Path(trainer.config.trainer.default_local_dir).parent/'validation128';out.mkdir(parents=True,exist_ok=True)
    path=out/'split.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('Validation split changed on resume')
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    trainer.opd_validation_dataset=Subset(dataset,val);trainer.opd_validation_output=out
    trainer.opd_validation_source_indices=val
    print(f'[FIXED128] {len(train)} training prompts; 128 fixed validation prompts excluded including duplicates.',flush=True)
    return Subset(dataset,train)


def generate(trainer,rows):
    from verl import DataProto
    from verl.utils.dataset.rl_dataset import collate_fn
    from verl.trainer.ppo.ray_trainer import compute_response_mask
    batch=DataProto.from_single_dict(collate_fn(rows))
    gen=trainer._get_gen_batch(batch)
    gen.meta_info=dict(eos_token_id=trainer.tokenizer.eos_token_id,pad_token_id=trainer.tokenizer.pad_token_id,
                      do_sample=True,validate=False,recompute_log_prob=False,global_steps=trainer.global_steps)
    if int(trainer.config.actor_rollout_ref.rollout.n)!=1:raise ValueError('Study explicitly repeats paths; rollout.n must be 1')
    output=trainer.actor_rollout_wg.generate_sequences(gen)
    batch=batch.union(output);batch.batch['response_mask']=compute_response_mask(batch)
    batch.meta_info['temperature']=float(trainer.config.actor_rollout_ref.rollout.temperature)
    student=trainer.actor_rollout_wg.compute_log_prob(batch)
    batch=batch.union(student)
    ref=trainer.actor_rollout_wg if trainer.ref_in_actor else trainer.ref_policy_wg
    batch=batch.union(ref.compute_ref_log_prob(batch))
    return batch


def readout(trainer,checkpoint):
    from verl.workers.actor.opd_top16 import reverse_kl
    dataset=trainer.opd_validation_dataset
    cache=getattr(trainer,'opd_boundary_validation',None)
    if cache is not None and cache['checkpoint']==checkpoint:
        batch=cache['batch'];report=cache['report'];trainer.opd_boundary_validation=None
        report={**report,'reused_zero_adapter_handoff':True}
        (trainer.opd_validation_output/f'checkpoint_{checkpoint:03d}.json').write_text(json.dumps(report,ensure_ascii=False),encoding='utf-8')
        print(f'[FIXED128] checkpoint={checkpoint}: reusing identical warmup boundary rollout; no second generation.',flush=True)
        return batch
    batch=generate(trainer,[dataset[i] for i in range(128)])
    mask=batch.batch['response_mask'];kl=reverse_kl(batch.batch['opd_student_top16_logits'],batch.batch['opd_teacher_top16_logits'])
    per=((kl*mask).sum(1)/mask.sum(1).clamp_min(1)).cpu().tolist()
    reward=trainer.val_reward_fn(batch,return_dict=True)['reward_tensor'].sum(-1).cpu().tolist()
    records=[]
    for i in range(128):
        records.append(dict(source_index=trainer.opd_validation_source_indices[i],kl=per[i],score=reward[i],
                            response_tokens=int(mask[i].sum()),response_ids=batch.batch['responses'][i][mask[i].bool()].cpu().tolist()))
    report=dict(checkpoint_optimizer_updates=checkpoint,fixed_prompts=128,fresh_on_policy_rollout=True,
                kl=float(np.mean(per)),accuracy=float(np.mean(reward)),responses=records)
    if trainer.config.data.get('opd_task', 'math') == 'code':
        report.update(task='code', accuracy_definition='fraction of responses passing all code tests')
    target=trainer.opd_validation_output/f'checkpoint_{checkpoint:03d}.json'
    target.write_text(json.dumps(report,ensure_ascii=False),encoding='utf-8')
    trainer.opd_last_validation_report=report
    print(f'[FIXED128] checkpoint={checkpoint} KL={report["kl"]:.6g} accuracy={report["accuracy"]:.4f}',flush=True)
    return batch


def attach(trainer,batch):
    started=time.perf_counter()
    cfg=trainer.config.actor_rollout_ref.actor.policy_loss.get('gradient_study',{})
    if not cfg.get('enabled',False):return
    reuse=int(trainer.config.actor_rollout_ref.actor.policy_loss.get('jump_reuse_updates',1))
    checkpoint=int(cfg.get('update_offset',0))+(trainer.global_steps-1)*reuse
    if checkpoint%int(cfg.get('validation_interval',10)):return
    validation=readout(trainer,checkpoint)
    if cfg.get('validation_only',False):return {'timing/validation128_rollout_readout_seconds':time.perf_counter()-started}
    payload=dict(validation=validation, validation_source_indices=trainer.opd_validation_source_indices,
                 geometry=checkpoint in cfg.get('geometry_steps',[10,30,60,100]))
    if checkpoint in cfg.get('path_steps',[10,50,100]):
        p=int(cfg.get('path_prompts',4));k=int(cfg.get('path_repeats',4))
        # Training pool, separate from validation. Fixed prompt panel across checkpoints.
        ids=np.random.default_rng(int(cfg.get('seed',42))+1).choice(len(trainer.train_dataset),p,replace=False)
        payload['paths']=generate(trainer,[trainer.train_dataset[int(i)] for i in ids for _ in range(k)])
        payload['path_training_indices']=ids.tolist()
    batch.meta_info['gradient_study_payload']=pickle.dumps(payload,protocol=pickle.HIGHEST_PROTOCOL)
    return {'timing/validation128_and_path_rollout_readout_seconds':time.perf_counter()-started}


def terminal(trainer):
    cfg=trainer.config.actor_rollout_ref.actor.policy_loss.get('gradient_study',{})
    if cfg.get('enabled',False):
        started=time.perf_counter()
        reuse=int(trainer.config.actor_rollout_ref.actor.policy_loss.get('jump_reuse_updates',1))
        checkpoint=int(cfg.get('update_offset',0))+trainer.global_steps*reuse
        batch=readout(trainer,checkpoint)
        if cfg.get('save_handoff',False) and checkpoint%int(cfg.get('validation_interval',10))==0:
            # Temporary CPU-only transfer, not a model checkpoint or ZIP payload.
            batch=batch.to('cpu')
            target=trainer.opd_validation_output/'boundary_validation.pt'
            torch.save(dict(checkpoint=checkpoint,batch=batch,report=trainer.opd_last_validation_report),target)
        elapsed=time.perf_counter()-started
        (trainer.opd_validation_output/'terminal_timing.json').write_text(json.dumps(dict(checkpoint=checkpoint,seconds=elapsed)),encoding='utf-8')
        return elapsed
    return 0.


def load_handoff_data(trainer):
    """Resume the consumed prompt cursor, not the warmup optimizer or local step."""
    source=trainer.config.trainer.get('opd_adapter_handoff_dir')
    if not source:return
    path=Path(source)
    expected=json.loads((path/'split.json').read_text())
    actual=json.loads((trainer.opd_validation_output/'split.json').read_text())
    if expected!=actual:raise ValueError('Adapter handoff validation/training split changed')
    trainer.train_dataloader.load_state_dict(torch.load(path/'data.pt',map_location='cpu',weights_only=False))
    boundary=path/'boundary_validation.pt'
    if boundary.exists():
        cache=torch.load(boundary,map_location='cpu',weights_only=False)
        cfg=trainer.config.actor_rollout_ref.actor.policy_loss['gradient_study']
        if cache['checkpoint']!=int(cfg['update_offset']):raise ValueError('Validation boundary checkpoint mismatch')
        trainer.opd_boundary_validation=cache
    print('[ADAPTER HANDOFF] Restored training prompt cursor and fixed validation split. '
          'Fresh answers will be generated by the current student. New parameter A gets its own Adam state ONCE; '
          'FULL parameter moments are not copied into unrelated A coordinates.',flush=True)
