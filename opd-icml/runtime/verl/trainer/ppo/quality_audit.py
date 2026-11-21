"""Lossless text/token evidence, written before the actor update. No model replay."""
import gzip
import hashlib
import json
import math
import os
import importlib.metadata
from pathlib import Path
import numpy as np
import torch


def serial(value):
    if isinstance(value, torch.Tensor):
        return serial(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return serial(value.tolist())
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)  # Record nonfinite diagnostics explicitly, do not crash training.
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    return value


def redact(value):
    if isinstance(value, dict):
        return {k: ('[REDACTED]' if any(s in k.lower() for s in ('api_key','password','secret','access_token')) else redact(v)) for k,v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    return serial(value)


def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(serial(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(tmp,path)


def ensure_run_manifest(directory, config, tokenizer):
    path=Path(directory)/'run_manifest.json'
    if path.exists():return
    from omegaconf import OmegaConf
    cfg=OmegaConf.to_container(config,resolve=True)
    asset=Path(cfg['actor_rollout_ref']['actor']['policy_loss']['direction_reward']['asset_path'])
    payload=json.loads(asset.read_text(encoding='utf-8'))
    def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    code_root=Path(__file__).resolve().parents[2]
    source_names=('trainer/ppo/ray_trainer.py','trainer/ppo/quality_audit.py','workers/actor/dp_actor.py','workers/actor/quality_fusion_reward.py','workers/actor/trajectory_policy_loss.py','workers/fsdp_workers.py')
    versions={}
    for name in ('torch','transformers','vllm','ray','tensordict','numpy'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    train_files=cfg.get('data',{}).get('train_files',[])
    if isinstance(train_files,str):train_files=[train_files]
    data_hashes={str(p):digest(Path(p)) for p in train_files if Path(p).is_file()}
    atomic_json(path,dict(schema=1,recording_revision='20260920-quality-v8-full-audit-v1',config=redact(cfg),reward_manifest=payload,
        train_file_sha256=data_hashes,
        source_sha256={name:digest(code_root/name) for name in source_names},versions=versions,
        asset_sha256=digest(asset),weights_sha256=digest(asset.parent/payload['weights_file']),
        tokenizer_class=type(tokenizer).__name__,tokenizer_name=tokenizer.name_or_path,
        chat_template=tokenizer.chat_template,special_tokens=tokenizer.special_tokens_map,
        tokenizer_sha256=hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
        feature_layout='quality_features: [13 NLL/length,35 raw spectrum,35 prompt-relative spectrum,16 PCA]; unstandardized; apply readout_mean/scale from saved scorer to reproduce margin',
        token_nll_trace_definition='Frozen Base conditional NLL at generated token, NOT actor logp; use quality_token_valid_mask, special tokens are excluded and filled with zero',
        note='Sample step k is generated before update k, from k-1 completed updates. No hidden tensors saved. Exact token IDs and compact scorer inputs are saved.'))


def dump_opd_fixed_rollout(directory, step, batch):
    """Save only the already-sampled OPD token IDs for the common-prefix basis.

    Unlike the signed-reward audit below, OPD has no direction_reward tensor.
    This function performs no model forward and never changes the batch.
    """
    out=Path(directory);out.mkdir(parents=True,exist_ok=True)
    t=batch.batch
    count=len(t['responses']);n=t['responses'].shape[1]
    path=out/f'pre_update_{step:04d}.jsonl.gz';tmp=path.with_name(path.name+'.tmp')
    with gzip.open(tmp,'wt',encoding='utf-8') as stream:
        for index in range(count):
            response_mask=t['response_mask'][index].bool()
            prompt_mask=t['attention_mask'][index,:-n].bool()
            prompt=t['input_ids'][index,:-n][prompt_mask].detach().cpu().tolist()
            response=t['responses'][index][response_mask].detach().cpu().tolist()
            if not prompt or not response:
                raise ValueError(f'empty OPD fixed rollout at row {index}')
            stream.write(json.dumps(dict(step=step,batch_row=index,
                prompt_token_ids=prompt,response_token_ids=response),
                ensure_ascii=False)+'\n')
    os.replace(tmp,path)


def dump_pre_update(directory, step, batch, tokenizer):
    """All rows (no sampling). Vector fields retained, never averaged to a scalar."""
    out=Path(directory);out.mkdir(parents=True,exist_ok=True)
    t={k:v.detach().cpu() for k,v in batch.batch.items()}
    nt=batch.non_tensor_batch
    count=len(t['responses']);n=t['responses'].shape[1]
    total=int(t['response_mask'].sum())
    path=out/f'pre_update_{step:04d}.jsonl.gz';tmp=path.with_name(path.name+'.tmp')
    with gzip.open(tmp,'wt',encoding='utf-8') as f:
        for i in range(count):
            rm=t['response_mask'][i].bool();pm=t['attention_mask'][i,:-n].bool()
            pid=t['input_ids'][i,:-n][pm].tolist();rid=t['responses'][i][rm].tolist()
            prompt=tokenizer.decode(pid,skip_special_tokens=False)
            response=tokenizer.decode(rid,skip_special_tokens=False)
            ids=pid+rid
            row=dict(step=step,policy_updates_before_rollout=step-1,batch_row=i,
                uid=serial(nt['uid'][i]) if 'uid' in nt else None,
                prompt_sha256=hashlib.sha256(json.dumps(pid).encode()).hexdigest(),
                trajectory_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                prompt_token_ids=pid,response_token_ids=rid,
                prompt=prompt,response=response,
                prompt_tokens=len(pid),response_tokens=len(rid),
                reaches_response_cap=len(rid)==n,
                last_token_is_eos=bool(rid and rid[-1]==tokenizer.eos_token_id),
                response_capacity=n,global_response_token_denominator=total,
                token_normalized_sequence_weight=len(rid)/max(total,1))
            for key in ('data_source','extra_info','reward_model','raw_prompt','boxed_format','acc'):
                if key in nt:row[key]=serial(nt[key][i])
            if 'advantages' in t:
                a=t['advantages'][i][rm]
                row['actor_advantage_min']=float(a.min()) if len(a) else 0.
                row['actor_advantage_max']=float(a.max()) if len(a) else 0.
            for key,value in t.items():
                if key.startswith(('direction_','quality_')):
                    v=value[i]
                    if key in ('quality_token_nll_trace','quality_token_valid_mask'):v=v[rm]
                    row[key]=serial(v)
            if 'token_level_scores' in t:
                row['math_monitor_score']=float(t['token_level_scores'][i].sum())
            reward=float(t['direction_reward'][i])
            row['signed_sequence_weight']=reward*len(rid)/max(total,1)
            row['monitor_score_is_training_reward']=False
            row['response_prefix_80']=response[:80]
            row['response_suffix_160']=response[-160:]
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    os.replace(tmp,path)


def dump_post_update(directory, step, metrics):
    # A pre_update file without a matching post_update is explicitly an unconfirmed update.
    atomic_json(Path(directory)/f'post_update_{step:04d}.json',dict(step=step,update_completed=True,metrics=redact(metrics)))


def commit_post_update(directory, step, metrics, publish_checkpoint):
    """Publish a restart point only after this step's local measurements are recorded."""
    if directory:
        dump_post_update(directory, step, metrics)
    if publish_checkpoint is not None:
        publish_checkpoint()
