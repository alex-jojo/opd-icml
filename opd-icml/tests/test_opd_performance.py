"""Topology, exact diagnostic optimizations, and complete multi-rank saves."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest
import torch

from test_parameter_tokens import ROOT,audit_module,make_actor,make_data,optimizer_update,reference_energy

spec=importlib.util.spec_from_file_location('performance_launcher',ROOT/'train.py')
launcher=importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.mark.parametrize('gpus,tp',[(1,1),(2,2),(4,4),(8,8),(4,1),(8,2)])
@pytest.mark.parametrize('task',['math','code'])
def test_gpu_count_and_rollout_tp_keep_global_training_protocol(gpus,tp,task):
    profile=ROOT/'configs/deepseek-coder-1.3b'
    _,cfg=launcher.training_config(profile,'paired-full',task)
    original=copy.deepcopy(cfg)
    overrides=launcher.apply_hardware_profile(cfg,'a100x1',gpus,tp)
    launcher.validate_gpu_topology(cfg)
    assert cfg['trainer']['n_gpus_per_node']==gpus
    assert cfg['actor_rollout_ref']['rollout']['tensor_model_parallel_size']==tp
    assert cfg['data']==original['data']
    expected=copy.deepcopy(original['actor_rollout_ref']['actor']['policy_loss'])
    expected['replication17']['distributed_world_size']=gpus
    assert cfg['actor_rollout_ref']['actor']['policy_loss']==expected
    assert cfg['actor_rollout_ref']['actor']['ppo_mini_batch_size']==128
    assert cfg['trainer']['opd_pause_after_rollout_rounds']==106
    assert len(overrides)>0


@pytest.mark.parametrize('gpus,tp',[(0,1),(-1,1),(4,3),(2,4),(2,0)])
def test_invalid_topology_fails_before_mutating_configuration(gpus,tp):
    _,cfg=launcher.training_config(ROOT/'configs/4b','paired-full','math')
    before=copy.deepcopy(cfg)
    with pytest.raises(ValueError,match='positive|divisible'):
        launcher.apply_hardware_profile(cfg,'a100x1',gpus,tp)
    assert cfg==before


def test_nondivisible_global_batch_is_rejected_without_changing_batch():
    _,cfg=launcher.training_config(ROOT/'configs/4b','paired-full','math')
    launcher.apply_hardware_profile(cfg,'a100x1',3,1)
    with pytest.raises(ValueError,match='divide evenly'):
        launcher.validate_gpu_topology(cfg)
    assert cfg['data']['train_batch_size']==128


def test_model_head_constraints(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps(dict(num_attention_heads=16,num_key_value_heads=2)))
    launcher.validate_model_tp(tmp_path,8)
    with pytest.raises(ValueError,match='attention heads'):
        launcher.validate_model_tp(tmp_path,3)
    (tmp_path/'config.json').write_text(json.dumps(dict(text_config=dict(num_attention_heads=16,num_key_value_heads=3))))
    with pytest.raises(ValueError,match='KV heads'):
        launcher.validate_model_tp(tmp_path,4)


@pytest.mark.parametrize('family',['llama','qwen3','gemma3_text'])
def test_prefix_only_diagnostic_matches_full_sequence_gradients_and_next_update(family):
    from transformers import (LlamaConfig,LlamaForCausalLM,Qwen3Config,Qwen3ForCausalLM,
                              Gemma3TextConfig,Gemma3ForCausalLM)
    torch.manual_seed(90)
    args=dict(hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,
              num_key_value_heads=1,head_dim=8,vocab_size=32,max_position_embeddings=64,attention_dropout=0.)
    config_type,model_type={'llama':(LlamaConfig,LlamaForCausalLM),'qwen3':(Qwen3Config,Qwen3ForCausalLM),
                           'gemma3_text':(Gemma3TextConfig,Gemma3ForCausalLM)}[family]
    config=config_type(**args)
    config._attn_implementation='eager'
    actor,data=make_actor(),make_data()
    actor.actor_module=model_type(config)
    actor.actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    actor.actor_optimizer=torch.optim.AdamW(actor.actor_module.parameters(),lr=.001)
    optimizer_update(actor,data)
    baseline=copy.deepcopy(actor)
    state=copy.deepcopy(actor.actor_optimizer.state_dict())
    full=audit_module.FullParameterTokenAudit(actor,dict(mode='exact_all',causal_prefix=False,graph_group_size=1)).measure(data,1,0)
    prefix=audit_module.FullParameterTokenAudit(actor,dict(mode='exact_all',causal_prefix=True,graph_group_size=1)).measure(data,1,0)
    assert prefix['causal_prefix'] and prefix['forward_input_tokens']<full['forward_input_tokens']
    np.testing.assert_allclose(prefix['energy'],full['energy'],rtol=3e-5,atol=1e-7)
    np.testing.assert_allclose(prefix['diagnostic_kl'],full['diagnostic_kl'],rtol=3e-5,atol=1e-7)
    assert prefix['backwards']==full['backwards']==5
    for index,values in state['state'].items():
        for key,value in values.items():
            assert torch.equal(value,actor.actor_optimizer.state_dict()['state'][index][key])
    optimizer_update(actor,data);optimizer_update(baseline,data)
    for actual,expected in zip(actor.actor_module.parameters(),baseline.actor_module.parameters()):
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_device_accumulation_matches_original_block_norm_and_does_not_touch_gradients():
    torch.manual_seed(7)
    actor,data=make_actor(),make_data()
    expected=reference_energy(actor,data)
    from test_parameter_tokens import objective
    objective(actor,data)[0,0].backward()
    before=[p.grad.clone() for p in actor.actor_module.parameters()]
    actual=audit_module.GradientEnergy(actor.actor_module).squared_norm()
    assert actual==pytest.approx(expected[0],rel=1e-12)
    for p,original in zip(actor.actor_module.parameters(),before):
        assert torch.equal(p.grad,original)


def checkpoint_module():
    spec=importlib.util.spec_from_file_location('performance_checkpoint',ROOT/'runtime/verl/trainer/ppo/opd_replication_checkpoint.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('world',[1,2,4,8])
@pytest.mark.parametrize('final_only',[True,False])
def test_checkpoint_requires_every_rank_before_publishing(tmp_path,world,final_only):
    base=tmp_path/'checkpoints';current=base/'global_step_106';actor=current/'actor';actor.mkdir(parents=True)
    (actor/'fsdp_config.json').write_text(json.dumps(dict(world_size=world)))
    (current/'opd_optimizer_updates.json').write_text(json.dumps(dict(rollout_round=106,optimizer_updates_done=106,budget_unit='rollout_rounds')))
    if not final_only:
        (current/'data.pt').write_bytes(b'data')
    for rank in range(world):
        for kind in (['model'] if final_only else ['model','optim','extra_state']):
            if rank==world-1 and kind=='model':continue
            (actor/f'{kind}_world_size_{world}_rank_{rank}.pt').write_bytes(b'state')
    module=checkpoint_module()
    with pytest.raises(ValueError,match='Incomplete checkpoint'):
        module.publish_and_rotate(base,106,1,final_model_only=final_only)
    assert not (base/'latest_checkpointed_iteration.txt').exists()
    (actor/f'model_world_size_{world}_rank_{world-1}.pt').write_bytes(b'state')
    module.publish_and_rotate(base,106,1,final_model_only=final_only)
    assert (base/'latest_checkpointed_iteration.txt').read_text()=='106'
    if final_only:
        assert json.loads((current/'final_model_ready.json').read_text())['world_size']==world


def test_interrupted_record_write_retains_checkpoint_until_replay(tmp_path):
    spec = importlib.util.spec_from_file_location('checkpoint_audit', ROOT/'runtime/verl/trainer/ppo/quality_audit.py')
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    checkpoint = checkpoint_module()
    base = tmp_path / 'checkpoints'
    for step in (10, 20):
        current = base / f'global_step_{step}'
        (current / 'actor').mkdir(parents=True)
        for kind in ('model', 'optim', 'extra_state'):
            (current / 'actor' / f'{kind}_world_size_1_rank_0.pt').write_bytes(b'shard')
        (current / 'data.pt').write_bytes(b'cursor')
        (current / 'opd_optimizer_updates.json').write_text(json.dumps(dict(
            rollout_round=step, optimizer_updates_done=step, budget_unit='rollout_rounds')))
    tracker = base / 'latest_checkpointed_iteration.txt'
    checkpoint.publish_and_rotate(base, 10, 1)
    records = tmp_path / 'records'
    records.mkdir()
    # A directory at the temporary filename simulates a failed atomic JSON write.
    obstruction = records / 'post_update_0020.json.tmp'
    obstruction.mkdir()
    def publish():
        assert json.loads((records / 'post_update_0020.json').read_text())['step'] == 20
        checkpoint.publish_and_rotate(base, 20, 1)
    with pytest.raises(OSError):
        audit.commit_post_update(records, 20, {}, publish)
    assert tracker.read_text() == '10' and (base / 'global_step_10').is_dir()
    obstruction.rmdir()
    audit.commit_post_update(records, 20, {}, publish)
    assert tracker.read_text() == '20' and not (base / 'global_step_10').exists()
