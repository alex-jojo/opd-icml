"""DeepSeek Base->Instruct runtime, token IDs/EOS, profiles and Llama gradients."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import model_assets
from deepseek_assets import prepare_deepseek_student


def pair(tmp_path):
    vocab={'<unk>':0,'<bos>':1,'<end>':2,'<|EOT|>':3,'<pad>':4}
    vocab.update({f't{i}':i for i in range(5,32)})
    core=Tokenizer(WordLevel(vocab,unk_token='<unk>'))
    student=tmp_path/'base';teacher=tmp_path/'instruct'
    student.mkdir();teacher.mkdir()
    template="{{ bos_token }}{% for m in messages %}{{ '### Instruction: '+m['content']+'\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '### Response:' }}{% endif %}"
    for path,eos,chat in ((student,'<end>',None),(teacher,'<|EOT|>',template)):
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=core,unk_token='<unk>',bos_token='<bos>',eos_token=eos,pad_token='<pad>')
        tokenizer.chat_template=chat
        tokenizer.save_pretrained(path)
        config=LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
            num_attention_heads=2,num_key_value_heads=1,bos_token_id=1,eos_token_id=vocab[eos],pad_token_id=4)
        model=LlamaForCausalLM(config)
        model.save_pretrained(path,safe_serialization=path==teacher)
    return student,teacher


def test_runtime_shares_teacher_template_eos_and_preserves_all_base_tensors(tmp_path):
    torch.manual_seed(41)
    student,teacher=pair(tmp_path)
    original_config=(student/'config.json').read_bytes()
    original_tokenizer=(student/'tokenizer_config.json').read_bytes()
    source=AutoModelForCausalLM.from_pretrained(student,attn_implementation='eager')
    prepared=prepare_deepseek_student(student,teacher,tmp_path/'runtime')
    runtime=AutoModelForCausalLM.from_pretrained(prepared,attn_implementation='eager')
    a=AutoTokenizer.from_pretrained(prepared);b=AutoTokenizer.from_pretrained(teacher)
    assert a.get_vocab()==b.get_vocab() and a.eos_token_id==b.eos_token_id==3
    assert a.chat_template==b.chat_template
    assert a.apply_chat_template([{'role':'user','content':'code'}],tokenize=False,add_generation_prompt=True).endswith('### Response:')
    for name,tensor in source.state_dict().items():
        assert torch.equal(tensor,runtime.state_dict()[name])
    assert os.path.samefile(student/'pytorch_model.bin',prepared/'pytorch_model.bin')
    assert (student/'config.json').read_bytes()==original_config
    assert (student/'tokenizer_config.json').read_bytes()==original_tokenizer
    assert AutoTokenizer.from_pretrained(student).eos_token_id==2
    assert prepare_deepseek_student(student,teacher,prepared)==prepared
    marker=json.loads((prepared/'opd_pair_runtime.json').read_text())
    assert marker['weights_modified'] is False and marker['weights_linked'] is True
    (prepared/'config.json').write_text('{}')
    with pytest.raises(ValueError,match='runtime files changed'):
        prepare_deepseek_student(student,teacher,prepared)


def test_preparation_integrates_with_existing_model_api(tmp_path):
    student,teacher=pair(tmp_path)
    result=model_assets.prepare_models('deepseek-coder-1.3b',student,teacher,model_dir=tmp_path/'prepared')
    assert Path(result['teacher'])==teacher
    assert Path(result['student'])!=student
    assert AutoTokenizer.from_pretrained(result['student']).eos_token=='<|EOT|>'


def test_mismatched_vocab_and_existing_directory_are_rejected(tmp_path):
    student,teacher=pair(tmp_path)
    occupied=tmp_path/'occupied';occupied.mkdir();(occupied/'keep').write_text('user data')
    with pytest.raises(ValueError,match='empty'):
        prepare_deepseek_student(student,teacher,occupied)
    tokenizer=AutoTokenizer.from_pretrained(teacher);tokenizer.add_tokens(['new-token']);tokenizer.save_pretrained(teacher)
    with pytest.raises(ValueError,match='identical'):
        prepare_deepseek_student(student,teacher,tmp_path/'out')
    assert (occupied/'keep').read_text()=='user data'


@pytest.mark.parametrize('experiment',['paired-full','original-full','early-stop','reuse'])
def test_deepseek_code_profile_and_launcher(experiment):
    result=subprocess.run([sys.executable,str(ROOT/'train.py'),'--scale','deepseek-coder-1.3b','--task','code',
        '--experiment',experiment,'--hardware','l40x4','--dry-run'],check=True,capture_output=True,text=True)
    plan=json.loads(result.stdout)
    assert plan['models']['student']['repo']=='deepseek-ai/deepseek-coder-1.3b-base'
    assert plan['models']['teacher']['repo']=='deepseek-ai/deepseek-coder-1.3b-instruct'
    assert plan['gpu_count']==4 and plan['task']=='code'
    assert plan['rounds_cap']==(53 if experiment=='reuse' else 106)
    if experiment=='paired-full':
        assert plan['parameter_token_gradients']=={'enabled':True,'mode':'exact_all'}
    assert plan['models']['teacher']['allow_patterns']==['*.json','*.safetensors','tokenizer.model']


def test_hydra_composition_and_full_llama_parameter_gradients(tmp_path):
    from hydra import compose,initialize_config_dir
    name='deepseek-coder-1.3b'
    search='hydra.searchpath='+json.dumps(['file://'+(ROOT/'runtime/verl/trainer/config').as_posix()],ensure_ascii=False)
    with initialize_config_dir(config_dir=str(ROOT/'configs'/name),version_base=None):
        config=compose(config_name='code-paired-full',overrides=[search])
    assert config.actor_rollout_ref.actor.policy_loss.replication17.distributed_world_size==4
    assert config.actor_rollout_ref.actor.policy_loss.gradient_study.parameter_tokens.enabled
    from test_parameter_tokens import make_data,make_actor,reference_energy,audit_module
    student,teacher=pair(tmp_path)
    prepared=prepare_deepseek_student(student,teacher,tmp_path/'runtime')
    actor=make_actor()
    actor.actor_module=AutoModelForCausalLM.from_pretrained(prepared,attn_implementation='eager')
    actor.actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    actor.actor_optimizer=torch.optim.AdamW(actor.actor_module.parameters(),lr=.001)
    data=make_data();expected=reference_energy(actor,data)
    result=audit_module.FullParameterTokenAudit(actor,{'mode':'exact_all'}).measure(data,1,0)
    np.testing.assert_allclose(result['energy'],expected,rtol=3e-5,atol=1e-7)
    assert all(p.grad is None for p in actor.actor_module.parameters())
