"""Real rollout method with a scripted backend; no GPU or external downloads."""
import ast
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import types

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('generation_eos_cpu', ROOT/'runtime/verl/utils/generation_eos.py')
eos = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eos)


def test_dual_eos_extends_existing_stops_and_preserves_sampling_settings():
    config = types.SimpleNamespace(opd_response_eos_token_ids=[32021,32014])
    original = dict(temperature=1.,max_tokens=3072,stop_token_ids=[99,32021],ignore_eos=False)
    fixed = eos.extend_sampling_eos(original,config)
    assert fixed['stop_token_ids'] == [99,32021,32014]
    assert original['stop_token_ids'] == [99,32021]
    assert fixed['max_tokens'] == 3072 and fixed['temperature'] == 1.
    assert eos.extend_sampling_eos(original,types.SimpleNamespace()) == original
    ignored = dict(original,ignore_eos=True)
    assert eos.extend_sampling_eos(ignored,config) == ignored


def test_masks_include_terminal_eos_and_exclude_all_padding_even_when_ids_are_identical():
    response = torch.tensor([[7,32014,32014,32014], [8,32021,32014,32014],
                             [9,10,11,12], [32014,32014,32014,32014], [32014]*4])
    mask = eos.response_mask_from_lengths(response,[2,2,4,1,0])
    assert mask.tolist() == [[1,1,0,0],[1,1,0,0],[1,1,1,1],[1,0,0,0],[0,0,0,0]]
    assert mask.sum(1).tolist() == [2,2,4,1,0]
    with pytest.raises(ValueError,match='outside'):
        eos.response_mask_from_lengths(response,[2,2,5,1,0])


class TensorBatch(dict):
    def __init__(self,values,batch_size=None):
        super().__init__(values)


class Proto:
    def __init__(self,batch,non_tensor_batch=None):
        self.batch=batch
        self.non_tensor_batch=non_tensor_batch or {}
        self.meta_info={}


def padded(rows,pad_id,max_length):
    return torch.tensor([list(row)+[pad_id]*(max_length-len(row)) for row in rows])


def real_rollout_method():
    path=ROOT/'runtime/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='vLLMRollout')
    method=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='generate_sequences')
    method.decorator_list=[]
    namespace=dict(torch=torch,np=np,DataProto=Proto,TensorDict=TensorBatch,
                   pad_2d_list_to_length=padded,response_mask_from_lengths=eos.response_mask_from_lengths)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(path),'exec'),namespace)
    return namespace['generate_sequences']


@pytest.mark.parametrize('logprobs',[True,False])
def test_actual_sync_rollout_and_trainer_use_returned_lengths_for_native_and_instruct_eos(logprobs):
    returned=[[10,32014],[11,12,32021],[13,14,15,16],[32014]]
    @contextmanager
    def update(**kwargs):
        yield
    def generate(**kwargs):
        return [types.SimpleNamespace(outputs=[types.SimpleNamespace(token_ids=tokens,
            logprobs=[{token:types.SimpleNamespace(logprob=-.1)} for token in tokens])]) for tokens in returned]
    actor=types.SimpleNamespace(pad_token_id=32014,lora_kwargs={},
        config=types.SimpleNamespace(response_length=4,calculate_log_probs=logprobs),
        sampling_params=object(),update_sampling_params=update,
        inference_engine=types.SimpleNamespace(generate=generate))
    prompts=Proto(dict(input_ids=torch.tensor([[1,2]]*4),attention_mask=torch.ones(4,2,dtype=torch.long),
                       position_ids=torch.tensor([[0,1]]*4)),dict(raw_prompt_ids=np.array([[1,2]]*4,dtype=object)))
    prompts.meta_info={'eos_token_id':32021}
    result=real_rollout_method()(actor,prompts)
    assert result.batch['attention_mask'][:,-4:].sum(1).tolist()==[2,3,4,1]
    assert result.batch['responses'][0].tolist()==[10,32014,32014,32014]
    # Downstream trainer derives response_mask from this exact attention_mask.
    path=ROOT/'runtime/verl/trainer/ppo/ray_trainer.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    method=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='compute_response_mask')
    namespace={'DataProto':Proto}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(path),'exec'),namespace)
    assert namespace['compute_response_mask'](result).sum(1).tolist()==[2,3,4,1]


def test_runtime_v2_generation_policy_survives_hf_save_and_reload(tmp_path):
    from transformers import AutoModelForCausalLM,GenerationConfig
    from test_deepseek_opd import pair
    import sys
    sys.path.insert(0,str(ROOT))
    from deepseek_assets import prepare_deepseek_student
    student,teacher=pair(tmp_path)
    prepared=prepare_deepseek_student(student,teacher,tmp_path/'runtime-v2')
    model=AutoModelForCausalLM.from_pretrained(prepared,attn_implementation='eager')
    model.save_pretrained(tmp_path/'hf-export')
    assert GenerationConfig.from_pretrained(tmp_path/'hf-export').eos_token_id==[3,2]
    assert model.config.opd_response_eos_token_ids==[3,2]


def test_installed_vllm_v1_scheduler_stops_on_both_eos_ids():
    pytest.importorskip('vllm')
    from vllm import SamplingParams
    from vllm.v1.core.sched.utils import check_stop
    config=types.SimpleNamespace(opd_response_eos_token_ids=[32021,32014])
    params=SamplingParams(**eos.extend_sampling_eos(dict(max_tokens=3072,ignore_eos=False),config))
    for token in [32014,32021,42]:
        request=types.SimpleNamespace(num_tokens=3,num_output_tokens=1,max_tokens=3072,
            eos_token_id=32021,sampling_params=params,output_token_ids=[token])
        assert check_stop(request,5120) is (token in [32014,32021])
        if token in [32014,32021]:
            assert request.status.name=='FINISHED_STOPPED'
