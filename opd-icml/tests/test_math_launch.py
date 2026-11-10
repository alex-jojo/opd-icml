"""Run the actual math shell entry point in dry-run mode, without GPU or downloads."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
pytestmark=pytest.mark.skipif(os.name=='nt',reason='Bash entry point runs on the Linux training server')


def run_script(**settings):
    env=os.environ.copy()
    for name in ['SCALE','EXPERIMENT','HARDWARE','ROLLOUT_BATCH_SIZE','MINI_BATCH_SIZE','MICRO_BATCH_SIZE',
                 'LEARNING_RATE','ROLLOUT_MAX_NUM_SEQS','ROLLOUT_GPU_MEMORY_UTILIZATION',
                 'PARAMETER_TOKEN_GRADIENTS','PARAMETER_TOKEN_SAMPLES','STUDENT_PATH','TEACHER_PATH',
                 'DATA_PATH','OUTPUT_DIR','CUDA_VISIBLE_DEVICES']:
        env.pop(name,None)
    env.update(PYTHON_BIN=sys.executable,**settings)
    return subprocess.run(['bash',str(ROOT/'scripts/train_math.sh'),'--dry-run'],
                          env=env,capture_output=True,text=True,cwd='/tmp')


@pytest.mark.parametrize('scale',['1.7b','4b','8b','gemma3-12b'])
@pytest.mark.parametrize('experiment',['paired-full','original-full','early-stop','reuse'])
def test_math_model_and_experiment_matrix_preserves_recorded_budget(scale,experiment):
    recorded=json.loads((ROOT/'configs'/scale/(experiment+'.yaml')).read_text())
    result=run_script(SCALE=scale,EXPERIMENT=experiment)
    assert result.returncode==0,result.stderr
    plan=json.loads(result.stdout)
    assert plan['task']=='math' and plan['scale']==scale and plan['experiment']==experiment
    assert plan['data'].endswith('data/dapo14k_train.parquet')
    assert plan['gpu_count']==plan['rollout_tensor_parallel_size']==1
    assert plan['rollout_batch']==plan['mini_batch_size']==recorded['data']['train_batch_size']
    assert plan['rounds_cap']==recorded['trainer']['opd_pause_after_rollout_rounds']
    assert plan['updates_per_round']==recorded['actor_rollout_ref']['actor']['policy_loss']['jump_reuse_updates']
    if experiment=='paired-full':
        assert plan['parameter_token_gradients']==dict(enabled=True,mode='kl_stratified',sample_size=256,seed=42)
    else:
        assert plan['parameter_token_gradients']==recorded['actor_rollout_ref']['actor']['policy_loss'].get(
            'gradient_study',{}).get('parameter_tokens',{'enabled':False})


@pytest.mark.parametrize('scale',['1.7b','4b','8b','gemma3-12b'])
@pytest.mark.parametrize('hardware',['l40x4','recorded'])
def test_hardware_modes_keep_config_and_device_count_consistent(scale,hardware):
    result=run_script(SCALE=scale,HARDWARE=hardware)
    assert result.returncode==0,result.stderr
    plan=json.loads(result.stdout)
    expected=4 if hardware=='l40x4' else json.loads((ROOT/'configs'/scale/'paired-full.yaml').read_text())['trainer']['n_gpus_per_node']
    assert plan['gpu_count']==expected
    assert plan['hardware_profile']==('l40x4' if hardware=='l40x4' else None)


def test_batch_diagnostic_and_quoted_path_overrides():
    result=run_script(SCALE='4b',ROLLOUT_BATCH_SIZE='64',MICRO_BATCH_SIZE='2',
        PARAMETER_TOKEN_SAMPLES='128',ROLLOUT_MAX_NUM_SEQS='32',
        STUDENT_PATH='/tmp/student weights',TEACHER_PATH='/tmp/teacher weights',
        DATA_PATH='/tmp/math data/train.parquet',OUTPUT_DIR='/tmp/math runs/test')
    assert result.returncode==0,result.stderr
    plan=json.loads(result.stdout)
    assert plan['rollout_batch']==plan['mini_batch_size']==64
    assert plan['micro_batch_size_per_gpu']==2 and plan['rollout_max_num_seqs']==32
    assert plan['parameter_token_gradients']['sample_size']==128
    assert plan['data']=='/tmp/math data/train.parquet' and plan['output']=='/tmp/math runs/test'


def test_unsupported_hardware_probe_combination_fails_without_removing_probes():
    result=run_script(SCALE='1.7b',EXPERIMENT='original-full',HARDWARE='l40x4')
    assert result.returncode!=0
    assert 'single-GPU disposable shadow probes' in result.stderr


def test_inherit_exact_and_off_diagnostic_modes():
    for mode in ['inherit','exact-all','off']:
        result=run_script(PARAMETER_TOKEN_GRADIENTS=mode)
        assert result.returncode==0,result.stderr
        options=json.loads(result.stdout)['parameter_token_gradients']
        assert options['enabled']==(mode!='off')
        assert options['mode']=='exact_all'
