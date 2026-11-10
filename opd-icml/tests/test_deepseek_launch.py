"""Validate batch overrides against the actual launcher and shell entry point."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
BASE=[sys.executable,str(ROOT/'train.py'),'--scale','deepseek-coder-1.3b','--task','code',
      '--experiment','paired-full','--hardware','a100x1','--dry-run']


def launch(*args):
    return subprocess.run(BASE+list(args),capture_output=True,text=True,cwd=ROOT)


def test_explicit_micro_batch_wins_over_hardware_and_hydra_matches_report():
    from hydra import compose,initialize_config_dir
    result=launch('--rollout-batch-size','128','--mini-batch-size','128','--micro-batch-size','4',
                  '--learning-rate','2e-5','--rollout-max-num-seqs','32','--rollout-gpu-memory-utilization','.4')
    assert result.returncode==0,result.stderr
    report=json.loads(result.stdout)
    assert report['micro_batch_size_per_gpu']==4 and report['mini_batch_size']==128
    assert report['rollout_batch']==128 and report['learning_rate']==2e-5
    assert report['rollout_max_num_seqs']==32 and report['rollout_gpu_memory_utilization']==.4
    assert report['rounds_cap']==106 and report['parameter_token_gradients']['sample_size']==256
    command=shlex.split(report['command'])
    assert sum(c.startswith('++actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=') for c in command)==1
    overrides=[c for c in command if c.startswith('++')]
    search='hydra.searchpath='+json.dumps(['file://'+(ROOT/'runtime/verl/trainer/config').as_posix()],ensure_ascii=False)
    with initialize_config_dir(config_dir=str(ROOT/'configs/deepseek-coder-1.3b'),version_base=None):
        cfg=compose(config_name='code-paired-full',overrides=[search]+overrides)
    assert cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu==4
    assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size==128
    assert cfg.actor_rollout_ref.actor.optim.lr==2e-5
    assert cfg.actor_rollout_ref.rollout.max_num_seqs==32


@pytest.mark.parametrize('arguments',[
    ['--rollout-batch-size','64','--mini-batch-size','32'],
    ['--micro-batch-size','3'],['--mini-batch-size','0'],
    ['--rollout-max-num-seqs','-1'],['--learning-rate','inf'],
    ['--rollout-gpu-memory-utilization','1'],['--rollout-gpu-memory-utilization','nan'],
])
def test_invalid_batch_settings_fail_before_loading_models(arguments):
    result=launch(*arguments)
    assert result.returncode!=0
    assert 'error:' in result.stderr


def test_changing_rollout_budget_requires_explicit_matching_mini_batch():
    result=launch('--rollout-batch-size','64','--mini-batch-size','64','--micro-batch-size','2')
    assert result.returncode==0,result.stderr
    report=json.loads(result.stdout)
    assert report['rollout_batch']==report['mini_batch_size']==64
    assert report['micro_batch_size_per_gpu']==2
    assert report['rounds_cap']==106


@pytest.mark.skipif(os.name=='nt',reason='Exercise the real bash entry point on the Linux server')
def test_shell_default_and_environment_overrides():
    script=ROOT/'scripts/train_deepseek_code_a100.sh'
    subprocess.run(['bash','-n',str(script)],check=True)
    for micro,sequences in [(1,16),(4,32)]:
        env=os.environ.copy()
        env.update(PYTHON_BIN=sys.executable,MICRO_BATCH_SIZE=str(micro),ROLLOUT_MAX_NUM_SEQS=str(sequences),
                   ROLLOUT_BATCH_SIZE='128',MINI_BATCH_SIZE='128',LEARNING_RATE='1e-5',
                   ROLLOUT_GPU_MEMORY_UTILIZATION='.3',PARAMETER_TOKEN_SAMPLES='256',
                   OUTPUT_DIR='runs/code/deepseek-coder-1.3b/script-config-test')
        result=subprocess.run(['bash',str(script),'--dry-run'],env=env,capture_output=True,text=True,check=True,cwd='/tmp')
        report=json.loads(result.stdout)
        assert report['micro_batch_size_per_gpu']==micro and report['rollout_max_num_seqs']==sequences
        assert report['mini_batch_size']==report['rollout_batch']==128
        assert report['gpu_count']==1 and report['parameter_token_gradients']['sample_size']==256
