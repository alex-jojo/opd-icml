"""Small per-rank RNG state at the adaptive FULL -> frozen Adapter boundary."""
from pathlib import Path
import random
import numpy as np
import torch
from . import opd_distributed as dp


def save_rng(worker,path):
    state=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
               world=dp.size(),rank=dp.rank())
    if torch.cuda.is_available():state['cuda']=torch.cuda.get_rng_state()
    for name in ('gen_random_states','torch_random_states'):
        if hasattr(worker,name):state[name]=getattr(worker,name).cpu()
    target=Path(path)/f'handoff_rng_rank_{dp.rank()}.pt'
    torch.save(state,target)


def load_rng(worker,path):
    state=torch.load(Path(path)/f'handoff_rng_rank_{dp.rank()}.pt',map_location='cpu',weights_only=False)
    if state['world']!=dp.size() or state['rank']!=dp.rank():raise ValueError('Handoff GPU topology changed')
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if 'cuda' in state:torch.cuda.set_rng_state(state['cuda'])
    for name in ('gen_random_states','torch_random_states'):
        if name in state:setattr(worker,name,state[name])
