"""Opt-in DP2 OPD: global batches/statistics, rank-local model computation."""
import numpy as np
import torch
import torch.distributed as dist

_CPU_GROUP=None


def cpu_group():
    """Keep CPU batches/diagnostic arrays OFF the NCCL CUDA staging path.

    NCCL object collectives pickle CPU payloads into CUDA byte buffers. With
    long rollouts and full hidden panels those buffers are needlessly large.
    All ranks create this group together; model/gradient collectives stay NCCL.
    """
    global _CPU_GROUP
    if size()==1 or dist.get_backend()!='nccl':return None
    world=dist.group.WORLD
    if _CPU_GROUP is None or _CPU_GROUP[0] is not world:
        if not dist.is_gloo_available():raise RuntimeError('DP2 CPU observation exchange requires Gloo')
        _CPU_GROUP=(world,dist.new_group(backend='gloo'))
    return _CPU_GROUP[1]


def size():return dist.get_world_size() if dist.is_initialized() else 1
def rank():return dist.get_rank() if dist.is_initialized() else 0
def primary():return rank()==0
def barrier():
    if size()>1:dist.barrier()


def exists(path):
    result=[path.exists() if primary() else None]
    if size()>1:dist.broadcast_object_list(result,src=0,group=cpu_group())
    return bool(result[0])


def primary_call(function):
    """Do CPU decompositions once; broadcast results or a shared failure."""
    if size()==1:return function()
    value=[None]
    if primary():
        try:value[0]={'result':function()}
        except Exception as exc:value[0]={'error':type(exc).__name__+': '+str(exc)}
    dist.broadcast_object_list(value,src=0,group=cpu_group())
    if 'error' in value[0]:raise RuntimeError('Global observation failed: '+value[0]['error'])
    return value[0]['result']


def gather(value):
    if size()==1:return [value]
    result=[None]*size();dist.all_gather_object(result,value,group=cpu_group());return result


def global_batch(data):
    return type(data).concat(gather(data)) if size()>1 else data


def shard(data):
    if len(data)%size():raise ValueError('Diagnostic/train trajectories must divide evenly across DP ranks')
    count=len(data)//size();offset=rank()*count
    return data.select_idxs(list(range(offset,offset+count))) if size()>1 else data,offset


def merge_fields(values):
    values=[v for v in values if v]
    if not values:raise ValueError('Missing distributed observation fields')
    if any(set(v)!=set(values[0]) for v in values):raise ValueError('Distributed observation fields differ')
    return {k:(torch.cat([v[k] for v in values]) if torch.is_tensor(values[0][k])
               else np.concatenate([v[k] for v in values])) for k in values[0]}


def gather_audit(observer):
    if size()==1:return
    payload=dict(fields={k:torch.cat(v) for k,v in observer.token_fields.items()},
        scores={k:np.concatenate([s[k] for s in observer.scores]) for k in observer.scores[0]},
        z=torch.cat(observer.raw_z),count=observer.count)
    parts=gather(payload)
    fields=merge_fields([p['fields'] for p in parts]);scores=merge_fields([p['scores'] for p in parts])
    observer.token_fields={k:[v] for k,v in fields.items()};observer.scores=[scores]
    observer.raw_z=[torch.cat([p['z'] for p in parts])];observer.count=sum(p['count'] for p in parts)


def total(value,average=False):
    if size()==1:return value
    device=torch.device('cuda',torch.cuda.current_device()) if dist.get_backend()=='nccl' else torch.device('cpu')
    result=torch.as_tensor(value,dtype=torch.float64,device=device).clone();dist.all_reduce(result)
    return float(result)/(size() if average else 1)


def sum_gram(gram,count):
    if size()==1:return gram,count
    dist.all_reduce(gram)
    n=torch.tensor(count,device=gram.device,dtype=torch.int64);dist.all_reduce(n)
    return gram,int(n)
