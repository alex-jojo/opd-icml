"""Official model/FSDP forward, compact fixed-support Linear output head.

Select the UNION of cached vocabulary IDs once, use a regular differentiable
Linear on those rows, then gather each position's own support. This preserves
weight gradients (including tied embeddings) without a custom backward or
per-token full-vocabulary gradient allocation. Top-k discovery stays dense.
Single-token audits select the prediction hidden and its support BEFORE the
Linear, while the official model still processes the complete input context.
"""
from contextlib import contextmanager
import torch
import torch.nn.functional as F


@contextmanager
def compact_head(head, ids=None, hidden_only=False, response_position=None):
    if not isinstance(head,torch.nn.Linear):raise TypeError('compact OPD head requires nn.Linear')
    if hasattr(head,'_opd_compact_active'):raise RuntimeError('nested compact head')
    local_forward=head.__dict__.get('forward')
    head._opd_compact_active=not hidden_only
    def call(hidden):
        if hidden_only:
            return hidden.new_zeros((*hidden.shape[:-1],1))
        if hidden.ndim!=3 or ids.ndim!=3 or hidden.shape[0]!=ids.shape[0] or hidden.shape[1]!=ids.shape[1]+1:
            raise ValueError('compact head requires response_length+1 hidden positions')
        prediction_hidden=hidden[:,:-1]
        support=ids
        if response_position is not None:
            # hidden starts one input token BEFORE the first response token.
            prediction_hidden=prediction_hidden[:,response_position:response_position+1]
            support=support[:,response_position:response_position+1]
        unique,inverse=torch.unique(support.reshape(-1),sorted=True,return_inverse=True)
        weight=head.weight.index_select(0,unique)
        bias=None if head.bias is None else head.bias.index_select(0,unique)
        logits=F.linear(prediction_hidden,weight,bias)
        selected=logits.gather(-1,inverse.reshape_as(support))
        # Keep a final unused slot, also when only one prediction is returned.
        return torch.cat((selected,selected.new_zeros((len(selected),1,ids.shape[-1]))),dim=1)
    head.forward=call
    try:yield
    finally:
        if local_forward is None:del head.forward
        else:head.forward=local_forward
        del head._opd_compact_active


def enabled(actor):
    return bool(getattr(actor.config,'policy_loss',{}).get('replication17',{}).get('compact_head',False))
