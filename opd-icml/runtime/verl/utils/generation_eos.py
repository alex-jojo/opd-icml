"""Generation stop policy and masks derived from actual returned token lengths."""
from numbers import Integral

import torch


def merge_eos_ids(*values):
    result = []
    for value in values:
        if value is None:
            continue
        items = [value] if isinstance(value, Integral) else value
        for item in items:
            if not isinstance(item, Integral) or item < 0:
                raise ValueError('EOS IDs must be nonnegative integers')
            if int(item) not in result:
                result.append(int(item))
    return result


def extend_sampling_eos(kwargs, model_config):
    """Opt-in prepared-model EOS policy; preserve unrelated sampling settings."""
    result = dict(kwargs)
    eos = getattr(model_config, 'opd_response_eos_token_ids', None)
    if eos is not None and not result.get('ignore_eos', False):
        result['stop_token_ids'] = merge_eos_ids(result.get('stop_token_ids'), eos)
    return result


def response_mask_from_lengths(response, lengths, dtype=torch.int64):
    """Count only returned tokens, including terminal EOS, even if EOS == PAD."""
    lengths = torch.as_tensor(lengths, device=response.device, dtype=torch.long)
    if response.ndim != 2 or lengths.ndim != 1 or len(lengths) != response.shape[0]:
        raise ValueError('Response lengths must match the returned response batch')
    if bool((lengths < 0).any()) or bool((lengths > response.shape[1]).any()):
        raise ValueError('Returned response length is outside the padded response width')
    positions = torch.arange(response.shape[1], device=response.device)
    return (positions[None, :] < lengths[:, None]).to(dtype)
