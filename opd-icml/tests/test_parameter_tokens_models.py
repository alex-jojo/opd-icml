"""Exact token gradients through the real supported HF model/checkpoint paths."""
import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, Gemma3TextConfig, Gemma3ForCausalLM
from test_parameter_tokens import make_actor, make_data, reference_energy, audit_module


@pytest.mark.parametrize('family', ['qwen3', 'gemma3'])
def test_checkpointed_hf_full_parameter_gradients(family):
    torch.manual_seed(50)
    common = dict(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                  num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                  vocab_size=32, max_position_embeddings=64, attention_dropout=0.)
    if family == 'qwen3':
        cfg = Qwen3Config(**common)
        cfg._attn_implementation = 'eager'
        model = Qwen3ForCausalLM(cfg)
    else:
        cfg = Gemma3TextConfig(**common, sliding_window=8, sliding_window_pattern=2,
                               query_pre_attn_scalar=8)
        cfg._attn_implementation = 'eager'
        model = Gemma3ForCausalLM(cfg)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    actor, data = make_actor(), make_data()
    actor.actor_module = model
    actor.actor_optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    expected = reference_energy(actor, data)
    result = audit_module.FullParameterTokenAudit(actor, {'mode': 'exact_all'}).measure(data, 1, 0)
    np.testing.assert_allclose(result['energy'], expected, rtol=3e-5, atol=1e-7)
    assert all(p.grad is None for p in model.parameters())
    assert model.training and model.model.training and model.is_gradient_checkpointing
