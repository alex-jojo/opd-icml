"""Exact token gradients through the real supported HF model/checkpoint paths."""
import numpy as np
import pytest
import torch
from transformers import (LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM,
                          Qwen3Config, Qwen3ForCausalLM, Gemma3TextConfig, Gemma3ForCausalLM)
from test_parameter_tokens import make_actor, make_data, reference_energy, audit_module, opd


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


@pytest.mark.parametrize('family', ['llama', 'qwen2', 'qwen3', 'gemma3'])
@pytest.mark.parametrize('prefix', [False, True])
def test_single_position_checkpointed_hf_gradients_match_dense_head(family, prefix):
    torch.manual_seed(51)
    config_type, model_type = {
        'llama': (LlamaConfig, LlamaForCausalLM),
        'qwen2': (Qwen2Config, Qwen2ForCausalLM),
        'qwen3': (Qwen3Config, Qwen3ForCausalLM),
        'gemma3': (Gemma3TextConfig, Gemma3ForCausalLM),
    }[family]
    common = dict(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                  num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                  vocab_size=64, max_position_embeddings=64, attention_dropout=0.,
                  tie_word_embeddings=True)
    if family == 'gemma3':
        # Exercise post-head model transformations, not just a plain Linear.
        common.update(sliding_window=8, sliding_window_pattern=2,
                      query_pre_attn_scalar=8, final_logit_softcapping=.1)
    config = config_type(**common)
    config._attn_implementation = 'eager'
    model = model_type(config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    actor, data = make_actor(), make_data().select_idxs([0])
    actor.actor_module = model
    actor.actor_optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    inputs = data.batch
    # Keep absolute positions after removing a common all-zero prompt prefix.
    for key in ('input_ids', 'attention_mask', 'position_ids'):
        inputs[key] = torch.cat((torch.zeros((1, 2), dtype=torch.long), inputs[key]), dim=1)
    inputs['opd_top16_ids'] = torch.arange(48).reshape(1, 3, 16).flip(-1)
    policy = actor.config.policy_loss['replication17']
    policy['trim_prompt_padding'] = True
    parameters = list(model.parameters())
    for position in range(3):
        teacher = inputs['opd_teacher_top16_logits'][:, position:position+1]
        policy['compact_head'] = False
        _, dense, _, _ = opd.forward(actor, inputs, 1., inputs['opd_top16_ids'], diagnostics=False)
        expected = dense[:, position:position+1]
        expected_kl = opd.reverse_kl(expected, teacher).sum()
        expected_gradients = torch.autograd.grad(expected_kl, parameters)

        policy['compact_head'] = True
        selected_inputs = audit_module.causal_token_prefix(inputs, position) if prefix else inputs
        _, selected, _, _ = opd.forward(actor, selected_inputs, 1., selected_inputs['opd_top16_ids'],
                                        diagnostics=False, response_position=position)
        actual_kl = opd.reverse_kl(selected, teacher).sum()
        actual_gradients = torch.autograd.grad(actual_kl, parameters)
        torch.testing.assert_close(selected, expected, rtol=3e-5, atol=1e-7)
        torch.testing.assert_close(actual_kl, expected_kl, rtol=3e-5, atol=1e-7)
        for actual, reference in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual, reference, rtol=5e-4, atol=1e-6)
    assert model.is_gradient_checkpointing and all(p.grad is None for p in parameters)
