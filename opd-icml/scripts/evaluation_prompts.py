"""Chat prompt encoding shared by math and code evaluation."""

CHAT_PROMPT_ENCODING = 'chat_template_token_ids_v1'


def encode_chat_prompt(tokenizer, messages):
    """Match training: let the template supply BOS and turn markers once."""
    rendered = tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    return rendered, tokenizer.encode(rendered, add_special_tokens=False)


def validate_chat_prompt_encoding(rows):
    if any(row.get('prompt_encoding') != CHAT_PROMPT_ENCODING for row in rows):
        raise ValueError('Cached chat generations use a different prompt encoding; '
                         'use a new output directory for generation or grade the saved texts only')
