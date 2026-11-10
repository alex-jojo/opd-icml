"""Pinned model preparation and lossless Gemma 3 text-checkpoint extraction."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GEMMA_CHAT_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}"
    "{% if message['role'] not in ['user', 'assistant'] %}"
    "{{ raise_exception('Gemma OPD accepts user/assistant messages only') }}{% endif %}"
    "{{ '<start_of_turn>' + ('model' if message['role'] == 'assistant' else 'user') + '\\n' }}"
    "{{ message['content'] | trim }}{{ '<end_of_turn>\\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<start_of_turn>model\\n' }}{% endif %}"
)


def model_profiles():
    return sorted(path.parent.name for path in (ROOT / 'configs').glob('*/models.json'))


def read_models(scale):
    return json.loads((ROOT / 'configs' / scale / 'models.json').read_text(encoding='utf-8'))


def text_key(key):
    for prefix in ('language_model.model.', 'model.language_model.'):
        if key.startswith(prefix):
            return 'model.' + key[len(prefix):]
    if key.startswith('language_model.lm_head.'):
        return key[len('language_model.'):]
    if key.startswith('lm_head.'):
        return key
    return None


def export_gemma_text(source, destination):
    """Stream text tensors one source shard at a time; no quantization or model allocation."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    source, destination = Path(source).resolve(), Path(destination).resolve()
    config_bytes = (source / 'config.json').read_bytes()
    config = json.loads(config_bytes)
    if config['model_type'] == 'gemma3_text':
        return source
    if config['model_type'] != 'gemma3':
        raise ValueError(f'Expected Gemma 3 checkpoint, got {config["model_type"]}')
    text_config = dict(config['text_config'])
    if text_config.get('final_logit_softcapping') is not None:
        raise ValueError('OPD analytic head diagnostics require no final logit softcapping')
    source_files = sorted(source.glob('*.safetensors'))
    if not source_files:
        raise ValueError(f'No safetensors weights in {source}')
    identity = dict(source=str(source), config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                    shards=[dict(name=p.name, bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in source_files],
                    protocol='gemma3-text-turn-eos-v1')
    marker = destination / 'opd_text_export.json'
    if marker.exists():
        saved = json.loads(marker.read_text(encoding='utf-8'))
        if saved['source'] != identity:
            raise ValueError(f'Export destination belongs to another checkpoint: {destination}')
        index = destination / 'model.safetensors.index.json'
        if saved.get('complete') and index.is_file():
            files = set(json.loads(index.read_text())['weight_map'].values())
            if all((destination / name).is_file() for name in files):
                return destination
    elif destination.exists() and any(destination.iterdir()):
        raise ValueError(f'Choose an empty text-export directory: {destination}')
    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True)
    stop_id = tokenizer.convert_tokens_to_ids('<end_of_turn>')
    if stop_id is None or stop_id == tokenizer.unk_token_id:
        raise ValueError('Gemma tokenizer has no <end_of_turn> token')
    tokenizer.chat_template = GEMMA_CHAT_TEMPLATE
    tokenizer.eos_token = '<end_of_turn>'
    destination.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(dict(source=identity, complete=False), indent=2), encoding='utf-8')
    groups = []
    for path in source_files:
        with safe_open(str(path), framework='pt', device='cpu') as handle:
            keys = [(key, text_key(key)) for key in handle.keys() if text_key(key) is not None]
        if keys:
            groups.append((path, keys))
    if not groups:
        raise ValueError('No Gemma language-model tensors found')
    weight_map, total_bytes = {}, 0
    for number, (path, keys) in enumerate(groups, 1):
        filename = f'model-{number:05d}-of-{len(groups):05d}.safetensors'
        with safe_open(str(path), framework='pt', device='cpu') as handle:
            tensors = {new: handle.get_tensor(old) for old, new in keys}
        if len(tensors) != len(keys) or any(key in weight_map for key in tensors):
            raise ValueError('Duplicate text weight names')
        total_bytes += sum(t.numel() * t.element_size() for t in tensors.values())
        temporary = destination / (filename + '.tmp')
        save_file(tensors, str(temporary), metadata={'format': 'pt'})
        temporary.replace(destination / filename)
        weight_map.update({key: filename for key in tensors})
        del tensors
    if not {'model.embed_tokens.weight', 'model.norm.weight'}.issubset(weight_map):
        raise ValueError('Incomplete Gemma text checkpoint')
    text_config.update(model_type='gemma3_text', architectures=['Gemma3ForCausalLM'],
                       eos_token_id=stop_id,
                       tie_word_embeddings=text_config.get('tie_word_embeddings', config.get('tie_word_embeddings', True)))
    (destination / 'config.json').write_text(json.dumps(text_config, indent=2) + '\n', encoding='utf-8')
    (destination / 'model.safetensors.index.json').write_text(
        json.dumps(dict(metadata=dict(total_size=total_bytes), weight_map=weight_map), indent=2) + '\n', encoding='utf-8')
    tokenizer.save_pretrained(str(destination))
    generation = source / 'generation_config.json'
    settings = json.loads(generation.read_text()) if generation.is_file() else {}
    settings.update(eos_token_id=stop_id, bos_token_id=tokenizer.bos_token_id, pad_token_id=tokenizer.pad_token_id)
    (destination / 'generation_config.json').write_text(json.dumps(settings, indent=2) + '\n', encoding='utf-8')
    marker.write_text(json.dumps(dict(source=identity, complete=True, tensors=len(weight_map),
                                    weight_bytes=total_bytes, eos_token_id=stop_id,
                                    vision_weights_included=False, quantization=False), indent=2) + '\n', encoding='utf-8')
    return destination


def prepare_models(scale, student=None, teacher=None, cache_dir=None, model_dir=None):
    models = read_models(scale)
    paths = {}
    for role, local in [('student', student), ('teacher', teacher)]:
        if local is not None and not (Path(local) / 'config.json').is_file():
            raise ValueError(f'Missing {role} config.json in {local}')
    for role, local in [('student', student), ('teacher', teacher)]:
        if local is not None:
            path = Path(local).resolve()
        else:
            from huggingface_hub import snapshot_download
            from huggingface_hub.errors import GatedRepoError
            spec = models[role]
            try:
                options = {'allow_patterns':spec['allow_patterns']} if 'allow_patterns' in spec else {}
                path = Path(snapshot_download(spec['repo'], revision=spec['revision'], cache_dir=cache_dir, **options))
            except GatedRepoError as error:
                raise ValueError(f"Access required for {spec['repo']}: accept its Hugging Face access terms and run huggingface-cli login, or pass a local --{role} directory") from error
        if models.get('text_only_export', False):
            target = Path(model_dir or ROOT / 'models') / scale / role
            path = export_gemma_text(path, target)
        paths[role] = str(path)
    if models.get('preparation') == 'deepseek-coder-shared-instruct-eot-v1':
        from deepseek_assets import prepare_deepseek_student
        target = Path(model_dir or ROOT / 'models') / scale / 'student-opd'
        paths['student'] = str(prepare_deepseek_student(paths['student'],paths['teacher'],target))
    if models.get('validate_tokenizers', False):
        from transformers import AutoTokenizer
        student_tokenizer, teacher_tokenizer = [AutoTokenizer.from_pretrained(paths[r], local_files_only=True) for r in ('student', 'teacher')]
        if student_tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
            raise ValueError('Top-16 OPD requires identical student/teacher token IDs and vocabulary')
        if not student_tokenizer.chat_template:
            raise ValueError('Student tokenizer has no chat template')
    return paths
