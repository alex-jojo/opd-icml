"""Weight-preserving DeepSeek Base runtime with the teacher's tokenizer policy."""
import hashlib
import json
import os
from pathlib import Path
import shutil
from project_paths import project_absolute, project_relative, saved_path_matches


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_deepseek_student(source, teacher, destination):
    """Share the Instruct template and retain BOTH native and instruction EOS."""
    from transformers import AutoTokenizer
    source, teacher, destination = [project_absolute(p) for p in (source, teacher, destination)]
    if destination.resolve() in (source.resolve(), teacher.resolve()):
        raise ValueError('DeepSeek OPD runtime must be separate from the original checkpoints')
    base_cfg = json.loads((source/'config.json').read_text(encoding='utf8'))
    teacher_cfg = json.loads((teacher/'config.json').read_text(encoding='utf8'))
    if base_cfg.get('model_type') != 'llama' or teacher_cfg.get('model_type') != 'llama':
        raise ValueError('Expected the DeepSeek-Coder Llama model family')
    for key in ('vocab_size','hidden_size','num_hidden_layers','num_attention_heads','num_key_value_heads'):
        if base_cfg.get(key) != teacher_cfg.get(key):
            raise ValueError('DeepSeek pair architecture differs: '+key)
    original = AutoTokenizer.from_pretrained(source, local_files_only=True)
    shared = AutoTokenizer.from_pretrained(teacher, local_files_only=True)
    if original.get_vocab() != shared.get_vocab():
        raise ValueError('DeepSeek OPD requires identical student/teacher token IDs and vocabulary')
    if not shared.chat_template or shared.eos_token != '<|EOT|>':
        raise ValueError('Expected the official Instruct chat template and <|EOT|> EOS')
    if original.convert_tokens_to_ids(shared.eos_token) != shared.eos_token_id:
        raise ValueError('Teacher EOT ID is absent from the Base vocabulary')
    weights = sorted(source.glob('*.safetensors'))
    if weights:
        indices = sorted(source.glob('*.safetensors.index.json'))
    else:
        weights = sorted(source.glob('pytorch_model*.bin'))
        indices = sorted(source.glob('pytorch_model*.bin.index.json'))
    if not weights:
        raise ValueError('No Base checkpoint weights found')
    stop_ids = list(dict.fromkeys([shared.eos_token_id, original.eos_token_id]))
    stop_ids = [int(token) for token in stop_ids if token is not None]
    identity = dict(protocol='deepseek-coder-dual-eos-v2', source=project_relative(source), teacher=project_relative(teacher),
        source_config_sha256=digest(source/'config.json'), teacher_config_sha256=digest(teacher/'config.json'),
        weights=[dict(name=p.name, bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in weights],
        teacher_template_sha256=hashlib.sha256(shared.chat_template.encode()).hexdigest(),
        eos_token_id=shared.eos_token_id, generation_eos_token_ids=stop_ids)
    marker = destination/'opd_pair_runtime.json'
    if marker.exists():
        previous = json.loads(marker.read_text(encoding='utf8'))
        saved = previous['identity']
        if (dict(saved, source=identity['source'], teacher=identity['teacher']) != identity
                or not saved_path_matches(saved.get('source', ''), source)
                or not saved_path_matches(saved.get('teacher', ''), teacher)):
            raise ValueError('DeepSeek runtime destination belongs to different source files')
        if previous.get('complete'):
            if (all((destination/p.name).is_file() and os.path.samefile(p, destination/p.name) for p in weights)
                and all((destination/name).is_file() and digest(destination/name)==sha for name,sha in previous['metadata_sha256'].items())):
                return destination
            raise ValueError('Prepared DeepSeek runtime files changed; use a new --model-dir')
    elif destination.exists() and any(destination.iterdir()):
        raise ValueError('Choose an empty DeepSeek runtime directory')
    destination.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(dict(identity=identity, complete=False),indent=2),encoding='utf8')
    for weight in weights:
        target = destination/weight.name
        if target.exists():
            if not os.path.samefile(weight,target):
                raise ValueError('Unexpected existing runtime weight file')
            continue
        try:
            os.link(weight.resolve(),target)
        except OSError:
            try:
                target.symlink_to(os.path.relpath(weight, target.parent))
            except OSError as error:
                raise ValueError('Cannot link weights; choose --model-dir on the same filesystem as the Base checkpoint') from error
    for index in indices:
        shutil.copy2(index,destination/index.name)
    shared.save_pretrained(destination)
    cfg = dict(base_cfg)
    cfg.update(eos_token_id=shared.eos_token_id, bos_token_id=shared.bos_token_id, pad_token_id=shared.pad_token_id)
    cfg['opd_response_eos_token_ids'] = stop_ids
    (destination/'config.json').write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf8')
    generation_file = source/'generation_config.json'
    generation = json.loads(generation_file.read_text(encoding='utf8')) if generation_file.exists() else {}
    generation.update(eos_token_id=stop_ids,bos_token_id=shared.bos_token_id,pad_token_id=shared.pad_token_id)
    (destination/'generation_config.json').write_text(json.dumps(generation,indent=2)+'\n',encoding='utf8')
    metadata = {p.name:digest(p) for p in destination.iterdir() if p.is_file() and p.name != marker.name and p not in [destination/w.name for w in weights]}
    marker.write_text(json.dumps(dict(identity=identity,complete=True,metadata_sha256=metadata,
        weights_linked=True,weights_modified=False,original_base_unchanged=True,
        prompt_protocol='official teacher Instruct template for BOTH student and teacher',
        source_eos_token_id=original.eos_token_id, runtime_eos_token_id=shared.eos_token_id,
        generation_eos_token_ids=stop_ids),indent=2)+'\n',encoding='utf8')
    return destination
