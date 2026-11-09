"""Prepare only the pinned Eurus-2-RL-Data code rows, preserving test cases."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATASET = 'PRIME-RL/Eurus-2-RL-Data'
SOURCE_REVISION = '9776b13264b5aaa0b16495fcf086a0a8d86fd655'
EXPECTED_ROWS = {'train': 25_276, 'validation': 1_024}
CODE_SOURCES = {'apps', 'codecontests', 'taco', 'codeforces'}
PROMPT_PROTOCOL = 'eurus-code-fold-system-v1'
SCHEMA = pa.schema([
    ('data_source', pa.string()),
    ('prompt', pa.list_(pa.struct([('role', pa.string()), ('content', pa.string())]))),
    ('ability', pa.string()),
    ('reward_model', pa.struct([('style', pa.string()), ('ground_truth', pa.string())])),
    ('extra_info', pa.struct([
        ('index', pa.int64()), ('split', pa.string()), ('source_index', pa.int64()),
        ('source_dataset', pa.string()), ('source_revision', pa.string()), ('source_split', pa.string()),
        ('source_datasource', pa.string()), ('source_prompt_sha256', pa.string()),
        ('prompt_protocol', pa.string()), ('test_count', pa.int64()),
    ])),
])


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def user_prompt(messages):
    """Fold the original system text into the user turn for all model families."""
    if hasattr(messages, 'tolist'):
        messages = messages.tolist()
    if not isinstance(messages, list) or not messages:
        raise ValueError('Missing Eurus prompt messages')
    systems, users = [], []
    for message in messages:
        role, content = message.get('role'), message.get('content')
        if role not in ('system', 'user') or not isinstance(content, str) or not content.strip():
            raise ValueError('Expected nonempty system/user prompt messages only')
        (systems if role == 'system' else users).append(content)
    if len(users) != 1:
        raise ValueError('Expected one original user problem')
    # Preserve instructions and problem text, including the original Python fence request.
    return [{'role': 'user', 'content': '\n\n'.join(systems + users)}]


def convert_row(row, split, source_index, index):
    if row.get('ability') != 'code' or row.get('data_source') not in CODE_SOURCES:
        raise ValueError('Expected an Eurus code row from a supported source')
    reward = row.get('reward_model') or {}
    ground_truth = reward.get('ground_truth')
    if reward.get('style') != 'rule' or not isinstance(ground_truth, str):
        raise ValueError(f'Invalid code test specification at source row {source_index}')
    tests = json.loads(ground_truth)
    inputs, outputs = tests.get('inputs'), tests.get('outputs')
    if not isinstance(inputs, list) or not isinstance(outputs, list) or not inputs or len(inputs) != len(outputs):
        raise ValueError(f'Empty/misaligned test cases at source row {source_index}')
    if tests.get('fn_name') is not None and not isinstance(tests['fn_name'], str):
        raise ValueError(f'Invalid function name at source row {source_index}')
    original_prompt = row['prompt'].tolist() if hasattr(row['prompt'], 'tolist') else row['prompt']
    prompt_hash = hashlib.sha256(json.dumps(original_prompt, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return dict(data_source=row['data_source'], prompt=user_prompt(original_prompt), ability='code',
                reward_model=dict(style='rule', ground_truth=ground_truth),
                extra_info=dict(index=index, split=split, source_index=source_index,
                    source_dataset=SOURCE_DATASET, source_revision=SOURCE_REVISION, source_split=split,
                    source_datasource=row['data_source'], source_prompt_sha256=prompt_hash,
                    prompt_protocol=PROMPT_PROTOCOL, test_count=len(inputs)))


def code_row_groups(parquet):
    candidates = []
    for group in range(parquet.metadata.num_row_groups):
        metadata = parquet.metadata.row_group(group)
        column = next(metadata.column(i) for i in range(metadata.num_columns)
                      if metadata.column(i).path_in_schema == 'ability')
        stats = column.statistics
        if stats is None or not stats.has_min_max or stats.min <= 'code' <= stats.max:
            candidates.append(group)
    return candidates


def validate_existing(output, split, expected_count):
    output = Path(output)
    manifest = output.with_suffix('.manifest.json')
    if not output.is_file() or not manifest.is_file():
        return False
    saved = json.loads(manifest.read_text(encoding='utf-8'))
    identity = (saved.get('source_dataset'), saved.get('source_revision'), saved.get('source_split'),
                saved.get('prompt_protocol'), saved.get('rows'))
    if identity != (SOURCE_DATASET, SOURCE_REVISION, split, PROMPT_PROTOCOL, expected_count):
        raise ValueError(f'Dataset protocol mismatch in {output}; choose another --output-dir')
    if pq.read_metadata(output).num_rows != expected_count or digest(output) != saved['sha256']:
        raise ValueError(f'Dataset checksum/count mismatch: {output}')
    return True


def prepare_split(source, output, split, expected_count=None):
    expected_count = EXPECTED_ROWS[split] if expected_count is None else expected_count
    output = Path(output)
    if validate_existing(output, split, expected_count):
        print(json.dumps(dict(phase='reuse_eurus_code', split=split, output=str(output), rows=expected_count)), flush=True)
        return
    if output.exists() or output.with_suffix('.manifest.json').exists():
        raise ValueError(f'Incomplete/existing dataset output: {output}; choose another --output-dir')
    output.parent.mkdir(parents=True, exist_ok=True)
    parquet = pq.ParquetFile(source)
    selected = code_row_groups(parquet)
    temporary = output.with_name(output.name + '.tmp')
    count, sources, call_based = 0, Counter(), 0
    try:
        with pq.ParquetWriter(temporary, SCHEMA, compression='zstd') as writer:
            for group in selected:
                offset = sum(parquet.metadata.row_group(i).num_rows for i in range(group))
                for batch in parquet.iter_batches(batch_size=64, row_groups=[group]):
                    converted = []
                    for local_index, row in enumerate(batch.to_pylist()):
                        if row['ability'] != 'code':
                            continue
                        item = convert_row(row, split, offset + local_index, count)
                        converted.append(item)
                        sources[item['data_source']] += 1
                        call_based += json.loads(item['reward_model']['ground_truth']).get('fn_name') is not None
                        count += 1
                    if converted:
                        writer.write_table(pa.Table.from_pylist(converted, schema=SCHEMA))
                    offset += len(batch)
                print(json.dumps(dict(phase='extract_eurus_code', split=split, row_group=group, code_rows=count)), flush=True)
        if count != expected_count:
            raise ValueError(f'Expected {expected_count} {split} code rows, got {count}; output not published')
        sha256 = digest(temporary)
        manifest = dict(source_dataset=SOURCE_DATASET, source_revision=SOURCE_REVISION, source_split=split,
                        prompt_protocol=PROMPT_PROTOCOL, rows=count, data_sources=dict(sources),
                        call_based_problems=call_based, stdin_problems=count-call_based,
                        sha256=sha256, ground_truth_preserved=True)
        temporary.replace(output)
        output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--source-dir', type=Path, help='Offline directory with original train.parquet and validation.parquet')
    parser.add_argument('--split', choices=['train', 'validation', 'all'], default='all')
    args = parser.parse_args()
    splits = ['train', 'validation'] if args.split == 'all' else [args.split]
    filesystem = None
    for split in splits:
        output = args.output_dir / f'eurus_code_{split}.parquet'
        if validate_existing(output, split, EXPECTED_ROWS[split]):
            print(json.dumps(dict(phase='reuse_eurus_code', split=split, output=str(output))), flush=True)
            continue
        if args.source_dir:
            prepare_split(args.source_dir / f'{split}.parquet', output, split)
        else:
            from huggingface_hub import HfFileSystem
            filesystem = filesystem or HfFileSystem()
            path = f'datasets/{SOURCE_DATASET}@{SOURCE_REVISION}/{split}.parquet'
            with filesystem.open(path, 'rb', block_size=1024 * 1024) as stream:
                prepare_split(stream, output, split)


if __name__ == '__main__':
    main()
