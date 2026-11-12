"""Separate GPU generation and CPU official code judging processes."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from tqdm import tqdm

from code_benchmarks import (BENCHMARKS, ROOT, digest, evalplus_grades, extract_solution, prompt_for,
    EVALUATION_PROTOCOL, read_jsonl, read_manifest, save_json, save_jsonl, summarize_pass8, validate_generations)
from evaluation_prompts import CHAT_PROMPT_ENCODING, encode_chat_prompt, validate_chat_prompt_encoding


def generate(args, questions, directory):
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    output = directory / 'generations.jsonl'
    style = getattr(args, 'prompt_style', 'chat')
    rows = list(read_jsonl(output)) if output.exists() else []
    validate_generations(questions[:len(rows)], rows, args.samples)
    if style == 'chat':
        validate_chat_prompt_encoding(rows)
    if len(rows) == len(questions):
        return
    journal = directory / 'generation.partial.jsonl'
    if journal.exists():
        rows = list(read_jsonl(journal))
        validate_generations(questions[:len(rows)], rows, args.samples)
        if style == 'chat':
            validate_chat_prompt_encoding(rows)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if style == 'completion':
        prompts = [prompt_for(q, style) for q in questions]
        tokenized = [tokenizer.encode(prompt, add_special_tokens=True) for prompt in prompts]
    else:
        if getattr(tokenizer, 'chat_template', True) is None:
            raise ValueError('Model has no chat template; use --prompt-style completion for a Base model')
        encoded = [encode_chat_prompt(tokenizer, [dict(role='user', content=prompt_for(q))]) for q in questions]
        prompts = [rendered for rendered, _ in encoded]
        tokenized = [ids for _, ids in encoded]
    for question, ids in zip(questions, tokenized):
        if len(ids) > args.max_prompt_tokens:
            raise ValueError(f'Prompt too long; refusing to truncate/drop {question["task_id"]}')
    model_started = time.perf_counter()
    model = LLM(model=args.model_path, tokenizer=args.model_path,
                tensor_parallel_size=torch.cuda.device_count(), max_num_seqs=args.max_num_seqs,
                max_model_len=args.max_prompt_tokens + args.max_tokens, gpu_memory_utilization=.9)
    model_seconds = time.perf_counter()-model_started
    generation_seconds = 0.
    generated_tokens = 0
    new_questions = 0
    generation_started_unix = None
    sampling = SamplingParams(n=args.samples, temperature=args.temperature, top_p=args.top_p,
                              seed=args.seed, max_tokens=args.max_tokens,
                              **({'stop': ['```']} if style == 'completion' and args.benchmark == 'livecodebench' else {}))
    with journal.open('a', encoding='utf-8') as stream, tqdm(
            total=len(questions), initial=len(rows), desc=f'{args.benchmark} | generate',
            unit='question', dynamic_ncols=True) as progress:
        progress.set_postfix(samples=args.samples, refresh=False)
        for start in range(len(rows), len(questions), args.batch_size):
            end = min(start + args.batch_size, len(questions))
            # Submit the exact IDs checked above; never tokenize template BOS again.
            inputs = [{'prompt_token_ids': ids} for ids in tokenized[start:end]]
            batch_started = time.perf_counter()
            if generation_started_unix is None:
                generation_started_unix = time.time()
            generated = model.generate(inputs, sampling_params=sampling, use_tqdm=False)
            generation_finished_unix = time.time()
            generation_seconds += time.perf_counter()-batch_started
            if len(generated) != end - start:
                raise ValueError('Incomplete vLLM generation batch')
            for index, (question, answer) in enumerate(zip(questions[start:end], generated)):
                if len(answer.outputs) != args.samples:
                    raise ValueError('Incomplete samples for a code task')
                row = dict(task_id=question['task_id'], prompt_sha256=hashlib.sha256(prompt_for(question, style).encode()).hexdigest(),
                           rendered_prompt=prompts[start + index],
                           responses=[result.text for result in answer.outputs],
                           response_lengths=[len(result.token_ids) for result in answer.outputs],
                           finish_reasons=[result.finish_reason for result in answer.outputs])
                if style != 'chat':
                    row.update(prompt_style=style, prompt_tokens=len(tokenized[start + index]))
                else:
                    row.update(prompt_encoding=CHAT_PROMPT_ENCODING, prompt_tokens=len(tokenized[start + index]))
                rows.append(row)
                generated_tokens += sum(row['response_lengths'])
                new_questions += 1
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
            progress.update(end-start)
            save_json(directory / 'generation.performance.json', dict(
                scope='this worker invocation; completed batches only; model warmup excluded from generation throughput',
                model_initialization_seconds=model_seconds, generation_seconds=generation_seconds,
                generation_started_unix=generation_started_unix, generation_finished_unix=generation_finished_unix,
                new_questions=new_questions, output_tokens=generated_tokens,
                output_tokens_per_second=generated_tokens/generation_seconds if generation_seconds else None,
                tensor_parallel_size=torch.cuda.device_count(), visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                batch_size=args.batch_size, max_num_seqs=args.max_num_seqs))
    validate_generations(questions, rows, args.samples)
    journal.replace(output)


def grade_evalplus(args, questions, generations, directory, data_directory):
    if args.benchmark != 'humanevalplus':
        raise ValueError('Only HumanEval+ is supported by the EvalPlus adapter')
    if importlib.metadata.version('evalplus') != '0.3.1':
        raise RuntimeError('Install the pinned code benchmark dependencies: scripts/setup_code_benchmarks.sh')
    dataset = 'humaneval'
    variable = 'HUMANEVAL_OVERRIDE_PATH'
    os.environ[variable] = str((data_directory / 'official.jsonl').resolve())
    samples = [dict(task_id=q['task_id'], solution=extract_solution(response, q, row.get('prompt_style', 'chat')))
               for q, row in zip(questions, generations) for response in row['responses']]
    sample_path = directory / 'evalplus.samples.jsonl'
    save_jsonl(sample_path, samples)
    # Import only AFTER setting the official dataset override.
    from evalplus.evaluate import evaluate
    # EvalPlus 0.3.1 derives its result filename from samples (no output_file argument).
    official_result = sample_path.with_name(sample_path.stem + '_eval_results.json')
    if official_result.exists():
        raw = json.loads(official_result.read_text(encoding='utf-8'))
        evalplus_grades(raw, [q['task_id'] for q in questions], args.samples)
    else:
        print(f'[JUDGE] {args.benchmark}: {len(questions)} questions x {args.samples} responses; '
              'official EvalPlus progress below counts responses', flush=True)
        evaluate(dataset=dataset, samples=str(sample_path), parallel=args.workers, i_just_wanna_run=True)
    result_path = directory / 'evalplus.results.json'
    official_result.replace(result_path)
    raw = json.loads(result_path.read_text(encoding='utf-8'))
    grades = evalplus_grades(raw, [q['task_id'] for q in questions], args.samples)
    return grades, ['evalplus.results.json', 'evalplus.samples.jsonl']


def lcb_one(task, codes, timeout):
    sys.path.insert(0, str(ROOT / 'third_party/livecodebench'))
    from lcb_runner.evaluation.compute_code_generation_metrics import evaluate_generations_by_problem
    results, metadata = evaluate_generations_by_problem((codes, {'input_output': task['input_output']}, False, timeout))
    expected_count = task['public_tests'] + task['private_tests']
    grades = [bool(len(statuses) == expected_count and all(value > 0 for value in statuses)) for statuses in results]
    return dict(task_id=task['task_id'], grades=grades, test_results=results, metadata=metadata,
                public_tests=task['public_tests'], private_tests=task['private_tests'])


def grade_lcb(args, questions, generations, directory, data_directory):
    journal = directory / 'livecodebench.results.jsonl'
    completed = {}
    if journal.exists():
        for row in read_jsonl(journal):
            if row['task_id'] in completed:
                raise ValueError('Duplicate task in LCB grading journal')
            completed[row['task_id']] = row
    question_map = {q['task_id']: q for q in questions}
    generation_map = {row['task_id']: row for row in generations}
    if not set(completed).issubset(question_map):
        raise ValueError('LCB grading journal task mismatch')
    pending, seen = {}, set()
    with journal.open('a', encoding='utf-8') as stream, tqdm(
            total=len(questions), initial=len(completed), desc=f'{args.benchmark} | judge',
            unit='question', dynamic_ncols=True) as progress, ProcessPoolExecutor(
            max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
        def commit(future):
            result = future.result()
            if result['task_id'] != pending.pop(future):
                raise ValueError('LCB task/result mismatch')
            completed[result['task_id']] = result
            stream.write(json.dumps(result, ensure_ascii=False) + '\n')
            stream.flush()
            progress.update(1)
        # Bound the number of in-flight test sets; LCB test payloads can be large.
        for task in read_jsonl(data_directory / 'tests.jsonl.gz'):
            task_id = task['task_id']
            if task_id not in question_map or task_id in seen:
                raise ValueError('LCB test task identity mismatch')
            seen.add(task_id)
            if task_id in completed:
                continue
            row = generation_map[task_id]
            codes = [extract_solution(text, question_map[task_id], row.get('prompt_style', 'chat')) for text in row['responses']]
            pending[pool.submit(lcb_one, task, codes, args.timeout)] = task_id
            if len(pending) >= args.workers * 2:
                commit(next(as_completed(pending)))
        while pending:
            commit(next(as_completed(pending)))
    if seen != set(question_map) or set(completed) != set(question_map):
        raise ValueError('LCB result coverage is incomplete')
    grades = [completed[q['task_id']]['grades'] for q in questions]
    if any(len(g) != args.samples for g in grades):
        raise ValueError('LCB sample coverage is incomplete')
    return grades, ['livecodebench.results.jsonl']


def grade(args, questions, directory):
    if os.name == 'nt':
        raise RuntimeError('Official code judges require a Linux/Unix evaluation environment')
    manifest = read_manifest(args.data_dir, args.benchmark)
    generation_path = directory / 'generations.jsonl'
    generations = list(read_jsonl(generation_path))
    validate_generations(questions, generations, args.samples)
    identity = dict(protocol=EVALUATION_PROTOCOL, metric='pass@8', generations_sha256=digest(generation_path), benchmark_manifest=manifest,
                    samples=args.samples, lcb_timeout=args.timeout,
                    evalplus_version='0.3.1', lcb_commit='28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24')
    marker = directory / 'grading.identity.json'
    if marker.exists() and json.loads(marker.read_text(encoding='utf-8')) != identity:
        raise ValueError('Grading data/generations/protocol changed; choose a fresh output directory')
    save_json(marker, identity)
    receipt = directory / 'grading.complete.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text(encoding='utf-8'))
        if saved['identity'] == identity and all(digest(directory / name) == checksum for name, checksum in saved['files'].items()):
            print(f'[GRADE] Reusing completed {args.benchmark}', flush=True)
            return
        raise ValueError('Completed code grading files changed')
    data_directory = Path(args.data_dir) / args.benchmark
    if args.benchmark == 'livecodebench':
        grades, files = grade_lcb(args, questions, generations, directory, data_directory)
    else:
        grades, files = grade_evalplus(args, questions, generations, directory, data_directory)
    summary = dict(benchmark=args.benchmark, **summarize_pass8(grades), data=manifest,
                   grading='official LCB public+private' if args.benchmark == 'livecodebench' else 'official EvalPlus base AND plus')
    save_json(directory / 'summary.json', summary)
    save_jsonl(directory / 'grades.jsonl', [dict(task_id=q['task_id'], grades=g) for q, g in zip(questions, grades)])
    files += ['summary.json', 'grades.jsonl']
    save_json(receipt, dict(identity=identity, files={name: digest(directory / name) for name in files}))


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from project_paths import project_absolute
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['generate', 'grade'], required=True)
    parser.add_argument('--benchmark', choices=BENCHMARKS, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model-path')
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--top-p', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prompt-style', choices=['chat', 'completion'], default='chat')
    parser.add_argument('--max-prompt-tokens', type=int, default=4096)
    parser.add_argument('--max-tokens', type=int, default=8192)
    parser.add_argument('--max-num-seqs', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--timeout', type=int, default=6)
    parser.add_argument('--shard-count', type=int, default=0)
    parser.add_argument('--shard-index', type=int, default=0)
    args = parser.parse_args()
    args.data_dir, args.output_dir = project_absolute(args.data_dir), project_absolute(args.output_dir)
    if args.model_path is not None:
        args.model_path = str(project_absolute(args.model_path))
    if args.samples < 8 or min(args.workers, args.batch_size, args.max_num_seqs, args.timeout) < 1:
        parser.error('Invalid pass@8 sample/worker/batch/timeout settings')
    questions = list(read_jsonl(args.data_dir / args.benchmark / 'questions.jsonl'))
    directory = args.output_dir / args.benchmark
    directory.mkdir(parents=True, exist_ok=True)
    if args.mode == 'generate':
        if args.shard_count > 0:
            from parallel_code_generation import shard_questions, shard_directory
            questions = shard_questions(directory, questions, args.samples, args.shard_count, args.shard_index)
            directory = shard_directory(directory, args.shard_count, args.shard_index)
            directory.mkdir(parents=True, exist_ok=True)
        elif (directory / 'generation.plan.json').exists() and not (directory / 'generations.jsonl').exists():
            raise ValueError('Data-parallel shards are in progress; resume with --parallel-mode data')
        generate(args, questions, directory)
    else:
        grade(args, questions, directory)


if __name__ == '__main__':
    main()

