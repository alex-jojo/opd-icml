"""Same final-box mean@8 scorer; parallel CPU grading and durable generation cache."""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import re
from fractions import Fraction
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
from evaluation_prompts import CHAT_PROMPT_ENCODING, encode_chat_prompt

BOXED_INSTRUCTION = r'Please reason step by step, and put your final answer within \boxed{}.'
LENIENT_MODE='lenient_answers_v1'


def complete_boxes(text):
    """Balanced boxed/fbox payloads, including whitespace before the brace."""
    found=[]
    for match in re.finditer(r'\\(?:boxed|fbox)\s*\{',text):
        start=match.end();depth=1
        for i in range(start,len(text)):
            if text[i]=='{' and (i==0 or text[i-1]!='\\'):depth+=1
            elif text[i]=='}' and (i==0 or text[i-1]!='\\'):
                depth-=1
                if depth==0:
                    found.append(dict(answer=text[start:i].strip(),source='boxed' if text[match.start():].startswith('\\boxed') else 'fbox',position=match.start()))
                    break
    return found


def answer_candidates(text):
    candidates=complete_boxes(text)
    pattern=r'(?i)(?:(?:the\s+)?(?:final\s+)?answer|最终答案|答案)\s*(?:is\b|[:：=]|为|是)\s*([^\n]*)'
    for match in re.finditer(pattern,text):
        value=match.group(1).strip().strip('*').strip()
        value=value.lstrip(':：= ').strip()
        if not value:
            tail=text[match.end():].strip().splitlines()
            value=tail[0].strip() if tail else ''
        value=value.rstrip('。').strip()
        if value.endswith('.') and not re.search(r'\d\.\d+$',value):value=value[:-1]
        if value:candidates.append(dict(answer=value,source='answer_statement',position=match.start()))
    if not candidates:
        lines=[x.strip() for x in text.strip().splitlines() if x.strip()]
        last=lines[-1] if lines else ''
        if last.startswith('**') and last.endswith('**'):last=last[2:-2].strip()
        # Last-line fallback only, never scan arbitrary body numbers against gold.
        if (0<len(last)<=160 and '?' not in last and re.search(r'\d|\\frac|\\sqrt|[π√]',last)
                and re.fullmatch(r'[0-9A-Za-z\\{}()\[\]+\-*/^_=.,:;%$<>|! \tπ√±×÷∞°]+',last)
                and not re.search(r'(?<!\\)\b[A-Za-z]{4,}\b',last)
                and last.count('{')==last.count('}')):
            candidates.append(dict(answer=last,source='short_math_last_line',position=len(text)-len(last)))
    unique={}
    for item in sorted(candidates,key=lambda x:x['position'],reverse=True):
        value=item['answer']
        if 0<len(value)<=300 and value not in unique:unique[value]=item
    return list(unique.values())


def strict_last_box(text):
    start=text.rfind(r'\boxed{')
    if start<0:return None
    depth=0
    for i in range(start,len(text)):
        if text[i]=='{':depth+=1
        elif text[i]=='}':
            depth-=1
            if depth==0:return text[start+7:i]
    return None


def simple_number(value):
    value=str(value).strip()
    if value.startswith('$') and value.endswith('$'):value=value.strip('$').strip()
    fraction=re.fullmatch(r'([+-]?)\\(?:dfrac|tfrac|frac)\s*\{([+-]?\d+)\}\s*\{([+-]?\d+)\}',value)
    if fraction:
        try:return (-1 if fraction[1]=='-' else 1)*Fraction(int(fraction[2]),int(fraction[3]))
        except (ValueError,ZeroDivisionError):return None
    if re.fullmatch(r'[+-]?\d+(?:\.\d+)?(?:/[+-]?\d+(?:\.\d+)?)?',value):
        try:
            if '/' in value:
                a,b=value.split('/');return Fraction(a)/Fraction(b)
            return Fraction(value)
        except (ValueError,ZeroDivisionError):pass
    return None


@lru_cache(maxsize=8192)
def fast_equivalent(left,right):
    left,right=str(left).strip(),str(right).strip()
    if not left or not right or len(left)>300 or len(right)>300:return False
    if left==right:return True
    a,b=simple_number(left),simple_number(right)
    if a is not None and b is not None:
        if a==b:return True
        if a.denominator==1 and b.denominator==1:return False
        # Preserve the original verifier's rounding/tolerance for approximate
        # decimal answers. Unequal non-integer rationals are NOT fast-rejected.
    from verl.utils.reward_score.math_verify import answers_equivalent
    return bool(answers_equivalent(left,right))


def score_row_lenient(row,equivalent=None):
    equivalent=equivalent or fast_equivalent
    result=dict(row);candidates=[];accs=[];strict=[];pred=[];cache={};hit_sources=[]
    def compare(value):
        if value is None:return False
        key=(str(row['answer']),value)
        if key not in cache:cache[key]=bool(equivalent(*key))
        return cache[key]
    for text in row['responses']:
        choices=answer_candidates(text);old=strict_last_box(text)
        # Preserve the strict result as a guaranteed subset of the lenient score.
        if old is not None and not any(c['answer']==old for c in choices):
            choices.append(dict(answer=old,source='strict_last_box',position=text.rfind(r'\boxed{')))
        strict.append(compare(old))
        hit=(dict(source='strict_last_box') if strict[-1]
             else next((c for c in choices if compare(c['answer'])),None))
        accs.append(hit is not None);hit_sources.append(hit['source'] if hit is not None else None)
        candidates.append(choices);pred.append(choices[0]['answer'] if choices else None)
    result.update(pred_answers=pred,answer_candidates=candidates,acc_list=accs,
        strict_last_box_acc_list=strict,lenient_hit_sources=hit_sources,**{f'mean@{len(accs)}':sum(accs)/len(accs)},
        scoring_config=dict(mode=LENIENT_MODE,metric='any_explicit_answer_candidate_mean',
            candidate_policy='all complete boxed/fbox + explicit answer statements; short last math line only if no explicit candidates',
            arbitrary_body_number_search=False,strict_score_preserved=True))
    return result


def automatic_workers():
    count=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count() or 1
    # Respect the server's CPU quota rather than its host's reported core count.
    try:
        quota,period=Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if quota!='max':count=min(count,max(1,int(quota)//int(period)))
    except (OSError,ValueError):pass
    return max(1,min(16,count))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def save_rows(path, rows):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def score_row(row, equivalent=None, extract=None, unbox=None):
    if equivalent is None:
        from verl.utils.reward_score.math_verify import answers_equivalent, last_boxed_only_string, remove_boxed
        equivalent, extract, unbox = answers_equivalent, last_boxed_only_string, remove_boxed
    result = dict(row)
    answers, accs, cache = [], [], {}
    for text in row['responses']:
        answer = unbox(extract(text))
        answers.append(answer)
        if answer is None:
            acc = False
        else:
            key = (str(row['answer']), answer)
            if key not in cache:
                cache[key] = bool(equivalent(*key))
            acc = cache[key]
        accs.append(acc)
    result.update(pred_answers=answers, acc_list=accs)
    result[f'mean@{len(accs)}'] = sum(accs) / len(accs)
    result['scoring_config'] = dict(metric='final_box_mean', auxiliary_box_and_majority_metrics=False,
        verifier='verl.utils.reward_score.math_verify.answers_equivalent', timeout='unchanged')
    return result


def grade(rows, journal, cache_hash, workers, scorer=score_row):
    journal = Path(journal)
    completed = {}
    if journal.exists():
        with journal.open('rb+') as stream:
            while True:
                start = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.endswith(b'\n'):
                    stream.truncate(start)  # Only discard an interrupted final journal line.
                    break
                item = json.loads(line)
                i = item['index']
                if item['generation_sha256'] != cache_hash or i not in range(len(rows)) or i in completed:
                    raise ValueError('Grading journal identity/order conflict')
                result = item['row']
                if any(result.get(k) != v for k, v in rows[i].items()) or len(result['acc_list']) != len(rows[i]['responses']):
                    raise ValueError('Grading journal row mismatch')
                completed[i] = result
    pending = [i for i in range(len(rows)) if i not in completed]
    print(f'[GRADE] cached={len(completed)}/{len(rows)}, workers={workers}; original verifier timeout unchanged', flush=True)
    started = time.monotonic()
    with journal.open('a', encoding='utf-8') as stream:
        def record(i, row):
            completed[i] = row
            stream.write(json.dumps(dict(index=i, generation_sha256=cache_hash, row=row), ensure_ascii=False) + '\n')
            stream.flush()
            if len(completed) % 10 == 0 or len(completed) == len(rows):
                print(f'[GRADE] {len(completed)}/{len(rows)} problems; elapsed={time.monotonic()-started:.1f}s', flush=True)
        if pending and workers == 1:
            for i in pending:
                record(i, scorer(rows[i]))
        elif pending:
            # Spawn: no inherited CUDA state; each verifier runs in the worker's main thread.
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
                futures = {pool.submit(scorer, rows[i]): i for i in pending}
                for future in as_completed(futures):
                    record(futures[future], future.result())
    return [completed[i] for i in range(len(rows))]


def sampling(args):
    return dict(max_prompt_tokens=args.max_prompt_tokens, max_tokens=args.max_tokens,
        temperature=args.temperature, top_p=args.top_p, seed=args.seed, n=args.n, compact_output=True)


def load_cache(cache, metadata, identity):
    if not metadata.exists():
        return None
    saved = json.loads(metadata.read_text(encoding='utf-8'))
    if saved['identity'] != identity or saved['sha256'] != digest(cache):
        raise ValueError('Generation cache model/dataset/sampling/prompt encoding identity changed; '
                         'use a new output file for generation or --regrade_only for saved texts')
    rows = read_rows(cache)
    if len(rows) != saved['rows'] or any(len(r['responses']) != identity['sampling']['n'] for r in rows):
        raise ValueError('Incomplete generation cache')
    return rows


def verify_rescore_identity(dest,old_hash):
    """Recover safely if scoring finished before its parent completion marker."""
    dest=Path(dest)
    if digest(dest)==old_hash:return
    backup=dest.with_suffix('.strict_last_box.jsonl')
    if not backup.exists() or digest(backup)!=old_hash:raise ValueError('Scored answers changed unexpectedly')
    old,new=read_rows(backup),read_rows(dest)
    fields=('problem','answer','benchmark','benchmark_index','responses','response_lengths','rendered_prompt','generation_config','model')
    if len(old)!=len(new) or any(any(a.get(k)!=b.get(k) for k in fields) for a,b in zip(old,new)):
        raise ValueError('Regrading changed generation inputs or response texts')


def generate(args, inputs):
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    toker = AutoTokenizer.from_pretrained(args.model_path)
    prompts, tokenized = [], []
    for item in inputs:
        problem = str(item['problem']).rstrip()
        if not problem.endswith(BOXED_INSTRUCTION):
            problem += '\n' + BOXED_INSTRUCTION
        rendered, ids = encode_chat_prompt(toker, [dict(role='user', content=problem)])
        prompts.append(rendered)
        tokenized.append(ids)
    for i, ids in enumerate(tokenized):
        if len(ids) > args.max_prompt_tokens:
            raise ValueError(f'Prompt {i} exceeds max_prompt_tokens; refusing truncation')
    llm = LLM(model=args.model_path, tokenizer=args.model_path, gpu_memory_utilization=0.95,
        tensor_parallel_size=torch.cuda.device_count(), max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_prompt_tokens + args.max_tokens)
    print(f'[GENERATE] {len(inputs)} problems x {args.n}; max_num_seqs={args.max_num_seqs}', flush=True)
    outputs = llm.generate([{'prompt_token_ids': ids} for ids in tokenized], sampling_params=SamplingParams(temperature=args.temperature,
        top_p=args.top_p, max_tokens=args.max_tokens, n=args.n, seed=args.seed))
    if len(outputs) != len(inputs):
        raise ValueError('Incomplete generation batch')
    rows = []
    for i, output in enumerate(outputs):
        if len(output.outputs) != args.n:
            raise ValueError('Incomplete samples')
        responses = [o.text.strip() for o in output.outputs]
        rows.append(dict(inputs[i], answer=str(inputs[i]['answer']), responses=responses,
            response_lengths=[len(toker.encode(r, add_special_tokens=False)) for r in responses],
            rendered_prompt=prompts[i], finish_reasons=[o.finish_reason for o in output.outputs],
            stop_reasons=[o.stop_reason for o in output.outputs], generation_config=sampling(args),
            generation_execution=dict(max_num_seqs=args.max_num_seqs, prompt_encoding=CHAT_PROMPT_ENCODING),
            model=Path(args.model_path).name))
    return rows


def main():
    p = argparse.ArgumentParser()
    for key in ('input_file', 'model_path', 'output_file'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--max_prompt_tokens', type=int, default=1024)
    p.add_argument('--max_tokens', type=int, default=8192)
    p.add_argument('--temperature', type=float, default=1.)
    p.add_argument('--top_p', type=float, default=1.)
    p.add_argument('--n', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_num_seqs', type=int, default=64)
    p.add_argument('--score_workers', type=int, default=automatic_workers())
    p.add_argument('--scoring_mode',choices=['strict_last_box',LENIENT_MODE],default='strict_last_box')
    p.add_argument('--regrade_only',action='store_true',help='Use saved responses only; never start GPU generation')
    p.add_argument('--compact_output', action='store_true')
    args = p.parse_args()
    if args.n < 1 or args.score_workers < 1 or args.max_num_seqs < 1 or (args.n > 1 and args.temperature <= 0):
        raise ValueError('Invalid sample/worker/concurrency settings')
    dest = Path(args.output_file)
    dest.parent.mkdir(parents=True, exist_ok=True)
    inputs = read_rows(args.input_file)
    model = Path(args.model_path)
    cache = dest.with_suffix('.generations.jsonl')
    metadata = dest.with_suffix('.generations.meta.json')
    if args.regrade_only:
        if metadata.exists():
            identity=json.loads(metadata.read_text(encoding='utf-8'))['identity']
            if identity['dataset_sha256']!=digest(args.input_file) or identity['sampling']!=sampling(args):
                raise ValueError('Cached response dataset/sampling changed')
        else:
            identity=dict(schema=1,dataset_sha256=digest(args.input_file),sampling=sampling(args),
                model_sha256=None,model_reference=str(model),origin='previously_scored_response_texts')
        print('[RESCORE] Text-only: preserve original generation identity; no model-weight hashing/loading, no GPU.',flush=True)
    else:
        identity = dict(schema=1, dataset_sha256=digest(args.input_file), sampling=sampling(args),
            prompt_encoding=CHAT_PROMPT_ENCODING,
            model_sha256={f.name: digest(f) for f in sorted(model.iterdir()) if f.is_file()})
    rows = load_cache(cache, metadata, identity)
    if rows is None:
        if args.regrade_only:
            if not dest.exists():raise ValueError('No saved answers; --regrade_only never generates')
            previous=read_rows(dest)
            if len(previous)!=len(inputs):raise ValueError('Saved answers do not match dataset')
            keys=('responses','response_lengths','rendered_prompt','finish_reasons','stop_reasons',
                  'generation_config','generation_execution','model')
            rows=[]
            for old,original in zip(previous,inputs):
                if any(old.get(k)!=v for k,v in original.items()) or len(old.get('responses',[]))!=args.n:
                    raise ValueError('Saved generation identity mismatch')
                if old.get('generation_config')!=sampling(args):raise ValueError('Saved sampling changed')
                rows.append(dict(original,**{k:old[k] for k in keys if k in old}))
            print('[RESCORE] Reusing saved response texts; GPU generation disabled.',flush=True)
        else:rows = generate(args, inputs)
        save_rows(cache, rows)
        save_json(metadata, dict(identity=identity, sha256=digest(cache), rows=len(rows)))
        print(f'[GENERATE] Saved all answers BEFORE grading: {cache}', flush=True)
    else:
        print(f'[GENERATE] Verified saved answers: {len(rows)} problems; skip GPU generation.', flush=True)
    # The generation cache, not scheduling/concurrency, identifies the grading inputs.
    scorer=score_row_lenient if args.scoring_mode==LENIENT_MODE else score_row
    journal=dest.with_suffix('.grading.lenient_v1.jsonl' if args.scoring_mode==LENIENT_MODE else '.grading.jsonl')
    results = grade(rows,journal,digest(cache),args.score_workers,scorer)
    if dest.exists() and args.scoring_mode==LENIENT_MODE:
        previous=read_rows(dest)
        if previous and previous[0].get('scoring_config',{}).get('mode')!=LENIENT_MODE:
            backup=dest.with_suffix('.strict_last_box.jsonl')
            if not backup.exists():save_rows(backup,previous)
    save_rows(dest, results)
    summary = {}
    for name in dict.fromkeys(r.get('benchmark', 'all') for r in results):
        group = [r for r in results if r.get('benchmark', 'all') == name]
        score = sum(sum(r['acc_list']) for r in group) / (args.n * len(group))
        summary[name] = dict(problems=len(group), samples=args.n * len(group), mean_at_8=score)
        print(f'{name}: mean@{args.n}={100*score:.2f}%', flush=True)
        if args.scoring_mode==LENIENT_MODE:
            count=args.n*len(group)
            strict=sum(sum(r['strict_last_box_acc_list']) for r in group)/count
            missing=sum(strict_last_box(text) is None for r in group for text in r['responses'])/count
            capped=sum(reason=='length' for r in group for reason in r.get('finish_reasons',[]))/count
            summary[name].update(scoring_mode=LENIENT_MODE,strict_last_box_mean_at_8=strict,
                missing_strict_box_rate=missing,length_finish_rate=capped,
                mean_response_tokens=sum(sum(r['response_lengths']) for r in group)/count)
            rescued={}
            for row in group:
                for good,was_good,source in zip(row['acc_list'],row['strict_last_box_acc_list'],row['lenient_hit_sources']):
                    if good and not was_good:rescued[source]=rescued.get(source,0)+1
            summary[name]['rescued_samples_by_source']=rescued
            print(f'  strict-last-box={100*strict:.2f}% | lenient={100*score:.2f}% | '
                f'no-complete-final-box={100*missing:.2f}% | length-stop={100*capped:.2f}%',flush=True)
            print(f'  additionally-correct samples by answer source: {rescued}',flush=True)
    save_json(dest.with_suffix('.summary.json'), summary)


if __name__ == '__main__':
    main()
