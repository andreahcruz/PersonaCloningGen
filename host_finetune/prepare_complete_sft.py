"""Prepare whole-source SFT candidates without changing frozen split ownership.

No model weights are loaded. Outputs are a review candidate, never selected as the
trainer default. Every rejection is retained in a provenance/disposition ledger.
Run with an immutable local tokenizer snapshot and a NEW EXP output directory.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys

from host_finetune.build_sft_from_cleaned_sources import SOURCES
from host_finetune.clean_scraped import clean_row
from host_finetune.filter_event_promos import _is_direct_event_announcement, _strip_trailing_cta
from host_finetune.llama_chat_format import training_text_from_row
from host_finetune.sft_chunk_utils import instruction_for
from host_finetune.split_groups import (
    assert_manifest_matches, file_sha256, load_rows, matches_gold_field,
    near_duplicate_pairs, normalize_text, write_assignment_file,
)
from spark_jobs.corpus_footer_scrub import is_event_promo

VERSION = 'complete-source-v3'
SPLITS = {'train', 'validation', 'test'}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def read_numbered(path: Path) -> list[tuple[int, dict]]:
    result = []
    with path.open(encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f'{path}:{line_no}: expected object')
                result.append((line_no, row))
    return result


def ownership(assignments: list[dict], prior_rows: list[dict]) -> dict:
    """Validate positional linkage as well as source keys; never reassign groups."""
    if len(assignments) != len(prior_rows):
        raise ValueError('Assignment and prior dataset lengths differ')
    result = defaultdict(list)
    for i, (a, row) in enumerate(zip(assignments, prior_rows)):
        key = (row.get('source_file'), row.get('source_line'))
        if (a['row_index'] != i or a['split'] not in SPLITS or
                key != (a.get('source_file'), a.get('source_line')) or not all(key)):
            raise ValueError(f'Invalid source/row assignment at {i}')
        result[key].append({**a, 'prior_output': row['output']})
    return dict(result)


def raw_lineage(raw_rows, medium, text_key, title_key):
    """Reproduce cleaned text, indexed by stable URL/id or complete text hash.

    Retain all matching raw records. Repeated raw copies do not erase provenance.
    """
    index = defaultdict(list)
    for line, row in raw_rows:
        kind = 'youtube' if medium.startswith('youtube') else medium
        kwargs = {'linkedin': True} if medium == 'linkedin' else (
            {'unescape': True} if kind == 'youtube' else {})
        cleaned, reason, _ = clean_row(kind, row, text_key, title_key, kwargs)
        if cleaned is None or reason:
            continue
        text = str(cleaned.get(text_key) or '').strip()
        stable = str(row.get('url') or row.get('video_url') or row.get('id') or '')
        key = (stable, digest(text))
        index[key].append({'raw_line': line, 'raw_record_sha256': digest(json.dumps(row, sort_keys=True, ensure_ascii=False)),
                           'raw_text_sha256': digest(str(row.get(text_key) or ''))})
    return index


def completeness_reason(text: str) -> str | None:
    # Conservative triage: ambiguous valid lists/social endings remain reviewable.
    end = text.rstrip().rstrip('\"\u201d\u2019\' )]}*')
    if not end or end.endswith((':', ';', ',', '-', '\u2014', '...', '\u2026')):
        return 'ambiguous_or_incomplete_ending'
    if end[-1] not in '.!?':
        return 'ending_requires_review'
    if '\ufffd' in text:
        return 'replacement_character'
    if any(marker in text for marker in ('<|start_header_id|>', '<|eot_id|>', '<|end_of_text|>')):
        return 'chat_control_token'
    return None


def candidate(row, medium, text_key, title_key, tokenizer):
    """Keep full text, never truncate/rewrite/derive a topic from the answer."""
    text = str(row.get(text_key) or '').strip()
    title = str(row.get(title_key) or '').strip() if title_key else ''
    reasons = []
    if medium.startswith('youtube'):
        reasons.append('unverified_speaker')
    if medium == 'linkedin' and str(row.get('author', '')).casefold() not in {'jason m. lemkin', 'jason lemkin'}:
        reasons.append('uncertain_author')
    if medium == 'x' and (row.get('username') != 'jasonlk' or row.get('is_reply') or row.get('is_repost_or_quote')):
        reasons.append('uncertain_author_or_repost')
    if not title:
        reasons.append('missing_independent_topic')
    if len(title) > 500:
        reasons.append('oversized_title')
    if len(text.split()) < (12 if medium == 'x' else 50):
        reasons.append('below_existing_word_floor')
    instruction = instruction_for(medium, title) if title else ''
    if is_event_promo(text, title) or _is_direct_event_announcement({'instruction': instruction}):
        reasons.append('promotional_document')
    if _strip_trailing_cta(text)[1]:
        reasons.append('promotional_tail_requires_review')
    if re.search(r'\bsaastr\s*(?:(?:ai|202\d)\s+)?(?:annual|europa|deploy|summit)\b', title, re.I) or (
            re.search(r'\bannual\b',title,re.I) and re.search(r'\b(?:sponsor|expo|ticket|session)s?\b',title,re.I)):
        reasons.append('event_titled_document_requires_review')
    if re.search(r'\bwith\b.{0,100}\b(?:ceo|coo|cbo|cro|cmo|cto|founder|president)s?\b', title, re.I):
        reasons.append('guest_recap_attribution_requires_review')
    if re.search(r'pic\.twitter\.com/|https?://t\.co/|(?<!\w)@\w+|\b(?:chart|graph) (?:above|below)\b|\byou can see above\b',text,re.I):
        reasons.append('embedded_or_external_context_requires_review')
    if 'quora' in title.casefold() and sum(line.rstrip().endswith('?') for line in text.splitlines()) >= 8:
        reasons.append('question_link_roundup')
    if re.search(r'\btop\b.{0,35}\b(?:content|posts|videos)\b.{0,25}\b(?:week|month|year)\b', title, re.I):
        reasons.append('content_link_roundup')
    if re.search(r'\b(?:survey|poll|chart|graph|table|image|figure|data|number)s?\b[^\n.!?]{0,60}\b(?:above|below)\b', text, re.I):
        reasons.append('missing_visual_context_requires_review')
    if re.search(r'[A-Za-z]{3,}\n[b-hj-z](?:[.!?]|[ \t])', text):
        reasons.append('scraped_word_break_requires_review')
    if re.search(r'\ba related post here\s*:', text[-600:], re.I):
        reasons.append('trailing_related_link_requires_review')
    if normalize_text(text) in {normalize_text(title), normalize_text(instruction)}:
        reasons.append('title_or_instruction_only')
    if re.match(r'^write (?:a |an )?.{0,80}in the style of jason lemkin', text, re.I):
        reasons.append('instruction_echo_target')
    ending = completeness_reason(text)
    if ending: reasons.append(ending)
    n = None
    if not reasons:
        rendered = training_text_from_row(tokenizer, instruction, '', text)
        n = len(tokenizer(rendered, add_special_tokens=False, truncation=False)['input_ids'])
    return {'instruction': instruction, 'input': '', 'output': text}, reasons, n


def leakage_links(documents: list[dict], gold_rows: list[dict]):
    """Exact, lexical-near and gold checks on all owned full sources, not just accepted rows."""
    normalized = [normalize_text(d['text']) for d in documents]
    exact = defaultdict(list)
    for i, t in enumerate(normalized):
        if t: exact[t].append(i)
    edges = [(members[0], j, 'normalized_exact') for members in exact.values() for j in members[1:]]
    prior_families = defaultdict(list)
    for i, d in enumerate(documents):
        for group in d['record'].get('prior_group_ids', []):
            prior_families[group].append(i)
        if d['record'].get('source_url_or_id'):
            prior_families['identity:' + d['record']['source_url_or_id']].append(i)
    edges.extend((members[0], j, 'frozen_family_or_source_identity')
                 for members in prior_families.values() for j in members[1:])
    print(f'Checking full-source near duplicates: {len(documents)} documents', flush=True)
    edges.extend((i, j, 'token_jaccard_0.8') for i, j in near_duplicate_pairs(normalized))
    # Connected families, including indirect links, inherit all conflict flags.
    parent = list(range(len(documents)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, j, _ in edges:
        parent[find(j)] = find(i)
    groups = defaultdict(list)
    for i in range(len(documents)): groups[find(i)].append(i)
    conflicts = set()
    for members in groups.values():
        splits = {s for i in members for s in documents[i]['splits']}
        if len(splits) > 1: conflicts.update(members)
    gold_matches = []
    for i, document in enumerate(documents):
        text = document.get('text') or ''
        title = document.get('title') or ''
        for row in gold_rows:
            fields = [('topic', row.get('topic', '')), ('gold_reference', row.get('gold_reference', ''))]
            fields.extend(('expected_fact', value) for value in row.get('expected_facts') or [])
            for field, value in fields:
                matched, score = matches_gold_field(text, title, value, field)
                if matched:
                    gold_matches.append({
                        'document': i, 'gold_id': row.get('id'), 'field': field, 'jaccard': score,
                    })
    gold_hit = {m['document'] for m in gold_matches}
    gold_families = {find(i) for i in gold_hit}
    gold_hit = {i for i in range(len(documents)) if find(i) in gold_families}
    return edges, conflicts, gold_matches, gold_hit


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def write_jsonl(path: Path, rows):
    with path.open('w', encoding='utf-8', newline='\n') as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False) + '\n')


def command_output(args):
    return subprocess.check_output(args, text=True, encoding='utf-8', errors='replace').strip()


def run(args):
    if args.max_tokens <= 0: raise ValueError('max-tokens must be positive')
    if args.out.exists(): raise ValueError('Output directory must be new; refusing overwrite')
    _, assignments = assert_manifest_matches(args.assignments, args.prior_dataset)
    prior = load_rows(args.prior_dataset)
    owners = ownership(assignments, prior)
    tokenizer_files = sorted(p for p in args.tokenizer.iterdir() if p.is_file() and p.suffix in {'.json', '.jinja', '.model'})
    if not tokenizer_files: raise ValueError('Tokenizer must be an existing immutable local snapshot')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True, trust_remote_code=False)
    inputs = [args.prior_dataset, args.assignments, args.gold] + tokenizer_files
    inputs += [folder / name for folder in (args.raw_dir, args.cleaned_dir) for name in SOURCES]
    input_hashes = {str(p.resolve()): file_sha256(p) for p in inputs}
    args.out.mkdir(parents=True)
    for name in ('raw', 'config', 'metrics', 'figures'): (args.out/name).mkdir()
    code_paths = [Path(__file__), Path('host_finetune/clean_scraped.py'), Path('spark_jobs/corpus_footer_scrub.py'),
                  Path('host_finetune/split_groups.py'), Path('host_finetune/sft_chunk_utils.py'),
                  Path('host_finetune/filter_event_promos.py'), Path('host_finetune/llama_chat_format.py'),
                  Path('host_finetune/build_sft_from_cleaned_sources.py')]
    for p in code_paths:
        (args.out/'config'/p.name).write_bytes(p.read_bytes())
    write_json(args.out/'config'/'input_hashes.json', input_hashes)
    (args.out/'config'/'package_snapshot.txt').write_text(command_output([sys.executable, '-m', 'pip', 'freeze'])+'\n', encoding='utf-8')
    write_json(args.out/'config'/'parameters.json', {k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})
    manifest = {
        'experiment_id': args.out.name, 'timestamp': datetime.now(timezone.utc).isoformat(),
        'status': 'RUNNING', 'execution_commands': [subprocess.list2cmdline([sys.executable, '-m', 'host_finetune.prepare_complete_sft', *sys.argv[1:]])],
        'git': {'commit': command_output(['git','rev-parse','HEAD']), 'dirty': bool(command_output(['git','status','--porcelain']))},
        'data': {'source_manifest':'config/input_hashes.json', 'source_hashes':input_hashes,
                 'processed_dataset_hash':None, 'split_manifest_hash':None, 'gold_eval_hash':file_sha256(args.gold)},
        'rag': {k:'n/a: preparation only; no retrieval/profile construction' for k in ('persona_profile_hash','chroma_collection','index_version_or_snapshot','embedding_model')},
        'model': {'model_name':'unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit tokenizer only',
                  'model_or_adapter_hash':'n/a: no weights loaded', 'config':{'tokenizer_snapshot':str(args.tokenizer.resolve()), 'max_tokens':args.max_tokens, 'policy':VERSION}, 'random_seed':'n/a: deterministic'},
        'evaluation': {'evaluator_commit':command_output(['git','rev-parse','HEAD']),
                       'evaluator_source_hashes':{str(p):file_sha256(p) for p in code_paths},
                       'judge_model':'n/a: lexical diagnostics only','judge_config':'n/a'},
        'environment': {'python_version':sys.version, 'package_snapshot':'config/package_snapshot.txt',
                        'hardware':platform.platform()+'; '+platform.processor()+'; CPU preparation; no GPU model load'},
        'outputs': {'raw':'raw/', 'metrics':'metrics/summary.json', 'warnings_and_failures':[]}}
    # JSON is valid YAML; avoids an extra dependency for the manifest writer.
    write_json(args.out/'manifest.yaml', manifest)
    try:
        docs=[]; ledger=[]; seen=set()
        for name,(medium,text_key,title_key) in SOURCES.items():
            raw_index=raw_lineage(read_numbered(args.raw_dir/name),medium,text_key,title_key)
            for line,row in read_numbered(args.cleaned_dir/name):
                key=(name,line); source_owners=owners.get(key,[])
                seen.add(key)
                original_text=str(row.get(text_key) or '')
                text=original_text.strip()
                span_start=len(original_text)-len(original_text.lstrip())
                stable=str(row.get('url') or row.get('video_url') or row.get('id') or '')
                matches=raw_index.get((stable,digest(text)),[])
                record={'source_file':name,'source_line':line,'source':medium,
                        'source_id':digest(name+'\n'+(stable or digest(text))),
                        'cleaned_text_sha256':digest(text), 'cleaned_span':[span_start,span_start+len(text)],
                        'raw_matches':matches, 'source_url_or_id':stable,
                        'splits':sorted({a['split'] for a in source_owners}),
                        'prior_group_ids':sorted({a['group_id'] for a in source_owners}),
                        'attribution':'channel_only_unverified' if medium.startswith('youtube') else 'source_account_or_corpus_label_not_independent_authorship_verification',
                        'reasons':[],'cleaning_version':VERSION}
                if not source_owners: record['reasons'].append('not_in_frozen_dataset')
                if not matches: record['reasons'].append('raw_cleaned_lineage_unverified')
                if len(record['splits'])>1 or len(record['prior_group_ids'])>1: record['reasons'].append('frozen_ownership_conflict')
                if source_owners and any(normalize_text(a['prior_output']) not in normalize_text(text) for a in source_owners):
                    record['reasons'].append('prior_fragment_source_mismatch')
                sft,reasons,n=candidate(row,medium,text_key,title_key,tokenizer)
                record['reasons'].extend(reasons);record['chat_tokens']=n
                ledger.append(record)
                if source_owners:
                    docs.append({'record':record,'text':text,'title':str(row.get(title_key) or '') if title_key else '',
                                 'splits':record['splits'],'sft':sft})
            print('Inspected',name,flush=True)
        missing=set(owners)-seen
        if missing: raise ValueError(f'{len(missing)} frozen source keys missing from cleaned corpus')
        edges,conflicts,gold_matches,gold_hit=leakage_links(docs,load_rows(args.gold))
        for i in conflicts: docs[i]['record']['reasons'].append('full_source_cross_split_family')
        for i in gold_hit: docs[i]['record']['reasons'].append('gold_overlap_family')
        budget_counts={str(b):Counter() for b in (512,1024,2048,4096)}
        accepted=[]; derived_assignments=[]
        for d in docs:
            r=d['record']; n=r['chat_tokens']
            if not r['reasons'] and n is not None:
                for b in budget_counts:
                    if n<=int(b):budget_counts[b][r['splits'][0]]+=1
                if n>args.max_tokens:r['reasons'].append('over_context_budget')
            if not r['reasons']:
                idx=len(accepted); split=r['splits'][0]; group=r['prior_group_ids'][0]
                accepted.append({**d['sft'],**{k:r[k] for k in ('source','source_file','source_line','source_id','cleaned_text_sha256','cleaned_span','raw_matches','attribution','chat_tokens','cleaning_version')},'split':split,'group_id':group})
                derived_assignments.append({'row_index':idx,'split':split,'group_id':group,'source_platform':r['source'],'source_file':r['source_file'],'source_line':r['source_line'],'base_title':d['title'],'part_index':None,'part_count':None})
        for r in ledger:r['disposition']='quarantined' if r['reasons'] else 'candidate'
        raw=args.out/'raw'
        write_jsonl(raw/'dispositions.jsonl',ledger)
        write_jsonl(raw/'dataset.jsonl',accepted)
        write_assignment_file(raw/'split_assignments.jsonl',file_sha256(raw/'dataset.jsonl'),derived_assignments)
        write_jsonl(raw/'duplicate_links.jsonl',({'left_source_id':docs[i]['record']['source_id'],'right_source_id':docs[j]['record']['source_id'],'kind':kind} for i,j,kind in edges))
        write_jsonl(raw/'gold_overlap.jsonl',({**m,'source_id':docs[m['document']]['record']['source_id']} for m in gold_matches))
        # No held-out bodies enter review samples or future retrieval pools.
        for split in ('train','validation','test'):
            write_jsonl(raw/f'{split}.jsonl',(r for r in accepted if r['split']==split))
        samples=[]
        for split in ('train','validation'):
            pool=sorted((r for r in accepted if r['split']==split),key=lambda r:r['source_id'])
            samples.extend(pool[:8])
        write_jsonl(raw/'review_sample_train_validation.jsonl',samples)
        changed=[p for p,h in input_hashes.items() if file_sha256(Path(p))!=h]
        if changed:raise RuntimeError(f'Input mutation detected: {changed}')
        summary={'candidate_rows':len(accepted),'splits':dict(Counter(r['split'] for r in accepted)),
                 'mediums':dict(Counter(r['source'] for r in accepted)),
                 'dispositions':dict(Counter(r['disposition'] for r in ledger)),
                 'reason_counts':dict(Counter(reason for r in ledger for reason in r['reasons'])),
                 'full_source_duplicate_links':len(edges),'cross_split_family_documents':len(conflicts),
                 'gold_matching_documents_including_family':len(gold_hit), 'eligibility_by_chat_budget':{b:dict(c) for b,c in budget_counts.items()},
                 'input_hashes_unchanged':True,'training_ready':False,
                 'limitations':['Lexical Jaccard is not semantic or exhaustive substring detection.',
                                'Authorship uses source attribution, not independent author verification.',
                                'Whole-source candidates require manual review; no training authorized by output.',
                                'Cross-split families quarantined without reassigning any source.',
                                'Untitled social posts and transcripts deferred; this candidate is not medium-comparable to prior SFT.']}
        write_json(args.out/'metrics'/'summary.json',summary)
        manifest['status']='MEASURED_PREPARATION_REVIEW_REQUIRED'
        manifest['data'].update(processed_dataset_hash=file_sha256(raw/'dataset.jsonl'),split_manifest_hash=file_sha256(raw/'split_assignments.jsonl'))
        manifest['outputs']['warnings_and_failures']=summary['limitations']
        print(json.dumps(summary,indent=2),flush=True)
    except Exception as exc:
        manifest['status']='FAILED';manifest['outputs']['warnings_and_failures'].append(repr(exc));raise
    finally:
        write_json(args.out/'manifest.yaml',manifest)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-dir',type=Path,default=Path('data'))
    p.add_argument('--cleaned-dir',type=Path,default=Path('data/cleaned'))
    p.add_argument('--prior-dataset',type=Path,required=True)
    p.add_argument('--assignments',type=Path,required=True)
    p.add_argument('--gold',type=Path,default=Path('data/openai_ft/lemkin_gold_eval.jsonl'))
    p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--max-tokens',type=int,default=2048)
    p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())


if __name__=='__main__':main()
