"""Read-only input checks and independent validation of a complete-source run.

Adds verification evidence to the specified run; never rewrites its datasets.
Uses direct chat-template tokenization and brute-force cross-split Jaccard checks
instead of the preparation job's indexed near-duplicate implementation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import unicodedata


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def norm(text):
    return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())


def check(condition, message):
    if not condition:raise ValueError(message)


def chat_token_count(tokenizer, messages):
    # Transformers versions return either token IDs or a BatchEncoding mapping.
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    ids = encoded['input_ids'] if hasattr(encoded, 'keys') else encoded
    return len(ids)


def verify(run):
    from transformers import AutoTokenizer
    config=json.loads((run/'config/parameters.json').read_text(encoding='utf-8'))
    manifest=json.loads((run/'manifest.yaml').read_text(encoding='utf-8'))
    inputs=json.loads((run/'config/input_hashes.json').read_text(encoding='utf-8'))
    check(all(sha(p)==h for p,h in inputs.items()),'Input hash changed')
    dataset=rows(run/'raw/dataset.jsonl')
    assignments=rows(run/'raw/split_assignments.jsonl')
    check(assignments[0]['dataset_sha256']==sha(run/'raw/dataset.jsonl'),'Dataset hash mismatch')
    check(manifest['data']['processed_dataset_hash']==sha(run/'raw/dataset.jsonl'),'Manifest dataset hash mismatch')
    check(manifest['data']['split_manifest_hash']==sha(run/'raw/split_assignments.jsonl'),'Manifest split hash mismatch')
    check(assignments[0]['n_rows']==len(dataset)==len(assignments)-1,'Assignment count mismatch')
    prior=rows(config['assignments'])[1:]
    inherited={}
    for a in prior:
        inherited.setdefault((a['source_file'],a['source_line']),set()).add((a['split'],a['group_id']))
    cleaned={name:Path(config['cleaned_dir'],name).read_text(encoding='utf-8').split('\n')
             for name in {r['source_file'] for r in dataset}}
    tokenizer=AutoTokenizer.from_pretrained(config['tokenizer'],local_files_only=True,trust_remote_code=False)
    normalized=[]; token_sets=[]
    for i,(r,a) in enumerate(zip(dataset,assignments[1:])):
        check(a['row_index']==i,'Noncontiguous row index')
        check(inherited[(r['source_file'],r['source_line'])]=={(r['split'],r['group_id'])},'Ownership changed')
        check((a['split'],a['group_id'],a['source_file'],a['source_line'])==
              (r['split'],r['group_id'],r['source_file'],r['source_line']),'Row/assignment mismatch')
        original=json.loads(cleaned[r['source_file']][r['source_line']-1])['content']
        start,end=r['cleaned_span']
        check(original[start:end]==r['output']==original.strip(),'Target is not the full source body')
        check(hashlib.sha256(r['output'].encode()).hexdigest()==r['cleaned_text_sha256'],'Source hash mismatch')
        chat=[{'role':'user','content':r['instruction']},{'role':'assistant','content':r['output']}]
        n=chat_token_count(tokenizer,chat)
        check(n==r['chat_tokens'] and n<=config['max_tokens'],'Token count/overflow mismatch')
        value=norm(r['output']);normalized.append(value);token_sets.append(set(re.findall(r'[a-z0-9$%]+',value)))
    comparisons=0; crossings=[]
    for i, left in enumerate(dataset):
        for j in range(i+1,len(dataset)):
            if left['split']==dataset[j]['split']:continue
            comparisons+=1
            a,b=token_sets[i],token_sets[j]
            if normalized[i]==normalized[j]:crossings.append([i,j,'exact']);continue
            if not a or not b or min(len(a),len(b))/max(len(a),len(b))<0.8:continue
            if len(a&b)/len(a|b)>=0.8:crossings.append([i,j,'near'])
    check(not crossings,'Cross-split lexical duplicates remain')
    for split in ('train','validation','test'):
        check(rows(run/f'raw/{split}.jsonl')==[r for r in dataset if r['split']==split],'Partition export mismatch')
    samples=rows(run/'raw/review_sample_train_validation.jsonl')
    check(all(r['split'] in ('train','validation') and r in dataset for r in samples),'Test data entered review sample')
    old=list(csv.DictReader(Path('docs/data/artifact_hashes.csv').open(encoding='utf-8')))
    check(all(sha(r['path'])==r['sha256'] for r in old),'An audit-inventoried old artifact changed')
    report={'status':'PASSED', 'rows_checked':len(dataset),
            'splits':dict(Counter(r['split'] for r in dataset)),
            'cross_split_pairs_checked_brute_force':comparisons,'cross_split_exact_or_jaccard_0_8_matches':len(crossings),
            'normalized_exact_duplicate_rows_within_splits':len(normalized)-len(set(normalized)),
            'prior_audit_artifacts_unchanged':len(old),'all_input_hashes_unchanged':True,
            'full_source_targets_verified':True,'full_chat_token_counts_verified':True,
            'split_ownership_preserved':True,'review_sample_excludes_test':True,
            'scope':'Structural/token/lexical checks only; not semantic completeness or verified authorship.'}
    (run/'metrics/verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    (run/'config/verify_complete_sft.py').write_bytes(Path(__file__).read_bytes())
    manifest['verification']={'command':subprocess.list2cmdline([sys.executable,'-m','host_finetune.verify_complete_sft',str(run)]),
                              'source_sha256':sha(__file__),'report':'metrics/verification.json'}
    (run/'manifest.yaml').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    verify(p.parse_args().run)
