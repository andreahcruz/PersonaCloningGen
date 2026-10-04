"""CPU-only audit of a frozen relabel run. Writes only to a NEW experiment.

Never imports training, loads model weights, or changes input artifacts.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def norm(text):
    return ' '.join(text.split())


def audit(dataset, assignments, prior, prior_assignments, ledger, cleaned):
    rows, records = read(dataset), read(assignments)
    meta, owners = records[0], records[1:]
    old, old_owners = read(prior), read(prior_assignments)[1:]
    assert meta['dataset_sha256'] == sha(dataset)
    assert meta['n_rows'] == len(rows) == len(owners)
    prior_meta = read(prior_assignments)[0]
    assert prior_meta['dataset_sha256'] == sha(prior)
    flags = {(r['source_file'], r['source_line']): r['reasons'] for r in read(ledger)}
    source_rows = {name: read(cleaned / name) for name in {r['source_file'] for r in rows}}
    counts = defaultdict(Counter)
    findings = []
    family_splits = defaultdict(set)
    for i, (r, a) in enumerate(zip(rows, owners)):
        j = a['prior_row_index']
        p, pa = old[j], old_owners[j]
        assert a['row_index'] == i and pa['row_index'] == j
        assert r['output'] == p['output'], f'target changed: {i}'
        assert a['split'] == pa['split'] and a['group_id'] == pa['group_id']
        assert (r['source_file'], r['source_line']) == (p['source_file'], p['source_line'])
        key = (r['source_file'], r['source_line'])
        family_splits[a['group_id']].add(a['split'])
        split, medium, role = a['split'], r['source'], r['sft_role']
        counts[split]['rows'] += 1
        counts[split][f'medium:{medium}'] += 1
        counts[split][f'role:{role}'] += 1
        reasons = []
        source = source_rows[key[0]][int(key[1])-1]
        body = source.get('content', source.get('text', source.get('transcript_text', ''))) or ''
        if role == 'unchanged' and norm(r['output']) != norm(body):
            reasons.append('unchanged_target_not_whole_cleaned_body')
        if medium.startswith('youtube'):
            reasons.append('transcript_retained')
        if role == 'continuation' and 'Previous section ending:' not in r['instruction']:
            reasons.append('continuation_without_previous_context')
        if role != 'unchanged' and r['output'].rstrip().endswith((':', '...', '\u2026')):
            reasons.append('ambiguous_section_ending')
        prefix = ' '.join(r['output'].split()[:12])
        if len(prefix.split()) >= 8 and prefix.casefold() in norm(r['instruction']).casefold():
            reasons.append('target_prefix_in_prompt')
        for flag in flags.get(key, []):
            if flag in {'full_source_cross_split_family', 'gold_overlap_family',
                        'guest_recap_attribution_requires_review', 'content_link_roundup',
                        'scraped_word_break_requires_review', 'missing_visual_context_requires_review'}:
                reasons.append('source_flag:' + flag)
        for reason in reasons:
            counts[split][reason] += 1
        if reasons:
            findings.append({'row_index': i, 'prior_row_index': j, 'split': split,
                             'source_file': key[0], 'source_line': key[1], 'reasons': reasons})
    assert all(len(s) == 1 for s in family_splits.values())
    return {'checks': {'targets_unchanged': True, 'split_ownership_unchanged': True,
                       'original_groups_do_not_cross_splits': True},
            'counts_by_split': dict(counts),
            'limitations': ['Source flags reuse the v3 full-source audit; no new semantic leakage search.',
                           'Flags are review candidates, not automatic deletion decisions.',
                           'No model evaluation or tokenizer run; no test bodies manually reviewed.']}, findings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise SystemExit('Use a new output directory')
    base = Path('experiments/EXP-20261003-004-relabel-continuations/raw')
    inputs = dict(dataset=base/'dataset.jsonl', assignments=base/'split_assignments.jsonl',
                  prior=Path('host_finetune/data/dataset_fit512_keepbreaks_balanced.jsonl'),
                  prior_assignments=Path('experiments/EXP-20261001-008-repaired-balanced-split/raw/split_assignments.jsonl'),
                  ledger=Path('experiments/EXP-20261003-003-complete-source-sft-v3/raw/dispositions.jsonl'))
    cleaned = Path('data/cleaned')
    paths = list(inputs.values()) + list(cleaned.glob('*.jsonl')) + [Path(__file__)]
    hashes = {str(x): sha(x) for x in paths}
    for folder in ('raw', 'metrics', 'config', 'figures'):
        (args.out/folder).mkdir(parents=True, exist_ok=True)
    dump = lambda path, obj: path.write_text(json.dumps(obj, indent=2)+'\n', encoding='utf-8')
    cmd = lambda xs: subprocess.check_output(xs, text=True, encoding='utf-8').strip()
    (args.out/'config/audit_relabel_run.py').write_bytes(Path(__file__).read_bytes())
    (args.out/'config/package_snapshot.txt').write_text(cmd([sys.executable,'-m','pip','freeze']),encoding='utf-8')
    dump(args.out/'config/input_hashes.json', hashes)
    manifest = {'experiment_id': args.out.name, 'timestamp': datetime.now(timezone.utc).isoformat(),
        'status': 'RUNNING', 'git': {'commit': cmd(['git','rev-parse','HEAD']), 'dirty': bool(cmd(['git','status','--porcelain']))},
        'execution_commands': [subprocess.list2cmdline([sys.executable,'-m','host_finetune.audit_relabel_run',*sys.argv[1:]])],
        'data': {'source_manifest':'config/input_hashes.json','source_hashes':hashes,
                 'processed_dataset_hash':sha(inputs['dataset']), 'split_manifest_hash':sha(inputs['assignments']),
                 'gold_eval_hash':sha('data/openai_ft/lemkin_gold_eval.jsonl')},
        'model': {'model_name':'n/a: dataset audit','model_or_adapter_hash':'n/a: no weights loaded','config':{},'random_seed':'n/a: deterministic'},
        'rag': {k:'n/a: no retrieval' for k in ('persona_profile_hash','chroma_collection','index_version_or_snapshot','embedding_model')},
        'evaluation': {'evaluator_commit':cmd(['git','rev-parse','HEAD']), 'source_sha256':sha(__file__), 'judge_model':'n/a','judge_config':'n/a'},
        'environment': {'python_version':sys.version, 'hardware':platform.platform()+' '+platform.processor()+' CPU only', 'package_snapshot':'config/package_snapshot.txt'},
        'outputs': {'raw':'raw/findings.jsonl','metrics':'metrics/audit.json','warnings_and_failures':[]}}
    try:
        report, findings = audit(**inputs, cleaned=cleaned)
        assert all(sha(x) == h for x,h in hashes.items()), 'Input changed'
        report['all_inputs_unchanged'] = True
        dump(args.out/'metrics/audit.json', report)
        (args.out/'raw/findings.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in findings),encoding='utf-8')
        manifest['status'] = 'MEASURED'
        manifest['outputs']['warnings_and_failures'] = report['limitations']
        print(json.dumps(report, indent=2))
    except Exception as exc:
        manifest['status'] = 'FAILED'
        manifest['outputs']['warnings_and_failures'].append(repr(exc))
        raise
    finally:
        dump(args.out/'manifest.yaml',manifest)


if __name__ == '__main__':
    main()
