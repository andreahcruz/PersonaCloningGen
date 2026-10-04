from host_finetune.generation_diagnostics import stop_metadata, summarize_stops


def test_eos_at_budget_is_eos_not_cutoff():
    assert stop_metadata([1, 9], [9, 10], 2)['stop_reason'] == 'eos'
    assert stop_metadata([1, 2], [9, 10], 2)['stop_reason'] == 'length'


def test_short_output_without_eos_is_unknown():
    assert stop_metadata([1], 9, 20)['stop_reason'] == 'other_or_unknown'
    assert stop_metadata([], None, 20)['last_token_id'] is None


def test_early_eos_diagnostic_and_missing_legacy_metadata():
    trace = stop_metadata([1, 9], 9, 20)
    assert trace['early_eos']
    report = summarize_stops([trace, {'answer': 'Legacy output.'}])
    assert report['stop_metadata_available'] == 1
    assert report['early_eos_rate'] == 1
    assert summarize_stops([{}])['early_eos_rate'] is None


def test_jsonl_unicode_separator_is_content(tmp_path):
    import json
    from host_finetune.audit_relabel_run import read
    path = tmp_path/'rows.jsonl'
    path.write_text(json.dumps({'output':'a\u2028b'}, ensure_ascii=False)+'\n',encoding='utf-8')
    assert read(path) == [{'output':'a\u2028b'}]
