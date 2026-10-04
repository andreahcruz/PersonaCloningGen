"""Observable stopping diagnostics, deliberately separate from completion quality."""


def stop_metadata(token_ids, eos_token_ids, max_new_tokens):
    ids = list(token_ids)
    eos = {eos_token_ids} if isinstance(eos_token_ids, int) else set(eos_token_ids or [])
    # EOS at exactly the budget is still an observed EOS, not inferred truncation.
    reason = ('eos' if ids and ids[-1] in eos else
              'length' if len(ids) >= max_new_tokens else 'other_or_unknown')
    return {'generated_tokens_including_stop': len(ids),
            'last_token_id': ids[-1] if ids else None,
            'effective_eos_token_ids': sorted(eos), 'stop_reason': reason,
            'budget_fraction': len(ids) / max_new_tokens if max_new_tokens else None,
            'early_eos': reason == 'eos' and len(ids) < max_new_tokens / 2}


def summarize_stops(traces):
    observed = [r for r in traces if r.get('stop_reason') in {'eos', 'length'}]
    return {'n': len(traces), 'stop_metadata_available': len(observed),
            'early_eos_rate': (sum(bool(r.get('early_eos')) for r in observed) / len(observed)
                               if observed else None),
            'length_limit_rate': (sum(r['stop_reason'] == 'length' for r in observed) / len(observed)
                                  if observed else None),
            'interpretation': 'Early EOS and absent final punctuation are not semantic incompleteness.'}
