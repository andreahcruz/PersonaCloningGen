import pytest

from host_finetune.prepare_complete_sft import (
    candidate, completeness_reason, leakage_links, ownership, raw_lineage,
)


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return 'system overhead ' + ' '.join(m['content'] for m in messages) + ' eos'

    def __call__(self, text, **kwargs):
        assert kwargs['truncation'] is False
        return {'input_ids': text.split()}


def test_complete_body_newlines_and_unicode_are_unchanged():
    body = 'A concrete operator lesson. ' * 20 + '\n\n1. Price early.\u2028\n2. Hire slowly.'
    sft, reasons, n = candidate({'title':'Pricing','content':body}, 'blog','content','title',Tokenizer())
    assert sft['output'] == body
    assert not reasons
    assert n > len(body.split())  # Full chat, not just answer tokens.


def test_oversized_sentence_is_not_split_or_silently_truncated():
    body = 'word ' * 3000 + 'end.'
    sft, reasons, n = candidate({'title':'Pricing','content':body}, 'blog','content','title',Tokenizer())
    assert not reasons and n > 2048
    assert sft['output'] == body


@pytest.mark.parametrize('body', ['The list:', 'An unfinished thought', 'We need...', 'We need\u2026'])
def test_incomplete_and_ambiguous_endings_are_deferred(body):
    assert completeness_reason(body)


def test_closed_quote_and_complete_list_ending_are_allowed():
    assert completeness_reason('He said, “Price earlier.”') is None
    assert completeness_reason('Key steps:\n1. Price early.\n2. Hire slowly.') is None
    assert completeness_reason('Key steps:\n1. Pricing\n2. Hiring') == 'ending_requires_review'


def test_untitled_social_never_uses_target_as_topic():
    body = 'Here is my advice. ' * 20
    sft, reasons, _ = candidate({'author':'Jason M. Lemkin','content':body},'linkedin','content',None,Tokenizer())
    assert 'missing_independent_topic' in reasons
    assert sft['instruction'] == ''
    assert sft['output'] == body.strip()


def test_transcripts_and_uncertain_authors_are_quarantined():
    _, reasons, _ = candidate({'video_title':'Advice','transcript_text':'Good advice. '*40},'youtube_jason','transcript_text','video_title',Tokenizer())
    assert 'unverified_speaker' in reasons
    _, reasons, _ = candidate({'author':'Guest','content':'Good advice. '*40},'linkedin','content',None,Tokenizer())
    assert 'uncertain_author' in reasons


def test_title_only_and_instruction_echo_are_not_targets():
    title = 'Advice. ' * 60
    _, reasons, _ = candidate({'title':title,'content':title},'blog','content','title',Tokenizer())
    assert 'title_or_instruction_only' in reasons
    _, reasons, _ = candidate({'title':'Advice','content':'Write a blog post in the style of Jason Lemkin about: Advice. ' * 10},'blog','content','title',Tokenizer())
    assert 'instruction_echo_target' in reasons


def test_ownership_rejects_unaligned_and_invalid_rows():
    prior=[{'source_file':'a','source_line':1,'output':'Body.'}]
    a={'row_index':0,'source_file':'a','source_line':1,'split':'train','group_id':'g'}
    assert ownership([a],prior)[('a',1)][0]['split']=='train'
    for change in ({'row_index':2},{'source_line':2},{'split':'unknown'}):
        with pytest.raises(ValueError): ownership([{**a,**change}],prior)


def test_lineage_replays_cleaner_without_using_raw_line_position():
    row={'url':'https://example.org/a','title':'Advice','content':'Sound advice. '*30}
    index=raw_lineage([(42,row)],'blog','content','title')
    assert next(iter(index.values()))[0]['raw_line']==42
    assert row['content']=='Sound advice. '*30


def doc(text, split, group, title=''):
    return {'text':text,'title':title,'splits':[split],
            'record':{'prior_group_ids':[group]}}


def test_duplicate_conflicts_propagate_through_frozen_family_without_reassignment():
    rows=[doc('same output.','train','a'),doc('same output.','test','b'),doc('Unrelated words.','train','a')]
    _, conflicts, _, _=leakage_links(rows,[])
    assert conflicts=={0,1,2}
    assert [r['splits'] for r in rows]==[['train'],['test'],['train']]


def test_near_duplicate_cross_platform_conflict():
    text=' '.join(f'word{i}' for i in range(25))
    _, conflicts, _, _=leakage_links([doc(text,'train','a'),doc(text+' extra','validation','b')],[])
    assert conflicts=={0,1}


def test_gold_title_match_propagates_to_frozen_family():
    rows=[doc('a different body','train','g','Pricing'),doc('other words','train','g')]
    _, _, matches, hits=leakage_links(rows,[{'id':'gold1','topic':'Pricing'}])
    assert matches[0]['field']=='topic'
    assert hits=={0,1}


@pytest.mark.parametrize('title,body,reason', [
    ('Can’t Miss Sessions at SaaStr Annual 2021!!', 'Advice. '*60, 'event_titled_document_requires_review'),
    ('80% of Sponsor Expo Slots Sold for Annual', 'Advice. '*60, 'event_titled_document_requires_review'),
    ('Going Upmarket with Stripe’s CBO', 'Advice. '*60, 'guest_recap_attribution_requires_review'),
    ('Market advice', 'Advice. '*60+' pic.twitter.com/123 More advice.', 'embedded_or_external_context_requires_review'),
    ('Popular Quora Answers', '\n'.join(['What is the next thing to do?']*12), 'question_link_roundup'),
    ('Top SaaStr Content of the Week', 'Advice. '*60, 'content_link_roundup'),
    ('Outbound advice', 'Advice. '*60+' A recent survey we did above supports this.', 'missing_visual_context_requires_review'),
    ('Fundraising advice', 'Advice. '*60+' Even bette\nr than before.', 'scraped_word_break_requires_review'),
    ('Sales advice', 'Advice. '*60+' A related post here:\nHow do we sell?', 'trailing_related_link_requires_review'),
])
def test_review_discovered_contaminants_are_quarantined(title,body,reason):
    _, reasons, _=candidate({'title':title,'content':body},'blog','content','title',Tokenizer())
    assert reason in reasons


@pytest.mark.parametrize('encoded', [[1,2,3], {'input_ids':[1,2,3], 'attention_mask':[1,1,1]}])
def test_verifier_supports_tokenizer_list_and_mapping_returns(encoded):
    from host_finetune.verify_complete_sft import chat_token_count
    class ChatTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return encoded
    assert chat_token_count(ChatTokenizer(),[])==3
