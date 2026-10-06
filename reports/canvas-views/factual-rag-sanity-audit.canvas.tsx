import {
  BarChart,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Stack,
  Stat,
  Table,
  Text,
} from "cursor/canvas";

const LENGTH_ROWS = [
  ["gold_001", "blog", "284", "244", "341", "310", "EOS, early", "EOS, early", "768", "44.4%", "40.4%"],
  ["gold_001", "linkedin", "16", "245", "24", "320", "EOS, early", "length", "320", "7.5%", "100%"],
  ["gold_001", "x", "16", "17", "24", "38", "EOS, early", "EOS, early", "160", "15.0%", "23.8%"],
  ["gold_001", "talk", "333", "278", "398", "364", "EOS", "EOS, early", "768", "51.8%", "47.4%"],
  ["gold_002", "blog", "276", "44", "389", "67", "EOS", "EOS, early", "768", "50.7%", "8.7%"],
  ["gold_002", "linkedin", "248", "76", "320", "115", "length", "EOS, early", "320", "100%", "35.9%"],
  ["gold_002", "x", "8", "57", "14", "91", "EOS, early", "EOS", "160", "8.8%", "56.9%"],
  ["gold_002", "talk", "320", "256", "440", "332", "EOS", "EOS, early", "768", "57.3%", "43.2%"],
];

export default function FactualRagSanityAudit() {
  return (
    <Stack gap={24}>
      <Stack gap={8}>
        <H1>Factual RAG sanity audit</H1>
        <Text tone="secondary">
          Relabel adapter, first two gold topics, four mediums. Retrieval-off is
          EXP-20261004-003. Retrieval-on is the 8-draft sanity run
          EXP-20261004-004-relabel-factual-rag-sanity. No new generation.
        </Text>
      </Stack>

      <Callout tone="warning" title="Do not run the 120-draft retrieval experiment">
        Retrieval did not systematically shorten these drafts. Two cells were
        already short with retrieval off. The real new failures are a copied
        headline tweet, two answers that echo the excerpt wrapper, and two
        gold_002 drafts that stop after a short list fragment.
      </Callout>

      <Grid columns={4} gap={16}>
        <Stat value="4 / 8" label="Retrieval-off drafts already early-EOS" />
        <Stat value="6 / 8" label="Retrieval-on drafts early-EOS" tone="warning" />
        <Stat value="-35" label="Mean word-count change, on minus off" />
        <Stat value="0.0087" label="gold_001 X near-copy cosine distance" tone="danger" />
      </Grid>

      <H2>Generated length, retrieval off versus on</H2>
      <Text tone="secondary" size="small">
        Word count of the decoded answer. Source: EXP-003 relabel.jsonl and the
        sanity relabel.jsonl. Same seeds 42–49. Token counts include the stop
        token. Early means EOS before half the max-token budget.
      </Text>
      <BarChart
        categories={[
          "001 blog",
          "001 LinkedIn",
          "001 X",
          "001 talk",
          "002 blog",
          "002 LinkedIn",
          "002 X",
          "002 talk",
        ]}
        series={[
          { name: "Retrieval off, words", data: [284, 16, 16, 333, 276, 248, 8, 320], tone: "neutral" },
          { name: "Factual RAG, words", data: [244, 245, 17, 278, 44, 76, 57, 256], tone: "info" },
        ]}
        height={240}
        valueSuffix=" words"
      />
      <Table
        headers={[
          "Topic",
          "Medium",
          "Off words",
          "On words",
          "Off tokens",
          "On tokens",
          "Off stop",
          "On stop",
          "Budget",
          "Off budget",
          "On budget",
        ]}
        columnAlign={[
          "left",
          "left",
          "right",
          "right",
          "right",
          "right",
          "left",
          "left",
          "right",
          "right",
          "right",
        ]}
        rows={LENGTH_ROWS}
        rowTone={[
          undefined,
          "info",
          "warning",
          undefined,
          "danger",
          "warning",
          "info",
          undefined,
        ]}
        striped
        stickyHeader
      />
      <Text tone="tertiary" size="small">
        Highlighted rows: gold_002 blog and LinkedIn got much shorter. gold_001 X
        was already a 16-word title. Blue rows echoed the factual-excerpt wrapper
        into the answer, so their higher word counts are not longer posts.
      </Text>

      <H2>Why some drafts stopped early</H2>
      <Text>
        The prompt did not consume the generation budget. The longest rendered
        prompt was 1,529 tokens, under the 2,048 context limit, and max-new-token
        budgets were unchanged. The model emitted EOS, or, in one case, filled
        the budget by copying the prompt.
      </Text>
      <Grid columns={2} gap={16}>
        <Card>
          <CardHeader>Already short with retrieval off</CardHeader>
          <CardBody>
            <Text>
              gold_001 X was 16 words and 24 tokens off, 17 words and 38 tokens on.
              The on answer is the retrieved tweet, including http://wp.me/p2Gf8o-19j.
              Token overlap with that hit is exact. gold_001 LinkedIn and gold_002 X
              were also title-only with retrieval off (16 and 8 words).
            </Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>New shortening on gold_002</CardHeader>
          <CardBody>
            <Text>
              Blog fell from 276 words to 44. The answer is one invented “#9”
              learning that is not in the four Monday.com excerpts, then EOS at 8.7%
              of the 768-token budget. LinkedIn fell from 248 words, at the length
              cap, to 76 words about Similarweb. That name appears inside a loosely
              related LinkedIn chunk, not in a Monday.com post.
            </Text>
          </CardBody>
        </Card>
      </Grid>
      <Text>
        Two answers pasted the wrapper “Factual excerpts from train-owned source
        text” and an excerpt into the completion: gold_001 LinkedIn, which then hit
        the 320-token cap, and gold_002 X. gold_001 blog and both talks stayed in
        the same length band. Talk crossed the early-EOS line only because 364 of
        768 tokens is just under half; the off draft used 51.8%.
      </Text>

      <H2>Near-copy: train_26148</H2>
      <Text tone="secondary" size="small">
        Read-only lookup in the EXP-004 assignments, the cleaned X row, the
        EXP-003 dispositions, and gold_overlap.jsonl. Nothing was deleted.
      </Text>
      <Table
        headers={["Field", "train_26148", "Held-out blog the gold topic matches"]}
        rows={[
          ["row_id", "26148", "10292–10294"],
          ["split", "train", "test"],
          ["group_id", "d845dcfa…df0f2f0a", "9b83196a…1e2d4ff"],
          ["source file", "jasonlk_originals.jsonl:23833", "jasonlemkin_blog.jsonl:2892"],
          ["URL", "https://x.com/jasonlk/status/395571048529534976", "https://www.saastr.com/can-an-8-person-startup-sell-to-a-cio-yes-if-you-understand-the-social-contract/"],
          ["Date", "2013-10-30", "2021-12-30, marked as an updated classic"],
          ["Normalized title", "Headline plus a WordPress shortlink, ASCII hyphens", "Same headline, em dash, no URL"],
          ["Gold overlap flag", "Not flagged. Jaccard 0.7619", "gold_overlap_family"],
        ]}
        striped
      />
      <Text>
        The gold file stores no source id, URL, or date. Its topic string is
        identical to the test-split blog’s assignment title. The 2013 tweet is that
        headline plus http://wp.me/p2Gf8o-19j. Sequence similarity of the normalized
        strings is 0.86. This is a cross-post of the same headline, in a different
        document family from the blog, not an independent topical match.
      </Text>
      <Callout tone="danger" title="The current family exclusion missed this spelling">
        Two later “Best of” tweets of the same headline were flagged because the
        em-dash topic string is contained in them, and their group was left out of
        the index. train_26148 uses “--” instead of “—”, so containment failed, and
        the URL tokens hold token Jaccard at 0.7619, under the 0.80 cutoff. A short
        tweet also cannot join the long blog near-duplicate family. The rule should
        have caught this row. This implementation did not.
      </Callout>

      <H2>Medium filter versus all-media evidence</H2>
      <Text>
        The query is the gold topic only. An unfiltered query therefore returns one
        ranking, and a second identical query returned the same four ids. Requested
        output medium changes the evidence only through the metadata filter. Dropping
        that filter would attach the same four chunks to blog, LinkedIn, X, and talk.
      </Text>
      <H3>All-media top 4, k=4, no rerank</H3>
      <Table
        headers={["Topic", "Rank", "ID", "Distance", "Platform", "Preview"]}
        columnAlign={["left", "right", "left", "right", "left", "left"]}
        rows={[
          ["gold_001", "1", "train_26148", "0.0087", "x", "Headline plus wp.me shortlink"],
          ["gold_001", "2", "train_20783", "0.3081", "x", "Selling SaaS to a big corporation, link"],
          ["gold_001", "3", "train_26362", "0.3093", "x", "Founders refusing to sell, Quora link"],
          ["gold_001", "4", "train_21878", "0.3159", "x", "CIOs on how startups sell to the enterprise"],
          ["gold_002", "1", "train_10049", "0.2178", "blog", "Monday.com check-in at $400m ARR"],
          ["gold_002", "2", "train_9156", "0.2184", "blog", "#7. Monday.com growing 68% at $550m ARR"],
          ["gold_002", "3", "train_14595", "0.2232", "x", "monday.com crosses $1B ARR"],
          ["gold_002", "4", "train_10774", "0.2378", "blog", "Monday.com IPO filing at $240m ARR"],
        ]}
        rowTone={["danger", undefined, undefined, undefined, "success", "success", "success", "success"]}
        striped
      />
      <Text tone="secondary" size="small">
        All eight hits are split=train. gold_001’s global top 4 are all X posts, so
        the X filter matches the unfiltered list and every other medium filter does
        not. gold_002’s global list is three Monday.com blog chunks plus one Monday
        tweet. The fourth blog chunk, train_10147 at distance 0.2446, falls just
        outside k=4.
      </Text>
      <H3>What the medium filter substitutes, best hit</H3>
      <Table
        headers={["Topic", "Medium", "Filtered best", "Distance", "All-media best", "Distance"]}
        columnAlign={["left", "left", "left", "right", "left", "right"]}
        rows={[
          ["gold_001", "blog", "train_565", "0.3253", "train_26148", "0.0087"],
          ["gold_001", "linkedin", "train_11587", "0.3729", "train_26148", "0.0087"],
          ["gold_001", "x", "train_26148", "0.0087", "train_26148", "0.0087"],
          ["gold_001", "talk", "train_26541", "0.4183", "train_26148", "0.0087"],
          ["gold_002", "blog", "train_10049", "0.2178", "train_10049", "0.2178"],
          ["gold_002", "linkedin", "train_12154", "0.3544", "train_10049", "0.2178"],
          ["gold_002", "x", "train_14595", "0.2232", "train_10049", "0.2178"],
          ["gold_002", "talk", "train_26541", "0.4248", "train_10049", "0.2178"],
        ]}
        rowTone={[
          undefined,
          "warning",
          "danger",
          "warning",
          undefined,
          "warning",
          undefined,
          "warning",
        ]}
        striped
      />
      <Text>
        Talk evidence is the weak case for the filter. Only 40 train-owned YouTube
        documents exist, and the best talk hit for both topics is a transcript
        fragment around distance 0.42 about selling a company or a demo, not the
        gold topic. For Monday.com, the unfiltered list is the on-topic blog and
        tweet evidence. For the CIO topic, the unfiltered list is better enterprise
        headlines only after the near-copy tweet. Removing the filter would also
        paste that tweet into the blog, LinkedIn, and talk prompts.
      </Text>

      <Divider />

      <H2>Recommendation</H2>
      <Text>
        Factual retrieval should drop the output-medium filter. The adapter is the
        style channel. Source platform is not a factual constraint, and it leaves
        talk with a 40-document corpus. Keep k=4, topic-only queries, and no reranker.
      </Text>
      <Text>
        One code change is required before another sanity run: stop passing the
        medium filter in the train-only retriever. Leave temperature, top-p,
        repetition penalty, token budgets, seeds, and the excerpt wording as they
        are. Do not rebuild the index for that run.
      </Text>
      <Card>
        <CardHeader>Proposed next sanity, after the filter change</CardHeader>
        <CardBody>
          <Text>
            Same relabel adapter, first two topics, factual retrieval on, scoring
            skipped, fixed style off. New directory
            experiments/EXP-20261004-008-relabel-factual-all-media. Eight drafts.
            The only intended change from the last sanity is that every medium
            receives the unfiltered top 4.
          </Text>
        </CardBody>
      </Card>
      <Callout tone="warning" title="Do not continue from that run into the full 120">
        gold_001’s unfiltered top hit will be train_26148 in all four mediums until
        that cross-post is excluded. Review the eight drafts. Exclude this
        dash-variant family before a 120-draft run. The 120 also waits on a
        judgment of the excerpt-wrapper echoes and the gold_002 early stops.
      </Callout>
    </Stack>
  );
}
