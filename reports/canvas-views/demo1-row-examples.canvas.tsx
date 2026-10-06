import {
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useCanvasState,
  useHostTheme,
} from "cursor/canvas";

type Tab = "clean" | "split" | "analytics" | "ml" | "flags";

const TABS: { id: Tab; label: string }[] = [
  { id: "clean", label: "1. Preprocessing" },
  { id: "split", label: "2. Train and test" },
  { id: "analytics", label: "3. Analytics" },
  { id: "ml", label: "4. ML results" },
  { id: "flags", label: "5. Generation flags" },
];

function Quote({ label, body }: { label: string; body: string }) {
  const theme = useHostTheme();
  return (
    <Stack gap={4}>
      <Text size="small" tone="tertiary" weight="semibold">
        {label}
      </Text>
      <div
        style={{
          background: theme.fill.tertiary,
          color: theme.text.primary,
          padding: "8px 10px",
          borderRadius: 6,
          whiteSpace: "pre-wrap",
        }}
      >
        <Text size="small">{body}</Text>
      </div>
    </Stack>
  );
}

function Clean() {
  return (
    <Stack gap={16}>
      <Text>
        Raw files were not edited. 32,835 rows became 30,370 cleaned rows.
      </Text>
      <Row gap={16} wrap>
        <Stat value="1,203" label="X posts under 25 characters" />
        <Stat value="623" label="LinkedIn feed cards" />
        <Stat value="450" label="Event promos, all sources" />
        <Stat value="5,492" label="SaaStr talk chunks removed later" tone="warning" />
      </Row>
      <H3>1. Too short to keep — X only, under 25 characters</H3>
      <Table
        headers={["Post", "Text that was dropped", "Characters"]}
        columnAlign={["left", "left", "right"]}
        rows={[
          ["2035910020949590443", "More here:", "10"],
          ["2038637529629692364", "Full deep dive here:", "20"],
          ["2032485949553488187", "A related post here:", "20"],
          ["2028229199048347878", "YouTube here:", "13"],
          ["2034649746057626071", "We're back!!", "12"],
        ]}
        rowTone={["danger", "danger", "danger", "danger", "danger"]}
      />
      <Text size="small" tone="secondary">
        Source: data/jasonlk_originals.jsonl. All 1,203 are from this file.
        Blog posts are not dropped by this rule.
      </Text>
      <H3>2. LinkedIn page chrome — body starts with “Feed post”</H3>
      <Quote
        label="Dropped, line 202 of jasonlemkinlinkedin.jsonl"
        body={'Feed post number 279 … merry Xmas!! Play Loaded: 100.00% Remaining time 0:04 … 3 comments 1 repost Like Comment Repost Send'}
      />
      <Quote
        label="Dropped, line 86 — a feed card that swallowed someone else’s show"
        body={'Feed post number 132 … Visible to anyone on or off LinkedIn Follow We\'re back!! Harry Stebbings … My 6 takeaways with Jason Lemkin…'}
      />
      <Text size="small" tone="secondary">
        623 of 2,675 LinkedIn rows. A post that does not start with “Feed post”
        is kept. 2,006 LinkedIn rows survived cleaning.
      </Text>
      <H3>3. Event promo — short post that is an ad</H3>
      <Quote
        label="Dropped from jasonlemkin_blog.jsonl"
        body="We’re 60 Days Out from SaaStr AI Annual 2026. Last Chance to Apply to Speak. And Why You Should Sponsor. — May 12-14. SF Bay. 40 acres. 300+ sessions…"
      />
      <H3>4. YouTube: empty captions dropped; SaaStr talks dropped later</H3>
      <Table
        headers={["Channel", "Raw", "After cleaning", "After balance"]}
        columnAlign={["left", "right", "right", "right"]}
        rows={[
          ["Jason’s channel", "462", "428 (34 empty transcripts)", "1,698 training chunks kept"],
          ["SaaStr channel", "199", "193 (5 empty, 1 promo)", "0 — all 5,492 chunks removed"],
        ]}
        rowTone={["success", "warning"]}
      />
      <H3>5. The label fix — same target, instruction now matches the slice</H3>
      <Quote
        label="Before, on every slice"
        body="Write a blog post in the style of Jason Lemkin about: 5 Ways to Turn Your Support Data Into a Customer Acquisition Channel with HappyFox’s Founder CEO"
      />
      <Quote
        label="After, a real continuation row (targets were not rewritten)"
        body={'Continue this blog post in the style of Jason Lemkin about: 5 Ways to Turn Your Support Data… — 1. Your Support Queue Already Has the Expansion Signals. Previous section ending: "Here are the 5 most actionable takeaways."'}
      />
    </Stack>
  );
}

function Split() {
  return (
    <Stack gap={16}>
      <Text>
        A group is one document family. The whole group gets one split, so a
        test row is not a later slice of a post the model already trained on.
      </Text>
      <Row gap={16} wrap>
        <Stat value="21,377" label="Train rows" />
        <Stat value="2,588" label="Validation rows" />
        <Stat value="2,580" label="Test rows" />
        <Stat value="0" label="Groups found in two splits" tone="success" />
      </Row>
      <H3>One post, seven slices, one split</H3>
      <Text>
        “5 Ways to Turn Your Support Data Into a Customer Acquisition Channel
        with HappyFox’s Founder CEO” is line 1 of jasonlemkin_blog.jsonl. It
        became 1 opening and 6 continuations. Same file, same line, same title,
        so the group stays in one split.
      </Text>
      <Quote
        label="Opening. The target stops after the setup."
        body="Shalin Jain built HappyFox to $20M+ ARR, bootstrapped, with about 100 employees. He went from 5 support reps at $1M ARR to 6 support reps at $20M ARR. Same story on sales: 6 sellers at $3M, still 6 sellers at $20M. He recently shared how his team did it by treating support data as their primary GTM engine. Here are the 5 most actionable takeaways."
      />
      <Quote
        label='Continuation. Instruction quotes the previous ending: "Here are the 5 most actionable takeaways."'
        body={"1. Your Support Queue Already Has the Expansion Signals. You Just Need Something Reading Them.\nMost B2B companies treat expansion as a sales-led or CS-led motion. Shalin found the highest-ROI signals were already sitting in support tickets.\nCustomers mention needing more seats. They ask about features on a higher plan. They reference products they don’t have yet."}
      />
      <Quote
        label='Next slice. Previous ending: "You’ll be surprised how much pipeline is hiding in plain sight."'
        body={"2. Bridge CRM Data Into Support (and Support Data Into Your CRM). Both Directions Matter.\nThis was one of the sharpest tactical moves in Shalin’s playbook. He built two agents that work in opposite directions.\nWhen a trial customer or active deal submits a support ticket, the agent pulls deal stage, MRR, and context from the CRM and surfaces it to the support rep. So support knows this isn’t a random free-tier user. It’s a $50K deal in evaluation."}
      />
      <H3>The four links, on this post</H3>
      <Table
        headers={["Rule", "What it caught here", "Why they share a split"]}
        rows={[
          [
            "Same source file and line",
            "The HappyFox post was cut into several 512-token slices from one cleaned line",
            "Part 2 is the rest of part 1",
          ],
          [
            "Same normalized title",
            "“(part i of n)” is stripped, so every slice keeps the HappyFox title",
            "Same article, even if the line ids differed",
          ],
          [
            "Same normalized answer",
            "An exact copy — a post scraped twice, or pasted from blog to LinkedIn",
            "Training on one and testing on the other is memorization",
          ],
          [
            "Word overlap 0.8 or more",
            "Shared unique words ÷ combined unique words. The shorter text must also be at least 80% as long",
            "A light rewrite of the same post stays with the original",
          ],
        ]}
      />
      <H3>What the X cap did inside this file</H3>
      <Text>
        After chunking, blog was 14,315 rows and X was larger. Whole X posts
        were kept, in source-line order, until X reached the blog count.
        Result: 14,316 X rows kept, 6,192 left out. The last post was kept
        intact, which is why X is one row over blog. This is a size cap, not
        a quality ranking.
      </Text>
      <Table
        headers={["Medium in the 31,328-row file", "Train", "Validation", "Test"]}
        columnAlign={["left", "right", "right", "right"]}
        rows={[
          ["Blog", "11,440", "1,440", "1,435"],
          ["X", "11,511", "1,401", "1,404"],
          ["LinkedIn", "755", "121", "123"],
          ["YouTube, Jason", "1,356", "171", "171"],
          ["YouTube, SaaStr", "0", "0", "0"],
        ]}
      />
      <Text size="small" tone="secondary">
        Source: EXP-20261001-008 split report, and the HappyFox rows in
        EXP-20261003-004. Relabel later dropped broken train rows and left
        21,377 / 2,588 / 2,580. Rows were not moved between splits. The scored
        index has 21,193 train documents: 21,377 train rows, minus 183
        gold-overlap families, minus train_26148.
      </Text>
    </Stack>
  );
}

function Analytics() {
  return (
    <Stack gap={16}>
      <Text>
        These figures describe the 21,193-document train-only index, collection
        lemkin_train_only. They are not the live PersonaRAG chunk count.
      </Text>
      <Row gap={16} wrap>
        <Stat value="41" label="Median words per document" />
        <Stat value="37%" label="Continuation slices" tone="warning" />
        <Stat value="24%" label="Under 20 words" tone="warning" />
        <Stat value="0" label="Empty retrievals on 30 topics" tone="success" />
      </Row>
      <Grid columns={2} gap={16}>
        <Stack gap={8}>
          <H3>What is in the index</H3>
          <Table
            headers={["Medium", "Documents", "Share"]}
            columnAlign={["left", "right", "right"]}
            rows={[
              ["X", "11,185", "52.8%"],
              ["Blog", "9,255", "43.7%"],
              ["LinkedIn", "713", "3.4%"],
              ["YouTube", "40", "0.2%"],
            ]}
          />
        </Stack>
        <Stack gap={8}>
          <H3>What the top 4 actually returned</H3>
          <Table
            headers={["Medium", "Top-4 slots", "Of 120 slots"]}
            columnAlign={["left", "right", "right"]}
            rows={[
              ["Blog", "69", "57.5%"],
              ["X", "39", "32.5%"],
              ["LinkedIn", "12", "10%"],
              ["YouTube", "0", "0%"],
            ]}
          />
          <Text size="small" tone="secondary">
            30 gold topics, k=4, topic string only. X is most of the index
            and is not most of the hits. 11 of 30 queries returned two chunks
            of the same post. Mean cosine distance 0.243.
          </Text>
        </Stack>
      </Grid>
      <H3>Five index facts that change how generation works</H3>
      <Table
        headers={["Fact", "Count", "Share of 21,193"]}
        columnAlign={["left", "right", "right"]}
        rows={[
          ["Under 20 words", "5,141", "24.3%"],
          ["Under 40 words", "10,427", "49.2%"],
          ["Continuation of a longer post", "7,895", "37.3%"],
          ["Shares a post with another chunk", "11,137", "52.6%"],
          ["No sentence-ending punctuation", "7,036", "33.2%"],
        ]}
        rowTone={["warning", "warning", "warning", "info", "warning"]}
      />
      <Text size="small" tone="secondary">
        Word counts: 10th percentile 15, median 41, 90th percentile 263.
        Largest post was cut into 52 chunks. Source: EXP-20261004-009.
      </Text>
      <Quote
        label='Real top hit for “Monday.com at $550M ARR” — a numbered fragment, not the whole post'
        body="#7. Monday.com is growing 68% (!) at $550m in ARR."
      />
      <Text size="small" tone="secondary">
        Relevance grades do not exist, so precision, recall, MRR, and nDCG
        are not reported.
      </Text>
    </Stack>
  );
}

function Flags() {
  return (
    <Stack gap={16}>
      <Text>
        Saved drafts from the scored runs.
      </Text>
      <Table
        headers={["Flag", "Where it showed up", "What changed"]}
        rows={[
          [
            "Cut off inside a word or a sentence",
            "Clone v6, clone v5, rank 32, repaired",
            "Relabel. Mid-sentence fell from 62.5% to 32.2%",
          ],
          [
            "Answer is only the title",
            "Relabel, then the RAG file",
            "Still there. Gates miss a finished 16-word title",
          ],
          [
            "Base model writes a study guide",
            "Untrained Llama 3.1, 119 of 120 drafts",
            "Stopped using that rubric to pick the adapter",
          ],
          [
            "RAG copies a training post",
            "2 of 120 final RAG drafts",
            "12-word overlap retries once. Both stayed flagged",
          ],
          [
            "RAG invents a number",
            "60 of 120 final RAG drafts",
            "Reported, not rewritten. Weaker LoRA was not deployed",
          ],
        ]}
        rowTone={["success", "danger", "info", "warning", "warning"]}
      />

      <H3>1. The draft stops inside a word</H3>
      <Grid columns={2} gap={16}>
        <Quote
          label="Clone v6, X, gold_008 — stops at the letters Pal"
          body={'The top Cloud and SaaS companies are still trading at insane multiples.\n\nYou can see it all over Twitter: “Pal'}
        />
        <Quote
          label="Clone v5, LinkedIn, gold_001 — stops at the letter C"
          body="If you are selling into a CIO or C"
        />
      </Grid>
      <Quote
        label="Rank-32 blog, gold_001 — the budget ends mid-sentence. 24 of these 30 blogs do the same"
        body="You have to be able to do it all the time. Not just when it’s convenient"
      />
      <Quote
        label="Repaired adapter, talk, gold_003 — a YouTube sign-off learned from captions"
        body="thanks everybody and uh be great see everyone later yeah bye bye [Music] you"
      />
      <Text size="small" tone="secondary">
        The prompt said “write the whole post” while the target was a fragment
        ending in end-of-text. Relabel kept the target text and changed the
        instruction to an opening or a continuation, and dropped mid-word cuts,
        other mid-sentence cuts, unfinished last slices, and 190 transcript
        fragments. Three-seed means, retrieval off: repaired mid-sentence
        62.5%, relabel 32.2%. The relabel file of 120 answers has no “[Music]”.
        LinkedIn still stops early.
      </Text>

      <H3>2. Relabel prints the title and stops</H3>
      <Quote
        label="Relabel, LinkedIn, gold_001 — the entire answer"
        body="Can an 8-Person StartUp Sell to a CIO? Yes — If You Understand The Social Contract."
      />
      <Quote
        label="Relabel, X, gold_003 — the entire answer"
        body="How to Truly Stand Out in Any Job Interview, from SDR to COO"
      />
      <Text size="small" tone="secondary">
        On the retrieval-off relabel file, 18 of 120 answers are nothing but
        the title: 6 LinkedIn and 12 X. The repaired adapter had 11, all X.
        The final RAG file has 17. This 16-word LinkedIn title passes
        completion and format.
      </Text>

      <H3>3. Untrained Llama wins the old rubric by writing a guide</H3>
      <Quote
        label="Llama 3.1, blog, gold_003 — markdown, a generic title, a first-person career story"
        body={"**The Ultimate Guide to Standing Out in Any Job Interview**\n\nAh, job interviews. The ultimate test of your skills, experience, and... charisma? As I've navigated my own career journey from early-stage CEO to venture capitalist..."}
      />
      <Quote
        label="Llama 3.1, X, gold_001 — it narrates the assignment, then stops mid-sentence"
        body={'Here\'s a possible X post:\n\n**"Can an 8-person startup sell to a CIO? YES! But only if you understand the social contract...**\n\nIt all comes down to understanding t'}
      />
      <Text size="small" tone="secondary">
        119 of 120 base answers use markdown bold or a heading. Gold-rubric
        overall: base 0.368, clone v6 0.264, repaired 0.244. BLEU stayed near
        0.004. That score rewards headings and length. The later selector uses
        blog voice, the gates, writing quality, and stance. The base model is
        not the deployed model.
      </Text>

      <H3>4. Factual RAG copies a training post</H3>
      <Quote
        label="Final RAG, blog, gold_004 — 27 words match train document train_7930"
        body="“Doubling Down” is a new series where we hear from top B2B SaaS investors on their most recent activities and takes on the current market. We had an incredible lineup for our first episode with Jay Levy (Zelkova), John Frankel (GGV) and Mary D’Onofrio (BV)."
      />
      <Quote
        label="Final RAG, blog, gold_020 — 15 words copied"
        body="Once you’ve made the decision to move on, the “Why” doesn’t matter anymore."
      />
      <Text size="small" tone="secondary">
        A copied span of 12 or more words retries generation once. gold_004
        retried and stayed flagged. Both drafts fail the copying gate, so they
        are out of the persona-utility average. The deployment path does not
        silently rewrite the draft.
      </Text>

      <H3>5. Factual RAG adds numbers the evidence does not have</H3>
      <Quote
        label="Final RAG, blog, gold_001 — both quantities are unsupported"
        body="one of SaaStr Fund’s portfolio companies that sells into enterprises, especially larger ones (think $100m+ ACV).\n\n…once we got past a few $500k-$1m deals, we needed a partner with us."
      />
      <Text size="small" tone="secondary">
        60 of 120 drafts have at least one unsupported quantity. That check is
        separate from claim support: 495 of 498 verifiable claims still matched
        a retrieved passage. Turning the adapter down to 0.25 made 3 of 4
        checked cells pass the number check, and gold_001 blog voice moved
        from 0.63 to 1.94, farther from Lemkin. No intermediate scale was
        deployed. A later style-rewrite trial restyled 5 drafts; all 5 were
        rejected for new quantities, and the full 120 was not started.
      </Text>
    </Stack>
  );
}

function Ml() {
  return (
    <Stack gap={16}>
      <Text>
        Selected model: Llama 3.1 8B Instruct, 4-bit, QLoRA rank 16, alpha 32,
        trained on the 21,377 relabeled rows. About 3.4 hours on one 16 GB GPU.
        Best eval loss 1.3316. That loss is not comparable to the repaired
        adapter’s 1.5311, because the validation prompts changed.
      </Text>
      <H3>1. The label fix changed how drafts end</H3>
      <Table
        headers={["", "Repaired adapter", "Relabel adapter"]}
        columnAlign={["left", "right", "right"]}
        rows={[
          ["Early end-of-text, mean of 3 seeds", "62.5%", "56.7%"],
          ["Mid-sentence stop, mean of 3 seeds", "62.5%", "32.2%"],
        ]}
        rowTone={[undefined, "success"]}
      />
      <Text size="small" tone="secondary">
        120 prompts each seed. Same decoding. Retrieval off. Sources:
        EXP-20261004-003, 005, and 006.
      </Text>
      <H3>2. The old gold rubric picked the untrained model</H3>
      <Table
        headers={["Model", "Gold-rubric overall"]}
        columnAlign={["left", "right"]}
        rows={[
          ["Llama 3.1, no adapter", "0.368"],
          ["Lemkin clone v6", "0.264"],
          ["Lemkin clone v5", "0.258"],
          ["Repaired QLoRA", "0.244"],
        ]}
      />
      <Text size="small" tone="secondary">
        EXP-20261001-010. The base model follows headings and length more
        often. BLEU stayed near 0.004.
      </Text>
      <H3>3. Retrieval made claims checkable and did not improve blog voice</H3>
      <Table
        headers={["On 120 prompts", "Retrieval off", "Relabel + train-only RAG"]}
        columnAlign={["left", "right", "right"]}
        rows={[
          ["Eligible after gates", "77", "72"],
          ["Completion failures", "41", "36"],
          ["Copying failures", "1", "2"],
          ["Grounding failures", "not scored", "14"],
          ["Blog voice distance (lower is closer)", "0.712", "0.777"],
          ["Writing quality, 1 to 5", "3.15", "3.06"],
          ["Persona utility, blogs that pass", "0.705", "0.688"],
          ["LinkedIn drafts eligible", "7 of 30", "9 of 30"],
        ]}
        rowTone={[
          undefined,
          "success",
          undefined,
          "warning",
          "warning",
          undefined,
          "info",
          "danger",
        ]}
      />
      <Quote
        label="Claim support on the RAG drafts"
        body="495 of 498 verifiable claims matched the retrieved passage (99.4%). Fourteen drafts failed the grounding gate. Sixty drafts contain a number the evidence does not support."
      />
    </Stack>
  );
}

export default function Demo1RowExamples() {
  const [tab, setTab] = useCanvasState<Tab>("tab", "clean");
  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>Demo 1 examples</H1>
        <Text tone="secondary">
          Rows from the scored files, October 2026.
        </Text>
      </Stack>
      <Row gap={8} wrap>
        {TABS.map((item) => (
          <Pill
            key={item.id}
            active={tab === item.id}
            onClick={() => setTab(item.id)}
          >
            {item.label}
          </Pill>
        ))}
      </Row>
      <Divider />
      {tab === "clean" && (
        <Stack gap={8}>
          <H2>Preprocessing — what was removed, and one label that changed</H2>
          <Clean />
        </Stack>
      )}
      {tab === "split" && (
        <Stack gap={8}>
          <H2>Train and test — one post stays in one split</H2>
          <Split />
        </Stack>
      )}
      {tab === "analytics" && (
        <Stack gap={8}>
          <H2>Analytics — the index is mostly short pieces</H2>
          <Analytics />
        </Stack>
      )}
      {tab === "ml" && (
        <Stack gap={8}>
          <H2>Machine learning — cutoff improved, voice did not</H2>
          <Ml />
        </Stack>
      )}
      {tab === "flags" && (
        <Stack gap={8}>
          <H2>Generation flags — the draft, the cause, and what we did</H2>
          <Flags />
        </Stack>
      )}
    </Stack>
  );
}
