import {
  Callout,
  Divider,
  H1,
  H3,
  Row,
  Select,
  Stack,
  Stat,
  Text,
  useCanvasState,
  useHostTheme,
  useMemo,
} from "cursor/canvas";

type Medium = "blog" | "talk" | "linkedin" | "x";

type Draft = {
  topic: string;
  medium: Medium;
  prompt: string;
  answer: string;
  words: number;
  stop: string;
  budget: number;
  penalty: number;
  copied: boolean;
};

const MEDIUM_LABEL: Record<Medium, string> = {
  blog: "Blog",
  talk: "Talk",
  linkedin: "LinkedIn",
  x: "X",
};

const ORDER: Medium[] = ["blog", "talk", "linkedin", "x"];

const DRAFTS: Draft[] = [{"topic": "Raising prices without losing customers", "medium": "blog", "prompt": "Write a blog post in the style of Jason Lemkin about Raising prices without losing customers. Write the post itself in about 250 words, with no title line.", "answer": "So I’ve been doing this SaaS thing for a while now, and one thing I’ve learned is that you can raise prices on your existing customers — if you do it right. And importantly, without them leaving. But how? Well, first of all, don’t raise prices on your existing customers unless you have to. It’s just too risky. You’ll lose some of them. Second, make sure the price increase is fair and justified. If you are raising prices 50% a year, that’s not fair or justified. But if you’ve added a ton of value — more features, better performance, etc. — then it can be OK to raise prices by say 10%-20%. More than that, and the customers will start to feel like they’re being taken advantage of. And third, make sure your customers know about the price increase well in advance. Don’t spring it on them. That’s just too risky. If you do all this right, most of your existing customers won’t leave. But some will. So what can you do? Well, first, don’t raise prices on your smallest customers. They are the ones that are least likely to churn if you raise prices. Second, make sure they know about it well in advance. Third, make sure they have a clear path out — if they want one. And fourth, make sure they know you’ll do better next time. If you do all this right, most of your existing customers won’t leave when you raise prices on them. But some will. So don’t get too worried about it.", "words": 259, "stop": "stop", "budget": 500, "penalty": 1.1, "copied": false}, {"topic": "Raising prices without losing customers", "medium": "talk", "prompt": "Write a spoken talk in the style of Jason Lemkin about Raising prices without losing customers. Use short spoken sentences to founders, about 160 words, with no title and no headings.", "answer": "So I've been doing this for a while now, investing in SaaS companies, helping them scale, and one thing that always comes up is how do you raise your prices? How do you get more money out of the customers without losing them? And it's tough because if you raise your price too much, they'll leave. But if you don't raise your price enough, you're leaving money on the table. So what can you do about it? Well first, let me tell you a little bit about how I think about this. When we were at Adobe Sign, which is now known as EchoSign, we had 50 customers when we started and then we grew to 500 and then eventually we got acquired by Adobe for $2 billion. And one of the things that was really important to me was making sure our customers loved us. We didn't want them to leave. So I always thought about it as how do you raise your prices without losing any customers? Because if they love you, they'll pay more. They'll buy more products from you and they won't leave. And that's what we did at Adobe Sign.", "words": 195, "stop": "stop", "budget": 400, "penalty": 1.15, "copied": false}, {"topic": "Raising prices without losing customers", "medium": "linkedin", "prompt": "About 80 words. The first line is a claim, not a headline. No hashtags. Write a LinkedIn post in the style of Jason Lemkin about Raising prices without losing customers.", "answer": "About 80% of SaaS companies raise prices at least once a year. But how do you do it without losing any customers? A few thoughts: - If your product is truly better, they won't care. They'll pay more. - If your product is truly worse, they won't care. They'll churn anyway. - If your product is the same but costs 20%+ more ... they will care. And you may lose them. - But if your product is the same but costs 10%-15% more ... they may not care. At least, not that much. - You can also raise prices on new customers and not existing ones. That's a subtle way to do it without angering anyone.", "words": 116, "stop": "stop", "budget": 320, "penalty": 1.1, "copied": false}, {"topic": "Raising prices without losing customers", "medium": "x", "prompt": "Write one X post in the style of Jason Lemkin. Exactly 3 sentences, under 60 words, no headline, and no hashtags. Topic: Raising prices without losing customers", "answer": "Raising prices without losing customers is a delicate dance. You have to do it right, or they'll leave. But done right, you can raise prices 10%-20% a year for years.", "words": 30, "stop": "stop", "budget": 120, "penalty": 1.1, "copied": false}, {"topic": "Hiring the first VP of Sales", "medium": "blog", "prompt": "Write a blog post in the style of Jason Lemkin about Hiring the first VP of Sales. Write the post itself in about 250 words, with no title line.", "answer": "So you’ve finally hired your first VP of Sales. And it’s time to get going. But … how? What do you need to do for him or her to succeed?\nWell, here are a few thoughts:\nFirst, make sure they have enough leads and opportunities\n. This is the #1 job of marketing. If they don’t have enough leads, no VP of Sales can hit their number. So if you haven’t hired a CMO yet — do that first. And second, make sure they have enough resources to hit the plan. A great VP of Sales will always want more people, and more budget. But at least make sure they have enough to hit the plan. If not, it’s your fault for hiring them.\nSecond, make sure they are aligned with the company goals\n. This is important. You need to be on the same page as your first VP of Sales. So if you haven’t done a formal business plan yet — do that now. And make sure everyone is aligned around the goals and metrics. A great VP of Sales will always want more, but at least make sure they are aligned with the company’s overall goals.\nThird, make sure they have enough support\n. This is important. Your first VP of Sales needs to be able to hit their number on their own. But they also need help from others in the company. So if you haven’t hired a VP of Marketing yet — do that now. And second, make sure your VP of Sales has enough support from engineering and product. They will need it.\nFourth, make sure they have enough time\n. This is important. Your first VP of Sales needs to be able to hit their number on their own. But they also need help from others in the company.", "words": 306, "stop": "stop", "budget": 500, "penalty": 1.1, "copied": false}, {"topic": "Hiring the first VP of Sales", "medium": "talk", "prompt": "Write a spoken talk in the style of Jason Lemkin about Hiring the first VP of Sales. Use short spoken sentences to founders, about 160 words, with no title and no headings.", "answer": "So I've been investing for almost two decades now, and one thing that's changed is how much more competitive it has become to hire a great VP of Sales. And this is true at every stage from seed through IPO. But the earlier you are in your journey as an entrepreneur, the less likely you are to have heard about all these different types of VPs of sales. So let me just give you a quick rundown of what I've learned over time. The first VP of Sales that most founders hire is often called the head of sales or the director of sales. And this person's job is really to help you scale from one, two, maybe three reps up to 10, 15, 20. So they're going to be a bit of an owner and a manager at the same time. They'll own all the revenue for that period of time, but they won't have any direct reports yet. And then as you get bigger, you start to hire more VPs of sales. You might hire one VP of Sales who owns everything from $1 million in ARR up to $10 million or even $20 million. Or maybe you'll split it into two different roles. One person will own the first five years of your journey and then another person comes in once you've crossed that threshold, but they're both VPs of sales. So let me just give you a quick rundown of what I've learned over time about how to hire these folks.", "words": 255, "stop": "stop", "budget": 400, "penalty": 1.15, "copied": false}, {"topic": "Hiring the first VP of Sales", "medium": "linkedin", "prompt": "About 80 words. The first line is a claim, not a headline. No hashtags. Write a LinkedIn post in the style of Jason Lemkin about Hiring the first VP of Sales.", "answer": "The #1 mistake I see founders make when they hire their first VP of Sales is hiring someone that doesn't really want to sell themselves anymore. They've done it before, and they are tired of it. And they just want to manage a team of reps instead. That's not what you need at the start. You need someone that still wants to close deals themselves.", "words": 65, "stop": "stop", "budget": 180, "penalty": 1.1, "copied": false}, {"topic": "Hiring the first VP of Sales", "medium": "x", "prompt": "Write one X post in the style of Jason Lemkin. Exactly 3 sentences, under 60 words, no headline, and no hashtags. Topic: Hiring the first VP of Sales", "answer": "Hiring your first VP of Sales is a big deal. It's a sign you are ready to scale. But it can also be risky if they don't really know how to sell themselves.", "words": 33, "stop": "stop", "budget": 120, "penalty": 1.1, "copied": false}, {"topic": "Net revenue retention under 100 percent", "medium": "blog", "prompt": "Write a blog post in the style of Jason Lemkin about Net revenue retention under 100 percent. Write the post itself in about 250 words, with no title line.", "answer": "So I’ve been doing this SaaS thing for a while now, and one thing that’s become clear is that you can’t hide from your NRR. It’s just too important to ignore. And it’s also a leading indicator of how well you are going to do in the long run. Because if your NRR is low, it’s hard to get it up later. So what happens when your NRR is less than 100%? Well, first, it’s not uncommon for new customers to churn at a higher rate than existing ones. That’s natural. But that doesn’t mean you can’t do better. If your NRR is < 100%, then you are losing money on the customers you already have. And that’s just a recipe for disaster. Because if you lose money on every customer, and they churn at a higher rate than new ones … well, it’s hard to see how you make any money at all. So what do you do? Well, first, don’t hide from your NRR. It’s not uncommon for folks to hide from their NRR when it’s less than 100%. But that doesn’t help. In fact, it makes things worse. Because if you hide from your NRR, then you can’t fix the problems underpinning it. So what do you do? Well, first, make sure you have a real CSM program in place. Not just a few folks doing upsells and renewals. But a true customer success team that is there to help every single customer succeed. Because if they don’t, then they’ll churn. And second, make sure your NRR is measured monthly. Not quarterly or annually.", "words": 267, "stop": "stop", "budget": 500, "penalty": 1.1, "copied": false}, {"topic": "Net revenue retention under 100 percent", "medium": "talk", "prompt": "Write a spoken talk in the style of Jason Lemkin about Net revenue retention under 100 percent. Use short spoken sentences to founders, about 160 words, with no title and no headings.", "answer": "So I've been investing for almost two decades now, and one thing that's changed is we're all more aware of churn than ever before. And it's a good thing because you can't hide from it anymore. You can't just say, \"Oh, my customers are happy.\" No, no, they're not. They leave. And if your net revenue retention is under 100 percent, that means the average customer leaves and takes their money with them. So I want to talk about what happens when you have a low NRR. What's it like? Well, first of all, it's hard to scale because every time you add a new customer, they're leaving. And so your growth is just decelerating over time. It's not accelerating. You can't get that many customers in the door if they leave and take their money with them. So I want to talk about what happens when NRR is under 100 percent. What does it feel like? Well, first of all, you're constantly worried about your existing customer base because every time a new one comes in, another one leaves. And so that's just constant stress and anxiety. You can't relax for a second. It's hard to hire sales reps when they know the customers are leaving. They don't want to work there. So it's really tough.", "words": 217, "stop": "stop", "budget": 400, "penalty": 1.15, "copied": false}, {"topic": "Net revenue retention under 100 percent", "medium": "linkedin", "prompt": "About 80 words. The first line is a claim, not a headline. No hashtags. Write a LinkedIn post in the style of Jason Lemkin about Net revenue retention under 100 percent.", "answer": "About 80% of SaaS companies have net negative churn / NRR < 100%. So you are probably doing worse than this. What can you do to improve it?", "words": 28, "stop": "stop", "budget": 180, "penalty": 1.1, "copied": false}, {"topic": "Net revenue retention under 100 percent", "medium": "x", "prompt": "Write one X post in the style of Jason Lemkin. Exactly 3 sentences, under 60 words, no headline, and no hashtags. Topic: Net revenue retention under 100 percent", "answer": "Net revenue retention under 100 percent is a sign of churn. It's not a death blow, but it's a sign you have to do better here. You can't grow at scale if your existing customers don't buy more from you.", "words": 40, "stop": "stop", "budget": 120, "penalty": 1.1, "copied": false}, {"topic": "When a SaaS startup should raise a Series A", "medium": "blog", "prompt": "Write a blog post in the style of Jason Lemkin about When a SaaS startup should raise a Series A. Write the post itself in about 250 words, with no title line.", "answer": "So when is it time to raise your first VC round? The answer isn’t always clear. But here’s my simple test:\nIf you have at least $1m in ARR growing at least 100% a year, and you can show that growth will continue for the next 12 months … then it’s probably time to raise a Series A.\nWhy? Because if you are growing 100% or more a year, with $1m+ in ARR, you have a real shot at an exit. And VCs know this. They know they can make money on a deal that is growing 100% or faster. Even if it’s not yet profitable. So if you’ve crossed $1m in ARR and are growing 100% or more … then it’s time to raise your Series A.\nBut, but, but …\nIf you don’t have at least $1m in ARR, and/or aren’t growing at least 100% a year, then it’s probably too early. You need to get there first. And if you are growing slower than that, then you may not be ready for a Series A yet either.\nWhy? Because VCs know they can make money on a deal that is growing 100% or faster. But they don’t know how to make money on one that isn’t. So it’s too early in many cases if you aren’t there yet. And the VC will likely lose money on the deal, which is why they won’t do it.\nSo if you have at least $1m in ARR growing 100% or more a year … then it’s time to raise your Series A. But if you don’t have that yet, then wait until you get there first. It will be worth it.", "words": 281, "stop": "stop", "budget": 500, "penalty": 1.1, "copied": false}, {"topic": "When a SaaS startup should raise a Series A", "medium": "talk", "prompt": "Write a spoken talk in the style of Jason Lemkin about When a SaaS startup should raise a Series A. Use short spoken sentences to founders, about 160 words, with no title and no headings.", "answer": "So I've been investing for almost two decades now, and one thing that's changed is how much more capital there is in the market today than when we started.\nAnd so it used to be, you'd raise a seed round, maybe $2 million or $3 million, and then you'd go out and try to raise your Series A. And if you didn't get funded by say, 18 months after that, you were probably in trouble because the next round would just take too long.\nBut today, there's so much more capital available for startups than ever before. So I think it's a good time to update some of our classic SaaStr learnings on when exactly should you raise your Series A?\nSo first off, let me tell you what doesn't count as a Series A. It does not count if the money is going into sales and marketing.\nIt does not count if the money is going into engineering or product development. And it definitely does not count if the money is going to go towards burn rate or runway. That's just too expensive, right? So what does count as a Series A?\nWell, first off, you need to have at least some early traction in your market.\nYou can't raise a Series A without any customers and no revenue. You're not ready for that round of capital yet. But if you do have some early traction, then the next thing is you need to be able to show that you are going to scale quickly with this new money.", "words": 260, "stop": "stop", "budget": 400, "penalty": 1.15, "copied": false}, {"topic": "When a SaaS startup should raise a Series A", "medium": "linkedin", "prompt": "About 80 words. The first line is a claim, not a headline. No hashtags. Write a LinkedIn post in the style of Jason Lemkin about When a SaaS startup should raise a Series A.", "answer": "When a SaaS startup should raise a Series A. It’s simple: when you have at least 2 good months of bookings in a row, and you are growing at least 50% a month. And ideally, more than that. If you don’t have this, it’s too early to raise a Series A.", "words": 51, "stop": "stop", "budget": 180, "penalty": 1.1, "copied": false}, {"topic": "When a SaaS startup should raise a Series A", "medium": "x", "prompt": "Write one X post in the style of Jason Lemkin. Exactly 3 sentences, under 60 words, no headline, and no hashtags. Topic: When a SaaS startup should raise a Series A", "answer": "When a SaaS startup should raise a Series A\n\nIt’s when you have something that is working well, but it’s not enough.\n\nYou need more capital to scale faster.", "words": 29, "stop": "stop", "budget": 120, "penalty": 1.1, "copied": false}, {"topic": "Competing with a free AI feature", "medium": "blog", "prompt": "Write a blog post in the style of Jason Lemkin about Competing with a free AI feature. Write the post itself in about 250 words, with no title line.", "answer": "So I’ve been doing this for a while now, and one thing is clear: if you are going to compete with an AI feature that is Free … you have to be at least as good or better than ChatGPT. Or even better. Because the vast majority of folks will just use ChatGPT. It’s free, it’s easy, and it works well enough for most simple tasks. So if you are going to compete with a Free AI feature, make sure your AI is at least as good as ChatGPT. And ideally, better. I’ve seen this play out in real time across the SaaStr ecosystem. Some founders have been hesitant to add an “AI” feature because they’re worried about competing with ChatGPT or other free tools. But here’s the thing: if you can build something that’s genuinely better than what’s available for free, people will pay for it. They’ll pay for security, reliability, integration, and most importantly — results. Because at the end of the day, folks don’t care about AI per se. They just want to get their job done faster and easier. And if your product can do that better than ChatGPT or any other free tool, they’ll pay top dollar for it. So don’t let fear hold you back from adding an “AI” feature. Instead, focus on building something that’s genuinely better than what’s available for free. Because in the end, that’s all that matters.", "words": 237, "stop": "stop", "budget": 500, "penalty": 1.1, "copied": false}, {"topic": "Competing with a free AI feature", "medium": "talk", "prompt": "Write a spoken talk in the style of Jason Lemkin about Competing with a free AI feature. Use short spoken sentences to founders, about 160 words, with no title and no headings.", "answer": "So I've been doing this for a while now, but it's only gotten more intense over the past year or so.\nAnd one thing that has become super common is when you're in market, especially if you have a lot of customers, someone will just add an AI feature to your competitor and then they'll undercut you on price.", "words": 59, "stop": "stop", "budget": 400, "penalty": 1.15, "copied": false}, {"topic": "Competing with a free AI feature", "medium": "linkedin", "prompt": "About 80 words. The first line is a claim, not a headline. No hashtags. Write a LinkedIn post in the style of Jason Lemkin about Competing with a free AI feature.", "answer": "The #1 thing I see founders get wrong today is competing with a Free AI Feature. It's just too hard to win when you are charging $50-$100 a month or more, and your competitor has no costs at all. And it's not even close. You can't compete on price here.", "words": 50, "stop": "stop", "budget": 180, "penalty": 1.1, "copied": false}, {"topic": "Competing with a free AI feature", "medium": "x", "prompt": "Write one X post in the style of Jason Lemkin. Exactly 3 sentences, under 60 words, no headline, and no hashtags. Topic: Competing with a free AI feature", "answer": "If you can't beat an AI feature that is Free ... then maybe it's not worth doing at all.  It's just too much work to compete with something that costs nothing.", "words": 31, "stop": "stop", "budget": 120, "penalty": 1.1, "copied": false}];

export default function MediumGrid() {
  const theme = useHostTheme();
  const [medium, setMedium] = useCanvasState("medium", "all");
  const [topic, setTopic] = useCanvasState("topic", "all");
  const topics = useMemo(() => {
    const seen: string[] = [];
    for (const row of DRAFTS) {
      if (!seen.includes(row.topic)) seen.push(row.topic);
    }
    return seen;
  }, []);
  const shown = useMemo(() => {
    return DRAFTS.filter((row) => {
      if (medium !== "all" && row.medium !== medium) return false;
      if (topic !== "all" && row.topic !== topic) return false;
      return true;
    }).sort((a, b) => {
      const topicOrder = topics.indexOf(a.topic) - topics.indexOf(b.topic);
      if (topicOrder !== 0) return topicOrder;
      return ORDER.indexOf(a.medium) - ORDER.indexOf(b.medium);
    });
  }, [medium, topic, topics]);
  const words = shown.reduce((sum, row) => sum + row.words, 0);
  const copied = shown.filter((row) => row.copied).length;
  const mean = shown.length ? Math.round(words / shown.length) : 0;

  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>lemkin-relabel, five topics</H1>
        <Text tone="secondary">
          One template per medium. Only the topic changes.           Twenty separate chats
          on lemkin-relabel, temperature 0, seed 42, 5 October 2026.
        </Text>
      </Stack>
      <Text size="small" tone="tertiary">
        API http://127.0.0.1:11434/api/chat. top_p 0.9. num_ctx 4096.
        Repeat penalty 1.1, except talk 1.15. Token budget: blog 500, talk 400,
        LinkedIn 180 to 320, X 120. LinkedIn and X put the format rules before the topic.
      </Text>
      <Row gap={12} wrap align="end">
        <Stack gap={4} style={{ minWidth: 280, flex: 1 }}>
          <Text size="small" tone="tertiary">Topic</Text>
          <Select
            value={topic}
            onChange={setTopic}
            options={[
              { value: "all", label: "All 5 topics" },
              ...topics.map((item) => ({ value: item, label: item })),
            ]}
          />
        </Stack>
        <Stack gap={4} style={{ minWidth: 180 }}>
          <Text size="small" tone="tertiary">Medium</Text>
          <Select
            value={medium}
            onChange={setMedium}
            options={[
              { value: "all", label: "All mediums" },
              { value: "blog", label: "Blog" },
              { value: "talk", label: "Talk" },
              { value: "linkedin", label: "LinkedIn" },
              { value: "x", label: "X" },
            ]}
          />
        </Stack>
      </Row>
      <Row gap={16} wrap>
        <Stat value={String(shown.length)} label="Drafts in view" />
        <Stat value={String(mean)} label="Mean words" />
        <Stat value={String(copied)} label="Copied the format line" tone={copied > 0 ? "warning" : "success"} />
      </Row>
      <Divider />
      {shown.map((row) => (
        <Stack key={`${row.topic}-${row.medium}`} gap={8}>
          <Row gap={8} align="center" wrap>
            <Text weight="semibold">{MEDIUM_LABEL[row.medium]}</Text>
            <Text size="small" tone="tertiary">
              {`${row.words} words · ${row.stop} · budget ${row.budget} · penalty ${row.penalty}`}
            </Text>
          </Row>
          <H3>{row.topic}</H3>
          <Text size="small" tone="tertiary">{row.prompt}</Text>
          {row.copied ? (
            <Callout tone="warning">
              This draft repeats the format sentence instead of only writing the post.
            </Callout>
          ) : null}
          <div style={{ background: theme.bg.elevated, padding: 16 }}>
            <Text style={{ whiteSpace: "pre-wrap", lineHeight: 1.6 }}>{row.answer}</Text>
          </div>
          <Divider />
        </Stack>
      ))}
    </Stack>
  );
}
