import { Card, PageHeader } from "@/components/ui";

const SECTIONS = [
  {
    title: "Objective",
    body: "Write LinkedIn posts, blog drafts, X threads and YouTube scripts in the voice of Jason Lemkin (SaaStr), grounded in his real writing.",
  },
  {
    title: "RAG architecture",
    body: "A brief is embedded with nomic-embed-text, the closest passages are pulled from Chroma, and they are placed in the prompt so the model writes from real material.",
  },
  {
    title: "QLoRA fine-tuning",
    body: "Llama 3.1 8B is tuned with a small QLoRA adapter on instruction and response pairs built from the corpus, merged, converted to GGUF and served in Ollama as lemkin-clone.",
  },
  {
    title: "Dataset sources",
    body: "X posts, SaaStr blog posts, LinkedIn posts and YouTube transcripts from two channels, about 32,800 raw rows and about 31,400 documents after filtering.",
  },
  {
    title: "Pipeline",
    body: "An Airflow DAG loads the files into MinIO, Spark cleans, chunks and embeds them, and the chunks are loaded into Chroma. A separate export feeds fine-tuning.",
  },
  {
    title: "Limitations",
    body: "Draft quality has not been fully measured yet. Faithfulness scores are low in the reported run, and the four-way model comparison is still pending.",
  },
  {
    title: "Future improvements",
    body: "Complete the model comparison, add streaming output, add authentication before any cloud deployment, and expose health checks for every service.",
  },
];

const TEAM = ["Vrushabh Bodarya", "Pukhraj Rathkanthiwar", "Kevin Gao", "Andreah Cruz"];

export default function AboutPage() {
  return (
    <>
      <PageHeader
        title="About the project"
        subtitle="DATA 298B, Group 2. AI-powered content generation inspired by Jason Lemkin's writing style."
      />
      <div className="grid gap-4 md:grid-cols-2">
        {SECTIONS.map((s) => (
          <Card key={s.title}>
            <h2 className="font-semibold">{s.title}</h2>
            <p className="mt-2 text-sm leading-relaxed text-mute">{s.body}</p>
          </Card>
        ))}
        <Card>
          <h2 className="font-semibold">Team</h2>
          <ul className="mt-2 space-y-1 text-sm text-mute">
            {TEAM.map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
        </Card>
      </div>
    </>
  );
}
