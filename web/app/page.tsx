import Link from "next/link";
import { Card, buttonClass } from "@/components/ui";

const FEATURES = [
  {
    title: "RAG-powered retrieval",
    body: "Every draft starts from real passages pulled from Jason Lemkin's blog, LinkedIn, X and YouTube writing.",
  },
  {
    title: "Persona fine-tuning",
    body: "A QLoRA-tuned Llama 3.1 8B model learns his tone, rhythm and habits of argument.",
  },
  {
    title: "Multi-format generation",
    body: "LinkedIn posts, blog articles, X threads and YouTube scripts from one content brief.",
  },
  {
    title: "Source-grounded outputs",
    body: "See which passages were retrieved and how closely each one matched your brief.",
  },
];

export default function HomePage() {
  return (
    <div className="space-y-12">
      <section className="rounded-3xl border border-line bg-linear-to-br from-card via-card to-primary/10 px-6 py-14 text-center sm:px-12">
        <p className="text-sm font-medium text-primary">Lemkin Studio</p>
        <h1 className="mx-auto mt-3 max-w-3xl text-4xl font-semibold tracking-tight sm:text-5xl">
          Create content in the voice of a SaaS expert
        </h1>
        <p className="mx-auto mt-4 max-w-2xl text-base text-mute">
          Retrieve relevant content from Jason Lemkin&apos;s knowledge base and generate LinkedIn
          posts, blogs, X threads and YouTube scripts using RAG and QLoRA fine-tuning.
        </p>
        <div className="mt-8 flex flex-wrap justify-center gap-3">
          <Link href="/generate" className={buttonClass("primary", "px-6 py-3 text-base")}>
            Start Creating
          </Link>
          <Link href="/about" className={buttonClass("ghost", "px-6 py-3 text-base")}>
            How it works
          </Link>
        </div>
      </section>

      <section aria-label="Features" className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {FEATURES.map((f) => (
          <Card key={f.title}>
            <h2 className="font-semibold">{f.title}</h2>
            <p className="mt-2 text-sm text-mute">{f.body}</p>
          </Card>
        ))}
      </section>
    </div>
  );
}
