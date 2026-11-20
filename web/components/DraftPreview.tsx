import type { FormatId } from "@/lib/types";

/**
 * Shows a finished draft the way its format is usually read: a post card, a thread, an article or a script.
 * This is a layout preview only. It uses a neutral "your name" placeholder, not anyone's real profile.
 */
export default function DraftPreview({ format, content }: { format: FormatId; content: string }) {
  const blocks = content
    .split(/\n\s*\n/)
    .map((block) => block.trim())
    .filter(Boolean);

  if (format === "linkedin_post") {
    return (
      <div className="mx-auto max-w-xl rounded-2xl border border-line bg-card p-5 shadow-soft">
        <div className="flex items-center gap-3">
          <span className="grid h-11 w-11 place-items-center rounded-full bg-card-2 text-sm font-semibold text-mute">
            You
          </span>
          <div>
            <p className="text-sm font-semibold">Your name</p>
            <p className="text-xs text-mute">Preview of a LinkedIn-style post</p>
          </div>
        </div>
        <div className="mt-4 space-y-3 text-[15px] leading-relaxed">
          {blocks.map((block, i) => (
            <p key={i} className="whitespace-pre-wrap">
              {block}
            </p>
          ))}
        </div>
      </div>
    );
  }

  if (format === "x_thread") {
    return (
      <ol className="mx-auto max-w-xl space-y-3">
        {blocks.map((block, i) => (
          <li key={i} className="flex gap-3 rounded-2xl border border-line bg-card p-4 shadow-soft">
            <span className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-card-2 text-xs font-semibold text-mute">
              {i + 1}
            </span>
            <p className="whitespace-pre-wrap text-[15px] leading-relaxed">{block}</p>
          </li>
        ))}
      </ol>
    );
  }

  if (format === "blog_draft") {
    const [first, ...rest] = blocks;
    const isTitle = first?.startsWith("#");
    return (
      <article className="mx-auto max-w-2xl rounded-2xl border border-line bg-card p-8 shadow-soft">
        {isTitle ? (
          <h2 className="font-display text-4xl leading-tight">{first.replace(/^#+\s*/, "")}</h2>
        ) : null}
        <div className="mt-5 space-y-4 text-[16px] leading-[1.8]">
          {(isTitle ? rest : blocks).map((block, i) =>
            block.startsWith("##") ? (
              <h3 key={i} className="pt-2 font-sans text-sm font-semibold uppercase tracking-wider text-primary">
                {block.replace(/^#+\s*/, "")}
              </h3>
            ) : (
              <p key={i} className="whitespace-pre-wrap">
                {block}
              </p>
            ),
          )}
        </div>
      </article>
    );
  }

  // YouTube script: show each beat (HOOK, INTRO, OUTRO...) with its label.
  return (
    <ol className="mx-auto max-w-2xl space-y-3">
      {blocks.map((block, i) => {
        const match = block.match(/^([A-Z][A-Z ]{1,14}):\s*([\s\S]*)$/);
        return (
          <li key={i} className="flex gap-4 rounded-2xl border border-line bg-card p-4 shadow-soft">
            <span className="w-16 shrink-0 pt-0.5 text-xs font-semibold uppercase tracking-wider text-primary">
              {match ? match[1] : `Beat ${i + 1}`}
            </span>
            <p className="whitespace-pre-wrap text-[15px] leading-relaxed">{match ? match[2] : block}</p>
          </li>
        );
      })}
    </ol>
  );
}
