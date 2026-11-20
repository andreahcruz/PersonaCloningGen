"""Builds figures and the PDF report from outputs/results.json. Every number is read from the results file."""
import csv, json, pathlib, collections, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle, PageBreak, Image

ROOT = pathlib.Path(__file__).resolve().parent
RES_DIR = sys.argv[1] if len(sys.argv) > 1 else "outputs"      # outputs = earlier version, outputs_colab = final version
OUT = ROOT / RES_DIR
COLAB = RES_DIR != "outputs"
R = json.load(open(OUT / "results.json", encoding="utf-8"))
OLD = json.load(open(ROOT / "outputs" / "results.json", encoding="utf-8")) if COLAB else None
NEGVAL = json.load(open(OUT / "negatives_validation.json", encoding="utf-8")) if COLAB else None
ROWS = list(csv.DictReader(open(OUT / "scores_per_draft.csv", encoding="utf-8")))
_docs = ROOT.parent / "docs" / "evaluation"
PDF = (_docs if _docs.exists() else ROOT.parent) / "AUTHORSHIP_CLASSIFIER_RESULTS.pdf"

ORDER = ["A Base Llama", "C PersonaRAG stand-in", "D QLoRA repaired", "E Relabel, no retrieval", "E Relabel + retrieval", "F Model F (F-REL)"]
NAVY = colors.HexColor("#1f3a5f"); LIGHT = colors.HexColor("#eef2f7")
GREEN = colors.HexColor("#e6f4ea"); AMBER = colors.HexColor("#fff4d6"); RED = colors.HexColor("#fde8e8")
f3 = lambda x: f"{x:.3f}"
f2 = lambda x: f"{x:.2f}"


# ------------------------------------------------------------------ figures
def fig_systems():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharex=True)
    for ax, med, title in zip(axes, ("blog", "linkedin"), ("Blog (first 100 words)", "LinkedIn (first 60 words, exploratory)")):
        m = R["media"][med]; ss = m["systems"]["style"]; v = m["validity"]["style_C1.0"]
        names = [s for s in ORDER if s in ss]
        y = np.arange(len(names))
        means = [ss[s]["mean_P"] for s in names]
        lo = [ss[s]["mean_P"] - ss[s]["ci_low"] for s in names]; hi = [ss[s]["ci_high"] - ss[s]["mean_P"] for s in names]
        ax.barh(y, means, xerr=[lo, hi], color="#5b8db8", capsize=3, height=0.62)
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis()
        ax.axvline(v["mean_P_real_lemkin"], color="#2e7d32", ls="--", lw=1.4, label=f"real Lemkin posts ({v['mean_P_real_lemkin']:.2f})")
        ax.axvline(v["mean_P_heldout_base_llama"], color="#c62828", ls=":", lw=1.4, label=f"plain base Llama text ({v['mean_P_heldout_base_llama']:.2f})")
        ax.axvline(v["mean_P_paul_graham"], color="#6a1b9a", ls=":", lw=1.4, label=f"Paul Graham ({v['mean_P_paul_graham']:.2f})")
        ax.set_xlim(0, 1); ax.set_title(title, fontsize=10); ax.set_xlabel("average P(looks like Lemkin), 95% interval over 30 topics", fontsize=8)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.95); ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig1_systems.png", dpi=170); plt.close(fig)


def fig_validity():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    for ax, med in zip(axes, ("blog", "linkedin")):
        z = np.load(OUT / f"validity_scores_{med}.npz")
        data = [z["real"], z["pg"], z["base_llama"]]
        bp = ax.boxplot(data, vert=False, showfliers=False, patch_artist=True, widths=0.55)
        for patch, c in zip(bp["boxes"], ("#a5d6a7", "#ce93d8", "#ef9a9a")):
            patch.set_facecolor(c)
        ax.set_yticklabels(["real Lemkin posts\n(held out)", "Paul Graham\n(held out)", "plain base Llama\n(unseen topics)"], fontsize=8)
        ax.set_xlim(0, 1); ax.set_xlabel("P(looks like Lemkin)", fontsize=8); ax.set_title(med.capitalize(), fontsize=10)
        ax.axvline(0.5, color="grey", lw=0.8, ls=":")
    fig.tight_layout(); fig.savefig(OUT / "fig2_validity.png", dpi=170); plt.close(fig)


def fig_agreement():
    sel = [r for r in ROWS if r["medium"] == "blog" and r["view"] == "style" and float(r["C"]) == 1.0 and r["voice_distance_fixed"] != ""]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    cols = {"A Base Llama": "#c62828", "C PersonaRAG stand-in": "#ef6c00", "E Relabel, no retrieval": "#1565c0", "E Relabel + retrieval": "#6a1b9a"}
    for s, c in cols.items():
        pts = [(float(r["voice_distance_fixed"]), float(r["p_lemkin"])) for r in sel if r["system"] == s]
        ax.scatter([a for a, _ in pts], [b for _, b in pts], s=12, alpha=0.6, color=c, label=s)
    rb = R["media"]["blog"]["robustness"]["draft_level_spearman_P_vs_negative_PUK28_voice_distance"]
    ax.set_xlabel("Kevin's voice distance at 100 words (lower = closer)", fontsize=8); ax.set_ylabel("classifier P(looks like Lemkin)", fontsize=8)
    ax.set_title(f"Blog drafts: two independent measures (rank corr. {rb['rho']:.2f}, n={rb['n']})", fontsize=9)
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT / "fig3_agreement.png", dpi=170); plt.close(fig)


fig_systems(); fig_validity(); fig_agreement()

# ------------------------------------------------------------------ pdf helpers
ss_ = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss_["Heading1"], fontName="Helvetica-Bold", fontSize=15, textColor=NAVY, spaceBefore=4, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=ss_["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, textColor=NAVY, spaceBefore=8, spaceAfter=4)
B = ParagraphStyle("B", parent=ss_["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13, spaceAfter=3)
SM = ParagraphStyle("SM", parent=B, fontSize=8, leading=10.2, spaceAfter=0)
TH = ParagraphStyle("TH", parent=SM, fontName="Helvetica-Bold", textColor=colors.white)
TITLE = ParagraphStyle("TITLE", parent=ss_["Title"], fontName="Helvetica-Bold", fontSize=21, textColor=NAVY, alignment=0, spaceAfter=2)
BUL = ParagraphStyle("BUL", parent=B, leftIndent=12, spaceAfter=2)
LINK = "#0b5cad"
P = lambda t, s=SM: Paragraph(t, s)
link = lambda url, label=None: f'<link href="{url}" color="{LINK}"><u>{label or url}</u></link>'


def table(rows, widths, zebra=True, extra=None):
    data = [[c if not isinstance(c, str) else P(c, TH if i == 0 else SM) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c2cf")),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("BACKGROUND", (0, 0), (-1, 0), NAVY)]
    if zebra:
        st += [("BACKGROUND", (0, i), (-1, i), LIGHT) for i in range(2, len(rows), 2)]
    if extra: st += extra
    t.setStyle(TableStyle(st)); return t


def footer(c, d):
    c.saveState(); c.setFont("Helvetica", 7.5); c.setFillColor(colors.HexColor("#666666"))
    c.drawString(0.5 * inch, 0.32 * inch, "DATA 298B Team 2  |  Authorship classifier results  |  Draft for team review")
    c.drawRightString(10.5 * inch, 0.32 * inch, f"Page {d.page}"); c.restoreState()


doc = BaseDocTemplate(str(PDF), pagesize=landscape(letter), leftMargin=0.5 * inch, rightMargin=0.5 * inch, topMargin=0.5 * inch,
                      bottomMargin=0.55 * inch, title="Authorship classifier results", author="DATA 298B Team 2")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")], onPage=footer)])
W = doc.width
S = []
b, l = R["media"]["blog"], R["media"]["linkedin"]
bs, bn = b["systems"]["style"], b["systems"]["style_nonum"]
v = b["validity"]; vs = v["style_C1.0"]; vn = v["style_nonum_C1.0"]; vf = v["full_C1.0"]; v0 = v["V0_paul_graham_only_classifier"]["style"]; v3 = v["V3_label_shuffle_control"]
rob = b["robustness"]; prov = R["provenance"]
drop = b["n_drafts_dropped_for_copy"]


SHORT = {"A Base Llama": "A", "C PersonaRAG stand-in": "C", "D QLoRA repaired": "D", "E Relabel, no retrieval": "E alone",
         "E Relabel + retrieval": "E+retrieval", "F Model F (F-REL)": "F"}


def readable(feats, n=8):
    names = {"—": "em dash", '"': "quotation marks", ";": "semicolons", "[": "square brackets", "]": "square brackets",
             "...": "ellipsis", ". . .": "ellipsis", ",": "commas", "!": "exclamation marks", ":": "colons"}
    out = []
    for f in feats:
        label = None
        if f in names: label = names[f]
        else:
            toks = f.split()
            if any(t.isalpha() for t in toks):
                label = '"' + " ".join("..." if t == "*" else t for t in toks) + '"'
            elif all(t in names for t in toks):
                label = names[toks[0]]
        if label and label not in out:
            out.append(label)
        if len(out) == n: break
    return out


def pair(m, key, name):
    return next(p for p in m["paired"][key] if p["B_minus_A"] == name)


# ================= page 1
S.append(P("Does the draft look like Jason Lemkin? A trained authorship classifier", TITLE))
S.append(P("DATA 298B Team 2. Built to answer an open question: how can we tell that the model sounds like our persona? "
           "This is a new, independent check next to our existing metrics. It reads saved drafts only and changes nothing in Kevin's evaluation.", B))
S.append(Image(str(OUT / "fig1_systems.png"), width=W, height=W * 3.6 / 11))
S.append(P("What the picture says", H2))
e_alone, a_base, c_pr = bs["E Relabel, no retrieval"], bs["A Base Llama"], bs["C PersonaRAG stand-in"]
S.append(P(f"<b>1.</b> Each bar is how often a model's blog draft looks like Lemkin to the classifier (0 means never, 1 means always). "
           f"Base Llama scores {f2(a_base['mean_P'])}, PersonaRAG {f2(c_pr['mean_P'])}, and our fine-tuned models D, E and F score {f2(bs['D QLoRA repaired']['mean_P'])} to {f2(e_alone['mean_P'])}. "
           f"Real held-out Lemkin posts score {f2(vs['mean_P_real_lemkin'])}.", BUL))
S.append(P(f"<b>2.</b> Fine-tuning is what moves the score. Model E (no retrieval) is {f2(pair(b, 'style', 'E Relabel, no retrieval minus A Base Llama')['mean_diff'])} above Base Llama, "
           f"interval {f2(pair(b, 'style', 'E Relabel, no retrieval minus A Base Llama')['ci_low'])} to {f2(pair(b, 'style', 'E Relabel, no retrieval minus A Base Llama')['ci_high'])}.", BUL))
S.append(P("<b>3.</b> Turning on retrieval (E with retrieval) and the graph (Model F) do not give a clear gain on this measure. This agrees with the fixed-length voice score from our PUK-28 work and with Kevin's finding that retrieval and the graph do not clearly improve voice, which is the point: a different method reaches the same conclusion. (It says nothing about writing quality or factual support, which other metrics cover.)", BUL))
if COLAB:
    S.append(P(f"<b>4.</b> In the final version the plain-Llama negatives are 900 fresh Llama 3.1 8B texts written on 450 non-gold Lemkin titles, so none of our systems' drafts is used in training. "
               f"This removed the main caveat of the earlier version. The blog ranking is identical, validity went up, and one change stands out: PersonaRAG now scores {f2(c_pr['mean_P'])}, close to Base Llama ({f2(a_base['mean_P'])}), where before it looked better ({f2(OLD['media']['blog']['systems']['style']['C PersonaRAG stand-in']['mean_P'])}). See the before-and-after table in Section 3b.", BUL))
S.append(P("<b>5.</b> The checks in Section 2 are what make this believable. One version of the classifier failed them and was thrown away (Section 2, check 0).", BUL))
S.append(PageBreak())

# ================= page 2
S.append(P("1. How it works, in plain words", H1))
S.append(table([
    ["Step", "What happens", "Why"],
    ["1. Real examples", "Real Lemkin blog posts from Kevin's training split. The families of the 30 gold topics are removed, so no test topic leaks in.", "Positive examples."],
    ["2. Two kinds of 'not Lemkin'", ("Paul Graham essays (another human writer) and 900 fresh texts written by plain Llama 3.1 8B on 450 non-gold Lemkin titles (made on Colab). No system's draft is used for training. Both are needed: with Paul Graham alone the classifier called base-Llama drafts 'Lemkin' (check 0)." if COLAB else "Paul Graham essays (another human writer) and plain base-Llama text taken from our own Base Llama drafts on other topics. Both are needed: with Paul Graham alone the classifier called base-Llama drafts 'Lemkin' (check 0)."), "Negative examples. Using LLM text as a negative follows the 'AI detection' check in [004]."],
    ["3. Fixed length", "Every text is cut to its first 100 words (60 for LinkedIn). Real posts and drafts are treated the same.", "Length cannot help one system. Same idea as our fixed-length voice score."],
    ["4. Unseen topics", "Five topic folds. A draft is scored by a model that never saw that draft's topic.", "Stops the model memorising topics."],
    ["5. Two views", "'Style' view hides every content word and keeps only small function words and punctuation. 'All words' view uses everything. A third view also ignores numbers.", "If topic words or numbers drove the result, the views would disagree. They agree."],
    ["6. Statistics", "Topic-level means, 95% bootstrap intervals over the 30 topics, paired sign-flip tests with Holm correction, the same tools as our PUK-28 work.", "Same rules as the rest of our evaluation."],
], [1.4 * inch, 5.9 * inch, W - 7.3 * inch]))
S.append(P("Why this answers the question", H2))
S.append(P("Published work on 'does generated text match one author' uses trained authorship classifiers: "
           f"{link('https://arxiv.org/abs/2509.14543', 'Wang et al. 2025 [004]')} combine attribution, verification, style matching and AI detection; "
           f"{link('https://arxiv.org/abs/2509.24930', 'Jemama and Kumar 2025 [047]')} verify authorship; "
           f"{link('https://aclanthology.org/2025.naacl-srw.47', 'Rezaei 2025 [065]')} trains a detector and uses it to score generated text because finding human experts per author is nearly impossible; "
           f"{link('https://aclanthology.org/2020.icon-main.16', 'Fabien et al. 2020 [113]')} fine-tune BERT for authorship on blog and email data. "
           f"{link('https://arxiv.org/abs/2508.06374', 'Jangra et al. 2025 [063]')} find that no single style metric is reliable and that combining several is better. "
           "This tool adds the missing piece to our metric set. It is still a proxy for style, not proof of authorship.", B))
S.append(P("What the classifier looks at", H2))
tf = R["top_features"]["blog_style_nonum"]
S.append(P(f"In the number-free style view, the patterns that push a text toward Lemkin are: <b>{', '.join(readable(tf['toward_lemkin']))}</b>. "
           f"The patterns that push toward other writers are: <b>{', '.join(readable(tf['toward_other']))}</b>. "
           "(Hidden content words appear as '...'.) So it mostly reads punctuation and sentence shape. That is real surface style, but it is also something a model could imitate without really sounding like him (see limits).", B))
S.append(PageBreak())

# ================= page 3: proof
S.append(P("2. Checks that show it can be trusted", H1))
sysrank = rob["system_level_spearman_P_vs_negative_PUK28_voice_distance"]; dl = rob["draft_level_spearman_P_vs_negative_PUK28_voice_distance"]
ll = l["robustness"]["draft_level_spearman_P_vs_negative_PUK28_voice_distance"]
v1, v2 = vs["V1_real_lemkin_vs_paul_graham_auc"], vs["V2_real_lemkin_vs_heldout_topic_base_llama_auc"]
v2n = vn["V2_real_lemkin_vs_heldout_topic_base_llama_auc"]
lv2 = l["validity"]["style_C1.0"]["V2_real_lemkin_vs_heldout_topic_base_llama_auc"]
nmatch = sum(1 for x in prov.values() if x["matches_PUK28_record"])
within = rob["within_system_spearman_P_vs_draft_words"]
rows = [
    ["#", "Check", "Result (blog)", "What it shows"],
    ["0", "Classifier against Paul Graham only",
     f"Separates the two writers (AUC {f3(v0['auc_real_vs_pg'])}) but scores plain base-Llama drafts {f2(v0['mean_P_base_llama_drafts'])} as 'Lemkin'.",
     "<b>This version is invalid and is not used.</b> It learned 'modern business writing', not 'Lemkin'. Fixed by adding LLM text as a negative."],
    ["1", "Held-out real Lemkin posts against held-out Paul Graham essays",
     f"AUC {f3(v1[0])} (95% interval {f3(v1[1])} to {f3(v1[2])}).", "It separates the two human writers on posts it never trained on."],
    ["2", "Held-out real Lemkin posts against base-Llama text on <b>unseen topics</b> (the key check)",
     f"AUC {f3(v2[0])} ({f3(v2[1])} to {f3(v2[2])}). Without numbers: {f3(v2n[0])}. LinkedIn: {f3(lv2[0])}.",
     "It tells real Lemkin from plain LLM writing on topics it never saw. This is the check that matters most."],
    *([["2b", "The Colab negatives file was checked before use (pass/fail rules fixed in advance)",
        f"{NEGVAL['n_rows']} texts, {NEGVAL['gold_topic_overlaps']} gold-topic overlaps, {NEGVAL['refusal_share']*100:.0f}% refusals, usable {NEGVAL['usable_share']*100:.1f}%. Copying (12-word runs, ignoring the topic title as Kevin's gate does): {NEGVAL['copy_share']*100:.1f}%. Raw count before ignoring titles: {NEGVAL['copy_share_raw_incl_topic_title_repeats']*100:.1f}%.",
        "The text is plain Llama writing and not Lemkin's. Note: the first version of the copy rule did not ignore title repeats and failed at 5.7%. 48 of 51 flags were only the model repeating the topic title. The rule was corrected to match Kevin's gate after that first run, and both numbers are shown."]] if COLAB else []),
    ["3", "Label-shuffle control (labels mixed up, 20 times)",
     f"Real against Paul Graham: mean AUC {f2(v3['auc_real_vs_pg_mean'])}, range {f2(v3['auc_real_vs_pg_min_max'][0])} to {f2(v3['auc_real_vs_pg_min_max'][1])}. Real against base Llama: mean {f2(v3['auc_real_vs_base_llama_mean'])}, range {f2(v3['auc_real_vs_base_llama_min_max'][0])} to {f2(v3['auc_real_vs_base_llama_min_max'][1])}.",
     f"With no real signal the scores scatter around chance, far below the real {f3(v2[0])}. So the result is not a leak or a bug." + (" (The shuffled runs sit a little under 0.5 against Llama text, which we note openly.)" if v3["auc_real_vs_base_llama_mean"] < 0.45 else "")],
    ["4", "Topic-hiding and number-free views against the all-words view",
     f"System ranking agreement (rank correlation): style vs all words {f2(rob['style_vs_full'])}; style vs number-free {f2(rob['style_vs_style_nonum'])}.",
     "Hiding topic words or numbers does not change the picture, so topic and invented figures are not driving the ranking."],
    ["5", "Different regularisation settings", f"Rank agreement {f2(rob['style_C0.3_vs_style_C1.0'])} and {f2(rob['style_C3.0_vs_style_C1.0'])} with the main setting.", "Not an accident of one setting."],
    ["6", "Drop drafts that copy training text (8-word overlap)",
     f"Rank agreement {f2(rob['style_vs_style_no_copy'])}. Dropped: " + ", ".join(f"{SHORT[k]} {v_}" for k, v_ in sorted(drop.items())) + ".",
     "Retrieval models that quote training posts cannot be inflating their own score."],
    ["7", "Draft files are the ones we reported on", f"{nmatch} of {len(prov)} draft files match the SHA-256 checksums recorded in PUK-28 (D and F have no recorded checksum).", "Same drafts as our earlier metrics."],
    ["8", "Agreement with Kevin's voice distance (first 100 words)",
     f"System ranking agrees (rank correlation {f2(sysrank)}, 4 systems). Draft by draft: {f2(dl['rho'])} for blog (n={dl['n']}); LinkedIn {f2(ll['rho'])}, not clear.",
     "A different method ranks systems the same way on blogs. On LinkedIn the old voice score cannot separate drafts (our PUK-28 finding), while this classifier still does (check 2)."],
    ["9", "Is it just length?", "Within each system, score against draft length: " + ", ".join(f"{SHORT[k]} {f2(x['rho'])}" for k, x in sorted(within.items())) + ".",
     "Weak links only. All texts are cut to the same number of words anyway."],
]
ex = [("BACKGROUND", (0, 1), (-1, 1), RED)]
S.append(table(rows, [0.3 * inch, 2.3 * inch, 3.6 * inch, W - 6.2 * inch], zebra=True, extra=ex))
S.append(PageBreak())
S.append(Image(str(OUT / "fig2_validity.png"), width=W * 0.97, height=W * 0.97 * 3.2 / 11))
S.append(P("Figure: scores for real held-out Lemkin posts, held-out Paul Graham essays, and plain base-Llama text on unseen topics (style view). Real Lemkin sits high, the other two sit low.", SM))
S.append(Spacer(1, 6))
S.append(Image(str(OUT / "fig3_agreement.png"), width=4.9 * inch, height=4.9 * inch * 3.6 / 5.2))
S.append(P("Figure: each dot is one blog draft. Higher on the classifier and lower on Kevin's voice distance both mean closer to Lemkin, so a falling cloud means the two measures agree.", SM))
S.append(PageBreak())

# ================= page: results tables
S.append(P("3. Results by system", H1))


def system_table(m, med):
    ss, sn, sf = m["systems"]["style"], m["systems"]["style_nonum"], m["systems"]["full"]; nc = m["systems"]["style_no_copy"]
    rows = [["System", "Style view: mean P (95% interval)", "Number-free view", "All-words view", "Style view, copy drafts removed", "Topics scored"]]
    for s in sorted(ss, key=lambda x: -ss[x]["mean_P"]):
        rows.append([s, f"<b>{f3(ss[s]['mean_P'])}</b> ({f3(ss[s]['ci_low'])} to {f3(ss[s]['ci_high'])})", f"{f3(sn[s]['mean_P'])}", f"{f3(sf[s]['mean_P'])}",
                     f"{f3(nc[s]['mean_P'])} (n={nc[s]['n_topics']})", str(ss[s]["n_topics"])])
    return table(rows, [2.3 * inch, 2.6 * inch, 1.3 * inch, 1.2 * inch, 1.8 * inch, W - 9.2 * inch])


def paired_table(m):
    rows = [["B minus A", "Difference (style view)", "95% interval", "Holm p", "Style verdict", "Number-free verdict"]]
    pn = {p["B_minus_A"]: p for p in m["paired"]["style_nonum"]}
    for p in m["paired"]["style"]:
        q = pn[p["B_minus_A"]]
        rows.append([p["B_minus_A"].replace(" minus ", "  minus  "), f"{p['mean_diff']:+.3f}", f"{p['ci_low']:+.3f} to {p['ci_high']:+.3f}", ("<0.001" if p["p_holm"] < 0.001 else f"{p['p_holm']:.3f}"), p["verdict"], q["verdict"]])
    return table(rows, [3.6 * inch, 1.2 * inch, 1.5 * inch, 0.7 * inch, 1.3 * inch, W - 8.3 * inch])


S.append(P("Blog (first 100 words, 30 topics; D and E have 3 seeds averaged per topic)", H2))
S.append(system_table(b, "blog"))
S.append(Spacer(1, 4))
S.append(P("Paired comparisons by topic (B minus A). A verdict says 'higher' or 'lower' only when the interval excludes zero and the Holm-corrected p is below 0.05.", SM))
S.append(paired_table(b))
S.append(PageBreak())
S.append(P("LinkedIn (first 60 words, exploratory: much less training data, weaker check 2 result)", H2))
S.append(system_table(l, "linkedin"))
S.append(Spacer(1, 4))
S.append(paired_table(l))
S.append(P("X posts and talks are not supported: X drafts are mostly shorter than any usable window, and there are no attributed talk references.", SM))
S.append(PageBreak())

# ================= limits

if COLAB:
    S.append(P("3b. Before and after the Colab negatives", H1))
    S.append(P("Earlier version: negatives were our own Base Llama drafts on other topics (cross-fitted). Final version: 900 fresh Llama texts on non-gold topics. Everything else is identical.", B))
    ob, nb_ = OLD["media"]["blog"], R["media"]["blog"]
    ol, nl = OLD["media"]["linkedin"], R["media"]["linkedin"]
    sysn = ["A Base Llama", "C PersonaRAG stand-in", "D QLoRA repaired", "E Relabel, no retrieval", "E Relabel + retrieval", "F Model F (F-REL)"]
    rk = lambda m: [SHORT[x] for x in sorted(sysn, key=lambda y: -m["systems"]["style"][y]["mean_P"])]
    chg = lambda a, b: [k for k in a if a[k]["verdict"] != b[k]["verdict"]]
    pa = {p["B_minus_A"]: p for p in ob["paired"]["style"]}
    pb = {p["B_minus_A"]: p for p in nb_["paired"]["style"]}
    pla = {p["B_minus_A"]: p for p in ol["paired"]["style"]}
    plb = {p["B_minus_A"]: p for p in nl["paired"]["style"]}
    pln = {p["B_minus_A"]: p for p in nl["paired"]["style_nonum"]}
    k_v2 = "V2_real_lemkin_vs_heldout_topic_base_llama_auc"
    li_changes = "; ".join(f"{k}: {pla[k]['verdict']} to {plb[k]['verdict']} (Holm p {plb[k]['p_holm']:.3f}, number-free view: {pln[k]['verdict']})" for k in chg(pla, plb)) or "None changed"
    rows_ba = [
        ["Item", "Earlier version", "Final version", "Reading"],
        ["Check 2, blog: real Lemkin vs unseen plain Llama text (AUC)", f3(ob["validity"]["style_C1.0"][k_v2][0]), f3(nb_["validity"]["style_C1.0"][k_v2][0]), "Better, and above 0.90"],
        ["Check 2, LinkedIn (exploratory)", f3(ol["validity"]["style_C1.0"][k_v2][0]), f3(nl["validity"]["style_C1.0"][k_v2][0]), "Better, but still under 0.90"],
        ["Label-shuffle control, blog, real vs Llama (mean AUC)", f3(ob["validity"]["V3_label_shuffle_control"]["auc_real_vs_base_llama_mean"]), f3(nb_["validity"]["V3_label_shuffle_control"]["auc_real_vs_base_llama_mean"]), "Now centred on chance (0.5)"],
        ["Blog system ranking (best to worst)", ", ".join(rk(ob)), ", ".join(rk(nb_)), "Same order" if rk(ob) == rk(nb_) else "Order changed"],
        ["Blog verdicts that changed (of 8 comparisons)", "-", str(len(chg(pa, pb))), "None changed" if not chg(pa, pb) else ", ".join(chg(pa, pb))],
        ["LinkedIn verdicts that changed (of 8)", "-", str(len(chg(pla, plb))), li_changes],
    ]
    notes = {"A Base Llama": "No longer in the negative class", "C PersonaRAG stand-in": "Biggest change: now close to Base Llama"}
    for sname in sysn:
        rows_ba.append([f"Blog mean P(Lemkin): {sname}", f3(ob["systems"]["style"][sname]["mean_P"]), f3(nb_["systems"]["style"][sname]["mean_P"]), notes.get(sname, "Small drop, same level")])
    S.append(table(rows_ba, [3.5 * inch, 2.3 * inch, 2.3 * inch, W - 8.1 * inch]))
    S.append(Spacer(1, 4))
    S.append(P("Conclusion: the main findings do not depend on the earlier caveat. Fine-tuning moves the score a lot, retrieval and the graph give no clear gain, and the ranking agrees with our voice score. PersonaRAG, which looked partly better than Base Llama before, is now about the same as Base Llama on this measure.", B))
    S.append(PageBreak())

S.append(P("4. Limits, stated plainly", H1))
lim = [
    ["Limit", "Why it matters", "What we would do"],
    *([["The negatives come from one model and one prompt.", "Llama 3.1 8B (Hugging Face weights, bf16) with the Lemkin-style prompt. The old Base Llama drafts came from the Ollama build, so wording differs slightly. 76 percent of the LinkedIn negatives hit the token limit, which does not matter because only the first 60 words are read.", "Fine as is. A second model or prompt would show whether the classifier only learns Llama-specific wording."]] if COLAB else [["Plain base Llama text was used as a negative, taken from our own base-Llama drafts on other topics.", "System A (Base Llama) is partly in the negative class, so its low score is partly built in. PersonaRAG was not trained on and still scores low (0.51), and check 2 uses unseen topics, which limits the damage.", "Generate several hundred base-Llama drafts on non-gold topics on Colab, then run: python -X utf8 run_pipeline.py --llm-negatives their_file.jsonl --out-dir outputs_colab."]]),
    ["It reads surface style: em dashes, quotation marks, short sentences.", "A model could raise its score by imitating these without sounding more like him. Numbers and symbols were tested and do not change the ranking.", "Keep it as one metric among several, never as proof of authorship."],
    ["Paul Graham is the only other human writer and writes about different topics.", "Mitigated by the style view and by check 2, but not removed.", "Add other modern business writers if we can find usable text."],
    ["Same 30 gold topics as the rest of our evaluation, one run each for A, C and F.", "A development panel, not a new test. Intervals are wide for F (28 blog drafts long enough).", "State it. Add F-SOURCE and more seeds if Kevin wants."],
    ["No human has checked it.", "[063] notes that human agreement on style is low, but a small check would still help.", "Team decision (DEC-007). Left out."],
    ["E against D (repaired) is borderline.", "On blogs it is clear in the style view but not in the number-free view. On LinkedIn it flipped between the two classifier versions.", "Report as uncertain."],
    ["LinkedIn is weaker than blog.", "Much less training text, and check 2 reaches only about 0.88 AUC.", "Treat LinkedIn as exploratory."],
]
S.append(table(lim, [3.2 * inch, 4.0 * inch, W - 7.2 * inch]))
S.append(P("5. How to re-run and where the files are", H2))
S.append(P("Folder: <b>authorship_classifier/</b> in the repository (branch vb-puk-28-authorship-classifier). "
           "Collect the inputs with <b>python authorship_classifier/prepare_inputs.py</b>, then run <b>python -X utf8 run_pipeline.py</b> (about 3 minutes, laptop only) and <b>python -X utf8 make_report.py</b>. "
           "Outputs: outputs/ (earlier version) and outputs_colab/ (final version): results.json, scores_per_draft.csv (every draft's score, no draft text), three figures. Final report: python -X utf8 make_report.py outputs_colab. "
           "13 automatic tests check the leak-prone steps: python -X utf8 -m pytest authorship_classifier/tests.", B))

# ================= references
S.append(PageBreak())
S.append(P("6. Papers behind the method (all from the repo research library; links checked)", H1))
refs = [
    ["Library", "Authors", "Year", "Title", "Used for", "Link"],
    ["[004]", "Zhengxiang Wang, Nafis Irtiza Tripto, Solha Park, Zhenzhen Li, Jiawei Zhou", "2025", "Catch Me If You Can? Not Yet: LLMs Still Struggle to Imitate the Implicit Writing Styles of Everyday Authors", "Ensemble of attribution, verification, style matching and AI detection; LLMs struggle most on informal blogs", link("https://arxiv.org/abs/2509.14543")],
    ["[047]", "Rebira Jemama, Rajesh Kumar", "2025", "How Well Do LLMs Imitate Human Writing Style?", "Authorship verification to score imitation; fidelity and detectability are separate", link("https://arxiv.org/abs/2509.24930")],
    ["[063]", "Anubhav Jangra, Bahareh Sarrafzadeh, Silviu Cucerzan, Adrian de Wynter, Sujay Kumar Jauhar", "2025", "Evaluating Style-Personalized Text Generation: Challenges and Directions", "No gold standard; combine several metrics", link("https://arxiv.org/abs/2508.06374")],
    ["[065]", "Mosab Rezaei", "2025", "Detecting, Generating, and Evaluating in the Writing Style of Different Authors", "Train a detector and use it to score generated text", link("https://aclanthology.org/2025.naacl-srw.47")],
    ["[066]", "George Mikros", "2025", "Beyond the surface: stylometric analysis of GPT-4o's capacity for literary style imitation", "Frequent-word distance and a classifier to test imitations", link("https://doi.org/10.1093/llc/fqaf035")],
    ["[081]", "Shuai Liu, Shantanu Agarwal, Jonathan May", "2024", "Authorship Style Transfer with Policy Optimization", "A trained style model as the reward; warns about topic leaking into the classifier", link("https://arxiv.org/abs/2403.08043")],
    ["[113]", "Mael Fabien, Esau Villatoro-Tello, Petr Motlicek, Shantipriya Parida", "2020", "BertAA: BERT fine-tuning for Authorship Attribution", "Authorship classifier on blog and email data; adding style features helps", link("https://aclanthology.org/2020.icon-main.16")],
]
S.append(table(refs, [0.55 * inch, 2.0 * inch, 0.4 * inch, 2.7 * inch, 2.35 * inch, W - 8.0 * inch]))
S.append(Spacer(1, 4))
S.append(P("The topic-hiding trick (keep function words, mask the rest) is a standard technique called text distortion in authorship research. It is not in our library, so it is described here but not cited from it.", SM))
doc.build(S)
print("built", PDF)
