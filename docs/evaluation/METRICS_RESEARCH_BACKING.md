# Research backing for our evaluation metrics

Written 2026-11-14 for the DATA 298B Team 2 group (PUK-28, metrics).

Purpose: make sure our metrics are based on the research papers we have, and give a published-research answer to how we can tell if a draft sounds like our persona. This file maps each metric to the published papers in the repo research library (docs/research/papers on Kevin's branch kevin/lightrag-model-f), says what each paper really says, and says where we differ or fall short.

How I checked: I extracted the text of 40 library PDFs and read the abstract plus the method, limits and conclusion parts that matter for about 25 of them (the ones cited below). I did not read every page of every paper. File numbers like [063] are the library numbers.

---

## 1. The short answer

1. **There is no accepted gold standard for measuring "sounds like this author".** The 2025 Microsoft and Columbia study of style metrics [063] says this directly and says human raters also agree with each other poorly on style.
2. **The accepted practice is a group of complementary checks, not one number.** [004] combines authorship attribution, authorship verification, style matching and AI detection. [063] finds that combining several metrics beats any single one. Our setup (gates plus separate scores, no single headline number for voice) follows that practice.
3. **What we already do has published support:** feature and word-frequency distance to the author's real texts [066] [047] [048] [065], a rubric-based AI judge from a different model family [103] [102] [084], and claim-by-claim support checking [025] [034].
4. **What is missing, and what the papers expect:** a trained authorship classifier that scores our drafts [004] [047] [065] [113]. This is the most direct way to answer "does it sound like him", and it needs no human raters. We do not have it yet.
5. **We must state the limits.** Every one of these is a proxy for style, not proof of authorship. The papers themselves say so.

---

## 2. Metric by metric

"Our metric" is what Kevin's unified_eval does today. "Backing" is what the papers support. "Limit" is where we differ or where the paper warns us.

### 2.1 Voice distance (surface features against real held-out posts)

Our metric: 29 features per draft (9 shape features such as sentence length, plus the rates of 20 common function words), mean absolute z-score against held-out real posts of the same medium. Trusted as a selector for blogs only.

| Paper | What it says | How it backs us |
|---|---|---|
| [066] Mikros, GPT-4o literary style imitation | Measures how close imitations are to an author with Cosine Delta on the 1,000 most frequent words, then adds a random forest authorship test and several feature sets. Finds GPT-4o copies some surface features but misses the full stylistic signature, and imitations still overlap with generic GPT text. | Distance on frequent-word frequencies against the author's own texts is a standard stylometric method. Also shows the limit: surface closeness is partial. |
| [047] Jemama and Kumar 2025 | Uses character n-gram TF-IDF plus embeddings to verify authorship without training. Names sentence length, punctuation habits and word preferences as style cues. Finds style fidelity and detectability by machine are separate things. | Same family of cues as our features. |
| [048] Reinhart et al. 2025 | Compares human and LLM text with many lexical and grammatical features. LLMs differ systematically from humans, and instruction-tuned models differ more than base models. | Feature profiles can tell machine writing from human writing. Our Llama 3.1 Instruct base is the kind of model this applies to. |
| [065] Rezaei, NAACL 2025 student workshop | Evaluates generated text by training an author detector and also comparing feature distributions between original and generated text. Says expert human evaluation for each author is nearly impossible. | Matches our setup closely: automatic evaluation because no human can judge each author. |

Limits to state:
- [063] finds single metrics only moderately effective at telling authors apart inside one domain.
- [004] finds LLMs struggle most with informal blogs and forums, which is our domain.
- Our own measurement: the score favours longer drafts (correlation about -0.66). We cut drafts to 100 words to compare. [066] says Cosine Delta on normalised frequency vectors is more robust to length differences, so it is a good extra check (see section 4).

### 2.2 Authorship classifier (NOT in our system yet)

| Paper | What it says |
|---|---|
| [004] Wang et al. 2025 | More than 40,000 generations per model and 400+ real authors across news, email, forums and blogs. Scores style imitation with an ensemble: authorship attribution, authorship verification, style matching and AI detection. Their verification models reach 87.7 to 91.4 percent accuracy on the four test sets (their Table 1). Finds LLMs approximate style in structured formats but struggle with informal blogs and forums. |
| [047] Jemama and Kumar 2025 | Authorship verification with 97.5 percent accuracy on essays and 94.5 percent across domains, with no supervised training. |
| [113] Fabien et al. 2020 (BertAA) | Fine-tunes BERT for authorship attribution, tested on Blog Authorship, Enron email and IMDb. Adding stylometric features in an ensemble raises macro F1 by 2.7 percent relative on average. |
| [065] Rezaei 2025 | Trains a detector and uses it to score generated text. |
| [081] Liu, Agarwal and May 2024 | Uses a trained style model as the scoring signal for style transfer (see section 5). |

Our own gap: Kevin's older eval_rag.py has a TF-IDF authorship score, but his own notes list problems with it (the vectorizer is fit before cross-validation, the negative class can be a synthetic fallback), so it is not used as a selector.

### 2.3 Writing quality with an AI judge

Our metric: judge qwen2.5:14b, temperature 0, blinded to the model, six items scored 1 to 5 (coherence, fluency, logical progression, relevance, specificity, usefulness). Not a selector until a sanity check passes. Voice is collected only for that check.

| Paper | What it says | How it backs us |
|---|---|---|
| [103] Liu et al. 2023 (G-Eval) | A rubric prompt plus step-by-step reasoning and form filling. Spearman correlation 0.514 with humans on summarization. Warns that LLM judges can favour LLM-written text. | The rubric-with-fixed-items design. |
| [102] Zheng et al. 2023 (MT-Bench) | Strong judges agree with humans over 80 percent on chat answers, the same as humans agree with each other. Lists position, verbosity and self-enhancement bias. | Supports using a judge at all. The listed biases match what we measured: our judge favours long text. |
| [084] Liu et al. 2024 (Narcissistic Evaluators) | LM-based metrics favour text from their own model family. | Supports Kevin's choice of a Qwen judge for Llama drafts. |
| [085] He et al. 2023 (Blind Spots) | Stress tests show metrics can miss real errors. For example BERTScore is confused by truncated summaries. | Supports our completion gate, which checks that a draft ends on a full sentence. |

Limits to state:
- These papers test summarization, dialogue and chat, not author style.
- [063] tried LLM judges on authorship attribution and got 0.238 and 0.491 accuracy for two prompts, close to random. This supports Kevin's rule that the judge is not used to score voice.
- Nobody has checked our judge against human raters (our PUK-28 gap G12). [063] reports that style preferences are more subjective than content preferences, so human agreement on style would be low anyway.

### 2.4 Claim support (grounding)

Our metric: split a draft into claims, label each as supported, unsupported, contradicted or not verifiable against the retrieved passages. A retrieval draft passes if at least one claim could be checked, none is contradicted, and at least half of the checked claims are supported.

| Paper | What it says | How it backs us |
|---|---|---|
| [025] Min et al. 2023 (FActScore) | Breaks long text into atomic facts and scores the share supported by a source. An automatic estimator is within 2 percent of the human score. | Same idea: claim-level support, not one yes or no for the whole draft. |
| [034] Es et al. (Ragas) | Faithfulness: extract statements from the answer, then check each can be inferred from the retrieved context. No human labels needed. | Same structure as our claim judge. |
| [104] Manakul et al. 2023 (SelfCheckGPT) | Detects hallucination by checking whether repeated samples agree. | Another accepted approach. We do not use it. |

Limit: all three use different judge models and domains (biographies, general question answering). Kevin's claim judge was not checked against human labels. His deterministic number check (numbers must appear in the evidence) is our own addition; none of these papers covers it.

### 2.5 Copying gate (12 or more shared words with training text)

[110] Hashimoto et al. 2019 explains why a quality check alone is not enough: a model that copies training text would pass a human quality check but has no real generalisation. This backs having a copy check next to the quality score.

### 2.6 BLEU, ROUGE, BERTScore as diagnostics only

| Paper | What it says |
|---|---|
| [016] Papineni 2002, [017] Lin 2004, [015] Zhang et al. (BERTScore) | The original definitions. Built for translation and summarization with reference texts. |
| [103] G-Eval | BLEU and ROUGE correlate poorly with human judgement on tasks that need creativity and variety. |
| [087] Logacheva et al. 2022 | Manual and automatic scores correlate only weakly for style transfer. ChrF and BERTScore are usable as a rough proxy only. |
| [086] Callan and Foster 2023 | BERTScore agrees poorly with human views of how interesting or coherent stories are. |
| [063] Jangra et al. 2025 | Tests BLEU, embeddings and LLM judges for style tasks and finds them weak when used alone. |

This backs Kevin's rule that these stay as diagnostics. In his run, BLEU 0.011 and ROUGE-L 0.077 could not separate any system.

### 2.6b Stance consistency (advice claims against held-out posts)

Weakest backing. [053] AXCEL and [054] DCR-Consistency are about checking consistency between texts at sentence level with LLMs and explaining the mismatch. They are indirect support at best. No library paper tests "stance matches an author's known position". Say so plainly in the report.

### 2.7 Reporting separate scores, no single number

[063] finds ensembles of diverse metrics beat single ones, using majority voting and performance-weighted voting. Our eligibility gates plus separate scores follow the same logic. Note: our Persona Utility (weights 45, 20, 25, 10) is a single weighted number, the weights were a project choice and have not been tested. No library paper backs those weights.

### 2.8 Human evaluation

[110] treats human judgement as the usual gold standard for quality but shows it misses diversity problems. [063] and [087] show style judgements by humans are subjective and correlate weakly with automatic scores. [065] says finding experts per author is nearly impossible. Together these support Kevin's decision (DEC-007) to drop human voice preference from model selection, but they also mean a small human check would still be valuable if the team could do one.

### 2.9 Statistics (bootstrap, Holm correction)

No paper in the library covers these. Cite standard statistics references separately if the report needs them.

---

## 3. Which claims are safe to make

| Claim | Safe? |
|---|---|
| "We measure style as distance to the author's real posts, a standard stylometric approach" | Yes, with [066] [047] [065] and the limit that it is a proxy |
| "We separate checks instead of one score, as recommended" | Yes for gates and the separate scores, with [063] [004]. Not for Persona Utility weights |
| "Our AI judge follows G-Eval and avoids same-family bias" | Yes, with [103] [084] [102]. Add that it is not human-validated |
| "Claims are checked one by one like FActScore" | Yes, with [025] [034] |
| "Our metrics prove the model sounds like Lemkin" | No. No paper supports that and [063] says no gold standard exists |
| "Our stance metric is backed by research" | Only weakly |

---

## 4. What we can improve (ranked)

1. **Add an authorship classifier score** ([004] [047] [065] [113]). Train on real Lemkin posts (Kevin's train split only) against other writers, check it on held-out real posts first, then score drafts from every system. Main risk: if topics differ between the two sides, the classifier learns the topic and not the style. [081] gives a similar warning in its footnote 7 for its own task. Paul Graham essays are in the repo (master branch) but cover different topics, so this limit must be stated.
2. **Add a length-robust distance check** (Cosine Delta on the most frequent words, as in [066]) beside the current z-score distance, because ours is length-sensitive.
3. **Run judge bias checks** named in [102] [084]: compare scores against draft length, and confirm the judge family differs from the writer (already true). We already know about the length effect, so report it.
4. **Test the Persona Utility weights.** Change the weights and see whether the ranking changes. If it holds, say so. If not, report it as a limit.
5. **Say the limits in the report.** All of these are proxies for style, with no gold standard ([063]).

---

## 5. Reinforcement learning, what the library offers

- [112] Ouyang et al. 2022 (InstructGPT): the standard reinforcement learning from human feedback recipe. It needs human rankings of outputs, which Kevin does not want to collect.
- [081] Liu, Agarwal and May 2024: the closest match for us. Tune first, then optimise with a reward from a trained style model, no human raters. They compare PPO, DPO and CPO and describe DPO and CPO as more stable and cheaper than PPO. Their own limits: low-resource individual author style stays much weaker than high-resource tasks, and small data may be too little for individual authors.
- Both papers fit the plan of a classifier as the reward (section 4, item 1), but then the classifier becomes the thing the model is trained to please, so it can no longer also serve as an independent evaluation score.

---

## 6. Papers not to cite (wrong PDFs in the library)

| Library file | Problem |
|---|---|
| [006] Shao 2024, Authorship Style Transfer ITDA | The PDF is a different paper (manifest title: topological superconductivity). |
| [026] Przystalski 2026, Stylometry Human vs LLM | The PDF is a different paper (manifest title: ventilator pressure prediction). |
| [078] Wang 2023, Authorship Representation Stylistic Features | I opened it: the PDF is an emotion recognition in conversations paper. |

---

## 7. Reference list (as stated in the PDFs)

- [004] Wang, Tripto, Park, Li, Zhou (2025). Catch Me If You Can? Not Yet: LLMs Still Struggle to Imitate the Implicit Writing Styles of Everyday Authors. arXiv:2509.14543.
- [015] Zhang et al. BERTScore: Evaluating Text Generation with BERT. ICLR 2020 submission copy.
- [016] Papineni, Roukos, Ward, Zhu (2002). BLEU.
- [017] Lin (2004). ROUGE.
- [025] Min et al. (2023). FActScore. arXiv:2305.14251.
- [034] Es, James, Espinosa-Anke, Schockaert. Ragas. arXiv:2309.15217.
- [047] Jemama and Kumar (2025). How Well Do LLMs Imitate Human Writing Style? IEEE UEMCON 2025. arXiv:2509.24930.
- [048] Reinhart et al. (2025). Do LLMs write like humans? Variation in grammatical and rhetorical styles. Journal name not in the extracted text; check the PDF.
- [053] Sreekar et al. (2024). AXCEL. [054] Cui et al. (2024). DCR-Consistency.
- [063] Jangra, Sarrafzadeh, Cucerzan, de Wynter, Jauhar (2025). Evaluating Style-Personalized Text Generation: Challenges and Directions. arXiv:2508.06374.
- [065] Rezaei (2025). Detecting, Generating, and Evaluating in the Writing Style of Different Authors. NAACL 2025 Student Research Workshop, pages 485 to 491.
- [066] Mikros. Beyond the surface: stylometric analysis of GPT-4o's capacity for literary style imitation. Year and venue not in the extracted text; check the PDF.
- [081] Liu, Agarwal, May (2024). Authorship Style Transfer with Policy Optimization. arXiv:2403.08043.
- [084] Liu, Moosavi, Lin (2024). LLMs as Narcissistic Evaluators. arXiv:2311.09766.
- [085] He et al. (2023). On the Blind Spots of Model-Based Evaluation Metrics for Text Generation. arXiv:2212.10020.
- [086] Callan and Foster (2023). How interesting and coherent are the stories generated by a large-scale neural language model? Expert Systems.
- [087] Logacheva et al. (2022). A Study on Manual and Automatic Evaluation for Text Style Transfer: The Case of Detoxification. HumEval workshop.
- [102] Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS 2023. arXiv:2306.05685.
- [103] Liu et al. (2023). G-Eval. arXiv:2303.16634.
- [104] Manakul, Liusie, Gales (2023). SelfCheckGPT. arXiv:2303.08896.
- [110] Hashimoto, Zhang, Liang (2019). Unifying Human and Statistical Evaluation for Natural Language Generation. NAACL-HLT 2019.
- [112] Ouyang et al. (2022). Training language models to follow instructions with human feedback.
- [113] Fabien, Villatoro-Tello, Motlicek, Parida (2020). BertAA: BERT fine-tuning for Authorship Attribution. ICON 2020.

Before citing any paper in the workbook, open the PDF once and confirm the details above.
