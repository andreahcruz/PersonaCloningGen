# Development retrieval annotation

This sheet is for development questions only. Do not open the corpus validation or test split. Do not grade a document because its id was preselected. Leave every grade null until you have read the document.

The annotator file is `host_finetune/benchmarks/dev_retrieval_benchmark.jsonl`. The file `dev_retrieval_benchmark_withheld.jsonl` contains the old review notes. Open that file only after the grades are assigned.

## Scale

| Grade | Meaning |
|---:|---|
| 0 | Irrelevant. The document does not help answer the question. |
| 1 | Tangential. The document shares a topic and does not answer the question. |
| 2 | Relevant. The document contains information that answers part of the question. |
| 3 | Highly relevant. The document contains the information the question asks for. |

Precision, recall, and MRR later treat grades 2 and 3 as relevant. Grade 1 is not relevant. nDCG uses the numeric grade, so a 3 can outrank a 2. Until a person fills the grades, those metrics stay null.

## What to do with each question

1. Read the query. Ignore the phenomenon tag when deciding a grade. The tag records why the question was written.
2. Read each candidate. The candidate list mixes saved pilot retrieval ids and author-preselected ids. Provenance says which. Neither origin is a grade.
3. Assign one grade, your name, and an optional note.
4. If a relevant source is missing, add its id to `missing_relevant_source_ids`. Do not invent an id. Use a real `gd_` id from the development corpus.
5. Do not add a reference answer. Do not copy a number from memory into the note as if it were the label.

A question with an empty candidate list is waiting for sources. Add ids only when you have identified them, and leave their grades null until a second pass reads them.

Saved pilot hits for the Olympic question are in the pool because a retriever returned them. Their grades are empty. Do not assign 0 because an older note said they were irrelevant. Read them, or leave them null.

## What this sheet is not

It is not the 30-topic gold generation file. It is not a score. Overlap with a preselected id is not recall. A longer candidate list is not a better question.
