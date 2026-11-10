# Dataset Maker Scripts

This folder contains scripts for turning arXiv mathematics papers into NaturalPRISM retrieval data. The workflow extracts mathematical statements and proof text from LaTeX, constructs partial-proof/query and cited-statement/answer pairs, adds same-paper hard negatives, and prepares training or evaluation files.

## Requirements

Use Python 3. The scripts use the standard library, plus:

- `regex` for recursive-pattern support in the LaTeX extractor;
- `tqdm` for progress bars in the extractor and evaluation-data preparation script.

Install the third-party dependencies in the environment used to run the scripts:

```text
pip install regex tqdm
```

There is no dependency lock file, so package versions are not pinned. The scripts expect the arXiv LaTeX source corpus and, when category filtering is used, an arXiv metadata snapshot to be prepared separately.

## Workflow

Configure the constants near the top of each script before running it. Unless a path is absolute, files are resolved relative to the current working directory.

1. Extract statements, proof environments, and proof sections from the papers' LaTeX source.
2. Combine proof-environment and proof-section records for use as proof input. Historically, this was done by manually copying/appending the JSONL records; there is no merge script in this folder. Keep one JSON object per line.
3. Match proof references to extracted statements and construct query/positive pairs.
4. Select other statements from the same paper as hard negatives.
5. Optionally filter pairs to one arXiv category.
6. Convert pairs and negatives to FlagEmbedding's training JSONL format, and/or construct retrieval evaluation files.

**Path note:** Default input and output filenames may not match between consecutive steps. When running the workflow, configure each input path to point to the intended output from the preceding step.

## Scripts

### `latex_files_to_thms_and_proofs_newcmd_replace.py`

Extracts theorem-like statements, proof environments, and sections whose headings indicate a proof. It also replaces supported document-specific LaTeX commands without positional arguments in the extracted text.

**Configure:**

- `ROOT_FOLDER`: root directory containing the extracted paper folders. The checked-in value is machine-specific and must be changed.
- `STATEMENTS_OUT`, `PROOFS_OUT`, `PROOF_SECTIONS_OUT`: output paths for the three JSONL record types.
- If needed, adjust extraction and context settings: `CONTEXT_GRAB_LEN`, `MIN_STMT_LENGTH`, `MAX_STMT_LENGTH`, `MAX_PROOF_LENGTH`, and `ATTACH_GAP_HIGH_CONF`.

The script searches below `ROOT_FOLDER` for directories containing `.tex` or `.ltx` files, then concatenates matching files directly in each directory in sorted filename order. The directory's relative path is used as its `paper_id`. This is source-level parsing rather than full TeX compilation; the script does not follow `\\input` or `\\include` dependencies, and custom-command replacement is not a general macro expander.

The outputs are three JSONL files: statement records with fields such as `paper_id`, `stmt_id`, `type`, `labels`, and `text`; proof records with `paper_id`, `proof_id`, `text`, and reference/attachment metadata; and proof-section records with compatible proof text and reference fields.

### `thms_and_proofs_to_query_pairs_v5.py`

Uses references in proof text to identify target statements and builds a query from the surrounding partial proof. It replaces the target citation with `[TARGET]`, other citations with `[REF]`, and filters ambiguous or low-content examples.

**Configure:**

- `STATEMENTS_IN`: path to the statement records from the extraction step.
- `PROOFS_IN`: path to the combined proof and proof-section records.
- `PAIRS_OUT`: output path for query/positive-pair records.
- Query construction settings: `QUERY_AFTER_MODE`, `QUERY_WINDOW_AFTER`, `AFTER_MAX_CHARS`, and `INCLUDE_SENTENCE_BOUNDARY`.
- Query/answer quality thresholds and filters, including `MIN_QUERY_LEN`, `MIN_POSITIVE_LEN`, minimum word/math counts, and reference limits.

The statement input needs `paper_id`, `stmt_id`, `labels`, `text`, and `type`. Proof records need `paper_id`, `proof_id`, and `text`; `attached_stmt_id` is used when available. Each output JSONL record contains the query, positive statement text and ID, source proof metadata, and invoked statement IDs. The default query mode extends past the target reference to a detected thought or sentence boundary, subject to a character limit.

### `query_answer_pairs_to_negatives.py`

For each query/positive pair, selects other statements from the same paper as hard negatives, excluding the positive and statements already invoked in the proof.

**Configure:**

- `STATEMENTS_IN`: path to the statement records used to find candidate negatives.
- `PAIRS_IN`: path to the query/positive-pair records.
- `NEGATIVES_OUT`: output path for hard-negative records.
- `MAX_NEGATIVES_PER_POSITIVE`, `RANDOM_SAMPLE_WHEN_CAPPED`, `SKIP_MALFORMED_NEGATIVES`, and `MIN_NEG_LEN`.

The output is JSONL with one record per pair, including its `pair_id` and a `negatives` list. If the eligible candidates exceed the configured cap, sampling is deterministic per pair ID.

### `filter_pairs_by_math_category.py`

Optionally retains pairs from papers assigned to a selected arXiv mathematics category.

**Configure:**

- `PAIRS_IN`: path to the query/positive-pair records to filter.
- `METADATA`: path to an arXiv metadata JSONL snapshot containing `id` and `categories` fields.
- `TARGET_CATEGORY`, `PREFIX_MATCH`, and `PAIRS_OUT`.

The output is a filtered JSONL pair file. The script normalizes arXiv IDs before matching them with `paper_id` values.

### `query_pairs_and_negatives_to_Flag_format.py`

Joins query/positive pairs with their hard negatives by `pair_id` and writes the contrastive training format used by FlagEmbedding.

**Configure:**

- `PAIRS_IN`: path to the selected query/positive-pair records.
- `NEGATIVES_IN`: path to the corresponding hard-negative records.
- `OUT`: output path for training JSONL.
- `MAX_PAIRS`, `RANDOM_SAMPLE`, and `SEED` if sampling is needed.
- `MAX_NEGATIVES_PER_PAIR`, `MIN_NEGATIVES`, and `QUERY_PROMPT`.

Input pair records need `pair_id`, `query`, and `positive_text`; negative records need matching `pair_id` values and a `negatives` array whose items contain `text`. Each output record contains `query`, `pos` (a list), and `neg` (a list). It also contains `prompt` when `QUERY_PROMPT` is set. Pair and negative inputs should come from compatible processing versions; records without a matching or sufficient set of negatives may be skipped.

This script prepares training data; it does not launch FlagEmbedding training. The training command and experiment parameters must be configured separately.

### `eval_set_maker.py`

Builds query, corpus, and relevance files for retrieval evaluation, excluding queries found in the listed training files.

**Configure:**

- `POSITIVE_PAIRS` and `NEGATIVE_PAIRS`: paths to the evaluation pair and hard-negative records.
- `TRAIN_FILES`: paths to all training JSONL files whose queries should be excluded from evaluation.
- `MAX_NUM`: maximum number of evaluation queries.
- `CORPUS_FILE`, `QUERY_FILE`, and `RELEVANCE_FILE`: output paths.

The outputs are JSONL files containing evaluation queries, unique candidate statements, and positive query/document relevance judgments. Negative statements are added to the candidate corpus but are not written as explicit zero-relevance qrels. Missing training files are skipped with a warning, so verify that every intended training file is present. This script prepares evaluation data; it does not encode or rank candidates or calculate retrieval metrics.

## Reproduction Notes

- The scripts do not download arXiv papers or metadata; prepare those inputs separately.
- This folder contains data-preparation scripts, not the FlagEmbedding training invocation or model-evaluation code.See more in the [FlagEmbedding repository](https://github.com/FlagOpen/FlagEmbedding).
- The JSONL files containing the extracted proofs and proof sections are merged manually by copying and appending their contents. This step is not captured by a script here.
