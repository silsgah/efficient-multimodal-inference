# Full DocVQA corpus build protocol

**Status:** Builder implemented and unit-tested; full corpus materialization not yet run.  
**Pinned source revision:** 13619c00b32b50729b6e50b4bb0aa4b355698f3f  
**Dataset:** pixparse/docvqa-single-page-questions

## 1. Verified source structure

The source was inspected through both the Hugging Face Dataset Viewer and actual Parquet shards at the pinned revision.

| Official split | Questions | Parquet shards | Parquet bytes reported by Dataset Viewer |
|---|---:|---:|---:|
| Train | 39,463 | 36 | 9,326,611,904 |
| Validation | 5,349 | 4 | 1,551,197,162 |
| Test | 5,188 | Not loaded | 1,521,245,171 |

The test split is deliberately not loaded by the builder.

The first train and validation shards were inspected. Their Arrow image field is struct<bytes: binary, path: string>. Question rows include:
- question_id, question, and answers (a list of accepted answer strings).
- data_split.
- ocr_results, containing page dimensions/orientation and line entries; each line has text, a polygon bounding box, and word records with text and bounding boxes.
- other_metadata, containing doc_id, source image filename, ucsf_document_id, and ucsf_document_page_no.

ucsf_document_id is the source-document grouping field. doc_id is page-level and is not used as a substitute for document identity.

## 2. Corpus construction

The builder is scripts/build_docvqa_corpus.py. It reads the source Parquet shards directly using PyArrow, with one-row batches and threaded decoding disabled. It performs separate sequential passes for question/OCR columns and the image column, writing question/OCR metadata to a temporary per-shard JSONL file before reopening the shard for the image-only pass. The streams are joined by question_id, then the metadata staging file is removed. This avoids materializing nested OCR and image-binary columns together. It does not call datasets.load_dataset() and does not load the official test split.

Each source image is decoded and canonicalized using the same operation used by the original pilot preparation: Pillow RGB conversion followed by PNG encoding. SHA-256 is computed over the canonical PNG bytes. The canonical PNG is stored once per unique image hash.

The question-level records retain every accepted answer in answers; answer is also populated with the first reference for compatibility with the existing benchmark loader. Page-level OCR and source metadata are stored once per unique image in pages.jsonl.

All 50 historical question IDs from the 18 November request CSV are matched to the existing pilot manifest. Their source images are decoded to RGB and compared using a pixel-content SHA-256 over image dimensions and raw RGB pixels. This is independent of PNG compression settings. A mismatch stops the build rather than silently changing the historical-development set.

When identical canonical page hashes occur under different ucsf_document_id values, those source-document keys are merged into one document group. This prevents a duplicate page from crossing partitions under distinct document IDs.

## 3. Partition assignments

The builder creates the following question manifests:

| Output | Source | Purpose |
|---|---|---|
| policy_train.jsonl | Official train | Policy learning |
| policy_validation.jsonl | Official train, held out by deterministic document-group hash | Policy selection and tuning |
| historical_development.jsonl | Official validation documents containing any of the 50 historical IDs | Historical development and audit |
| locked_evaluation.jsonl | Remaining official validation documents with no train overlap | Locked evaluation; do not use for tuning |
| cross_split_validation_excluded.jsonl | Official validation documents also present in official train, excluding historical documents | Explicit leakage exclusion |
| historical_overlap_train_excluded.jsonl | Official train rows whose document groups are in historical development | Explicit historical overlap exclusion |

Policy training/validation assignment uses SHA-256 of the stable document-group ID, with seed 42 and a requested 10% policy-validation fraction. All questions from a document group receive the same assignment.

Before output is finalized, the builder verifies zero shared document groups and zero shared image hashes between:
- policy training and policy validation;
- policy training and locked evaluation;
- policy validation and locked evaluation;
- historical development and policy training;
- historical development and policy validation.

The document catalog records all partition memberships. A source document can legitimately have more than one membership where one official split's rows are retained and another official split's duplicate rows are explicitly excluded.

## 4. Output layout

Default output directory:

data/docvqa/full_corpus_2026-11-21/

It contains:
- Six question JSONL partition manifests listed above.
- pages.jsonl: one record per unique canonical page, containing the image path/hash, OCR structure, page metadata, source split membership, and final document-group ID.
- documents.jsonl: one record per document group, including page/question counts and all partition memberships.
- images/<sha256>.png: one canonical RGB PNG per unique page.
- corpus_summary.json: pinned revision, input hashes, source shard hashes, partition counts, leakage checks, and output artifact hashes.

The build writes to a sibling .building directory and renames it to the final output path only after all count and leakage checks pass. Existing final or partial build directories are not overwritten automatically. A failed build preserves the partial directory for diagnosis; do not delete it before inspecting the error.

## 5. Reproduction command

Run from the repository's isolated data environment on the existing CPU Pod:

~~~bash
ROOT=/workspace/projects/efficient-multimodal-inference
WT=/tmp/emi-anls-audit-20261121
ENV=/workspace/venvs/docvqa-data
CACHE=/workspace/hf-cache

source "$ENV/bin/activate"

python "$WT/scripts/build_docvqa_corpus.py" \
  --revision 13619c00b32b50729b6e50b4bb0aa4b355698f3f \
  --repo-root "$ROOT" \
  --cache-dir "$CACHE" \
  --output-dir "$ROOT/data/docvqa/full_corpus_2026-11-21" \
  --policy-validation-fraction 0.1 \
  --seed 42
~~~

The minimal data dependencies are pinned separately in requirements/data.txt; inference/GPU requirements are unchanged.

## 6. Scope and limitations

This is a corpus-preparation artifact, not an experiment result. No policy has been trained, no new model inference has been run, and no energy/latency claims follow from corpus construction.

The locked_evaluation.jsonl must remain untouched by policy training, threshold selection, ROI design, and error-driven iteration. The official test split has not been loaded and remains outside this builder.

Stored PNG artifacts are identified by byte-level SHA-256. Historical image equality is checked with SHA-256 over decoded RGB pixels plus dimensions, so equivalent page pixels do not fail solely because of different PNG compression encoding. If a source image's RGB pixel digest does not match its historical pilot page, stop and investigate revision/image identity; do not weaken the check or silently remap historical samples.

The initial combined image/OCR reader was killed during the first training shard on the CPU Pod. It has been replaced by the sequential disk-staged two-pass reader. Before restarting the full build, the isolated one-row image-only decode check must pass. The original partial `.building` directory is preserved for diagnosis and must not be overwritten or deleted.
