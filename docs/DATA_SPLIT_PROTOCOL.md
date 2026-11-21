# Document-level data split protocol

## Existing 200-question manifest

The existing `data/docvqa/manifest.jsonl` contains 200 questions selected from the DocVQA validation split. Fifty question IDs have already been used in fixed-resolution and energy experiments. Four document-image hashes were observed in both the 50-question development group and the 150-question candidate holdout group.

The script `scripts/freeze_validation_holdout.py` resolves each image path and computes SHA-256 over image bytes. Every sample whose image hash occurs among the 50 historical development questions is assigned to the development side. This moves all questions on those pages together and prevents page-level leakage.

Run from the repository root:

```bash
python scripts/freeze_validation_holdout.py
```

It reads, but never modifies:
- `data/docvqa/manifest.jsonl`
- `publication/energy_measurement_2026-11-18/energy_requests.csv`

It writes:
- `publication/rl_data_split_2026-11-21/development.jsonl`
- `publication/rl_data_split_2026-11-21/holdout.jsonl`
- `publication/rl_data_split_2026-11-21/split_summary.json`

The summary records input file hashes, sample IDs, question counts, image counts and the remaining overlap count. A successful run must report `shared_image_hashes: 0`.

## Important scope limitation

This freezes a page-disjoint split of the existing 200-question validation sample only. It does not create the larger policy-training corpus needed for RL and should not be described as the final benchmark evaluation by itself.

The next data-preparation stage should load the full DocVQA train and validation partitions, preserve OCR metadata and group by document image hash. The historically used validation pages must be excluded from the locked evaluation partition. Policy training examples should be sampled only from the training partition; validation data used for model selection must remain separate from the final locked evaluation set.

Dataset schema and split sizes are documented by
[pixparse/docvqa-single-page-questions](https://huggingface.co/datasets/pixparse/docvqa-single-page-questions).
The dataset card states that the dataset includes question IDs, question text, multiple answer strings, OCR results with line/word bounding boxes, and train/validation/test splits. It declares an MIT licence; any redistributed assets must still be checked against the upstream terms and attribution requirements.
