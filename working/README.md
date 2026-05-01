# Working Pipeline

This folder is the new working area for the simplified notebook pipeline.

## Notebook Order

1. `00_n8n_collection_scaffold.ipynb`
2. `01_domain_codomain_ingest.ipynb`
3. `02_preprocessing.ipynb`
4. `03_sentiment_analysis.ipynb`
5. `04_eda.ipynb`
6. `05_tfidf.ipynb`
7. `06_topic_modeling.ipynb`
8. `07_local_llm.ipynb`

Optional:

- `98_codomain_experiments.ipynb`
- `dashboard/app.py`

## Design

- One shared library lives in `working/lib/pipeline.py`.
- The notebooks stay thin and sequential.
- Each notebook can target any supported subject: `costco`, `keir_starmer`, or `nfl`.
- Outputs are written under `working/outputs/<subject_slug>/<stage>/`.

## Naming Convention

Artifacts use:

`<subject_slug>__<dataset>__<artifact>.<ext>`

Examples:

- `costco__domain__corpus.csv`
- `costco__codomain__preprocessed.parquet`
- `costco__domain__sentiment_counts.csv`
- `costco__codomain__tfidf_metrics_tidy.parquet`
- `costco__domain__topic_summaries.csv`

Figures are saved inside a `figures/` subfolder inside the stage directory using the same convention.

## Main Stages

- `00_collection`: n8n scaffold and workflow summaries
- `01_ingest`: standardized copies of existing domain/codomain source files
- `02_preprocess`: cleaned text, tokens, lemmas, stopword-removed text, QC summaries
- `03_sentiment`: RoBERTa sentiment scores and counts
- `04_eda`: summary tables and figures
- `05_tfidf`: aggregate TF-IDF, weighted IDF, polarity tables and figures
- `06_topic_modeling`: grid search, fitted LDA outputs, summaries, exemplars, optional pyLDAvis
- `07_local_llm`: JSONL prompt bundle plus optional local-model inference outputs

## Notes

- `00_n8n_collection_scaffold.ipynb` is intentionally scaffold-first. The repo only contains exported n8n workflow JSON, not a guaranteed runnable local automation setup.
- `98_codomain_experiments.ipynb` is outside the main pipeline. It is a staging area for testing codomain improvements without destabilizing the main sequence.
- `dashboard/app.py` is a Streamlit UI for browsing outputs and running the main pipeline with a progress bar.
