# Social Media Topic Modeling

<a href="https://deepwiki.com/TLindenIII/Social-Media-Topic-Modeling">
  <img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki">
</a>
  
This repository contains a notebook-first workflow for comparing topic-specific X/Twitter corpora against codomain corpora built from users connected to those topics. The current project covers three subjects:

- Costco
- Keir Starmer
- NFL

The repo is organized around the streamlined working pipeline in `working/`, a local Streamlit dashboard for browsing outputs, exported `n8n` workflow JSON for collection scaffolding, and the final poster/presentation deliverables at the repository root.

## Repository Layout

- `working/data/`
  Canonical source inputs for each subject. These are the CSV files the pipeline ingests.
- `working/notebooks/`
  Sequential notebooks for collection scaffolding, ingest, preprocessing, sentiment, EDA, TF-IDF, topic modeling, and local LLM interpretation.
- `working/lib/pipeline.py`
  Shared pipeline logic used by both the notebooks and the dashboard.
- `working/outputs/`
  Standardized pipeline outputs, figures, manifests, and local-LLM artifacts.
- `working/dashboard/app.py`
  Streamlit dashboard for reviewing outputs and running the pipeline.
- `dashboard.sh`
  Convenience launcher for the local Streamlit dashboard.
- `n8n/`
  Exported `n8n` workflow JSON files used as collection references/scaffolds.
- `Conference Poster Landscape.pdf`
  Poster deliverable retained in the repo.
- `ASA SDSS 2026 Lightning Presentation.pptx`
  Presentation deck retained in the repo.

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

## What Is Tracked

The repository is set up to track:

- source data in `working/data/`
- pipeline code and notebooks in `working/`
- generated outputs in `working/outputs/`
- `n8n` workflow exports
- the poster PDF and PowerPoint deck at the repo root

The `presentation/` workspace is intentionally ignored by Git, along with local virtual environments, notebook checkpoints, and OS metadata files.

## Quick Start

Install the working environment dependencies:

```bash
python3 -m venv working/.venv
source working/.venv/bin/activate
pip install -r working/requirements.txt
```

## Running the Dashboard

From the project root:

```bash
source working/.venv/bin/activate
streamlit run working/dashboard/app.py
```

Or use the launcher script:

```bash
./dashboard.sh
```

If you prefer not to activate the environment first:

```bash
working/.venv/bin/streamlit run working/dashboard/app.py
```

## Notes

- The collection stage is scaffold-first. The repo stores exported `n8n` workflows, not a guaranteed runnable local collection stack.
- The codomain workflow is designed to support both direct comparison and adjacent-interest experimentation.
- Output artifact names follow the convention:

  `<subject_slug>__<dataset>__<artifact>.<ext>`
