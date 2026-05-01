from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKING_ROOT = PROJECT_ROOT / "working"
NOTEBOOK_ROOT = WORKING_ROOT / "notebooks"


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip() + "\n",
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip() + "\n",
    }


def figure_preview_cell(stage: str) -> dict:
    return code_cell(
        f"""
        preview_subject = subject if "subject" in globals() else xp.get_subject_config(SUBJECT)
        _ = xp.show_stage_figures(preview_subject, "{stage}")
        """
    )


BOOTSTRAP = """
from pathlib import Path
import sys
import pandas as pd
from IPython.display import display

cwd = Path.cwd().resolve()
candidates = [cwd, *list(cwd.parents[:3])]
PROJECT_ROOT = next((path for path in candidates if (path / "working" / "lib" / "pipeline.py").exists()), cwd)
LIB_ROOT = PROJECT_ROOT / "working" / "lib"
if str(LIB_ROOT) not in sys.path:
    sys.path.insert(0, str(LIB_ROOT))

import pipeline as xp

xp.ensure_working_tree()
print("Project root:", PROJECT_ROOT)
display(xp.list_subjects())
"""


NOTEBOOK_SPECS = [
    {
        "filename": "00_n8n_collection_scaffold.ipynb",
        "cells": [
            markdown_cell(
                """
                # 00 n8n Collection Scaffold

                This notebook is intentionally scaffold-first.

                Use it to:

                - inspect the saved n8n workflow exports
                - record the subject query and date window you want
                - write a reproducible collection scaffold into `working/outputs/<subject>/00_collection/`

                It does **not** assume the exported web workflows can run unchanged in local `n8n`.
                """
            ),
            code_cell(BOOTSTRAP),
            code_cell(
                """
                SUBJECT = "costco"
                CUSTOM_QUERY = '@costco OR "costco" OR #costco'
                SINCE = None
                UNTIL = None
                RUN_LOCAL_N8N = False
                """
            ),
            code_cell(
                """
                subject = xp.get_subject_config(SUBJECT)
                workflow_df = pd.DataFrame(xp.load_n8n_workflow_summaries())
                display(workflow_df)

                scaffold = xp.save_collection_scaffold(
                    subject,
                    custom_query=CUSTOM_QUERY,
                    since=SINCE,
                    until=UNTIL,
                    run_local_n8n=RUN_LOCAL_N8N,
                )
                scaffold
                """
            ),
            markdown_cell(
                """
                The next notebook does not depend on this stage.

                If local n8n execution is still blocked, continue by standardizing the source corpora in `01_domain_codomain_ingest.ipynb`.
                """
            ),
            figure_preview_cell("00_collection"),
        ],
    },
    {
        "filename": "01_domain_codomain_ingest.ipynb",
        "cells": [
            markdown_cell(
                """
                # 01 Domain and Codomain Ingest

                This notebook copies the existing repo corpora into a clean, standardized output structure.
                """
            ),
            code_cell(BOOTSTRAP),
            code_cell(
                """
                SUBJECT = "costco"
                COPY_SEED_FILES = True
                """
            ),
            code_cell(
                """
                subject = xp.get_subject_config(SUBJECT)
                manifest = xp.ingest_legacy_subject_data(subject, copy_seed_files=COPY_SEED_FILES)
                manifest
                """
            ),
            code_cell(
                """
                domain_df = xp.load_stage_table(subject, "01_ingest", "domain", "corpus", "csv")
                codomain_df = xp.load_stage_table(subject, "01_ingest", "codomain", "corpus", "csv")

                display(
                    pd.DataFrame(
                        [
                            xp.summarize_frame(domain_df, "domain"),
                            xp.summarize_frame(codomain_df, "codomain"),
                        ]
                    )
                )
                display(domain_df.head(2))
                """
            ),
            figure_preview_cell("01_ingest"),
        ],
    },
    {
        "filename": "02_preprocessing.ipynb",
        "cells": [
            markdown_cell(
                """
                # 02 Preprocessing

                Runs shared preprocessing for the standardized domain and codomain corpora.
                """
            ),
            code_cell(BOOTSTRAP),
            code_cell(
                """
                SUBJECT = "costco"
                DATASETS = ["domain", "codomain"]
                EXTRA_STOPWORDS = []
                DROP_DUPLICATE_CONTENT = True
                TARGET_LANGUAGE = "en"
                """
            ),
            code_cell(
                """
                subject = xp.get_subject_config(SUBJECT)
                results = []
                for dataset in DATASETS:
                    results.append(
                        xp.preprocess_stage(
                            subject,
                            dataset,
                            extra_stopwords=EXTRA_STOPWORDS,
                            drop_duplicate_content=DROP_DUPLICATE_CONTENT,
                            target_language=TARGET_LANGUAGE,
                        )
                    )
                display(pd.DataFrame(results))
                """
            ),
            code_cell(
                """
                preview_dataset = DATASETS[0]
                preview_df = xp.load_stage_table(subject, "02_preprocess", preview_dataset, "preprocessed", "parquet")
                display(preview_df.head(2))
                """
            ),
            figure_preview_cell("02_preprocess"),
        ],
    },
    {
        "filename": "03_sentiment_analysis.ipynb",
        "cells": [
            markdown_cell(
                """
                # 03 Sentiment Analysis

                Scores preprocessed tweets with CardiffNLP RoBERTa.
                """
            ),
            code_cell(BOOTSTRAP),
            code_cell(
                """
                SUBJECT = "costco"
                DATASETS = ["domain", "codomain"]
                MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
                BATCH_SIZE = 32
                MAX_LENGTH = 280
                """
            ),
            code_cell(
                """
                subject = xp.get_subject_config(SUBJECT)
                results = []
                for dataset in DATASETS:
                    results.append(
                        xp.sentiment_stage(
                            subject,
                            dataset,
                            model_name=MODEL_NAME,
                            batch_size=BATCH_SIZE,
                            max_length=MAX_LENGTH,
                        )
                    )
                display(pd.DataFrame(results))
                """
            ),
            code_cell(
                """
                preview_dataset = DATASETS[0]
                counts_df = xp.load_stage_table(subject, "03_sentiment", preview_dataset, "sentiment_counts", "csv")
                display(counts_df)
                """
            ),
            figure_preview_cell("03_sentiment"),
        ],
    },
    {
        "filename": "04_eda.ipynb",
        "cells": [
            markdown_cell(
                """
                # 04 EDA

                Generates a reduced, repeatable EDA set from the sentiment outputs.

                Notes:

                - `top_tokens` is still the raw QC-style token view.
                - additional filtered-token and bigram views are saved for a more interpretable read.
                - this is display-only; it does not change downstream TF-IDF or topic-modeling inputs.
                """
            ),
            code_cell(BOOTSTRAP),
            code_cell(
                """
                SUBJECT = "costco"
                DATASETS = ["domain", "codomain"]
                TOP_K = 20
                """
            ),
            code_cell(
                """
                subject = xp.get_subject_config(SUBJECT)
                results = [xp.eda_stage(subject, dataset, top_k=TOP_K) for dataset in DATASETS]
                display(pd.DataFrame(results))
                """
            ),
            markdown_cell(
                """
                Figures are saved to:

                `working/outputs/<subject>/<04_eda>/figures/`
                """
            ),
            figure_preview_cell("04_eda"),
        ],
    },
    {
        "filename": "05_tfidf.ipynb",
        "cells": [
            markdown_cell(
                """
                # 05 TF-IDF

                Fits the simplified TF-IDF stage and writes tidy/wide term metrics.
                """
            ),
            code_cell(BOOTSTRAP),
            code_cell(
                """
                SUBJECT = "costco"
                DATASETS = ["domain", "codomain"]
                MIN_DF = 5
                MAX_DF = 0.80
                TOP_K = 20
                """
            ),
            code_cell(
                """
                subject = xp.get_subject_config(SUBJECT)
                results = [
                    xp.tfidf_stage(
                        subject,
                        dataset,
                        min_df=MIN_DF,
                        max_df=MAX_DF,
                        top_k=TOP_K,
                    )
                    for dataset in DATASETS
                ]
                display(pd.DataFrame(results))
                """
            ),
            code_cell(
                """
                preview_dataset = DATASETS[0]
                top_terms_df = xp.load_stage_table(subject, "05_tfidf", preview_dataset, "top_terms", "csv")
                display(top_terms_df.head(10))
                """
            ),
            figure_preview_cell("05_tfidf"),
        ],
    },
    {
        "filename": "06_topic_modeling.ipynb",
        "cells": [
            markdown_cell(
                """
                # 06 Topic Modeling

                Runs a simplified LDA workflow, writes topic summaries/exemplars, and attempts `pyLDAvis` if available.
                """
            ),
            code_cell(BOOTSTRAP),
            code_cell(
                """
                SUBJECT = "costco"
                DATASETS = ["domain", "codomain"]
                TOPIC_GRID = [4, 6, 8, 10]
                N_TOPICS = "grid_min"
                N_TOP_WORDS = 12
                MIN_DF = 10
                MAX_DF = 0.80
                MIN_TOPIC_PROP = 0.15
                """
            ),
            code_cell(
                """
                subject = xp.get_subject_config(SUBJECT)
                results = []
                for dataset in DATASETS:
                    results.append(
                        xp.topic_model_stage(
                            subject,
                            dataset,
                            topic_grid=TOPIC_GRID,
                            n_topics=N_TOPICS,
                            n_top_words=N_TOP_WORDS,
                            min_df=MIN_DF,
                            max_df=MAX_DF,
                            min_topic_prop=MIN_TOPIC_PROP,
                        )
                    )
                display(pd.DataFrame(results))
                """
            ),
            code_cell(
                """
                for preview_dataset in DATASETS:
                    print(f"\\nTopic summaries: {preview_dataset}")
                    summary_df = xp.load_stage_table(subject, "06_topic_modeling", preview_dataset, "topic_summaries", "csv")
                    grid_df = xp.load_stage_table(subject, "06_topic_modeling", preview_dataset, "topic_grid_search", "csv")
                    display(grid_df)
                    display(summary_df.head(10))
                """
            ),
            figure_preview_cell("06_topic_modeling"),
        ],
    },
    {
        "filename": "07_local_llm.ipynb",
        "cells": [
            markdown_cell(
                """
                # 07 Local LLM

                This notebook builds the prompt bundle from TF-IDF and topic outputs, then optionally runs a local model.
                """
            ),
            code_cell(BOOTSTRAP),
            markdown_cell(
                """
                ## Local Model Options

                Default path: **Ollama + `isotnek/qwen3.5:9B-Unsloth-UD-Q4_K_XL`**

                Why this is the default:

                - easiest for most people to install
                - available across macOS, Linux, and Windows
                - stronger model quality while still staying practical for local use

                Optional upgrade:

                - `qwen2.5:14b-instruct` if you want to stay on a smaller official Ollama model family
                """
            ),
            code_cell(
                """
                display(xp.local_llm_options())
                print("Detected backends:", xp.available_local_llm_backends())
                """
            ),
            code_cell(
                """
                SUBJECT = "costco"
                DATASETS = ["domain", "codomain"]
                TOP_TERMS = 12
                TOP_TOPICS = None
                BACKEND = "ollama"
                MODEL = "isotnek/qwen3.5:9B-Unsloth-UD-Q4_K_XL"
                RUN_LLM = True
                MAX_TOKENS = 3072
                AUTO_INSTALL_OLLAMA = False
                """
            ),
            markdown_cell(
                """
                ## Ollama Setup

                This cell detects the OS, checks whether Ollama is already installed, and only attempts the matching install path if `AUTO_INSTALL_OLLAMA = True`.
                """
            ),
            code_cell(
                """
                import shutil
                import subprocess

                system = xp.detect_os()
                print("Detected OS:", system)
                print("Ollama installed:", xp.ollama_installed())
                print("Model storage:", xp.ollama_model_storage_hint()["model_dir"])

                if not xp.ollama_installed():
                    if not AUTO_INSTALL_OLLAMA:
                        print("Ollama is not installed. Set AUTO_INSTALL_OLLAMA=True to let this cell install it automatically.")
                    else:
                        if system == "darwin":
                            cmd = ["brew", "install", "ollama"]
                        elif system == "linux":
                            cmd = ["bash", "-lc", "curl -fsSL https://ollama.com/install.sh | sh"]
                        elif system == "windows":
                            cmd = ["winget", "install", "--id", "Ollama.Ollama"]
                        else:
                            raise RuntimeError(f"Unsupported OS for automated install: {system}")

                        print("Running install command:", " ".join(cmd))
                        subprocess.run(cmd, check=True)

                print("Ollama installed after setup:", xp.ollama_installed())
                """
            ),
            code_cell(
                """
                server_ok = xp.start_ollama_server(timeout_seconds=20)
                print("Ollama server running:", server_ok)
                """
            ),
            code_cell(
                """
                model_status = xp.ensure_ollama_model(MODEL)
                model_status
                """
            ),
            code_cell(
                """
                if xp.ollama_installed() and xp.ollama_server_running():
                    smoke = xp.run_local_llm("Reply with READY only.", backend="ollama", model=MODEL, max_tokens=32)
                    print(smoke)
                else:
                    print("Skipping smoke test because Ollama is not ready.")
                """
            ),
            code_cell(
                """
                subject = xp.get_subject_config(SUBJECT)
                bundles = {}
                bundle_rows = []
                for dataset in DATASETS:
                    bundles[dataset] = xp.prepare_local_llm_stage(
                        subject,
                        dataset,
                        top_terms=TOP_TERMS,
                        top_topics=TOP_TOPICS,
                    )
                    bundle_rows.append(
                        {
                            "dataset": dataset,
                            "prompt_path": bundles[dataset]["prompt_path"],
                            "deterministic_output_path": bundles[dataset]["deterministic_output_path"],
                        }
                    )
                display(pd.DataFrame(bundle_rows))
                """
            ),
            code_cell(
                """
                ollama_ready = xp.ollama_installed() and xp.ollama_server_running() if BACKEND == "ollama" else True

                results = []
                for dataset in DATASETS:
                    if RUN_LLM and ollama_ready:
                        result = xp.run_local_llm_stage(
                            subject,
                            dataset,
                            backend=BACKEND,
                            model=MODEL,
                            max_tokens=MAX_TOKENS,
                        )
                    elif RUN_LLM and BACKEND == "ollama" and not ollama_ready:
                        result = {
                            "dataset": dataset,
                            "backend": BACKEND,
                            "model": MODEL,
                            "note": "Skipping LLM run because Ollama is not installed or the server is not running.",
                            "prompt_path": bundles[dataset]["prompt_path"],
                        }
                    else:
                        result = {
                            "dataset": dataset,
                            "backend": BACKEND,
                            "model": MODEL,
                            "note": "Prompt bundle created. Set RUN_LLM=True to execute a local backend.",
                            "prompt_path": bundles[dataset]["prompt_path"],
                        }
                    results.append(result)

                display(
                    pd.DataFrame(
                        [
                            {
                                key: value
                                for key, value in result.items()
                                if key != "parsed_output"
                            }
                            for result in results
                        ]
                    )
                )
                """
            ),
            code_cell(
                """
                import json
                from pathlib import Path

                for result in results:
                    print(f"\\nLLM output: {result.get('dataset', 'unknown')}")
                    if result.get("parsed_output") is not None:
                        display(result["parsed_output"])
                    elif result.get("result_path"):
                        output_path = Path(result["result_path"])
                        print(output_path.read_text(encoding="utf-8")[:4000])
                    else:
                        print("No LLM output was produced. If that was intentional, set RUN_LLM=False. Otherwise check the setup cells above.")
                """
            ),
            markdown_cell(
                """
                ## Optional Shutdown

                You do not have to stop the Ollama server after the notebook finishes, but you can if you want to free memory.
                """
            ),
            code_cell(
                """
                # Optional
                # xp.stop_ollama_server()
                """
            ),
            figure_preview_cell("07_local_llm"),
        ],
    },
    {
        "filename": "98_codomain_experiments.ipynb",
        "cells": [
            markdown_cell(
                """
                # 98 Codomain Experiments

                This notebook is optional and intentionally outside the main sequence.

                Use it to prototype codomain changes before promoting anything into the main pipeline.
                """
            ),
            code_cell(BOOTSTRAP),
            code_cell(
                """
                display(xp.codomain_improvement_candidates())
                """
            ),
            code_cell(
                """
                SUBJECT = "costco"
                VARIANT_NAME = "costco_adjacent_interest_v1"
                DROP_RETWEETS = False
                DROP_REPLIES = False
                RUN_COMPARISON_GRID = True
                defaults = xp.codomain_filter_defaults(SUBJECT)
                INCLUDE_TERMS = defaults.get("include_terms", [])
                EXCLUDE_TERMS = defaults.get("exclude_terms", [])
                MAX_POSTS_PER_USER = defaults.get("max_posts_per_user", 12)
                MIN_INCLUDE_MATCHES = defaults.get("min_include_matches", 1)
                MAX_EXCLUDE_MATCHES = defaults.get("max_exclude_matches", 0)
                MIN_RELEVANCE_SCORE = defaults.get("min_relevance_score", 1)
                MIN_AUTHOR_INCLUDE_HITS = max(2, defaults.get("min_author_include_hits", 2))
                MIN_AUTHOR_FOCUS_SHARE = defaults.get("min_author_focus_share", 0.20)
                MAX_AUTHOR_URL_SHARE = 0.75
                DROP_DIRECT_SUBJECT_POSTS = True
                DROP_LINK_HEAVY_NOISE = True
                MAX_URLS = 1
                DROP_CASHTAGS = True
                MIN_ALPHA_WORDS_WITH_URL = 6
                DOMAIN_SIMILARITY_QUANTILE = None
                DOMAIN_SIMILARITY_MIN = None
                """
            ),
            code_cell(
                """
                subject = xp.get_subject_config(SUBJECT)
                codomain_df = xp.load_stage_table(subject, "01_ingest", "codomain", "corpus", "csv")

                if RUN_COMPARISON_GRID:
                    comparison_specs = [
                        {
                            "variant_name": f"{SUBJECT}_qualified_users_all_posts",
                            "drop_direct_subject_posts_from_kept_users": False,
                            "drop_link_heavy_noise": False,
                            "max_author_url_share": None,
                            "domain_similarity_quantile": None,
                            "domain_similarity_min": None,
                        },
                        {
                            "variant_name": f"{SUBJECT}_adjacent_interest_posts",
                            "drop_direct_subject_posts_from_kept_users": True,
                            "drop_link_heavy_noise": False,
                            "max_author_url_share": None,
                            "domain_similarity_quantile": None,
                            "domain_similarity_min": None,
                        },
                        {
                            "variant_name": f"{SUBJECT}_adjacent_interest_clean",
                            "drop_direct_subject_posts_from_kept_users": True,
                            "drop_link_heavy_noise": True,
                            "max_author_url_share": MAX_AUTHOR_URL_SHARE,
                            "domain_similarity_quantile": None,
                            "domain_similarity_min": None,
                        },
                    ]
                    comparison_runs = []
                    comparison_rows = []
                    for spec in comparison_specs:
                        result = xp.build_codomain_variant(
                            subject,
                            codomain_df,
                            variant_name=spec["variant_name"],
                            include_terms=INCLUDE_TERMS,
                            exclude_terms=EXCLUDE_TERMS,
                            drop_retweets=DROP_RETWEETS,
                            drop_replies=DROP_REPLIES,
                            max_posts_per_user=MAX_POSTS_PER_USER,
                            min_include_matches=MIN_INCLUDE_MATCHES,
                            max_exclude_matches=MAX_EXCLUDE_MATCHES,
                            min_relevance_score=MIN_RELEVANCE_SCORE,
                            min_author_include_hits=MIN_AUTHOR_INCLUDE_HITS,
                            min_author_focus_share=MIN_AUTHOR_FOCUS_SHARE,
                            max_author_url_share=spec["max_author_url_share"],
                            drop_direct_subject_posts_from_kept_users=spec["drop_direct_subject_posts_from_kept_users"],
                            drop_link_heavy_noise=spec["drop_link_heavy_noise"],
                            max_urls=MAX_URLS,
                            drop_cashtags=DROP_CASHTAGS,
                            min_alpha_words_with_url=MIN_ALPHA_WORDS_WITH_URL,
                            domain_similarity_quantile=spec["domain_similarity_quantile"],
                            domain_similarity_min=spec["domain_similarity_min"],
                        )
                        comparison_runs.append(result)
                        final_stage = result["audit"].iloc[-1].to_dict()
                        comparison_rows.append(
                            {
                                "variant": result["variant"],
                                "rows": final_stage.get("rows"),
                                "unique_authors": final_stage.get("unique_authors"),
                                "drop_direct_subject_posts": spec["drop_direct_subject_posts_from_kept_users"],
                                "drop_link_heavy_noise": spec["drop_link_heavy_noise"],
                                "max_author_url_share": spec["max_author_url_share"],
                                "mean_domain_similarity": final_stage.get("mean_domain_similarity"),
                                "threshold": result.get("similarity_summary", {}).get("threshold"),
                                "manifest_path": result["manifest_path"],
                            }
                        )
                    display(pd.DataFrame(comparison_rows))
                else:
                    comparison_runs = []
                    print("Comparison grid skipped.")
                """
            ),
            code_cell(
                """
                experiment = xp.build_codomain_variant(
                    subject,
                    codomain_df,
                    variant_name=VARIANT_NAME,
                    include_terms=INCLUDE_TERMS,
                    exclude_terms=EXCLUDE_TERMS,
                    drop_retweets=DROP_RETWEETS,
                    drop_replies=DROP_REPLIES,
                    max_posts_per_user=MAX_POSTS_PER_USER,
                    min_include_matches=MIN_INCLUDE_MATCHES,
                    max_exclude_matches=MAX_EXCLUDE_MATCHES,
                    min_relevance_score=MIN_RELEVANCE_SCORE,
                    min_author_include_hits=MIN_AUTHOR_INCLUDE_HITS,
                    min_author_focus_share=MIN_AUTHOR_FOCUS_SHARE,
                    max_author_url_share=MAX_AUTHOR_URL_SHARE,
                    drop_direct_subject_posts_from_kept_users=DROP_DIRECT_SUBJECT_POSTS,
                    drop_link_heavy_noise=DROP_LINK_HEAVY_NOISE,
                    max_urls=MAX_URLS,
                    drop_cashtags=DROP_CASHTAGS,
                    min_alpha_words_with_url=MIN_ALPHA_WORDS_WITH_URL,
                    domain_similarity_quantile=DOMAIN_SIMILARITY_QUANTILE,
                    domain_similarity_min=DOMAIN_SIMILARITY_MIN,
                )
                experiment
                """
            ),
            code_cell(
                """
                display(experiment["audit"])
                print("Filters")
                display(pd.DataFrame([experiment.get("filters", {})]))
                print("Similarity summary")
                display(pd.DataFrame([experiment.get("similarity_summary", {})]))
                print("Noise filter meta")
                display(pd.DataFrame([experiment.get("noise_filter_meta", {})]))
                display(experiment["author_summary"].head(20))
                """
            ),
            code_cell(
                """
                print("Kept sample")
                display(experiment["kept_sample"])
                print("Dropped sample")
                display(experiment["dropped_sample"])
                """
            ),
            figure_preview_cell("98_experiments"),
        ],
    },
]


def notebook_payload(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_ROOT.mkdir(parents=True, exist_ok=True)
    for spec in NOTEBOOK_SPECS:
        payload = notebook_payload(spec["cells"])
        output_path = NOTEBOOK_ROOT / spec["filename"]
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
