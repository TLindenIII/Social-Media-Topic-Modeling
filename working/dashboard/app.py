from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


LIB_ROOT = Path(__file__).resolve().parents[1] / "lib"
if str(LIB_ROOT) not in sys.path:
    sys.path.insert(0, str(LIB_ROOT))

import pipeline as xp


st.set_page_config(page_title="X Sentiment Dashboard", layout="wide")


DATASETS = ["domain", "codomain"]
DATASET_LABELS = {"domain": "Domain", "codomain": "Codomain"}
STAGE_LABELS = {
    "01_ingest": "Ingest",
    "02_preprocess": "Preprocess",
    "03_sentiment": "Sentiment",
    "04_eda": "EDA",
    "05_tfidf": "TF-IDF",
    "06_topic_modeling": "Topics",
    "07_local_llm": "Local LLM",
}
STAGE_DESCRIPTIONS = {
    "01_ingest": "Legacy subject exports loaded into the standardized working structure.",
    "02_preprocess": "Text cleaning, language filtering, normalization, and token-level preparation.",
    "03_sentiment": "Tweet-level sentiment scores and label distributions.",
    "04_eda": "Exploratory charts covering corpus size, token usage, and descriptive distributions.",
    "05_tfidf": "Term-level signal views showing weighted distinctiveness and polarity.",
    "06_topic_modeling": "Topic-model diagnostics, prevalence patterns, and grid-search outputs.",
    "07_local_llm": "Interpretive layer built on the deterministic term and topic outputs.",
}
SENTIMENT_ORDER = ["negative", "neutral", "positive"]
SENTIMENT_LABELS = {"negative": "Negative", "neutral": "Neutral", "positive": "Positive"}
FIGURE_DESCRIPTIONS = {
    "04_eda": {
        "sentiment_distribution": "Sentiment balance across scored posts.",
        "tweets_per_user": "How concentrated posting volume is across authors.",
        "word_counts": "Distribution of post length after text normalization.",
        "top_tokens_raw_negative": "Most frequent raw tokens in negative posts.",
        "top_tokens_raw_neutral": "Most frequent raw tokens in neutral posts.",
        "top_tokens_raw_positive": "Most frequent raw tokens in positive posts.",
        "top_tokens_filtered_negative": "Display-filtered token view for negative posts.",
        "top_tokens_filtered_neutral": "Display-filtered token view for neutral posts.",
        "top_tokens_filtered_positive": "Display-filtered token view for positive posts.",
        "top_bigrams_negative": "Most common bigrams in negative posts.",
        "top_bigrams_neutral": "Most common bigrams in neutral posts.",
        "top_bigrams_positive": "Most common bigrams in positive posts.",
    },
    "05_tfidf": {
        "term_polarity": "Most polarized terms when positive and negative weighted IDF are contrasted.",
        "top_weighted_idf_negative": "Most distinctive weighted-IDF terms in negative posts.",
        "top_weighted_idf_neutral": "Most distinctive weighted-IDF terms in neutral posts.",
        "top_weighted_idf_positive": "Most distinctive weighted-IDF terms in positive posts.",
    },
    "06_topic_modeling": {
        "topic_grid_search": "Perplexity by candidate topic count. Lower values indicate the preferred model size.",
        "topic_prevalence": "Average topic prevalence by sentiment group.",
    },
    "98_experiments": {
        "filter_audit_rows": "How many rows survive each codomain filtering step.",
        "relevance_score_distribution": "Distribution of subject-link relevance before author qualification.",
        "domain_similarity_distribution": "How closely the filtered codomain still overlaps with domain language.",
    },
}


st.markdown(
    """
    <style>
    :root {
        --dashboard-muted: color-mix(in srgb, CanvasText 78%, Canvas 22%);
        --dashboard-subtle: color-mix(in srgb, CanvasText 62%, Canvas 38%);
        --dashboard-border: color-mix(in srgb, CanvasText 12%, Canvas 88%);
        --dashboard-divider: color-mix(in srgb, CanvasText 10%, Canvas 90%);
        --dashboard-surface: color-mix(in srgb, CanvasText 5%, Canvas 95%);
        --dashboard-surface-strong: color-mix(in srgb, CanvasText 7%, Canvas 93%);
        --dashboard-pill-bg: color-mix(in srgb, CanvasText 7%, Canvas 93%);
    }
    .block-container {
        max-width: 1480px;
        padding-top: 1.6rem;
        padding-bottom: 3rem;
    }
    .section-intro {
        color: var(--dashboard-muted);
        font-size: 0.98rem;
        line-height: 1.5;
        margin-bottom: 0.8rem;
    }
    .subtle {
        color: var(--dashboard-subtle);
        font-size: 0.9rem;
    }
    .insight-card {
        border: 1px solid var(--dashboard-border);
        border-radius: 12px;
        padding: 0.9rem 1rem 0.85rem 1rem;
        margin-bottom: 0.8rem;
        background: var(--dashboard-surface);
    }
    .insight-title {
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .insight-meta {
        color: var(--dashboard-subtle);
        font-size: 0.86rem;
        margin-bottom: 0.35rem;
    }
    .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin: 0.2rem 0 0.8rem 0;
    }
    .pill {
        border-radius: 999px;
        border: 1px solid var(--dashboard-border);
        padding: 0.25rem 0.55rem;
        font-size: 0.82rem;
        background: var(--dashboard-pill-bg);
    }
    .quality-high {
        border-left: 4px solid #c0392b;
        padding-left: 0.7rem;
        margin-bottom: 0.8rem;
        background: color-mix(in srgb, var(--dashboard-surface) 88%, transparent);
        border-radius: 8px;
        padding-top: 0.35rem;
        padding-bottom: 0.35rem;
    }
    .quality-medium {
        border-left: 4px solid #b9770e;
        padding-left: 0.7rem;
        margin-bottom: 0.8rem;
        background: color-mix(in srgb, var(--dashboard-surface) 88%, transparent);
        border-radius: 8px;
        padding-top: 0.35rem;
        padding-bottom: 0.35rem;
    }
    .quality-low {
        border-left: 4px solid #2471a3;
        padding-left: 0.7rem;
        margin-bottom: 0.8rem;
        background: color-mix(in srgb, var(--dashboard-surface) 88%, transparent);
        border-radius: 8px;
        padding-top: 0.35rem;
        padding-bottom: 0.35rem;
    }
    .readiness-list {
        border: 1px solid var(--dashboard-border);
        border-radius: 14px;
        background: var(--dashboard-surface-strong);
        overflow: hidden;
        margin-top: 0.25rem;
    }
    .readiness-row {
        display: block;
        padding: 0.58rem 0.92rem 0.56rem 0.92rem;
        border-bottom: 1px solid var(--dashboard-divider);
    }
    .readiness-row:last-child {
        border-bottom: none;
    }
    .readiness-main {
        min-width: 0;
    }
    .readiness-title {
        font-weight: 600;
        margin-bottom: 0.18rem;
    }
    .readiness-description {
        color: var(--dashboard-subtle);
        font-size: 0.84rem;
        line-height: 1.35;
    }
    .readiness-status {
        text-align: left;
    }
    .section-label {
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    [data-testid="stToolbar"] button[title="Deploy"],
    [data-testid="stToolbar"] a[title="Deploy"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_json(path_str: str) -> dict[str, Any] | None:
    path = Path(path_str)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_table(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.DataFrame()


def artifact(subject: str, stage: str, dataset: str | None, name: str, ext: str) -> Path:
    return xp.artifact_path(subject, stage, dataset, name, ext)


def llm_artifact_paths(subject: str, dataset: str) -> dict[str, Path]:
    return {
        "json": artifact(subject, "07_local_llm", dataset, "local_llm_output", "json"),
        "txt": artifact(subject, "07_local_llm", dataset, "local_llm_output", "txt"),
        "prompt": artifact(subject, "07_local_llm", dataset, "prompt", "txt"),
        "deterministic": artifact(subject, "07_local_llm", dataset, "deterministic_output", "json"),
        "topic_debug": artifact(subject, "07_local_llm", dataset, "topic_call_status", "json"),
    }


def llm_artifact_warning(paths: dict[str, Path]) -> str | None:
    json_path = paths["json"]
    raw_path = paths["txt"]
    if raw_path.exists() and raw_path.stat().st_size == 0:
        return "The latest LLM run returned an empty response. This page is still showing the previous structured JSON artifact."
    if raw_path.exists() and json_path.exists() and raw_path.stat().st_mtime > json_path.stat().st_mtime:
        return "The latest raw LLM output is newer than the saved JSON artifact. This page may still be showing the previous successful structured output."
    return None


def section_intro(text: str) -> None:
    st.markdown(f"<div class='section-intro'>{text}</div>", unsafe_allow_html=True)


def section_label(text: str) -> None:
    st.markdown(f"<div class='section-label'>{text}</div>", unsafe_allow_html=True)


def subtle_divider() -> None:
    st.markdown(
        "<div style='height:1px;background:var(--dashboard-divider);margin:0.95rem 0 1rem 0;'></div>",
        unsafe_allow_html=True,
    )


def pretty_label(value: str) -> str:
    return str(value).replace("_", " ").strip().title()


def severity_label(value: str) -> str:
    return pretty_label(str(value).lower())


def dataset_label(value: str) -> str:
    return DATASET_LABELS.get(value, pretty_label(value))


def format_float(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):,.{digits}f}"


def format_percent(value: Any, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:,.{digits}f}%"


def compact_int(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{int(value):,}"


def safe_round(df: pd.DataFrame, columns: Iterable[str], digits: int = 3) -> pd.DataFrame:
    output = df.copy()
    for column in columns:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce").round(digits)
    return output


def show_table(
    df: pd.DataFrame,
    rename_map: dict[str, str] | None = None,
    columns: list[str] | None = None,
    round_columns: list[str] | None = None,
    height: int | None = None,
) -> None:
    table = df.copy()
    if columns is not None:
        keep = [column for column in columns if column in table.columns]
        table = table.loc[:, keep]
    if round_columns:
        table = safe_round(table, round_columns)
    if rename_map:
        table = table.rename(columns=rename_map)
    kwargs = {"hide_index": True}
    if height is not None:
        kwargs["height"] = height
    try:
        st.dataframe(table, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(table, use_container_width=True, **kwargs)


def stretch_button(label: str, *, key: str | None = None, type: str = "secondary") -> bool:
    try:
        return st.button(label, key=key, type=type, width="stretch")
    except TypeError:
        return st.button(label, key=key, type=type, use_container_width=True)


def stretch_image(image: str | Path) -> None:
    try:
        st.image(str(image), width="stretch")
    except TypeError:
        st.image(str(image), use_container_width=True)


def segmented_control_compat(
    label: str,
    options: list[str],
    *,
    default: str,
    selection_mode: str,
    format_func=None,
    key: str,
):
    try:
        return st.segmented_control(
            label,
            options=options,
            default=default,
            selection_mode=selection_mode,
            format_func=format_func,
            key=key,
            width="stretch",
        )
    except TypeError:
        return st.segmented_control(
            label,
            options=options,
            default=default,
            selection_mode=selection_mode,
            format_func=format_func,
            key=key,
        )


def collapse_quality_flags(flags: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not flags:
        return []
    severity_rank = {"low": 0, "medium": 1, "high": 2}
    by_flag = {str(item.get("flag", "")).strip().lower(): dict(item) for item in flags}
    generic = by_flag.pop("generic_terms", None)
    noisy = by_flag.pop("noise_topics", None)
    if generic and noisy:
        generic_rank = severity_rank.get(str(generic.get("severity", "medium")).lower(), 1)
        noisy_rank = severity_rank.get(str(noisy.get("severity", "medium")).lower(), 1)
        severity_value = max(generic_rank, noisy_rank)
        severity = next((name for name, rank in severity_rank.items() if rank == severity_value), "medium")
        by_flag["low_signal_language"] = {
            "flag": "low_signal_language",
            "severity": severity,
            "explanation": (
                "Generic filler terms and noisy topic clusters are both reducing signal quality, so the corpus reads as low-density rather than clearly thematic."
            ),
        }
    else:
        if generic:
            by_flag["generic_terms"] = generic
        if noisy:
            by_flag["noise_topics"] = noisy
    return list(by_flag.values())


def render_pyldavis_html(path: Path, *, height: int = 980, scale: float = 0.84) -> None:
    raw_html = path.read_text(encoding="utf-8", errors="ignore")
    wrapped = f"""
    <style>
      html, body {{
        margin: 0;
        padding: 0;
        background: white;
        overflow: auto;
      }}
      .pyldavis-shell {{
        width: 100%;
        overflow: auto;
        background: white;
      }}
      .pyldavis-scale {{
        transform: scale({scale});
        transform-origin: top left;
        width: {100 / scale:.3f}%;
        min-height: {int(height / scale)}px;
        background: white;
      }}
    </style>
    <div class="pyldavis-shell">
      <div class="pyldavis-scale">
        {raw_html}
      </div>
    </div>
    """
    components.html(wrapped, height=height, scrolling=True)


def metric_row(items: list[tuple[str, str, str | None]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items):
        with col:
            st.metric(label=label, value=value, help=help_text)


def stage_status_table(subject: str) -> pd.DataFrame:
    status_df = xp.pipeline_stage_status(subject)
    pivot = (
        status_df.pivot(index="stage", columns="dataset", values="exists")
        .reindex(list(STAGE_LABELS))
        .reset_index()
        .rename(columns={"stage": "Stage", "domain": "Domain", "codomain": "Codomain"})
    )
    pivot["Stage"] = pivot["Stage"].map(lambda value: STAGE_LABELS.get(value, value))
    for column in ("Domain", "Codomain"):
        if column in pivot.columns:
            pivot[column] = pivot[column].map(lambda value: "Ready" if bool(value) else "Missing")
    return pivot


def stage_status_map(subject: str) -> dict[str, dict[str, bool]]:
    status_df = xp.pipeline_stage_status(subject)
    output: dict[str, dict[str, bool]] = {}
    for stage in STAGE_LABELS:
        subset = status_df.loc[status_df["stage"].eq(stage)]
        output[stage] = {
            dataset: bool(subset.loc[subset["dataset"].eq(dataset), "exists"].iloc[0])
            if not subset.loc[subset["dataset"].eq(dataset)].empty
            else False
            for dataset in DATASETS
        }
    return output


def missing_stage_labels(subject: str, dataset: str) -> list[str]:
    readiness = stage_status_map(subject)
    return [STAGE_LABELS[stage] for stage in STAGE_LABELS if not readiness[stage].get(dataset, False)]


def readiness_help_text(subject: str, dataset: str) -> str | None:
    missing = missing_stage_labels(subject, dataset)
    if not missing:
        return None
    return "Missing stages: " + ", ".join(missing)


def render_stage_readiness(subject: str) -> None:
    rows = ["<div class='readiness-list'>"]
    for stage in STAGE_LABELS:
        rows.append(
            "<div class='readiness-row'>"
            "<div class='readiness-main'>"
            f"<div class='readiness-title'>{STAGE_LABELS[stage]}</div>"
            f"<div class='readiness-description'>{STAGE_DESCRIPTIONS.get(stage, '')}</div>"
            "</div>"
            "</div>"
        )
    rows.append("</div>")
    st.markdown("".join(rows), unsafe_allow_html=True)


def figure_inventory(subject: str) -> dict[str, dict[str, list[Path]]]:
    inventory: dict[str, dict[str, list[Path]]] = {}
    for stage in STAGE_LABELS:
        stage_map: dict[str, list[Path]] = {}
        for dataset in DATASETS:
            paths = xp.gather_stage_figures(subject, stage, datasets=dataset)
            if paths:
                stage_map[dataset] = paths
        if stage_map:
            inventory[stage] = stage_map
    return inventory


def segmented_choice(
    label: str,
    options: list[str],
    key: str,
    format_func=None,
) -> str | None:
    if not options:
        return None
    default = st.session_state.get(key)
    if default not in options:
        default = options[0]
    selection = segmented_control_compat(
        label,
        options,
        default=default,
        selection_mode="single",
        format_func=format_func,
        key=f"{key}__widget",
    )
    if selection is None:
        selection = default
    st.session_state[key] = selection
    return selection


def figure_slug(path: Path) -> str:
    parts = path.stem.split("__", 2)
    return parts[2] if len(parts) == 3 else path.stem


def figure_explanation(stage: str, path: Path) -> str:
    slug = figure_slug(path)
    if slug in FIGURE_DESCRIPTIONS.get(stage, {}):
        return FIGURE_DESCRIPTIONS[stage][slug]
    return pretty_label(slug)


def sync_figure_index(total: int, key: str) -> int:
    index_key = f"{key}__index"
    current = int(st.session_state.get(index_key, 0))
    if total <= 0:
        st.session_state[index_key] = 0
        return 0
    current %= total
    control_cols = st.columns([1, 1, 6])
    with control_cols[0]:
        if stretch_button("←", key=f"{key}__prev"):
            current = (current - 1) % total
    with control_cols[1]:
        if stretch_button("→", key=f"{key}__next"):
            current = (current + 1) % total
    with control_cols[2]:
        st.caption(f"Figure {current + 1} of {total}")
    st.session_state[index_key] = current
    return current


def render_single_figure_browser(paths: list[Path], key: str, label: str, stage: str) -> None:
    if not paths:
        return
    index = sync_figure_index(len(paths), key)
    current = paths[index]
    st.caption(f"{label}: {figure_explanation(stage, current)}")
    stretch_image(current)


def render_synced_figure_browser(subject: str, stage: str, key: str) -> None:
    path_map = {dataset: xp.gather_stage_figures(subject, stage, datasets=dataset) for dataset in DATASETS}
    total = max((len(paths) for paths in path_map.values()), default=0)
    if total == 0:
        return
    index = sync_figure_index(total, key)
    reference_path = None
    for dataset in DATASETS:
        paths = path_map[dataset]
        if paths:
            reference_path = paths[index % len(paths)]
            break
    if reference_path is not None:
        st.caption(figure_explanation(stage, reference_path))
    cols = st.columns(2)
    for col, dataset in zip(cols, DATASETS):
        with col:
            st.markdown(f"**{dataset_label(dataset)}**")
            paths = path_map[dataset]
            if not paths:
                st.info("No figures available.")
                continue
            current = paths[index % len(paths)]
            st.caption(f"Figure {(index % len(paths)) + 1} of {len(paths)}")
            stretch_image(current)


def render_manifest_figure_browser(figures: list[Path], key: str) -> None:
    if not figures:
        return
    index = sync_figure_index(len(figures), key)
    current = figures[index]
    st.caption(figure_explanation("98_experiments", current))
    stretch_image(current)


def render_stage_figure_studio(subject: str) -> None:
    inventory = figure_inventory(subject)
    if not inventory:
        st.info("No saved figures are available for this subject yet.")
        return
    stage = segmented_choice(
        "Stage",
        list(inventory),
        key=f"{subject}__figure_stage",
        format_func=lambda value: STAGE_LABELS.get(value, value),
    )
    if stage is None:
        return
    stage_map = inventory[stage]
    dataset_options = []
    if len(stage_map) > 1:
        dataset_options.append("all")
    dataset_options.extend([dataset for dataset in DATASETS if dataset in stage_map])
    dataset_choice = segmented_choice(
        "Dataset view",
        dataset_options,
        key=f"{subject}__figure_dataset",
        format_func=lambda value: "Domain + Codomain" if value == "all" else dataset_label(value),
    )
    st.caption(STAGE_DESCRIPTIONS.get(stage, "Saved figures from this stage."))
    if dataset_choice == "all":
        render_synced_figure_browser(subject, stage, key=f"{subject}__{stage}__all")
    elif dataset_choice in stage_map:
        render_single_figure_browser(
            stage_map[dataset_choice],
            key=f"{subject}__{stage}__{dataset_choice}",
            label=dataset_label(dataset_choice),
            stage=stage,
        )


def run_pipeline_with_progress(
    subject: str,
    datasets: list[str],
    llm_backend: str,
    llm_model: str,
    llm_max_tokens: int,
) -> dict[str, Any]:
    config = xp.get_subject_config(subject)
    steps: list[tuple[str, str, Any]] = [("01_ingest", "all", lambda: xp.ingest_legacy_subject_data(config))]
    for dataset in datasets:
        steps.extend(
            [
                ("02_preprocess", dataset, lambda d=dataset: xp.preprocess_stage(config, d)),
                ("03_sentiment", dataset, lambda d=dataset: xp.sentiment_stage(config, d)),
                ("04_eda", dataset, lambda d=dataset: xp.eda_stage(config, d)),
                ("05_tfidf", dataset, lambda d=dataset: xp.tfidf_stage(config, d)),
                ("06_topic_modeling", dataset, lambda d=dataset: xp.topic_model_stage(config, d, n_topics="grid_min")),
                (
                    "07_local_llm",
                    dataset,
                    (
                        lambda d=dataset: xp.run_local_llm_stage(
                            config,
                            d,
                            backend=llm_backend,
                            model=llm_model,
                            max_tokens=llm_max_tokens,
                        )
                        if llm_backend != "prompt_only"
                        else xp.prepare_local_llm_stage(config, d)
                    ),
                ),
            ]
        )

    display_steps: list[dict[str, str]] = []
    if llm_backend == "ollama":
        display_steps.append({"label": f"Prepare Ollama ({llm_model})"})
    for stage, dataset, _ in steps:
        dataset_text = "All" if dataset == "all" else dataset_label(dataset)
        display_steps.append({"label": f"{STAGE_LABELS.get(stage, stage)} / {dataset_text}"})

    total = len(display_steps)
    progress = st.progress(0.0, text="Starting pipeline")
    status = st.empty()
    checklist = st.empty()
    stage_rows = []
    step_index = 0

    def render_checklist(current_label: str | None = None) -> None:
        lines = []
        for index, item in enumerate(display_steps):
            if index < step_index:
                lines.append(f"- [x] {item['label']}")
            elif index == step_index and current_label is not None:
                lines.append(f"- [ ] {item['label']} (running)")
            else:
                lines.append(f"- [ ] {item['label']}")
        checklist.markdown("\n".join(lines))

    if llm_backend == "ollama":
        current_label = display_steps[step_index]["label"]
        status.write(current_label)
        render_checklist(current_label)
        if not xp.start_ollama_server():
            raise RuntimeError("Ollama server is not available.")
        ensure_status = xp.ensure_ollama_model(llm_model)
        step_index += 1
        render_checklist(None)
        progress.progress(step_index / total, text=f"{current_label}: {ensure_status.get('message', 'ready')}")

    for stage, dataset, fn in steps:
        current_label = display_steps[step_index]["label"]
        status.write(current_label)
        render_checklist(current_label)
        result = fn()
        stage_rows.append(
            {
                "stage": stage,
                "dataset": dataset,
                "status": "ok",
                "path": result.get("summary_path")
                or result.get("preprocessed_path")
                or result.get("scores_path")
                or result.get("prompt_path")
                or result.get("result_path")
                or result.get("manifest_path"),
            }
        )
        step_index += 1
        render_checklist(None)
        progress.progress(step_index / total, text=f"Completed {current_label}")

    manifest = {
        "subject": config.slug,
        "datasets": datasets,
        "llm_backend": llm_backend,
        "llm_model": llm_model,
        "stages": stage_rows,
    }
    manifest_path = xp.subject_output_dir(config) / "pipeline_run_manifest.json"
    xp.write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    render_checklist(None)
    progress.progress(1.0, text="Pipeline complete")
    status.write(f"Saved run manifest to {manifest_path}")
    return manifest


def run_llm_only_with_progress(
    subject: str,
    datasets: list[str],
    llm_backend: str,
    llm_model: str,
    llm_max_tokens: int,
) -> dict[str, Any]:
    config = xp.get_subject_config(subject)
    steps: list[tuple[str, Any]] = []
    if llm_backend == "ollama":
        steps.append(("Prepare Ollama", None))
    for dataset in datasets:
        steps.append((f"Local LLM / {dataset_label(dataset)}", dataset))

    total = len(steps)
    progress = st.progress(0.0, text="Starting LLM-only run")
    status = st.empty()
    checklist = st.empty()
    stage_rows = []
    step_index = 0

    def render_checklist(current_label: str | None = None) -> None:
        lines = []
        for index, (label, _) in enumerate(steps):
            if index < step_index:
                lines.append(f"- [x] {label}")
            elif index == step_index and current_label is not None:
                lines.append(f"- [ ] {label} (running)")
            else:
                lines.append(f"- [ ] {label}")
        checklist.markdown("\n".join(lines))

    if llm_backend == "ollama":
        current_label = steps[step_index][0]
        status.write(current_label)
        render_checklist(current_label)
        if not xp.start_ollama_server():
            raise RuntimeError("Ollama server is not available.")
        ensure_status = xp.ensure_ollama_model(llm_model)
        step_index += 1
        render_checklist(None)
        progress.progress(step_index / total, text=f"{current_label}: {ensure_status.get('message', 'ready')}")

    for label, dataset in steps[step_index:]:
        status.write(label)
        render_checklist(label)
        if llm_backend == "prompt_only":
            result = xp.prepare_local_llm_stage(config, dataset)
            stage_rows.append(
                {
                    "stage": "07_local_llm",
                    "dataset": dataset,
                    "status": "prompt_only",
                    "path": result.get("prompt_path"),
                }
            )
        else:
            result = xp.run_local_llm_stage(
                config,
                dataset,
                backend=llm_backend,
                model=llm_model,
                max_tokens=llm_max_tokens,
            )
            stage_rows.append(
                {
                    "stage": "07_local_llm",
                    "dataset": dataset,
                    "status": result.get("status", "ok"),
                    "path": result.get("result_path") or result.get("prompt_path"),
                    "note": result.get("note"),
                    "topic_debug_path": result.get("topic_debug_path"),
                }
            )
        step_index += 1
        render_checklist(None)
        progress.progress(step_index / total, text=f"Completed {label}")

    manifest = {
        "mode": "llm_only",
        "subject": config.slug,
        "datasets": datasets,
        "llm_backend": llm_backend,
        "llm_model": llm_model,
        "stages": stage_rows,
    }
    manifest_path = xp.subject_output_dir(config) / "llm_only_run_manifest.json"
    xp.write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    render_checklist(None)
    progress.progress(1.0, text="LLM-only run complete")
    status.write(f"Saved run manifest to {manifest_path}")
    return manifest


def render_sentiment_panel(subject: str, dataset: str) -> None:
    counts = load_table(str(artifact(subject, "03_sentiment", dataset, "sentiment_counts", "csv")))
    st.markdown(f"### {dataset_label(dataset)}")
    if counts.empty:
        st.info("No sentiment results found.")
        return

    total = counts["count"].sum()
    metric_items = []
    for label in SENTIMENT_ORDER:
        row = counts.loc[counts["rob_label"].eq(label)]
        count = int(row["count"].iloc[0]) if not row.empty else 0
        share = count / total if total else 0.0
        metric_items.append((SENTIMENT_LABELS[label], f"{count:,}", f"{share:.1%} of scored posts"))
    metric_row(metric_items)
    st.caption(f"{total:,} scored posts.")


def render_terms_panel(subject: str, dataset: str) -> None:
    tfidf = load_table(str(artifact(subject, "05_tfidf", dataset, "tfidf_metrics_tidy", "parquet")))
    st.markdown(f"### {dataset_label(dataset)}")
    if tfidf.empty:
        st.info("No TF-IDF output found.")
        return

    sentiment_filter = segmented_choice(
        "Sentiment filter",
        ["all", *SENTIMENT_ORDER],
        key=f"{subject}__{dataset}__tfidf_filter",
        format_func=lambda value: "All" if value == "all" else SENTIMENT_LABELS.get(value, pretty_label(value)),
    )
    filtered = tfidf.copy()
    if sentiment_filter and sentiment_filter != "all":
        filtered = filtered.loc[filtered["sentiment"].eq(sentiment_filter)]
    filtered = filtered.sort_values(["fused_score", "iwdf", "term"], ascending=[False, False, True]).reset_index(drop=True)
    st.caption(f"{len(filtered):,} rows shown. Sorted by signal score, then weighted IDF.")
    show_table(
        filtered,
        columns=["term", "sentiment", "iwdf", "fused_score"],
        rename_map={
            "term": "Term",
            "sentiment": "Sentiment",
            "iwdf": "Weighted IDF",
            "fused_score": "Signal score",
        },
        round_columns=["iwdf", "fused_score"],
        height=520,
    )


def topic_display_rows(subject: str, dataset: str) -> list[dict[str, Any]]:
    summaries = load_table(str(artifact(subject, "06_topic_modeling", dataset, "topic_summaries", "csv")))
    if summaries.empty:
        return []
    payload, _ = llm_payload(subject, dataset)
    interpretation_map: dict[int, dict[str, Any]] = {}
    if payload is not None:
        for item in payload.get("topic_interpretations", []):
            try:
                topic_id = int(item.get("topic_id"))
            except Exception:
                continue
            interpretation_map[topic_id] = item

    rows = []
    for _, row in summaries.sort_values("topic_id").iterrows():
        topic_id = int(row["topic_id"])
        interpretation = interpretation_map.get(topic_id, {})
        rows.append(
            {
                "topic_id": topic_id,
                "caption": str(interpretation.get("label") or row.get("label") or f"Topic {topic_id + 1}").strip(),
                "top_words": str(row.get("top_words", "")).strip(),
                "exemplar_1": str(row.get("exemplar_1", "")).strip(),
                "exemplar_2": str(row.get("exemplar_2", "")).strip(),
                "exemplar_3": str(row.get("exemplar_3", "")).strip(),
                "theme_summary": str(interpretation.get("theme_summary", "")).strip(),
            }
        )
    return rows


def render_topics_panel(subject: str, dataset: str) -> None:
    topic_summary = load_json(str(artifact(subject, "06_topic_modeling", dataset, "topic_model_summary", "json")))
    topic_rows = topic_display_rows(subject, dataset)

    st.markdown(f"### {dataset_label(dataset)}")
    if topic_summary is None or not topic_rows:
        st.info("No topic model outputs found.")
        return

    metric_row(
        [
            ("Selected topics", compact_int(topic_summary.get("n_topics")), "Chosen from the lowest perplexity in the topic grid."),
        ]
    )
    st.caption("Topic count is selected per corpus from the minimum perplexity in the grid search.")

    for row in topic_rows:
        title = f"Topic {int(row['topic_id']) + 1}: {row['caption']}"
        with st.expander(title, expanded=False):
            st.markdown(f"**Top words**: {row['top_words']}")
            exemplar_lines = [row.get("exemplar_1", ""), row.get("exemplar_2", ""), row.get("exemplar_3", "")]
            exemplar_lines = [line for line in exemplar_lines if str(line).strip()]
            if exemplar_lines:
                st.markdown("**Exemplar posts**")
                st.markdown("\n".join(f"- {line}" for line in exemplar_lines))
            else:
                st.caption("No exemplar posts met the topic-strength threshold.")


def render_topic_details(subject: str) -> None:
    with st.expander("Model details", expanded=False):
        cols = st.columns(2)
        for col, dataset in zip(cols, DATASETS):
            with col:
                st.markdown(f"**{dataset_label(dataset)}**")
                summaries = load_table(str(artifact(subject, "06_topic_modeling", dataset, "topic_summaries", "csv")))
                grid = load_table(str(artifact(subject, "06_topic_modeling", dataset, "topic_grid_search", "csv")))
                if summaries.empty:
                    st.info("No topic model outputs found.")
                    continue
                show_table(
                    summaries,
                    columns=["topic_id", "label", "top_words", "exemplar_1", "exemplar_2", "exemplar_3"],
                    rename_map={
                        "topic_id": "Topic ID",
                        "label": "Topic label",
                        "top_words": "Top words",
                        "exemplar_1": "Example 1",
                        "exemplar_2": "Example 2",
                        "exemplar_3": "Example 3",
                    },
                )
                if not grid.empty:
                    st.markdown("**Grid search**")
                    show_table(
                        grid,
                        columns=["n_topics", "perplexity", "log_likelihood"],
                        rename_map={
                            "n_topics": "Topic count",
                            "perplexity": "Perplexity",
                            "log_likelihood": "Log likelihood",
                        },
                        round_columns=["perplexity", "log_likelihood"],
                    )


def render_shared_pyldavis(subject: str) -> None:
    available = []
    path_map: dict[str, Path] = {}
    for dataset in DATASETS:
        topic_summary = load_json(str(artifact(subject, "06_topic_modeling", dataset, "topic_model_summary", "json")))
        path = Path(str(topic_summary.get("pyldavis_path", ""))) if topic_summary and topic_summary.get("pyldavis_path") else None
        if path is not None and path.exists():
            available.append(dataset)
            path_map[dataset] = path
    if not available:
        st.info("No PyLDAvis outputs found.")
        return

    with st.expander("Interactive PyLDAvis", expanded=False):
        dataset = segmented_choice(
            "Corpus",
            available,
            key=f"{subject}__shared_pyldavis_dataset",
            format_func=dataset_label,
        )
        if dataset is None:
            return
        st.caption(f"{dataset_label(dataset)} interactive topic view.")
        render_pyldavis_html(path_map[dataset], height=980, scale=0.82)


def experiment_selector(option_names: list[str], manifest_details: dict[str, dict[str, Any]], captions: list[str], key: str) -> str:
    state_key = f"{key}__selected"
    if st.session_state.get(state_key) not in option_names:
        st.session_state[state_key] = option_names[0]
    columns = st.columns(3)
    for index, name in enumerate(option_names):
        with columns[index % 3]:
            label = pretty_label((manifest_details.get(name) or {}).get("variant", name))
            button_type = "primary" if st.session_state[state_key] == name else "secondary"
            if stretch_button(label, key=f"{key}__{index}", type=button_type):
                st.session_state[state_key] = name
            st.caption(captions[index])
    return st.session_state[state_key]


def render_quality_flags(flags: list[dict[str, Any]]) -> None:
    display_flags = collapse_quality_flags(flags)
    if not display_flags:
        st.info("No quality flags were raised.")
        return
    for flag in display_flags:
        severity = str(flag.get("severity", "medium")).lower()
        css_class = f"quality-{severity if severity in {'low', 'medium', 'high'} else 'medium'}"
        body = (
            f"<div class='{css_class}'><strong>{pretty_label(flag.get('flag', 'flag'))}</strong><br>"
            f"{flag.get('explanation', '')}</div>"
        )
        st.markdown(body, unsafe_allow_html=True)


def render_opportunities(items: list[dict[str, Any]]) -> None:
    if not items:
        st.info("No high-confidence opportunities were produced from this output.")
        return
    for item in items:
        topic_ids = item.get("evidence_topic_ids", [])
        topic_text = ", ".join(str(int(topic_id) + 1) for topic_id in topic_ids) if topic_ids else "None"
        meta = f"{pretty_label(item.get('activation_type', 'monitoring'))} | Confidence: {severity_label(item.get('confidence', 'medium'))} | Evidence topics: {topic_text}"
        st.markdown(
            f"<div class='insight-card'><div class='insight-title'>{item.get('opportunity', 'Opportunity')}</div><div class='insight-meta'>{meta}</div>{item.get('rationale', '')}</div>",
            unsafe_allow_html=True,
        )


def render_llm_topic_interpretations(subject: str, dataset: str) -> None:
    rows = topic_display_rows(subject, dataset)
    if not rows:
        st.info("No topic interpretations were available.")
        return
    for row in rows:
        title = f"Topic {row['topic_id'] + 1}: {row['caption']}"
        with st.expander(title, expanded=False):
            if row["theme_summary"]:
                st.write(row["theme_summary"])
            else:
                st.caption("No saved interpretation for this topic yet.")


def render_llm_panel(subject: str, dataset: str) -> None:
    paths = llm_artifact_paths(subject, dataset)
    payload = load_json(str(paths["json"]))

    st.markdown(f"### {dataset_label(dataset)}")

    if payload is not None:
        overview = payload.get("overview", {})
        if overview:
            st.write(overview.get("summary", ""))
            pills = [
                f"Mode: {pretty_label(overview.get('analysis_mode', 'unknown'))}",
                f"Off-topic risk: {severity_label(overview.get('off_topic_risk', 'medium'))}",
                f"Confidence: {severity_label(overview.get('confidence', 'medium'))}",
            ]
            pill_html = "".join(f"<span class='pill'>{pill}</span>" for pill in pills)
            st.markdown(f"<div class='pill-row'>{pill_html}</div>", unsafe_allow_html=True)

        takeaways = payload.get("takeaways") or payload.get("poster_takeaways") or []
        if takeaways:
            st.markdown("**Takeaways**")
            st.markdown("\n".join(f"- {item}" for item in takeaways))

        st.markdown("**Quality flags**")
        render_quality_flags(payload.get("quality_flags", []))

        st.markdown("**Strategic opportunities**")
        render_opportunities(payload.get("strategic_opportunities", []))

        st.markdown("**Topic interpretations**")
        render_llm_topic_interpretations(subject, dataset)

        with st.expander("Artifact details", expanded=False):
            status_rows = [
                {"Artifact": name.upper(), "Exists": path.exists(), "Path": str(path)}
                for name, path in paths.items()
            ]
            show_table(pd.DataFrame(status_rows))
            st.json(payload)
        return

    raw_text = paths["txt"].read_text(encoding="utf-8") if paths["txt"].exists() else ""
    prompt_text = paths["prompt"].read_text(encoding="utf-8") if paths["prompt"].exists() else ""
    deterministic = load_json(str(paths["deterministic"]))

    if raw_text:
        st.warning("Structured JSON was not found. Showing the raw model response instead.")
        st.code(raw_text[:8000], language="json")
    elif prompt_text or deterministic is not None:
        st.info("No LLM response file is available yet. Showing the prepared bundle instead.")
    else:
        st.info("No local LLM artifacts found for this corpus yet.")

    with st.expander("Artifact details", expanded=False):
        status_rows = [
            {"Artifact": name.upper(), "Exists": path.exists(), "Path": str(path)}
            for name, path in paths.items()
        ]
        show_table(pd.DataFrame(status_rows))
        if deterministic is not None:
            st.markdown("**Deterministic bundle**")
            st.json(deterministic)
        if prompt_text:
            st.markdown("**Prompt preview**")
            st.code(prompt_text[:4000])


def llm_payload(subject: str, dataset: str) -> tuple[dict[str, Any] | None, dict[str, Path]]:
    paths = llm_artifact_paths(subject, dataset)
    return load_json(str(paths["json"])), paths


def summarize_experiment_variant(subject: str, manifest: dict[str, Any]) -> dict[str, Any]:
    corpus = load_table(str(manifest.get("corpus_csv_path", "")))
    audit = load_table(str(manifest.get("audit_path", "")))
    config = xp.get_subject_config(subject)
    defaults = xp.codomain_filter_defaults(config)
    include_terms = {term.lower() for term in defaults.get("include_terms", [])}
    exclude_terms = {term.lower() for term in defaults.get("exclude_terms", [])}

    if corpus.empty:
        return {
            "rows": 0,
            "authors": 0,
            "retention_rate": 0.0,
            "top_author_share": None,
            "url_share": None,
            "retweet_share": None,
            "top_phrases": [],
            "diagnostic_notes": ["The selected variant produced no retained posts."],
        }

    baseline_rows = int(audit.iloc[0]["rows"]) if not audit.empty else len(corpus)
    author_counts = corpus["Author ID"].value_counts() if "Author ID" in corpus.columns else pd.Series(dtype=int)
    top_author_share = float(author_counts.max() / len(corpus)) if not author_counts.empty else None
    content = corpus["Content"].fillna("").astype(str)
    url_share = float(content.str.contains(r"https?://|www\.", regex=True).mean()) if len(corpus) else None
    retweet_share = float(content.str.startswith("RT ").mean()) if len(corpus) else None

    top_phrases: list[str] = []
    try:
        from sklearn.feature_extraction.text import CountVectorizer

        prepared, _, _ = xp.preprocess_corpus(corpus, drop_duplicate_content=False)
        texts = prepared["clean_nostop"].fillna("").astype(str).str.strip()
        texts = texts.loc[texts.str.len().gt(0)].reset_index(drop=True)
        if not texts.empty:
            min_df = 2 if len(texts) >= 8 else 1
            vectorizer = CountVectorizer(
                stop_words=None,
                lowercase=False,
                ngram_range=(1, 2),
                min_df=min_df,
                max_df=0.9,
                token_pattern=r"(?u)\b\w\w+\b",
            )
            matrix = vectorizer.fit_transform(texts)
            vocab = vectorizer.get_feature_names_out()
            counts = matrix.sum(axis=0).A1
            ranked = sorted(zip(vocab, counts), key=lambda item: item[1], reverse=True)
            for term, _count in ranked:
                parts = term.lower().split()
                if any(part in include_terms or part in exclude_terms for part in parts):
                    continue
                top_phrases.append(term)
                if len(top_phrases) >= 8:
                    break
    except Exception:
        top_phrases = []

    notes = []
    retention_rate = float(len(corpus) / baseline_rows) if baseline_rows else 0.0
    if len(corpus) < 40:
        notes.append("The retained codomain is thin, so individual users can shape the result disproportionately.")
    if top_author_share is not None and top_author_share >= 0.30:
        notes.append("A small number of authors dominate the kept corpus, which weakens confidence in any broad audience inference.")
    if url_share is not None and url_share >= 0.60:
        notes.append("A high share of retained posts still contain links, which often signals publisher or amplification behavior rather than organic adjacent interests.")
    if not top_phrases:
        notes.append("The remaining text does not yet produce stable non-subject phrases, which suggests the methodology is still over-filtering or retaining heterogeneous authors.")

    return {
        "rows": len(corpus),
        "authors": int(corpus["Author ID"].nunique()) if "Author ID" in corpus.columns else None,
        "retention_rate": retention_rate,
        "top_author_share": top_author_share,
        "url_share": url_share,
        "retweet_share": retweet_share,
        "top_phrases": top_phrases,
        "diagnostic_notes": notes,
    }


def render_pipeline_page(subject: str) -> None:
    st.subheader("Pipeline overview")
    section_intro("Review stage readiness, inspect saved visual outputs, and launch the end-to-end pipeline from one place.")

    status_df = xp.pipeline_stage_status(subject)
    completion = status_df.groupby("dataset")["exists"].sum().to_dict()
    expected = status_df.groupby("dataset")["exists"].size().to_dict()
    metric_row(
        [
            (
                "Domain stages ready",
                f"{completion.get('domain', 0):.0f} / {expected.get('domain', 0):.0f}",
                readiness_help_text(subject, "domain"),
            ),
            (
                "Codomain stages ready",
                f"{completion.get('codomain', 0):.0f} / {expected.get('codomain', 0):.0f}",
                readiness_help_text(subject, "codomain"),
            ),
        ]
    )
    subtle_divider()

    section_label("Stage readiness")
    render_stage_readiness(subject)

    subtle_divider()
    section_label("Figure studio")
    render_stage_figure_studio(subject)

    if st.session_state.get("last_run_manifest"):
        with st.expander("Last run manifest", expanded=False):
            st.json(st.session_state["last_run_manifest"])


def render_sentiment_page(subject: str) -> None:
    st.subheader("Sentiment")
    section_intro("Compare sentiment balance across the two corpora, then review the synced EDA figures that explain how those distributions were formed.")
    cols = st.columns(2)
    for col, dataset in zip(cols, DATASETS):
        with col:
            render_sentiment_panel(subject, dataset)
    subtle_divider()
    if any(xp.gather_stage_figures(subject, "04_eda", datasets=dataset) for dataset in DATASETS):
        section_label("EDA figures")
        render_synced_figure_browser(subject, "04_eda", key=f"{subject}__eda_sync")


def render_terms_page(subject: str) -> None:
    st.subheader("TF-IDF")
    section_intro("Review the full weighted-IDF term tables for each corpus, then use the synced figures to spot which sentiment-linked terms are most distinctive.")
    with st.expander("Full term table", expanded=False):
        cols = st.columns(2)
        for col, dataset in zip(cols, DATASETS):
            with col:
                render_terms_panel(subject, dataset)
    subtle_divider()
    if any(xp.gather_stage_figures(subject, "05_tfidf", datasets=dataset) for dataset in DATASETS):
        section_label("TF-IDF figures")
        render_synced_figure_browser(subject, "05_tfidf", key=f"{subject}__tfidf_sync")


def render_topics_page(subject: str) -> None:
    st.subheader("Topics")
    section_intro("Topic counts are selected automatically per corpus from the lowest-perplexity point in the grid search, then summarized with exemplars and prevalence visuals.")
    cols = st.columns(2)
    for col, dataset in zip(cols, DATASETS):
        with col:
            render_topics_panel(subject, dataset)
    subtle_divider()
    section_label("Model details")
    render_topic_details(subject)
    subtle_divider()
    section_label("Interactive PyLDAvis")
    render_shared_pyldavis(subject)
    subtle_divider()
    if any(xp.gather_stage_figures(subject, "06_topic_modeling", datasets=dataset) for dataset in DATASETS):
        section_label("Topic figures")
        render_synced_figure_browser(subject, "06_topic_modeling", key=f"{subject}__topic_sync")


def render_llm_page(subject: str) -> None:
    st.subheader("Local LLM")
    section_intro("The local model translates the deterministic pipeline outputs into a concise analyst readout. It should be read as interpretation layered on top of the saved topic and term evidence, not a replacement for it.")
    payloads = {dataset: llm_payload(subject, dataset)[0] for dataset in DATASETS}
    paths = {dataset: llm_artifact_paths(subject, dataset) for dataset in DATASETS}

    header_cols = st.columns(2)
    for col, dataset in zip(header_cols, DATASETS):
        with col:
            st.markdown(f"### {dataset_label(dataset)}")
            warning = llm_artifact_warning(paths[dataset])
            if warning:
                st.warning(warning, icon="⚠️")

    subtle_divider()
    section_label("Overview")
    cols = st.columns(2)
    for col, dataset in zip(cols, DATASETS):
        with col:
            payload = payloads[dataset]
            if payload is None:
                st.info("No structured output available.")
                continue
            overview = payload.get("overview", {})
            if overview:
                st.write(overview.get("summary", ""))
                pills = [
                    f"Mode: {pretty_label(overview.get('analysis_mode', 'unknown'))}",
                    f"Off-topic risk: {severity_label(overview.get('off_topic_risk', 'medium'))}",
                    f"Confidence: {severity_label(overview.get('confidence', 'medium'))}",
                ]
                pill_html = "".join(f"<span class='pill'>{pill}</span>" for pill in pills)
                st.markdown(f"<div class='pill-row'>{pill_html}</div>", unsafe_allow_html=True)

    subtle_divider()
    section_label("Takeaways")
    cols = st.columns(2)
    for col, dataset in zip(cols, DATASETS):
        with col:
            payload = payloads[dataset]
            takeaways = payload.get("takeaways") or payload.get("poster_takeaways") or [] if payload else []
            if takeaways:
                st.markdown("\n".join(f"- {item}" for item in takeaways))
            else:
                st.info("No takeaways available.")

    subtle_divider()
    section_label("Quality flags")
    cols = st.columns(2)
    for col, dataset in zip(cols, DATASETS):
        with col:
            payload = payloads[dataset]
            render_quality_flags(payload.get("quality_flags", []) if payload else [])

    subtle_divider()
    section_label("Strategic opportunities")
    cols = st.columns(2)
    for col, dataset in zip(cols, DATASETS):
        with col:
            payload = payloads[dataset]
            render_opportunities(payload.get("strategic_opportunities", []) if payload else [])

    subtle_divider()
    section_label("Topic interpretations")
    cols = st.columns(2)
    for col, dataset in zip(cols, DATASETS):
        with col:
            render_llm_topic_interpretations(subject, dataset)

    with st.expander("Artifact details", expanded=False):
        cols = st.columns(2)
        for col, dataset in zip(cols, DATASETS):
            with col:
                st.markdown(f"**{dataset_label(dataset)}**")
                status_rows = [
                    {"Artifact": name.upper(), "Exists": path.exists(), "Path": str(path)}
                    for name, path in paths[dataset].items()
                ]
                show_table(pd.DataFrame(status_rows))
                payload = payloads[dataset]
                if payload is not None:
                    st.json(payload)
                else:
                    raw_text = paths[dataset]["txt"].read_text(encoding="utf-8") if paths[dataset]["txt"].exists() else ""
                    prompt_text = paths[dataset]["prompt"].read_text(encoding="utf-8") if paths[dataset]["prompt"].exists() else ""
                    deterministic = load_json(str(paths[dataset]["deterministic"]))
                    if raw_text:
                        st.code(raw_text[:4000], language="json")
                    elif deterministic is not None:
                        st.json(deterministic)
                    elif prompt_text:
                        st.code(prompt_text[:3000])


def render_experiments(subject: str) -> None:
    st.subheader("Codomain lab")
    section_intro("This workspace is for codomain methodology testing. Use it to judge whether a filter stack is surfacing genuine adjacent interests or just trading one form of noise for another.")
    st.caption("Assessment on this page is deterministic dashboard logic, not a local LLM judgment.")
    config = xp.get_subject_config(subject)
    manifests = sorted((xp.stage_dir(config, "98_experiments")).glob(f"{config.slug}__*__variant_manifest.json"))
    if not manifests:
        st.info("No codomain experiment manifests found.")
        return

    manifest_options = {path.name: path for path in manifests}
    manifest_details = {name: load_json(str(path)) for name, path in manifest_options.items()}
    option_names = list(manifest_options)
    captions = []
    for name in option_names:
        info = manifest_details.get(name) or {}
        filters = info.get("filters", {})
        tokens = []
        tokens.append("keep RTs" if not filters.get("drop_retweets") else "drop RTs")
        if filters.get("drop_direct_subject_posts_from_kept_users"):
            tokens.append("remove direct-subject posts")
        if filters.get("drop_link_heavy_noise"):
            tokens.append("trim link-heavy noise")
        if filters.get("max_author_url_share") is not None:
            tokens.append(f"author URL share <= {int(float(filters['max_author_url_share']) * 100)}%")
        captions.append(", ".join(tokens))

    section_label("Experiment variants")
    selected = experiment_selector(option_names, manifest_details, captions, key=f"experiment_{subject}")
    manifest = manifest_details.get(selected)
    if manifest is None:
        st.info("Could not read the selected manifest.")
        return

    audit = load_table(str(manifest["audit_path"]))
    author_summary = load_table(str(manifest["author_summary_path"]))
    kept = load_table(str(manifest["kept_sample_path"]))
    dropped = load_table(str(manifest["dropped_sample_path"]))
    variant_summary = summarize_experiment_variant(subject, manifest)

    final_stage = audit.iloc[-1].to_dict() if not audit.empty else {}
    filters = manifest.get("filters", {})
    metric_row(
        [
            ("Final rows", compact_int(final_stage.get("rows")), None),
            ("Unique authors", compact_int(final_stage.get("unique_authors")), None),
            ("Retweets dropped", "Yes" if filters.get("drop_retweets") else "No", None),
            ("Direct subject posts removed", "Yes" if filters.get("drop_direct_subject_posts_from_kept_users") else "No", None),
        ]
    )
    subtle_divider()
    metric_row(
        [
            ("Retention rate", f"{variant_summary['retention_rate']:.1%}", "Share of baseline codomain posts still present after all filters."),
            (
                "Top author share",
                format_percent(variant_summary["top_author_share"]),
                "How much of the retained corpus comes from the single most represented author.",
            ),
            (
                "URL share",
                format_percent(variant_summary["url_share"]),
                "How much of the retained corpus still contains a link.",
            ),
            (
                "Retweet share",
                format_percent(variant_summary["retweet_share"]),
                "How much of the retained corpus is pure amplification rather than original posting.",
            ),
        ]
    )

    section_label("Signal readout")
    if variant_summary["top_phrases"]:
        st.markdown("**Top adjacent phrases**")
        st.markdown("\n".join(f"- `{phrase}`" for phrase in variant_summary["top_phrases"]))
    else:
        st.info("No stable adjacent phrases were detected from the retained corpus.")

    if variant_summary["diagnostic_notes"]:
        st.markdown("**Assessment**")
        st.markdown("\n".join(f"- {note}" for note in variant_summary["diagnostic_notes"]))

    subtle_divider()
    if not audit.empty:
        section_label("Filter audit")
        show_table(
            audit,
            columns=["stage", "rows", "unique_authors", "mean_include_matches", "mean_exclude_matches", "mean_relevance_score"],
            rename_map={
                "stage": "Stage",
                "rows": "Rows",
                "unique_authors": "Authors",
                "mean_include_matches": "Mean include matches",
                "mean_exclude_matches": "Mean exclude matches",
                "mean_relevance_score": "Mean relevance score",
            },
            round_columns=["mean_include_matches", "mean_exclude_matches", "mean_relevance_score"],
        )

    subtle_divider()
    if not author_summary.empty:
        section_label("Author summary")
        show_table(
            author_summary.head(20),
            columns=["author_username", "posts", "subject_posts", "focus_share", "url_share", "include_hits", "exclude_hits"],
            rename_map={
                "author_username": "Author",
                "posts": "Posts",
                "subject_posts": "Subject-linked posts",
                "focus_share": "Subject post share",
                "url_share": "URL share",
                "include_hits": "Matched subject terms",
                "exclude_hits": "Matched exclude terms",
            },
            round_columns=["focus_share", "url_share"],
        )

    subtle_divider()
    section_label("Example posts")
    sample_tabs = st.tabs(["Kept sample", "Dropped sample"])
    with sample_tabs[0]:
        if kept.empty:
            st.info("No kept sample available.")
        else:
            show_table(
                kept,
                columns=["Author Username", "include_match_count", "exclude_match_count", "topic_relevance_score", "url_count", "Content"],
                rename_map={
                    "Author Username": "Author",
                    "include_match_count": "Include matches",
                    "exclude_match_count": "Exclude matches",
                    "topic_relevance_score": "Relevance score",
                    "url_count": "URLs",
                    "Content": "Post text",
                },
            )
    with sample_tabs[1]:
        if dropped.empty:
            st.info("No dropped sample available.")
        else:
            show_table(
                dropped,
                columns=["Author Username", "include_match_count", "exclude_match_count", "topic_relevance_score", "url_count", "Content"],
                rename_map={
                    "Author Username": "Author",
                    "include_match_count": "Include matches",
                    "exclude_match_count": "Exclude matches",
                    "topic_relevance_score": "Relevance score",
                    "url_count": "URLs",
                    "Content": "Post text",
                },
            )

    figures = [Path(path) for path in manifest.get("figures", []) if Path(path).exists()]
    if figures:
        subtle_divider()
        section_label("Experiment figures")
        render_manifest_figure_browser(figures, key=f"{subject}__experiment_figures")

    with st.expander("Manifest details", expanded=False):
        st.json(manifest)


subject_lookup = {
    row["display_name"]: row["slug"]
    for _, row in xp.list_subjects()[["display_name", "slug"]].iterrows()
}


st.title("X Sentiment Dashboard")
st.caption("A working review surface for domain and codomain analysis, local LLM summaries, and codomain filter experiments.")

with st.sidebar:
    display_name = st.selectbox("Subject", list(subject_lookup))
    subject = subject_lookup[display_name]
    selected_datasets = st.multiselect("Datasets to run", DATASETS, default=DATASETS)
    llm_backend = st.selectbox("LLM backend", ["ollama", "prompt_only"], index=0)
    llm_model = st.text_input("LLM model", value="isotnek/qwen3.5:9B-Unsloth-UD-Q4_K_XL")
    llm_max_tokens = st.number_input("Max tokens", min_value=512, max_value=8192, value=3072, step=256)
    run_pipeline = stretch_button("Run main pipeline", key="run_main_pipeline", type="secondary")
    run_llm_only = stretch_button("Run LLM only", key="run_llm_only", type="secondary")

if run_pipeline:
    st.session_state["last_run_manifest"] = run_pipeline_with_progress(
        subject=subject,
        datasets=selected_datasets or DATASETS,
        llm_backend=llm_backend,
        llm_model=llm_model,
        llm_max_tokens=int(llm_max_tokens),
    )
elif run_llm_only:
    st.session_state["last_run_manifest"] = run_llm_only_with_progress(
        subject=subject,
        datasets=selected_datasets or DATASETS,
        llm_backend=llm_backend,
        llm_model=llm_model,
        llm_max_tokens=int(llm_max_tokens),
    )

overview_tab, sentiment_tab, terms_tab, topics_tab, llm_tab, experiments_tab = st.tabs(
    ["Pipeline", "Sentiment", "TF-IDF", "Topics", "LLM", "Codomain Lab"]
)

with overview_tab:
    render_pipeline_page(subject)

with sentiment_tab:
    render_sentiment_page(subject)

with terms_tab:
    render_terms_page(subject)

with topics_tab:
    render_topics_page(subject)

with llm_tab:
    render_llm_page(subject)

with experiments_tab:
    render_experiments(subject)
