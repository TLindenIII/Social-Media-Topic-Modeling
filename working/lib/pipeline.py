from __future__ import annotations

import ast
import csv
import json
import math
import platform
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKING_ROOT = PROJECT_ROOT / "working"
NOTEBOOK_ROOT = WORKING_ROOT / "notebooks"
OUTPUT_ROOT = WORKING_ROOT / "outputs"
DATA_ROOT = WORKING_ROOT / "data"
SCRIPT_ROOT = WORKING_ROOT / "scripts"
N8N_ROOT = PROJECT_ROOT / "n8n"

MAIN_STAGES = [
    "00_collection",
    "01_ingest",
    "02_preprocess",
    "03_sentiment",
    "04_eda",
    "05_tfidf",
    "06_topic_modeling",
    "07_local_llm",
    "98_experiments",
]

SENTIMENT_ORDER = ["negative", "neutral", "positive"]
SENTIMENT_COLORS = {
    "negative": "#c44e52",
    "neutral": "#7f7f7f",
    "positive": "#55a868",
}

DISPLAY_STOP_TOKENS = {
    "also",
    "back",
    "best",
    "come",
    "could",
    "day",
    "even",
    "get",
    "go",
    "going",
    "good",
    "great",
    "know",
    "look",
    "lot",
    "make",
    "many",
    "much",
    "need",
    "new",
    "one",
    "people",
    "really",
    "right",
    "say",
    "see",
    "still",
    "take",
    "thing",
    "think",
    "time",
    "use",
    "want",
    "way",
    "well",
    "would",
}

_MODEL_CACHE: dict[str, tuple[Any, Any, str]] = {}


@dataclass(frozen=True)
class SubjectConfig:
    slug: str
    display_name: str
    legacy_root_name: str
    domain_dir_name: str
    codomain_dir_name: str
    legacy_prefix: str
    aliases: tuple[str, ...]

    @property
    def legacy_root(self) -> Path:
        return PROJECT_ROOT / self.legacy_root_name

    @property
    def domain_dir(self) -> Path:
        return self.legacy_root / self.domain_dir_name

    @property
    def codomain_dir(self) -> Path:
        return self.legacy_root / self.codomain_dir_name

    @property
    def domain_corpus_path(self) -> Path:
        return self.domain_dir / f"{self.legacy_prefix}_d_corpus.csv"

    @property
    def codomain_corpus_path(self) -> Path:
        return self.codomain_dir / f"{self.legacy_prefix}_cd_corpus.csv"

    @property
    def codomain_seed_pre_path(self) -> Path:
        return self.codomain_dir / f"{self.legacy_prefix}_cd_sent_pre.csv"

    @property
    def codomain_seed_post_path(self) -> Path:
        return self.codomain_dir / f"{self.legacy_prefix}_cd_sent_post.csv"


SUBJECTS: dict[str, SubjectConfig] = {
    "costco": SubjectConfig(
        slug="costco",
        display_name="Costco",
        legacy_root_name="working/data/costco",
        domain_dir_name="domain",
        codomain_dir_name="codomain",
        legacy_prefix="costco",
        aliases=("costco",),
    ),
    "keir_starmer": SubjectConfig(
        slug="keir_starmer",
        display_name="Keir Starmer (UK PM)",
        legacy_root_name="working/data/keir_starmer",
        domain_dir_name="domain",
        codomain_dir_name="codomain",
        legacy_prefix="ks",
        aliases=("keir_starmer", "keir starmer", "keir starmer (uk pm)", "ks"),
    ),
    "nfl": SubjectConfig(
        slug="nfl",
        display_name="NFL",
        legacy_root_name="working/data/nfl",
        domain_dir_name="domain",
        codomain_dir_name="codomain",
        legacy_prefix="nfl",
        aliases=("nfl",),
    ),
}

SUBJECT_ALIASES = {
    alias.lower(): config.slug
    for config in SUBJECTS.values()
    for alias in config.aliases
}

CODOMAIN_FILTER_DEFAULTS = {
    "costco": {
        "include_terms": [
            "costco",
            "kirkland",
            "kirkland signature",
            "costco membership",
            "costco gas",
            "costco food court",
            "costco hot dog",
            "rotisserie chicken",
            "warehouse club",
        ],
        "exclude_terms": [
            "stock",
            "stocks",
            "market",
            "trade",
            "trading",
            "bitcoin",
            "btc",
            "eth",
            "crypto",
            "analyst",
            "earnings",
            "nasdaq",
            "dow",
            "s&p",
            "sp500",
            "trump",
            "democrat",
            "republican",
            "tesla",
            "ai",
        ],
        "max_posts_per_user": 12,
        "min_include_matches": 1,
        "max_exclude_matches": 0,
        "min_relevance_score": 1,
        "min_author_include_hits": 1,
        "min_author_focus_share": 0.10,
    },
    "keir_starmer": {
        "include_terms": [
            "keir",
            "starmer",
            "labour",
            "labour party",
            "prime minister",
            "pm",
            "downing street",
            "uk government",
            "britain",
            "british politics",
        ],
        "exclude_terms": [
            "stock",
            "bitcoin",
            "crypto",
            "nasdaq",
            "s&p",
            "sp500",
        ],
        "max_posts_per_user": 15,
        "min_include_matches": 1,
        "max_exclude_matches": 0,
        "min_relevance_score": 1,
        "min_author_include_hits": 2,
        "min_author_focus_share": 0.15,
    },
    "nfl": {
        "include_terms": [
            "nfl",
            "football",
            "quarterback",
            "touchdown",
            "super bowl",
            "playoffs",
            "draft",
            "coach",
            "team",
            "game",
            "season",
        ],
        "exclude_terms": [
            "stock",
            "bitcoin",
            "crypto",
            "nasdaq",
            "s&p",
            "sp500",
        ],
        "max_posts_per_user": 15,
        "min_include_matches": 1,
        "max_exclude_matches": 0,
        "min_relevance_score": 1,
        "min_author_include_hits": 2,
        "min_author_focus_share": 0.15,
    },
}

LOCAL_LLM_OPTIONS = [
    {
        "backend": "ollama",
        "recommended_models": [
            "isotnek/qwen3.5:9B-Unsloth-UD-Q4_K_XL",
            "qwen2.5:7b-instruct",
            "qwen2.5:14b-instruct",
            "llama3.1:8b-instruct",
            "mistral-small:24b-instruct-2501",
        ],
        "notes": "Simplest cross-platform path. The default here is an Ollama community build of Qwen3.5-9B; qwen2.5:7b-instruct remains the smaller fallback.",
    },
    {
        "backend": "mlx_lm",
        "recommended_models": [
            "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "mlx-community/Llama-3.1-8B-Instruct-4bit",
            "mlx-community/Mistral-Small-24B-Instruct-2501-4bit",
        ],
        "notes": "Best local performance path on Apple Silicon if you want an MLX-native stack.",
    },
    {
        "backend": "llama_cpp",
        "recommended_models": [
            "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
            "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        ],
        "notes": "Good when you want GGUF portability and fine-grained runtime control.",
    },
    {
        "backend": "prompt_only",
        "recommended_models": [],
        "notes": "Write the prompt bundle only. Useful when the prompt is ready but the local runtime is not.",
    },
]

CODOMAIN_IMPROVEMENTS = [
    {
        "name": "Adjacent-interest codomain view",
        "priority": "high",
        "why": "The codomain should reveal what subject-linked users care about besides the subject, not simply restate the domain.",
        "prototype": "Qualify users by subject linkage, then analyze their broader posts with optional removal of direct subject posts to surface adjacent interests.",
    },
    {
        "name": "Topic relevance filter",
        "priority": "high",
        "why": "User qualification still matters. Without it, codomain timelines can be dominated by users who barely care about the subject.",
        "prototype": "Use topic keywords, seed confidence, or lightweight relevance models to qualify users before analyzing their broader timelines.",
    },
    {
        "name": "Per-user cap and balancing",
        "priority": "high",
        "why": "A few prolific accounts can dominate codomain language. That hides community-level patterns.",
        "prototype": "Cap tweets per author and balance the positive/negative seed-user pools before building the codomain corpus.",
    },
    {
        "name": "Seed-confidence threshold",
        "priority": "high",
        "why": "Seed users are currently selected from single-tweet sentiment. Weak-confidence seeds contaminate the codomain with mislabeled users.",
        "prototype": "Keep only seed tweets with strong positive or negative RoBERTa margins and drop low-confidence or neutral cases.",
    },
    {
        "name": "Retweet and promo filtering",
        "priority": "medium",
        "why": "RTs, link drops, and stock-promo style posts dilute the semantic signal, especially for brands and sports.",
        "prototype": "Filter retweets, low-context link drops, cashtag-heavy posts, and obvious promo/news-bot patterns before downstream analysis.",
    },
    {
        "name": "Seed-tweet removal from codomain",
        "priority": "medium",
        "why": "If the timeline pull includes the same topic tweet that defined the user seed, the codomain can partially collapse back into the domain.",
        "prototype": "Drop exact matches or same Tweet IDs as the seed set before codomain analysis.",
    },
    {
        "name": "User-level aggregation",
        "priority": "medium",
        "why": "Document-level sentiment and TF-IDF overemphasize frequent posters.",
        "prototype": "Aggregate to user-level language summaries, then compare groups using one user = one vote.",
    },
]


def ensure_working_tree() -> None:
    WORKING_ROOT.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    SCRIPT_ROOT.mkdir(parents=True, exist_ok=True)
    for stage in MAIN_STAGES:
        (OUTPUT_ROOT / "_template" / stage).mkdir(parents=True, exist_ok=True)


def list_subjects() -> pd.DataFrame:
    rows = []
    for config in SUBJECTS.values():
        rows.append(
            {
                "slug": config.slug,
                "display_name": config.display_name,
                "source_domain_corpus": str(config.domain_corpus_path.relative_to(PROJECT_ROOT)),
                "source_codomain_corpus": str(config.codomain_corpus_path.relative_to(PROJECT_ROOT)),
            }
        )
    return pd.DataFrame(rows).sort_values("slug").reset_index(drop=True)


def get_subject_config(subject: str | SubjectConfig) -> SubjectConfig:
    if isinstance(subject, SubjectConfig):
        return subject
    key = str(subject).strip().lower()
    slug = SUBJECT_ALIASES.get(key, key)
    if slug not in SUBJECTS:
        raise KeyError(f"Unsupported subject: {subject!r}. Choose from {sorted(SUBJECTS)}.")
    return SUBJECTS[slug]


def subject_output_dir(subject: str | SubjectConfig) -> Path:
    config = get_subject_config(subject)
    path = OUTPUT_ROOT / config.slug
    path.mkdir(parents=True, exist_ok=True)
    return path


def stage_dir(subject: str | SubjectConfig, stage: str) -> Path:
    if stage not in MAIN_STAGES:
        raise KeyError(f"Unknown stage: {stage}")
    path = subject_output_dir(subject) / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


def artifact_path(
    subject: str | SubjectConfig,
    stage: str,
    dataset: str | None,
    artifact: str,
    ext: str,
    subdir: str | None = None,
) -> Path:
    base = stage_dir(subject, stage)
    if subdir:
        base = base / subdir
        base.mkdir(parents=True, exist_ok=True)

    parts = [get_subject_config(subject).slug]
    if dataset:
        parts.append(dataset)
    parts.append(artifact)
    filename = "__".join(parts) + f".{ext.lstrip('.')}"
    return base / filename


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle, ensure_ascii=False)
            handle.write("\n")
    return path


def load_table(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    if path.suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    raise ValueError(f"Unsupported table type: {path.suffix}")


def load_stage_table(
    subject: str | SubjectConfig,
    stage: str,
    dataset: str | None,
    artifact: str,
    ext: str,
    **kwargs: Any,
) -> pd.DataFrame:
    return load_table(artifact_path(subject, stage, dataset, artifact, ext), **kwargs)


def summarize_frame(df: pd.DataFrame, name: str) -> dict[str, Any]:
    return {
        "name": name,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "unique_authors": int(df["Author ID"].nunique()) if "Author ID" in df.columns else None,
    }


def _parse_query_parameters(node: dict[str, Any]) -> dict[str, Any]:
    params = {}
    for item in node.get("parameters", {}).get("queryParameters", {}).get("parameters", []):
        name = item.get("name")
        if name:
            params[name] = item.get("value")
    return params


def load_n8n_workflow_summaries() -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in sorted(N8N_ROOT.glob("*.json")):
        workflow = json.loads(path.read_text(encoding="utf-8"))
        nodes = workflow.get("nodes", [])
        for node in nodes:
            if node.get("type") != "n8n-nodes-base.httpRequest":
                continue
            summaries.append(
                {
                    "workflow_file": path.name,
                    "workflow_name": workflow.get("name"),
                    "node_name": node.get("name"),
                    "url": node.get("parameters", {}).get("url"),
                    "query_parameters": _parse_query_parameters(node),
                }
            )
    return summaries


def save_collection_scaffold(
    subject: str | SubjectConfig,
    custom_query: str | None = None,
    since: str | None = None,
    until: str | None = None,
    run_local_n8n: bool = False,
) -> dict[str, Any]:
    config = get_subject_config(subject)
    workflows = load_n8n_workflow_summaries()
    scaffold = {
        "subject": asdict(config),
        "custom_query": custom_query,
        "since": since,
        "until": until,
        "run_local_n8n": run_local_n8n,
        "local_n8n_binary_found": shutil.which("n8n") is not None,
        "status": "scaffold_only",
        "notes": [
            "The repo only stores exported n8n workflow JSON, not a verified local execution stack.",
            "Treat this stage as configuration + documentation until local credentials and package behavior are confirmed.",
            "Stage 01 can standardize the current source corpora without depending on this scaffold.",
        ],
        "workflow_summaries": workflows,
    }
    output = artifact_path(config, "00_collection", None, "collection_scaffold", "json")
    write_json(output, scaffold)
    return {"output_path": str(output), **scaffold}


def legacy_paths(subject: str | SubjectConfig) -> dict[str, Path]:
    config = get_subject_config(subject)
    return {
        "domain_corpus": config.domain_corpus_path,
        "codomain_corpus": config.codomain_corpus_path,
        "codomain_seed_pre": config.codomain_seed_pre_path,
        "codomain_seed_post": config.codomain_seed_post_path,
    }


def _read_standard_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"Author ID": "string", "Tweet ID": "string"})


def _parse_seed_post_labels(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    if not path.exists():
        return None, "missing"

    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        return None, "empty"

    label_values = {"positive", "neutral", "negative"}

    def is_label(value: str) -> bool:
        return str(value).strip().lower() in label_values

    if rows and len(rows[0]) >= 2 and all(len(row) >= 2 for row in rows[: min(25, len(rows))]):
        preview = rows[: min(25, len(rows))]
        if sum(is_label(row[1]) for row in preview) >= max(3, len(preview) // 2):
            frame = pd.DataFrame(rows, columns=["Author ID", "seed_label", *range(2, max(map(len, rows)))])
            frame = frame[["Author ID", "seed_label"]].copy()
            frame["Author ID"] = frame["Author ID"].astype("string")
            frame["seed_label"] = frame["seed_label"].str.lower()
            return frame, None

    header = [value.strip().lower() for value in rows[0]]
    if "rob_label" in header and "original" in header:
        return None, "tweet_level_output_not_author_label_map"

    return None, "unrecognized_format"


def ingest_legacy_subject_data(subject: str | SubjectConfig, copy_seed_files: bool = True) -> dict[str, Any]:
    config = get_subject_config(subject)
    ensure_working_tree()

    paths = legacy_paths(config)
    manifest: dict[str, Any] = {
        "subject": config.slug,
        "source_paths": {name: str(path) for name, path in paths.items()},
        "outputs": {},
        "issues": [],
    }

    for dataset, source_path in {
        "domain": paths["domain_corpus"],
        "codomain": paths["codomain_corpus"],
    }.items():
        if not source_path.exists():
            manifest["issues"].append(f"Missing {dataset} corpus: {source_path}")
            continue
        df = _read_standard_csv(source_path)
        if "Subject" not in df.columns:
            df.insert(0, "Subject", config.display_name)
        if "Dataset" not in df.columns:
            df.insert(1, "Dataset", dataset)
        output = artifact_path(config, "01_ingest", dataset, "corpus", "csv")
        df.to_csv(output, index=False)
        manifest["outputs"][f"{dataset}_corpus"] = {
            "path": str(output),
            "summary": summarize_frame(df, f"{config.slug}_{dataset}_corpus"),
        }

    if copy_seed_files and paths["codomain_seed_pre"].exists():
        pre_df = _read_standard_csv(paths["codomain_seed_pre"])
        pre_output = artifact_path(config, "01_ingest", "codomain", "seed_pre", "csv")
        pre_df.to_csv(pre_output, index=False)
        manifest["outputs"]["codomain_seed_pre"] = {
            "path": str(pre_output),
            "summary": summarize_frame(pre_df, f"{config.slug}_codomain_seed_pre"),
        }

    if copy_seed_files:
        labels_df, issue = _parse_seed_post_labels(paths["codomain_seed_post"])
        if labels_df is not None:
            labels_output = artifact_path(config, "01_ingest", "codomain", "seed_labels", "csv")
            labels_df.to_csv(labels_output, index=False)
            manifest["outputs"]["codomain_seed_labels"] = {
                "path": str(labels_output),
                "rows": int(len(labels_df)),
            }
        elif issue:
            manifest["issues"].append(f"Seed-label file issue: {issue}")

    manifest_path = artifact_path(config, "01_ingest", None, "manifest", "json")
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _clean_for_lang(text: str) -> str:
    value = str(text)
    value = re.sub(r"http\S+|www\S+", " ", value)
    value = re.sub(r"RT\s+@\w+:", " ", value)
    value = re.sub(r"[@#]\w+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _safe_detect_lang(text: str) -> str:
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = 0
    cleaned = _clean_for_lang(text)
    if sum(ch.isalpha() for ch in cleaned) < 10:
        return "unk"
    try:
        return detect(cleaned)
    except Exception:
        return "unk"


def ensure_nltk_resources() -> None:
    import nltk

    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for resource_path, download_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(download_name, quiet=True)


def preprocess_corpus(
    df: pd.DataFrame,
    extra_stopwords: Iterable[str] | None = None,
    drop_duplicate_content: bool = True,
    target_language: str = "en",
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    from nltk import pos_tag
    from nltk.corpus import stopwords, wordnet
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import TweetTokenizer

    ensure_nltk_resources()

    working = df.copy()
    working = working.dropna(subset=["Content"]).reset_index(drop=True)

    duplicate_mask = working.duplicated(subset="Content", keep="first")
    duplicates = working.loc[duplicate_mask].copy().reset_index(drop=True)
    if drop_duplicate_content:
        working = working.loc[~duplicate_mask].copy().reset_index(drop=True)

    working["lang"] = working["Content"].apply(_safe_detect_lang)
    english = working.loc[working["lang"].eq(target_language)].copy().reset_index(drop=True)

    url_re = re.compile(r"https?://\S+|www\.\S+")
    ordinal_re = re.compile(r"(\d+)(st|nd|rd|th)\b", flags=re.I)
    camel_re = re.compile(r"(?<=[a-z])(?=[A-Z])")

    def prepare_for_roberta(text: str) -> str:
        value = str(text)
        value = re.sub(r"http\S+|www\.\S+", "HTTPURL", value)
        value = re.sub(r"@\w+", "@USER", value)
        return value.strip()

    def clean_for_nlp(text: str) -> str:
        value = str(text).lower()
        value = url_re.sub(" ", value)
        value = ordinal_re.sub(r"\1", value)
        value = re.sub(r"@\w+", " ", value)

        def strip_hash(match: re.Match[str]) -> str:
            token = match.group(1)
            if camel_re.search(token):
                token = camel_re.sub(" ", token)
            return f" {token} "

        value = re.sub(r"#([A-Za-z0-9_]+)", strip_hash, value)
        value = re.sub(r"\brt\b", " ", value)
        value = re.sub(r"[^a-z0-9\s']", " ", value)
        value = re.sub(r"\s+", " ", value).strip()
        return value

    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
    lemmatizer = WordNetLemmatizer()

    def wn_pos(tag: str) -> str:
        if tag.startswith("J"):
            return wordnet.ADJ
        if tag.startswith("V"):
            return wordnet.VERB
        if tag.startswith("N"):
            return wordnet.NOUN
        if tag.startswith("R"):
            return wordnet.ADV
        return wordnet.NOUN

    neg_keep = {"no", "not", "nor", "n't", "never", "none"}
    domain_stops = {
        "people",
        "like",
        "just",
        "say",
        "think",
        "year",
        "time",
        "government",
        "country",
        "party",
        "today",
        "tonight",
        "tomorrow",
        "via",
        "amp",
        "gt",
    }
    if extra_stopwords:
        domain_stops |= {str(word).strip().lower() for word in extra_stopwords if str(word).strip()}

    stop_set = (set(stopwords.words("english")) - neg_keep) | domain_stops

    prep = english.copy().reset_index(drop=True)
    prep["Content_clean"] = prep["Content"].apply(clean_for_nlp)

    def preprocess_one(text: str) -> tuple[list[str], list[str], list[str]]:
        tokens = tokenizer.tokenize(text)
        tokens = [token for token in tokens if any(ch.isalpha() for ch in token)]
        tagged = pos_tag(tokens)
        lemmas = [lemmatizer.lemmatize(token, wn_pos(tag)) for token, tag in tagged]
        nostop = [token for token in lemmas if token not in stop_set and len(token) > 1]
        return tokens, lemmas, nostop

    if len(prep):
        processed = prep["Content_clean"].apply(preprocess_one)
        tokens, lemmas, nostop = zip(*processed)
    else:
        tokens, lemmas, nostop = [], [], []

    if "Tweet ID" in prep.columns:
        prep["Tweet ID"] = prep["Tweet ID"].astype("string")

    prep["text_for_sent"] = prep["Content"].fillna("").map(prepare_for_roberta)
    prep["clean"] = prep["Content_clean"]
    prep["tokens"] = list(tokens)
    prep["lemmas"] = list(lemmas)
    prep["no_stop"] = list(nostop)
    prep["clean_lemmas"] = prep["lemmas"].apply(" ".join)
    prep["clean_nostop"] = prep["no_stop"].apply(" ".join)

    text_cols = [
        "text_for_sent",
        "clean",
        "tokens",
        "lemmas",
        "no_stop",
        "clean_lemmas",
        "clean_nostop",
    ]
    original_cols = [column for column in prep.columns if column not in text_cols]
    prep = prep[original_cols + text_cols]

    empty_mask = prep["clean_nostop"].str.len().eq(0)
    prep = prep.loc[~empty_mask].copy().reset_index(drop=True)

    summary = {
        "raw_rows": int(len(df)),
        "non_empty_content_rows": int(df.dropna(subset=["Content"]).shape[0]),
        "duplicate_rows_removed": int(duplicates.shape[0]) if drop_duplicate_content else 0,
        "rows_after_duplicate_filter": int(working.shape[0]),
        "rows_after_language_filter": int(english.shape[0]),
        "rows_after_preprocess": int(prep.shape[0]),
        "target_language": target_language,
    }
    return prep, summary, duplicates


def preprocess_stage(
    subject: str | SubjectConfig,
    dataset: str,
    extra_stopwords: Iterable[str] | None = None,
    drop_duplicate_content: bool = True,
    target_language: str = "en",
) -> dict[str, Any]:
    config = get_subject_config(subject)
    source = artifact_path(config, "01_ingest", dataset, "corpus", "csv")
    raw = _read_standard_csv(source)
    prep, summary, duplicates = preprocess_corpus(
        raw,
        extra_stopwords=extra_stopwords,
        drop_duplicate_content=drop_duplicate_content,
        target_language=target_language,
    )

    prep_output = artifact_path(config, "02_preprocess", dataset, "preprocessed", "parquet")
    dup_output = artifact_path(config, "02_preprocess", dataset, "duplicates_removed", "csv")
    summary_output = artifact_path(config, "02_preprocess", dataset, "preprocess_summary", "json")

    prep.to_parquet(prep_output, index=False)
    duplicates.to_csv(dup_output, index=False)
    write_json(summary_output, summary)

    return {
        "subject": config.slug,
        "dataset": dataset,
        "rows": int(len(prep)),
        "preprocessed_path": str(prep_output),
        "duplicates_path": str(dup_output),
        "summary_path": str(summary_output),
        **summary,
    }


def _load_roberta_model(model_name: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cached = _MODEL_CACHE.get(model_name)
    if cached is not None:
        return cached

    tokenizer = None
    model = None

    # Prefer the local HF cache first. This avoids repeated remote metadata calls after the first download.
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            local_files_only=True,
            use_safetensors=False,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            use_safetensors=False,
        )

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    loaded = (tokenizer, model, device)
    _MODEL_CACHE[model_name] = loaded
    return loaded


def score_sentiment_frame(
    prep: pd.DataFrame,
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    batch_size: int = 32,
    max_length: int = 280,
) -> pd.DataFrame:
    import torch

    tokenizer, model, device = _load_roberta_model(model_name)

    id2label = model.config.id2label
    label_names = [id2label[index].lower() for index in range(model.config.num_labels)]
    label_to_idx = {label: index for index, label in enumerate(label_names)}
    pos_idx = label_to_idx.get("positive")
    neg_idx = label_to_idx.get("negative")

    texts = prep["text_for_sent"].fillna("").tolist()

    @torch.inference_mode()
    def predict(text_batch: Sequence[str]) -> np.ndarray:
        output: list[np.ndarray] = []
        for start in range(0, len(text_batch), batch_size):
            batch = text_batch[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            output.append(probs)
        if not output:
            return np.zeros((0, model.config.num_labels))
        return np.vstack(output)

    probs = predict(texts)
    scored = prep.copy().reset_index(drop=True)
    prob_cols = [f"rob_prob_{label}" for label in label_names]
    scored[prob_cols] = probs
    scored["rob_label"] = (
        pd.DataFrame(probs, columns=prob_cols)
        .idxmax(axis=1)
        .str.replace("rob_prob_", "", regex=False)
    )
    scored["rob_score"] = probs[:, pos_idx] - probs[:, neg_idx] if pos_idx is not None and neg_idx is not None else np.nan
    return scored


def sentiment_stage(
    subject: str | SubjectConfig,
    dataset: str,
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    batch_size: int = 32,
    max_length: int = 280,
) -> dict[str, Any]:
    config = get_subject_config(subject)
    prep = load_stage_table(config, "02_preprocess", dataset, "preprocessed", "parquet")
    scored = score_sentiment_frame(prep, model_name=model_name, batch_size=batch_size, max_length=max_length)

    score_output = artifact_path(config, "03_sentiment", dataset, "scores", "parquet")
    counts_output = artifact_path(config, "03_sentiment", dataset, "sentiment_counts", "csv")
    summary_output = artifact_path(config, "03_sentiment", dataset, "sentiment_summary", "json")

    scored.to_parquet(score_output, index=False)
    counts = (
        scored["rob_label"]
        .value_counts()
        .reindex(SENTIMENT_ORDER, fill_value=0)
        .rename_axis("rob_label")
        .reset_index(name="count")
    )
    counts.to_csv(counts_output, index=False)
    summary = {
        "subject": config.slug,
        "dataset": dataset,
        "model_name": model_name,
        "rows": int(len(scored)),
        "label_counts": dict(zip(counts["rob_label"], counts["count"])),
    }
    write_json(summary_output, summary)
    return {
        **summary,
        "scores_path": str(score_output),
        "counts_path": str(counts_output),
        "summary_path": str(summary_output),
    }


def _save_figure(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return str(path)


def _token_list(series: pd.Series) -> pd.Series:
    def ensure_tokens(value: Any) -> list[str]:
        if isinstance(value, np.ndarray):
            return [str(item).strip() for item in value.tolist() if str(item).strip()]
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if str(item).strip()]
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return []
        text = str(value).strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                parsed = None
            if isinstance(parsed, np.ndarray):
                return [str(item).strip() for item in parsed.tolist() if str(item).strip()]
            if isinstance(parsed, (list, tuple)):
                return [str(item).strip() for item in parsed if str(item).strip()]
        cleaned = re.sub(r"[\"'\[\]]", " ", text)
        return [part for part in re.split(r"[,\s]+", cleaned) if part]

    return series.apply(ensure_tokens)


def eda_stage(subject: str | SubjectConfig, dataset: str, top_k: int = 20) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    config = get_subject_config(subject)
    scores = load_stage_table(config, "03_sentiment", dataset, "scores", "parquet")

    figures: list[str] = []
    summary: dict[str, Any] = {
        "subject": config.slug,
        "dataset": dataset,
        "rows": int(len(scores)),
    }

    counts = scores["rob_label"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot(kind="bar", color=[SENTIMENT_COLORS[label] for label in SENTIMENT_ORDER], edgecolor="black", ax=ax)
    ax.set_title(f"{config.display_name} {dataset.title()} Sentiment Distribution")
    ax.set_ylabel("Tweet count")
    ax.set_xlabel("")
    figures.append(
        _save_figure(
            fig,
            artifact_path(config, "04_eda", dataset, "sentiment_distribution", "png", subdir="figures"),
        )
    )
    plt.close(fig)

    if "Author ID" in scores.columns:
        user_counts = (
            scores.dropna(subset=["Author ID"])
            .groupby(["rob_label", "Author ID"])
            .size()
            .reset_index(name="tweet_count")
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        for label in SENTIMENT_ORDER:
            subset = user_counts.loc[user_counts["rob_label"].eq(label), "tweet_count"].to_numpy()
            if subset.size:
                bins = min(30, int(np.ptp(subset)) + 1 if subset.size > 1 else 5)
                ax.hist(
                    subset,
                    bins=bins,
                    alpha=0.45,
                    color=SENTIMENT_COLORS[label],
                    label=label.title(),
                    edgecolor="white",
                )
        ax.set_title(f"{config.display_name} {dataset.title()} Tweets Per User")
        ax.set_xlabel("Tweets per user")
        ax.set_ylabel("Number of users")
        ax.legend()
        figures.append(
            _save_figure(
                fig,
                artifact_path(config, "04_eda", dataset, "tweets_per_user", "png", subdir="figures"),
            )
        )
        plt.close(fig)
        summary["unique_authors"] = int(scores["Author ID"].nunique())

    word_count = scores["clean_nostop"].fillna("").astype(str).str.split().map(len)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label in SENTIMENT_ORDER:
        subset = word_count[scores["rob_label"].eq(label)].to_numpy()
        subset = subset[subset > 0]
        if subset.size:
            ax.hist(subset, bins=30, density=True, histtype="step", color=SENTIMENT_COLORS[label], label=label.title())
    ax.set_title(f"{config.display_name} {dataset.title()} Word Counts")
    ax.set_xlabel("Words per tweet")
    ax.set_ylabel("Density")
    ax.legend()
    figures.append(
        _save_figure(
            fig,
            artifact_path(config, "04_eda", dataset, "word_counts", "png", subdir="figures"),
        )
    )
    plt.close(fig)

    token_series = _token_list(scores["no_stop"])
    token_rows = []
    bigram_rows = []
    for label in SENTIMENT_ORDER:
        counter = Counter()
        filtered_counter = Counter()
        bigram_counter = Counter()
        subset = token_series.loc[scores["rob_label"].eq(label)]
        for tokens in subset:
            counter.update(tokens)
            filtered_counter.update(token for token in tokens if token not in DISPLAY_STOP_TOKENS)
            bigram_counter.update(" ".join(pair) for pair in zip(tokens, tokens[1:]))
        for term, count in counter.most_common(top_k):
            token_rows.append({"rob_label": label, "term": term, "count": int(count)})
        for term, count in bigram_counter.most_common(top_k):
            bigram_rows.append({"rob_label": label, "term": term, "count": int(count)})
        if counter:
            terms, values = zip(*counter.most_common(top_k))
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(list(terms)[::-1], list(values)[::-1], color=SENTIMENT_COLORS[label])
            ax.set_title(f"{config.display_name} {dataset.title()} Raw Top Tokens ({label})")
            ax.set_xlabel("Count")
            figures.append(
                _save_figure(
                    fig,
                    artifact_path(config, "04_eda", dataset, f"top_tokens_raw_{label}", "png", subdir="figures"),
                )
            )
            plt.close(fig)
        if filtered_counter:
            terms, values = zip(*filtered_counter.most_common(top_k))
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(list(terms)[::-1], list(values)[::-1], color=SENTIMENT_COLORS[label])
            ax.set_title(f"{config.display_name} {dataset.title()} Filtered Top Tokens ({label})")
            ax.set_xlabel("Count")
            figures.append(
                _save_figure(
                    fig,
                    artifact_path(config, "04_eda", dataset, f"top_tokens_filtered_{label}", "png", subdir="figures"),
                )
            )
            plt.close(fig)
        if bigram_counter:
            terms, values = zip(*bigram_counter.most_common(top_k))
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(list(terms)[::-1], list(values)[::-1], color=SENTIMENT_COLORS[label])
            ax.set_title(f"{config.display_name} {dataset.title()} Top Bigrams ({label})")
            ax.set_xlabel("Count")
            figures.append(
                _save_figure(
                    fig,
                    artifact_path(config, "04_eda", dataset, f"top_bigrams_{label}", "png", subdir="figures"),
                )
            )
            plt.close(fig)

    top_tokens = pd.DataFrame(token_rows)
    top_bigrams = pd.DataFrame(bigram_rows)
    top_tokens_output = artifact_path(config, "04_eda", dataset, "top_tokens", "csv")
    top_bigrams_output = artifact_path(config, "04_eda", dataset, "top_bigrams", "csv")
    summary_output = artifact_path(config, "04_eda", dataset, "eda_summary", "json")
    top_tokens.to_csv(top_tokens_output, index=False)
    top_bigrams.to_csv(top_bigrams_output, index=False)
    summary["figures"] = figures
    summary["top_tokens_path"] = str(top_tokens_output)
    summary["top_bigrams_path"] = str(top_bigrams_output)
    write_json(summary_output, summary)
    summary["summary_path"] = str(summary_output)
    return summary


def _minmax(series: pd.Series) -> pd.Series:
    series = series.astype(float)
    return (series - series.min()) / (series.max() - series.min() + 1e-12)


def compute_tfidf_metrics(
    scores: pd.DataFrame,
    min_df: int = 5,
    max_df: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    text = scores["clean_nostop"].fillna("").astype(str)
    if not (text.str.len() > 0).any():
        raise ValueError("All documents are empty after preprocessing.")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=max_df,
        norm=None,
        sublinear_tf=True,
        stop_words=None,
    )
    matrix = vectorizer.fit_transform(text)
    vocab = np.array(vectorizer.get_feature_names_out())
    idf = vectorizer.idf_.astype(float)

    def subset_mask(label: str) -> np.ndarray:
        return np.asarray(scores["rob_label"].values == label, dtype=bool)

    def aggregate_importance(mask: np.ndarray) -> np.ndarray:
        if not np.any(mask):
            return np.zeros(matrix.shape[1], dtype=float)
        return matrix[mask].sum(axis=0).A1.astype(float)

    def weighted_idf(mask: np.ndarray) -> np.ndarray:
        if not np.any(mask):
            return np.zeros(matrix.shape[1], dtype=float)
        df_subset = (matrix[mask] > 0).sum(axis=0).A1.astype(float)
        return df_subset * idf

    rows = []
    for label in SENTIMENT_ORDER:
        mask = subset_mask(label)
        agg = aggregate_importance(mask)
        iwdf = weighted_idf(mask)
        label_df = pd.DataFrame(
            {
                "term": vocab,
                "sentiment": label,
                "agg": agg,
                "iwdf": iwdf,
            }
        )
        label_df["agg_norm"] = _minmax(label_df["agg"])
        label_df["iwdf_norm"] = _minmax(label_df["iwdf"])
        label_df["fused_score"] = (label_df["agg_norm"] + label_df["iwdf_norm"]) / 2.0
        rows.append(label_df)

    tidy = pd.concat(rows, ignore_index=True)
    wide = (
        tidy.pivot(index="term", columns="sentiment", values="iwdf")
        .reindex(columns=SENTIMENT_ORDER)
        .fillna(0.0)
        .reset_index()
    )
    top_terms = (
        tidy.sort_values(["sentiment", "fused_score", "iwdf"], ascending=[True, False, False])
        .groupby("sentiment", group_keys=False)
        .head(20)
        .reset_index(drop=True)
    )
    return tidy, wide, top_terms


def tfidf_stage(
    subject: str | SubjectConfig,
    dataset: str,
    min_df: int = 5,
    max_df: float = 0.80,
    top_k: int = 20,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    config = get_subject_config(subject)
    scores = load_stage_table(config, "03_sentiment", dataset, "scores", "parquet")
    tidy, wide, top_terms = compute_tfidf_metrics(scores, min_df=min_df, max_df=max_df)

    tidy_output = artifact_path(config, "05_tfidf", dataset, "tfidf_metrics_tidy", "parquet")
    wide_output = artifact_path(config, "05_tfidf", dataset, "weighted_idf_wide", "parquet")
    top_terms_output = artifact_path(config, "05_tfidf", dataset, "top_terms", "csv")
    summary_output = artifact_path(config, "05_tfidf", dataset, "tfidf_summary", "json")

    tidy.to_parquet(tidy_output, index=False)
    wide.to_parquet(wide_output, index=False)
    top_terms.to_csv(top_terms_output, index=False)

    figures = []
    for label in SENTIMENT_ORDER:
        subset = top_terms.loc[top_terms["sentiment"].eq(label)].head(top_k)
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 6))
        ordered = subset.sort_values("iwdf")
        ax.barh(ordered["term"], ordered["iwdf"], color=SENTIMENT_COLORS[label])
        ax.set_title(f"{config.display_name} {dataset.title()} Top Weighted IDF ({label})")
        ax.set_xlabel("Weighted IDF")
        figures.append(
            _save_figure(
                fig,
                artifact_path(config, "05_tfidf", dataset, f"top_weighted_idf_{label}", "png", subdir="figures"),
            )
        )
        plt.close(fig)

    polarity = (
        wide.rename(columns={"negative": "neg", "neutral": "neu", "positive": "pos"})
        .assign(delta=lambda frame: frame["pos"] - frame["neg"])
        .assign(abs_delta=lambda frame: frame["delta"].abs())
        .sort_values("abs_delta", ascending=False)
        .head(top_k)
        .sort_values("delta", ascending=True)
    )
    if not polarity.empty:
        fig, ax = plt.subplots(figsize=(7, max(6, 0.35 * len(polarity))))
        y = np.arange(len(polarity))
        for index, (_, row) in enumerate(polarity.iterrows()):
            color = SENTIMENT_COLORS["positive"] if row["delta"] >= 0 else SENTIMENT_COLORS["negative"]
            ax.plot([row["neg"], row["pos"]], [index, index], color=color, lw=2)
        ax.scatter(polarity["neg"], y, color=SENTIMENT_COLORS["negative"], label="Negative", s=35)
        ax.scatter(polarity["neu"], y, color=SENTIMENT_COLORS["neutral"], label="Neutral", s=30)
        ax.scatter(polarity["pos"], y, color=SENTIMENT_COLORS["positive"], label="Positive", s=35)
        ax.set_yticks(y, labels=polarity["term"])
        ax.set_title(f"{config.display_name} {dataset.title()} Most Polarized Terms")
        ax.set_xlabel("Weighted IDF")
        ax.invert_yaxis()
        ax.legend()
        figures.append(
            _save_figure(
                fig,
                artifact_path(config, "05_tfidf", dataset, "term_polarity", "png", subdir="figures"),
            )
        )
        plt.close(fig)

    summary = {
        "subject": config.slug,
        "dataset": dataset,
        "rows": int(len(tidy)),
        "tidy_path": str(tidy_output),
        "wide_path": str(wide_output),
        "top_terms_path": str(top_terms_output),
        "figures": figures,
    }
    write_json(summary_output, summary)
    summary["summary_path"] = str(summary_output)
    return summary


def _make_topic_label(top_terms: Sequence[str], n_words: int = 4) -> str:
    bigrams = [term for term in top_terms if " " in term]
    keep = bigrams[:n_words]
    if len(keep) < n_words:
        keep.extend(term for term in top_terms if " " not in term and term not in keep)
    return " · ".join(keep[:n_words])


def _topic_exemplars(
    doc_topic: np.ndarray,
    topic_term: np.ndarray,
    vocab: np.ndarray,
    corpus: Sequence[str],
    original_corpus: Sequence[str] | None = None,
    sentiments: Sequence[str] | None = None,
    n_top_words: int = 12,
    exemplars_per_topic: int = 3,
    min_prop: float = 0.15,
    preview_len: int = 160,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    top_idx = np.argsort(topic_term, axis=1)[:, ::-1][:, :n_top_words]
    top_words = [[str(vocab[index]) for index in indices] for indices in top_idx]

    rows = []
    for topic_id in range(topic_term.shape[0]):
        scores = doc_topic[:, topic_id]
        eligible = np.where(scores >= min_prop)[0]
        if not eligible.size:
            continue
        ranked = eligible[np.argsort(scores[eligible])[::-1]][:exemplars_per_topic]
        for rank, doc_idx in enumerate(ranked, start=1):
            preview_text = (
                str(original_corpus[doc_idx]) if original_corpus is not None else str(corpus[doc_idx])
            )
            entry = {
                "topic_id": int(topic_id),
                "rank": int(rank),
                "score": float(scores[doc_idx]),
                "doc_idx": int(doc_idx),
                "preview": preview_text[:preview_len],
            }
            if sentiments is not None:
                entry["sentiment"] = sentiments[doc_idx]
            rows.append(entry)

    exemplars = pd.DataFrame(rows)
    summary_rows = []
    for topic_id, terms in enumerate(top_words):
        subset = exemplars.loc[exemplars["topic_id"].eq(topic_id)].sort_values("rank")
        summary_rows.append(
            {
                "topic_id": int(topic_id),
                "label": _make_topic_label(terms),
                "top_words": ", ".join(terms),
                "exemplar_1": subset["preview"].iloc[0] if len(subset) > 0 else "",
                "exemplar_2": subset["preview"].iloc[1] if len(subset) > 1 else "",
                "exemplar_3": subset["preview"].iloc[2] if len(subset) > 2 else "",
            }
        )
    summaries = pd.DataFrame(summary_rows)
    return summaries, exemplars


def topic_model_stage(
    subject: str | SubjectConfig,
    dataset: str,
    topic_grid: Sequence[int] = (4, 6, 8, 10),
    n_topics: int | str = 6,
    n_top_words: int = 12,
    min_df: int = 10,
    max_df: float = 0.80,
    min_topic_prop: float = 0.15,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer

    config = get_subject_config(subject)
    scores = load_stage_table(config, "03_sentiment", dataset, "scores", "parquet")
    valid_mask = scores["clean_nostop"].fillna("").astype(str).str.strip().str.len() > 0
    filtered = scores.loc[valid_mask].copy().reset_index(drop=True)
    corpus = filtered["clean_nostop"].fillna("").astype(str).str.strip().reset_index(drop=True)
    original_corpus = filtered["Content"].fillna("").astype(str).str.strip().reset_index(drop=True)
    sentiments = filtered["rob_label"].fillna("unknown").astype(str).tolist()

    vectorizer = CountVectorizer(
        stop_words=None,
        lowercase=False,
        ngram_range=(1, 2),
        max_df=max_df,
        min_df=min_df,
        token_pattern=r"(?u)\b\w\w+\b",
    )
    matrix = vectorizer.fit_transform(corpus)
    vocab = np.array(vectorizer.get_feature_names_out())

    grid_rows = []
    for topic_count in topic_grid:
        lda = LatentDirichletAllocation(
            n_components=int(topic_count),
            learning_method="batch",
            random_state=0,
        )
        lda.fit(matrix)
        grid_rows.append(
            {
                "n_topics": int(topic_count),
                "perplexity": float(lda.perplexity(matrix)),
                "log_likelihood": float(lda.score(matrix)),
            }
        )
    grid_df = pd.DataFrame(grid_rows).sort_values("n_topics").reset_index(drop=True)

    if isinstance(n_topics, str):
        choice = n_topics.strip().lower()
        if choice == "grid_min":
            n_topics = int(grid_df.loc[grid_df["perplexity"].idxmin(), "n_topics"])
        else:
            raise ValueError(f"Unsupported n_topics strategy: {n_topics}")
    else:
        n_topics = int(n_topics)

    lda = LatentDirichletAllocation(
        n_components=int(n_topics),
        learning_method="batch",
        random_state=0,
    )
    doc_topic = lda.fit_transform(matrix)
    summaries, exemplars = _topic_exemplars(
        doc_topic=doc_topic,
        topic_term=lda.components_,
        vocab=vocab,
        corpus=corpus.tolist(),
        original_corpus=original_corpus.tolist(),
        sentiments=sentiments,
        n_top_words=n_top_words,
        exemplars_per_topic=3,
        min_prop=min_topic_prop,
    )

    doc_topic_df = pd.DataFrame(doc_topic, columns=[f"topic_{index}" for index in range(n_topics)])
    doc_topic_df["dominant_topic"] = doc_topic.argmax(axis=1)
    doc_topic_df["sentiment"] = sentiments
    doc_topic_df["preview"] = original_corpus.str.slice(0, 200).values

    prevalence = (
        doc_topic_df.groupby("sentiment")[[f"topic_{index}" for index in range(n_topics)]]
        .mean()
        .reset_index()
    )

    grid_output = artifact_path(config, "06_topic_modeling", dataset, "topic_grid_search", "csv")
    summary_output = artifact_path(config, "06_topic_modeling", dataset, "topic_summaries", "csv")
    exemplars_output = artifact_path(config, "06_topic_modeling", dataset, "topic_exemplars", "csv")
    doc_topic_output = artifact_path(config, "06_topic_modeling", dataset, "doc_topic", "parquet")
    prevalence_output = artifact_path(config, "06_topic_modeling", dataset, "topic_prevalence_by_sentiment", "csv")
    metadata_output = artifact_path(config, "06_topic_modeling", dataset, "topic_model_summary", "json")

    grid_df.to_csv(grid_output, index=False)
    summaries.to_csv(summary_output, index=False)
    exemplars.to_csv(exemplars_output, index=False)
    doc_topic_df.to_parquet(doc_topic_output, index=False)
    prevalence.to_csv(prevalence_output, index=False)

    summary_jsonl = artifact_path(config, "06_topic_modeling", dataset, "topic_summaries", "jsonl")
    exemplars_jsonl = artifact_path(config, "06_topic_modeling", dataset, "topic_exemplars", "jsonl")
    write_jsonl(
        summary_jsonl,
        [
            {
                "subject": config.slug,
                "dataset": dataset,
                "topic_id": int(row["topic_id"]),
                "label": row["label"],
                "top_words": row["top_words"],
                "exemplars": [row["exemplar_1"], row["exemplar_2"], row["exemplar_3"]],
            }
            for _, row in summaries.iterrows()
        ],
    )
    write_jsonl(
        exemplars_jsonl,
        [
            {
                "subject": config.slug,
                "dataset": dataset,
                "topic_id": int(row["topic_id"]),
                "rank": int(row["rank"]),
                "score": float(row["score"]),
                "doc_idx": int(row["doc_idx"]),
                "sentiment": row.get("sentiment", ""),
                "preview": row["preview"],
            }
            for _, row in exemplars.iterrows()
        ],
    )

    figures = []
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid_df["n_topics"], grid_df["perplexity"], marker="o")
    ax.set_title(f"{config.display_name} {dataset.title()} Topic Grid Search")
    ax.set_xlabel("Number of topics")
    ax.set_ylabel("Perplexity")
    figures.append(
        _save_figure(
            fig,
            artifact_path(config, "06_topic_modeling", dataset, "topic_grid_search", "png", subdir="figures"),
        )
    )
    plt.close(fig)

    prevalence_long = prevalence.melt(id_vars="sentiment", var_name="topic", value_name="mean_proportion")
    fig, ax = plt.subplots(figsize=(10, 5))
    for label in SENTIMENT_ORDER:
        subset = prevalence_long.loc[prevalence_long["sentiment"].eq(label)]
        if subset.empty:
            continue
        ax.plot(subset["topic"], subset["mean_proportion"], marker="o", label=label.title(), color=SENTIMENT_COLORS[label])
    ax.set_title(f"{config.display_name} {dataset.title()} Topic Prevalence by Sentiment")
    ax.set_ylabel("Mean topic proportion")
    ax.legend()
    figures.append(
        _save_figure(
            fig,
            artifact_path(config, "06_topic_modeling", dataset, "topic_prevalence", "png", subdir="figures"),
        )
    )
    plt.close(fig)

    pyldavis_path = artifact_path(config, "06_topic_modeling", dataset, "pyldavis", "html")
    pyldavis_status = "not_attempted"
    try:
        import pyLDAvis

        try:
            from pyLDAvis import sklearn as sklearn_lda

            vis = sklearn_lda.prepare(lda, matrix, vectorizer, mds="pcoa")
        except Exception:
            term_frequency = np.asarray(matrix.sum(axis=0)).ravel().astype(float)
            doc_lengths = np.asarray(matrix.sum(axis=1)).ravel().astype(float)
            topic_term = lda.components_.astype(float)
            topic_term_dists = topic_term / (topic_term.sum(axis=1, keepdims=True) + 1e-12)
            doc_topic_dists = doc_topic / (doc_topic.sum(axis=1, keepdims=True) + 1e-12)
            vis = pyLDAvis.prepare(topic_term_dists, doc_topic_dists, doc_lengths, vocab, term_frequency)
        pyLDAvis.save_html(vis, str(pyldavis_path))
        pyldavis_status = "saved"
    except Exception as exc:
        pyldavis_status = f"skipped: {exc}"

    summary = {
        "subject": config.slug,
        "dataset": dataset,
        "n_topics": int(n_topics),
        "topic_grid_path": str(grid_output),
        "topic_summaries_path": str(summary_output),
        "topic_exemplars_path": str(exemplars_output),
        "doc_topic_path": str(doc_topic_output),
        "topic_prevalence_path": str(prevalence_output),
        "topic_summaries_jsonl_path": str(summary_jsonl),
        "topic_exemplars_jsonl_path": str(exemplars_jsonl),
        "figures": figures,
        "pyldavis_status": pyldavis_status,
        "pyldavis_path": str(pyldavis_path),
    }
    write_json(metadata_output, summary)
    summary["summary_path"] = str(metadata_output)
    return summary


def local_llm_options() -> pd.DataFrame:
    return pd.DataFrame(LOCAL_LLM_OPTIONS)


def available_local_llm_backends() -> list[str]:
    available = ["prompt_only"]
    if shutil.which("ollama"):
        available.append("ollama")
    try:
        import mlx_lm  # noqa: F401

        available.append("mlx_lm")
    except Exception:
        pass
    try:
        import llama_cpp  # noqa: F401

        available.append("llama_cpp")
    except Exception:
        pass
    return available


def detect_os() -> str:
    return platform.system().strip().lower()


def ollama_installed() -> bool:
    return shutil.which("ollama") is not None


def ollama_server_running() -> bool:
    try:
        with urlopen("http://127.0.0.1:11434/api/tags", timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def start_ollama_server(timeout_seconds: int = 20) -> bool:
    if ollama_server_running():
        return True
    if not ollama_installed():
        return False

    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(max(1, timeout_seconds)):
        if ollama_server_running():
            return True
        import time

        time.sleep(1)
    return ollama_server_running()


def stop_ollama_server() -> bool:
    if not ollama_installed():
        return False
    system = detect_os()
    if system in {"darwin", "linux"}:
        subprocess.run(["pkill", "ollama"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif system == "windows":
        subprocess.run(
            ["taskkill", "/IM", "ollama.exe", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return not ollama_server_running()


def ollama_model_installed(model: str) -> bool:
    if not ollama_installed() or not ollama_server_running():
        return False
    result = subprocess.run(
        ["ollama", "show", model],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def ensure_ollama_model(model: str) -> dict[str, Any]:
    if not ollama_installed():
        return {"model": model, "installed": False, "downloaded": False, "message": "ollama_not_installed"}
    if not start_ollama_server():
        return {"model": model, "installed": False, "downloaded": False, "message": "ollama_server_not_running"}
    if ollama_model_installed(model):
        return {"model": model, "installed": True, "downloaded": False, "message": "already_present"}

    result = subprocess.run(
        ["ollama", "pull", model],
        text=True,
        capture_output=True,
        check=False,
    )
    ok = result.returncode == 0 and ollama_model_installed(model)
    return {
        "model": model,
        "installed": ok,
        "downloaded": ok,
        "message": "downloaded" if ok else "pull_failed",
        "stdout_tail": "\n".join(result.stdout.splitlines()[-10:]),
        "stderr_tail": "\n".join(result.stderr.splitlines()[-10:]),
    }


def ollama_model_storage_hint() -> dict[str, str]:
    system = detect_os()
    if system == "windows":
        base = str(Path.home() / ".ollama" / "models")
    else:
        base = str(Path.home() / ".ollama" / "models")
    return {"system": system, "model_dir": base}


def gather_stage_figures(
    subject: str | SubjectConfig,
    stage: str,
    datasets: str | Sequence[str] | None = None,
) -> list[Path]:
    config = get_subject_config(subject)
    figure_dir = stage_dir(config, stage) / "figures"
    if not figure_dir.exists():
        return []

    if datasets is None:
        prefixes = [f"{config.slug}__"]
    elif isinstance(datasets, str):
        prefixes = [f"{config.slug}__{datasets}__"]
    else:
        prefixes = [f"{config.slug}__{dataset}__" for dataset in datasets]

    matched = []
    for path in sorted(figure_dir.glob("*.png")):
        if any(path.name.startswith(prefix) for prefix in prefixes):
            matched.append(path)
    return matched


def show_stage_figures(
    subject: str | SubjectConfig,
    stage: str,
    datasets: str | Sequence[str] | None = None,
    width: int = 900,
) -> list[str]:
    paths = gather_stage_figures(subject, stage, datasets=datasets)
    if not paths:
        print(f"No figures found for subject={get_subject_config(subject).slug}, stage={stage}.")
        return []

    try:
        import ipywidgets as widgets
        from IPython.display import Image, Markdown, clear_output, display

        state = {"index": 0}
        out = widgets.Output()
        header = widgets.HTML()
        prev_btn = widgets.Button(description="←", layout=widgets.Layout(width="48px"))
        next_btn = widgets.Button(description="→", layout=widgets.Layout(width="48px"))

        def render() -> None:
            current = state["index"]
            header.value = (
                f"<b>Figure {current + 1} / {len(paths)}</b><br>"
                f"<code>{paths[current].name}</code>"
            )
            with out:
                clear_output(wait=True)
                display(Image(filename=str(paths[current]), width=width))

        def prev_click(_button) -> None:
            state["index"] = (state["index"] - 1) % len(paths)
            render()

        def next_click(_button) -> None:
            state["index"] = (state["index"] + 1) % len(paths)
            render()

        prev_btn.on_click(prev_click)
        next_btn.on_click(next_click)

        display(Markdown(f"### Figure Preview: {stage}"))
        display(widgets.VBox([header, widgets.HBox([prev_btn, next_btn]), out]))
        render()
    except Exception:
        from IPython.display import Image, Markdown, display

        display(Markdown(f"### Figure Preview: {stage}"))
        display(Image(filename=str(paths[0]), width=width))
        print("ipywidgets is unavailable. Showing the first figure only.")

    return [str(path) for path in paths]


LOCAL_LLM_SYSTEM_PROMPT = """You are an analyst. You receive JSONL files that describe a topic's language landscape.
The files are already computed. Use them to produce concise insights with evidence.
Be conservative. If the evidence looks noisy, generic, or off-topic, say so directly.
Do not convert weak or off-topic evidence into confident business advice.

Return valid JSON with:
{
  "overview": {
    "summary": "<2-4 sentences>",
    "analysis_mode": "domain|codomain_adjacent_interest",
    "off_topic_risk": "low|medium|high",
    "confidence": "low|medium|high"
  },
  "topic_interpretations": [
    {
      "topic_id": <int>,
      "label": "<string>",
      "relevance": "high|medium|low",
      "theme_summary": "<2-3 sentences explaining what the topic is, what kind of posts drive it, and why it matters or why it looks noisy>",
      "evidence": ["<short preview 1>", "<short preview 2>"]
    }
  ],
  "quality_flags": [
    {
      "flag": "<short label>",
      "severity": "low|medium|high",
      "explanation": "<what looks wrong or uncertain>"
    }
  ],
  "strategic_opportunities": [
    {
      "opportunity": "<short headline>",
      "activation_type": "offering|message|content|monitoring",
      "evidence_topic_ids": [<int>, <int>],
      "rationale": "<1-2 sentences>",
      "confidence": "low|medium|high"
    }
  ],
  "takeaways": ["<short takeaway>", "<short takeaway>"]
}
"""

LOCAL_LLM_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "overview": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "analysis_mode": {"type": "string", "enum": ["domain", "codomain_adjacent_interest"]},
                "off_topic_risk": {"type": "string", "enum": ["low", "medium", "high"]},
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            },
            "required": ["summary", "analysis_mode", "off_topic_risk", "confidence"],
        },
        "topic_interpretations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic_id": {"type": "integer"},
                    "label": {"type": "string"},
                    "relevance": {"type": "string", "enum": ["high", "medium", "low"]},
                    "theme_summary": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["topic_id", "label", "relevance", "theme_summary", "evidence"],
            },
        },
        "quality_flags": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "flag": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "explanation": {"type": "string"},
                },
                "required": ["flag", "severity", "explanation"],
            },
        },
        "strategic_opportunities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "opportunity": {"type": "string"},
                    "activation_type": {
                        "type": "string",
                        "enum": ["offering", "message", "content", "monitoring"],
                    },
                    "evidence_topic_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                    "rationale": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                },
                "required": ["opportunity", "activation_type", "evidence_topic_ids", "rationale", "confidence"],
            },
        },
        "takeaways": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["overview", "topic_interpretations", "quality_flags", "strategic_opportunities", "takeaways"],
}

LOCAL_LLM_TOPIC_REPAIR_SCHEMA = {
    "type": "object",
    "properties": {
        "topic_interpretations": LOCAL_LLM_JSON_SCHEMA["properties"]["topic_interpretations"],
    },
    "required": ["topic_interpretations"],
}


def _jsonl_text(records: Sequence[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(record, ensure_ascii=False) for record in records)


def _missing_topic_ids(summary_rows: Sequence[dict[str, Any]], items: Sequence[dict[str, Any]]) -> list[int]:
    expected = {int(row["topic_id"]) for row in summary_rows}
    present = {int(item["topic_id"]) for item in items if "topic_id" in item}
    return sorted(expected - present)


def _sanitize_enum(value: Any, allowed: Sequence[str], default: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in set(allowed):
        return normalized
    return default


def _llm_sentiment_terms(polarity_rows: Sequence[dict[str, Any]], terms_per_sentiment: int = 6) -> dict[str, list[dict[str, Any]]]:
    output = {label: [] for label in SENTIMENT_ORDER}
    for label in SENTIMENT_ORDER:
        subset = [row for row in polarity_rows if row.get("sentiment") == label][:terms_per_sentiment]
        output[label] = [
            {
                "term": str(row["term"]),
                "score": round(float(row["scores"]["fused_score"]), 6),
            }
            for row in subset
        ]
    return output


def _topic_relevance_diagnostics(
    config: SubjectConfig,
    dataset: str,
    polarity_rows: Sequence[dict[str, Any]],
    summary_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    defaults = CODOMAIN_FILTER_DEFAULTS.get(config.slug, {})
    include_terms = _normalize_terms(defaults.get("include_terms", []))
    exclude_terms = _normalize_terms(defaults.get("exclude_terms", []))

    unique_terms = []
    seen = set()
    for row in polarity_rows:
        term = str(row.get("term", "")).strip().lower()
        if not term or term in seen:
            continue
        seen.add(term)
        unique_terms.append(term)

    generic_terms = [term for term in unique_terms if term in DISPLAY_STOP_TOKENS]
    generic_term_share = float(len(generic_terms) / len(unique_terms)) if unique_terms else 0.0

    topic_rows = []
    subject_hits = 0
    noise_hits = 0
    for row in summary_rows:
        blob = " ".join(
            [
                str(row.get("label", "")),
                str(row.get("top_words", "")),
                *[str(value) for value in row.get("exemplars", [])],
            ]
        ).lower()
        matched_subject_terms = _matched_terms(blob, include_terms)
        matched_off_topic_terms = _matched_terms(blob, exclude_terms)
        if matched_subject_terms:
            subject_hits += 1
        if matched_off_topic_terms:
            noise_hits += 1
        topic_rows.append(
            {
                "topic_id": int(row["topic_id"]),
                "label": str(row["label"]),
                "matched_subject_terms": matched_subject_terms[:6],
                "matched_off_topic_terms": matched_off_topic_terms[:6],
            }
        )

    topic_count = max(1, len(summary_rows))
    subject_topic_share = float(subject_hits / topic_count)
    noise_topic_share = float(noise_hits / topic_count)

    if dataset == "domain":
        analysis_mode = "domain"
        if subject_topic_share >= 0.67 and noise_topic_share <= 0.25 and generic_term_share <= 0.35:
            off_topic_risk = "low"
            confidence = "high"
        elif subject_topic_share >= 0.34 and noise_topic_share <= 0.50 and generic_term_share <= 0.60:
            off_topic_risk = "medium"
            confidence = "medium"
        else:
            off_topic_risk = "high"
            confidence = "low"
    else:
        analysis_mode = "codomain_adjacent_interest"
        if noise_topic_share <= 0.25 and generic_term_share <= 0.45:
            off_topic_risk = "low"
            confidence = "high"
        elif noise_topic_share <= 0.50 and generic_term_share <= 0.65:
            off_topic_risk = "medium"
            confidence = "medium"
        else:
            off_topic_risk = "high"
            confidence = "low"

    flags = []
    if generic_terms:
        flags.append(
            {
                "flag": "generic_terms",
                "severity": "medium" if generic_term_share <= 0.50 else "high",
                "explanation": "Many top sentiment terms are generic verbs or filler tokens instead of subject-specific language.",
            }
        )
    if dataset == "domain" and subject_topic_share < 0.34:
        flags.append(
            {
                "flag": "weak_subject_focus",
                "severity": "high",
                "explanation": "The domain topics are not consistently anchored to the target subject.",
            }
        )
    if noise_topic_share > 0.34:
        flags.append(
            {
                "flag": "noise_topics",
                "severity": "high" if noise_topic_share > 0.67 else "medium",
                "explanation": "Several topic summaries contain excluded terms or look like noisy, low-signal content.",
            }
        )
    if dataset == "codomain" and subject_topic_share > 0.75:
        flags.append(
            {
                "flag": "codomain_collapse",
                "severity": "medium",
                "explanation": "The codomain still overlaps heavily with direct subject discussion instead of revealing adjacent interests.",
            }
        )

    return {
        "analysis_mode": analysis_mode,
        "off_topic_risk": off_topic_risk,
        "confidence": confidence,
        "generic_terms": generic_terms[:10],
        "generic_term_share": round(generic_term_share, 4),
        "subject_topic_share": round(subject_topic_share, 4),
        "noise_topic_share": round(noise_topic_share, 4),
        "topic_diagnostics": topic_rows,
        "quality_flags": flags,
    }


def _sanitize_overview(item: Any, default_mode: str, default_risk: str, default_confidence: str) -> dict[str, Any]:
    if not isinstance(item, dict):
        item = {}
    return {
        "summary": str(item.get("summary", "")).strip(),
        "analysis_mode": _sanitize_enum(item.get("analysis_mode"), ("domain", "codomain_adjacent_interest"), default_mode),
        "off_topic_risk": _sanitize_enum(item.get("off_topic_risk"), ("low", "medium", "high"), default_risk),
        "confidence": _sanitize_enum(item.get("confidence"), ("low", "medium", "high"), default_confidence),
    }


def _sanitize_topic_interpretations(items: Any, allowed_topic_ids: set[int]) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    cleaned = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            topic_id = int(item.get("topic_id"))
        except Exception:
            continue
        if topic_id not in allowed_topic_ids:
            continue
        evidence = item.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = []
        cleaned.append(
            {
                "topic_id": topic_id,
                "label": str(item.get("label", "")).strip(),
                "relevance": _sanitize_enum(item.get("relevance"), ("high", "medium", "low"), "medium"),
                "theme_summary": str(item.get("theme_summary", "")).strip(),
                "evidence": [str(value).strip() for value in evidence if str(value).strip()][:3],
            }
        )
    return cleaned


def _compact_preview(text: Any, limit: int = 140) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    value = re.sub(r"https?://\S+", "", value)
    value = re.sub(r"\s+", " ", value).strip(" \n\t-:,;")
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def _fallback_topic_theme_summary(row: dict[str, Any]) -> str:
    top_words = [token.strip() for token in str(row.get("top_words", "")).split(",") if token.strip()][:5]
    exemplar_previews = [_compact_preview(value, limit=96) for value in row.get("exemplars", []) if _compact_preview(value, limit=96)]

    parts = []
    if top_words:
        parts.append("This topic clusters around " + ", ".join(top_words) + ".")
    if exemplar_previews:
        parts.append("Representative posts mention " + "; ".join(exemplar_previews[:2]) + ".")
    return " ".join(parts).strip()


def _complete_topic_interpretations(
    summary_rows: Sequence[dict[str, Any]],
    items: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    item_map = {int(item["topic_id"]): item for item in items if "topic_id" in item}
    completed = []
    for row in summary_rows:
        topic_id = int(row["topic_id"])
        item = dict(item_map.get(topic_id, {}))
        label = str(item.get("label", "")).strip() or str(row.get("label", "")).strip()
        evidence = [str(value).strip() for value in item.get("evidence", []) if str(value).strip()][:3]
        if not evidence:
            evidence = [_compact_preview(value, limit=120) for value in row.get("exemplars", []) if _compact_preview(value, limit=120)][:3]
        theme_summary = str(item.get("theme_summary", "")).strip() or _fallback_topic_theme_summary(row)
        completed.append(
            {
                "topic_id": topic_id,
                "label": label,
                "relevance": _sanitize_enum(item.get("relevance"), ("high", "medium", "low"), "medium"),
                "theme_summary": theme_summary,
                "evidence": evidence,
            }
        )
    return completed


def _sanitize_quality_flags(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    cleaned = []
    for item in items:
        if not isinstance(item, dict):
            continue
        cleaned.append(
            {
                "flag": str(item.get("flag", "")).strip().lower(),
                "severity": _sanitize_enum(item.get("severity"), ("low", "medium", "high"), "medium"),
                "explanation": str(item.get("explanation", "")).strip(),
            }
        )
    return cleaned


def _sanitize_opportunities(items: Any, allowed_topic_ids: set[int] | None = None) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    cleaned = []
    for item in items:
        if not isinstance(item, dict):
            continue
        topic_ids = []
        for topic_id in item.get("evidence_topic_ids", []):
            try:
                parsed = int(topic_id)
            except Exception:
                continue
            if allowed_topic_ids is not None and parsed not in allowed_topic_ids:
                continue
            topic_ids.append(parsed)
        cleaned.append(
            {
                "opportunity": str(item.get("opportunity", "")).strip(),
                "activation_type": _sanitize_enum(
                    item.get("activation_type"),
                    ("offering", "message", "content", "monitoring"),
                    "monitoring",
                ),
                "evidence_topic_ids": topic_ids,
                "rationale": str(item.get("rationale", "")).strip(),
                "confidence": _sanitize_enum(item.get("confidence"), ("low", "medium", "high"), "medium"),
            }
        )
    return cleaned


def _sanitize_text_list(items: Any, max_items: int = 6) -> list[str]:
    if not isinstance(items, list):
        return []
    cleaned = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        cleaned.append(text)
    return cleaned[:max_items]


def _combine_quality_flags(model_flags: Sequence[dict[str, Any]], diagnostic_flags: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    severity_rank = {"low": 0, "medium": 1, "high": 2}
    combined: dict[str, dict[str, Any]] = {}
    for item in list(model_flags) + list(diagnostic_flags):
        flag = str(item.get("flag", "")).strip().lower()
        explanation = str(item.get("explanation", "")).strip()
        if not flag:
            continue
        candidate = {
            "flag": flag,
            "severity": _sanitize_enum(item.get("severity"), ("low", "medium", "high"), "medium"),
            "explanation": explanation,
        }
        current = combined.get(flag)
        if current is None:
            combined[flag] = candidate
            continue
        current_rank = severity_rank.get(current["severity"], 1)
        candidate_rank = severity_rank.get(candidate["severity"], 1)
        if candidate_rank > current_rank:
            combined[flag] = candidate
            continue
        if candidate_rank == current_rank and len(candidate["explanation"]) > len(current["explanation"]):
            combined[flag] = candidate
    generic = combined.pop("generic_terms", None)
    noisy = combined.pop("noise_topics", None)
    if generic and noisy:
        generic_rank = severity_rank.get(generic["severity"], 1)
        noisy_rank = severity_rank.get(noisy["severity"], 1)
        merged_rank = max(generic_rank, noisy_rank)
        merged_severity = next((name for name, rank in severity_rank.items() if rank == merged_rank), "medium")
        combined["low_signal_language"] = {
            "flag": "low_signal_language",
            "severity": merged_severity,
            "explanation": "Generic filler terms and noisy topic clusters are both reducing signal quality, so the corpus reads as low-density rather than clearly thematic.",
        }
    else:
        if generic is not None:
            combined["generic_terms"] = generic
        if noisy is not None:
            combined["noise_topics"] = noisy
    return list(combined.values())


def _merge_local_llm_output(bundle: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
    allowed_topic_ids = {int(row["topic_id"]) for row in bundle.get("topic_summary_rows", [])}
    diagnostics = bundle["deterministic_output"]["diagnostics"]
    cleaned_topics = _sanitize_topic_interpretations(parsed.get("topic_interpretations"), allowed_topic_ids)
    return {
        "subject": bundle["subject"],
        "dataset": bundle["dataset"],
        "overview": _sanitize_overview(
            parsed.get("overview"),
            default_mode=str(diagnostics.get("analysis_mode", "domain")),
            default_risk=str(diagnostics.get("off_topic_risk", "medium")),
            default_confidence=str(diagnostics.get("confidence", "medium")),
        ),
        "topic_interpretations": _complete_topic_interpretations(bundle.get("topic_summary_rows", []), cleaned_topics),
        "sentiment_terms": bundle["deterministic_output"]["sentiment_terms"],
        "quality_flags": _combine_quality_flags(
            _sanitize_quality_flags(parsed.get("quality_flags")),
            diagnostics.get("quality_flags", []),
        ),
        "strategic_opportunities": _sanitize_opportunities(
            parsed.get("strategic_opportunities"),
            allowed_topic_ids=allowed_topic_ids,
        ),
        "takeaways": _sanitize_text_list(parsed.get("takeaways") or parsed.get("poster_takeaways"), max_items=6),
        "diagnostics": diagnostics,
    }


def prepare_local_llm_stage(
    subject: str | SubjectConfig,
    dataset: str,
    top_terms: int = 12,
    top_topics: int | None = None,
) -> dict[str, Any]:
    config = get_subject_config(subject)
    tfidf = load_stage_table(config, "05_tfidf", dataset, "tfidf_metrics_tidy", "parquet")
    topic_summaries = load_stage_table(config, "06_topic_modeling", dataset, "topic_summaries", "csv")
    topic_exemplars = load_stage_table(config, "06_topic_modeling", dataset, "topic_exemplars", "csv")

    polarity_rows = []
    for label in SENTIMENT_ORDER:
        subset = (
            tfidf.loc[tfidf["sentiment"].eq(label)]
            .sort_values(["fused_score", "iwdf"], ascending=False)
            .head(top_terms)
        )
        for _, row in subset.iterrows():
            polarity_rows.append(
                {
                    "subject": config.slug,
                    "dataset": dataset,
                    "sentiment": label,
                    "term": row["term"],
                    "scores": {
                        "agg": float(row["agg"]),
                        "iwdf": float(row["iwdf"]),
                        "agg_norm": float(row["agg_norm"]),
                        "iwdf_norm": float(row["iwdf_norm"]),
                        "fused_score": float(row["fused_score"]),
                    },
                }
            )

    summary_source = topic_summaries if top_topics is None else topic_summaries.head(top_topics)
    summary_rows = []
    for _, row in summary_source.iterrows():
        summary_rows.append(
            {
                "subject": config.slug,
                "dataset": dataset,
                "topic_id": int(row["topic_id"]),
                "label": row["label"],
                "top_words": row["top_words"],
                "exemplars": [row.get("exemplar_1", ""), row.get("exemplar_2", ""), row.get("exemplar_3", "")],
            }
        )

    exemplar_rows = []
    exemplar_topics = {row["topic_id"] for row in summary_rows}
    for _, row in topic_exemplars.loc[topic_exemplars["topic_id"].isin(exemplar_topics)].iterrows():
        exemplar_rows.append(
            {
                "subject": config.slug,
                "dataset": dataset,
                "topic_id": int(row["topic_id"]),
                "rank": int(row["rank"]),
                "score": float(row["score"]),
                "doc_idx": int(row["doc_idx"]),
                "sentiment": row.get("sentiment", ""),
                "preview": row["preview"],
            }
        )

    polarity_path = artifact_path(config, "07_local_llm", dataset, "polar_terms", "jsonl")
    summary_path = artifact_path(config, "07_local_llm", dataset, "topic_summaries", "jsonl")
    exemplars_path = artifact_path(config, "07_local_llm", dataset, "topic_exemplars", "jsonl")
    deterministic_path = artifact_path(config, "07_local_llm", dataset, "deterministic_output", "json")
    prompt_path = artifact_path(config, "07_local_llm", dataset, "prompt", "txt")

    write_jsonl(polarity_path, polarity_rows)
    write_jsonl(summary_path, summary_rows)
    write_jsonl(exemplars_path, exemplar_rows)

    diagnostics = _topic_relevance_diagnostics(config, dataset, polarity_rows, summary_rows)
    deterministic_output = {
        "subject": config.slug,
        "dataset": dataset,
        "sentiment_terms": _llm_sentiment_terms(polarity_rows),
        "diagnostics": diagnostics,
    }
    write_json(deterministic_path, deterministic_output)

    prompt = (
        LOCAL_LLM_SYSTEM_PROMPT.strip()
        + "\n\nTASK CONTEXT\n"
        + f"subject={config.slug}\n"
        + f"dataset={dataset}\n"
        + (
            "analysis_goal=Summarize the direct language and themes in the subject conversation.\n"
            if dataset == "domain"
            else "analysis_goal=Summarize the adjacent interests of users who also talk about the subject. Codomain topics do not need to mention the subject directly. Focus on what these users care about besides the subject, and what that suggests for offerings, positioning, or talking points. Only treat content as low-quality when it looks spammy, generic, or dominated by excluded/noise terms.\n"
        )
        + "\nPOLAR_TERMS_JSONL\n"
        + _jsonl_text(polarity_rows)
        + "\n\nTOPIC_SUMMARIES_JSONL\n"
        + _jsonl_text(summary_rows)
        + "\n\nTOPIC_EXEMPLARS_JSONL\n"
        + _jsonl_text(exemplar_rows)
        + "\n\nDIAGNOSTICS_JSON\n"
        + json.dumps(diagnostics, ensure_ascii=False)
        + "\n\nEND_INPUT_DATA\n"
        + "Do not continue, copy, or transform any single input JSON row.\n"
        + "Synthesize across all of the supplied records.\n"
        + "Return exactly one JSON object matching the schema above.\n"
        + "Do not return subject, dataset, sentiment_terms, or diagnostics. Those fields are merged deterministically in code.\n"
        + "Use only topic_id values that appear in TOPIC_SUMMARIES_JSONL.\n"
        + "The diagnostics are heuristics, not truth, but you should respect them when judging off-topic risk and confidence.\n"
        + "Include one topic_interpretation for every topic_id in TOPIC_SUMMARIES_JSONL.\n"
        + "Each topic_interpretation label must be a short human-readable caption, not a bag-of-words list.\n"
        + "Each topic_interpretation theme_summary must be 2 to 3 sentences.\n"
        + "Sentence 1 should explain the shared theme in plain English.\n"
        + "Sentence 2 should explain what the exemplar posts suggest about the kind of discussion, audience, or behavior driving the topic.\n"
        + "Sentence 3, when useful, should explain why the topic matters for the subject or why it should be treated as noise, off-topic spillover, or low-signal chatter.\n"
        + "Do not simply restate the top words; translate them into a readable interpretation.\n"
        + "Even when a topic is weak or messy, still provide a concrete interpretation rather than omitting it.\n"
        + "Include 1 to 4 quality_flags, 1 to 4 strategic_opportunities, and 2 to 5 takeaways.\n"
        + "Use short evidence snippets drawn from the exemplar previews.\n"
        + "For codomain, the strategic opportunities should reflect adjacent interests of subject-linked users, not generic campaign or product advice.\n"
        + "For codomain, only include a strategic opportunity when it is supported by repeated evidence across multiple topics or exemplars. Do not turn single-account meme streams or fragmented chatter into an opportunity.\n"
        + "If the codomain evidence is too fragmented for a real opportunity, return an empty strategic_opportunities list.\n"
        + "Every strategic_opportunity must include an activation_type and evidence_topic_ids.\n"
        + "Do not write generic audience recommendations. Focus on interpretation, quality, opportunities, and presentation-ready takeaways.\n"
        + "Return JSON only.\n"
        + "\nOUTPUT_JSON\n"
    )
    prompt_path.write_text(prompt + "\n", encoding="utf-8")

    bundle = {
        "subject": config.slug,
        "dataset": dataset,
        "polar_terms_path": str(polarity_path),
        "topic_summaries_path": str(summary_path),
        "topic_exemplars_path": str(exemplars_path),
        "deterministic_output_path": str(deterministic_path),
        "prompt_path": str(prompt_path),
        "prompt": prompt,
        "deterministic_output": deterministic_output,
        "topic_summary_rows": summary_rows,
        "topic_exemplar_rows": exemplar_rows,
        "available_backends": available_local_llm_backends(),
        "model_options": LOCAL_LLM_OPTIONS,
    }
    write_json(artifact_path(config, "07_local_llm", dataset, "bundle_manifest", "json"), bundle)
    return bundle


def _topic_interpretations_prompt(bundle: dict[str, Any], topic_ids: Sequence[int] | None = None) -> str:
    if topic_ids is None:
        topic_ids = [int(row["topic_id"]) for row in bundle.get("topic_summary_rows", [])]
    target_set = {int(topic_id) for topic_id in topic_ids}
    summary_rows = [row for row in bundle.get("topic_summary_rows", []) if int(row["topic_id"]) in target_set]
    exemplar_rows = [row for row in bundle.get("topic_exemplar_rows", []) if int(row["topic_id"]) in target_set]
    return (
        "You are generating topic interpretations for a previously prepared corpus analysis.\n"
        "Return valid JSON only.\n"
        "Return exactly one object with a single key: topic_interpretations.\n"
        "Include one topic_interpretation for every topic_id listed below.\n"
        "Do not omit any listed topic_id.\n"
        "Each label must be a short human-readable caption, not a bag-of-words list.\n"
        "Each theme_summary must be 2 to 3 sentences.\n"
        "Sentence 1 should explain the theme in plain English.\n"
        "Sentence 2 should explain what the exemplar posts suggest about the discussion, audience, or behavior.\n"
        "Sentence 3, when useful, should explain why the topic matters or why it should be treated as noise.\n"
        "Use only the evidence provided below. Be conservative.\n"
        "\nTASK CONTEXT\n"
        f"subject={bundle['subject']}\n"
        f"dataset={bundle['dataset']}\n"
        f"topic_ids={','.join(str(topic_id) for topic_id in topic_ids)}\n"
        "\nTOPIC_SUMMARIES_JSONL\n"
        + _jsonl_text(summary_rows)
        + "\n\nTOPIC_EXEMPLARS_JSONL\n"
        + _jsonl_text(exemplar_rows)
        + "\n\nOUTPUT_JSON\n"
    )


def _generate_topic_interpretations(
    bundle: dict[str, Any],
    *,
    backend: str,
    model: str | None,
    max_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    topic_rows = bundle.get("topic_summary_rows", [])
    results: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []

    for row in topic_rows:
        topic_id = int(row["topic_id"])
        prompt = _topic_interpretations_prompt(bundle, [topic_id])
        output_text = run_local_llm(
            prompt,
            backend=backend,
            model=model,
            max_tokens=min(max_tokens, 1024),
            response_schema=LOCAL_LLM_TOPIC_REPAIR_SCHEMA,
        )
        record = {
            "topic_id": topic_id,
            "label": str(row.get("label", "")),
            "status": "empty_output",
            "raw_char_count": len(output_text or ""),
        }
        if output_text and output_text.strip():
            try:
                parsed = json.loads(output_text)
            except Exception as exc:
                record["status"] = "parse_error"
                record["error"] = str(exc)
            else:
                cleaned = _sanitize_topic_interpretations(parsed.get("topic_interpretations"), {topic_id})
                if cleaned:
                    results.append(cleaned[0])
                    record["status"] = "ok"
                else:
                    record["status"] = "missing_topic"
        debug_rows.append(record)
    return results, debug_rows


def run_local_llm(
    prompt: str,
    backend: str = "prompt_only",
    model: str | None = None,
    max_tokens: int = 3072,
    response_schema: dict[str, Any] | None = None,
) -> str | None:
    backend = backend.strip().lower()
    if backend == "prompt_only":
        return None

    if backend == "ollama":
        if not shutil.which("ollama"):
            raise RuntimeError("Ollama is not installed or not on PATH.")
        if not model:
            raise ValueError("A model name is required for Ollama.")
        if not start_ollama_server():
            raise RuntimeError("Ollama server is not running.")
        payload = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "format": response_schema or LOCAL_LLM_JSON_SCHEMA,
                "options": {
                    "num_predict": int(max_tokens),
                    "temperature": 0.2,
                },
            }
        ).encode("utf-8")
        request = Request(
            "http://127.0.0.1:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=600) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(detail or f"Ollama HTTP error: {exc.code}") from exc
        except URLError as exc:
            raise RuntimeError(f"Could not reach Ollama: {exc}") from exc
        return data.get("response", "")

    if backend == "mlx_lm":
        if not model:
            raise ValueError("A model name is required for MLX.")
        result = subprocess.run(
            [
                "python3",
                "-m",
                "mlx_lm.generate",
                "--model",
                model,
                "--prompt",
                prompt,
                "--max-tokens",
                str(max_tokens),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"mlx_lm exited with code {result.returncode}")
        return result.stdout

    if backend == "llama_cpp":
        if not model:
            raise ValueError("A model path is required for llama.cpp / llama_cpp_python.")
        from llama_cpp import Llama

        llm = Llama(model_path=model, n_ctx=8192, verbose=False)
        completion = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return completion["choices"][0]["message"]["content"]

    raise ValueError(f"Unsupported backend: {backend}")


def run_local_llm_stage(
    subject: str | SubjectConfig,
    dataset: str,
    backend: str = "prompt_only",
    model: str | None = None,
    max_tokens: int = 3072,
) -> dict[str, Any]:
    config = get_subject_config(subject)
    bundle = prepare_local_llm_stage(config, dataset)
    output_text = run_local_llm(bundle["prompt"], backend=backend, model=model, max_tokens=max_tokens)

    result = {
        "subject": config.slug,
        "dataset": dataset,
        "backend": backend,
        "model": model,
        "prompt_path": bundle["prompt_path"],
        "raw_result_path": None,
        "result_path": None,
        "parsed_output": None,
        "status": "not_run",
        "note": None,
        "raw_char_count": 0,
        "topic_debug_path": None,
    }

    if output_text is None:
        result["status"] = "prompt_only"
        return result

    raw_output_path = artifact_path(config, "07_local_llm", dataset, "local_llm_output", "txt")
    raw_output_path.write_text(output_text, encoding="utf-8")
    result["raw_result_path"] = str(raw_output_path)
    result["result_path"] = str(raw_output_path)
    result["raw_char_count"] = len(output_text)

    if not output_text.strip():
        result["status"] = "empty_output"
        result["note"] = "The model returned an empty response. The previous structured JSON, if any, was left unchanged."
        return result

    try:
        parsed = json.loads(output_text)
    except Exception as exc:
        parsed = None
        result["status"] = "parse_error"
        result["note"] = f"Model returned non-JSON output: {exc}"

    if parsed is not None:
        dedicated_topics, topic_debug_rows = _generate_topic_interpretations(
            bundle,
            backend=backend,
            model=model,
            max_tokens=max_tokens,
        )
        dedicated_topic_count = len(dedicated_topics)
        parsed["topic_interpretations"] = dedicated_topics

        topic_debug_path = artifact_path(config, "07_local_llm", dataset, "topic_call_status", "json")
        write_json(topic_debug_path, topic_debug_rows)
        result["topic_debug_path"] = str(topic_debug_path)

        parsed_path = artifact_path(config, "07_local_llm", dataset, "local_llm_output", "json")
        merged = _merge_local_llm_output(bundle, parsed)
        write_json(parsed_path, merged)
        result["result_path"] = str(parsed_path)
        result["parsed_output"] = merged
        result["status"] = "ok"
        unresolved = _missing_topic_ids(bundle.get("topic_summary_rows", []), merged.get("topic_interpretations", []))
        if dedicated_topic_count:
            result["note"] = f"Structured JSON saved successfully. Dedicated topic pass returned {dedicated_topic_count} topic interpretations."
        elif unresolved:
            result["note"] = f"Structured JSON saved successfully, but {len(unresolved)} topic interpretations still required deterministic fallback."
        else:
            result["note"] = "Structured JSON saved successfully."

    return result


def codomain_filter_defaults(subject: str | SubjectConfig) -> dict[str, Any]:
    config = get_subject_config(subject)
    defaults = CODOMAIN_FILTER_DEFAULTS.get(config.slug, {})
    return json.loads(json.dumps(defaults))


def drop_retweets_and_replies(
    df: pd.DataFrame,
    drop_retweets: bool = False,
    drop_replies: bool = False,
) -> pd.DataFrame:
    filtered = df.copy()
    if drop_retweets:
        filtered = filtered.loc[~filtered["Content"].fillna("").astype(str).str.startswith("RT ")]
    if drop_replies:
        filtered = filtered.loc[~filtered["Content"].fillna("").astype(str).str.startswith("@")]
    return filtered.reset_index(drop=True)


def cap_posts_per_user(df: pd.DataFrame, max_posts_per_user: int = 20) -> pd.DataFrame:
    if "Author ID" not in df.columns:
        return df.copy()
    working = df.copy()
    if "Date" in working.columns:
        working = working.assign(_parsed_date=pd.to_datetime(working["Date"], errors="coerce"))
        working = working.sort_values(["_parsed_date", "Author ID"], ascending=[False, True])
    else:
        working = working.sort_values("Author ID", ascending=True)
    limited = working.groupby("Author ID", group_keys=False).head(max_posts_per_user).reset_index(drop=True)
    return limited.drop(columns=["_parsed_date"], errors="ignore")


def _normalize_terms(terms: Iterable[str]) -> list[str]:
    output = []
    seen = set()
    for term in terms:
        normalized = str(term).strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def _matched_terms(text: str, terms: Sequence[str]) -> list[str]:
    lowered = str(text or "").lower()
    matches = []
    for term in terms:
        pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", flags=re.IGNORECASE)
        if pattern.search(lowered):
            matches.append(term)
    return matches


def score_topic_relevance_codomain(
    df: pd.DataFrame,
    include_terms: Iterable[str],
    exclude_terms: Iterable[str] = (),
) -> pd.DataFrame:
    include_terms_norm = _normalize_terms(include_terms)
    exclude_terms_norm = _normalize_terms(exclude_terms)
    working = df.copy()
    content = working["Content"].fillna("").astype(str)
    working["include_terms_matched"] = content.map(lambda value: _matched_terms(value, include_terms_norm))
    working["exclude_terms_matched"] = content.map(lambda value: _matched_terms(value, exclude_terms_norm))
    working["include_match_count"] = working["include_terms_matched"].map(len)
    working["exclude_match_count"] = working["exclude_terms_matched"].map(len)
    working["topic_relevance_score"] = working["include_match_count"] - working["exclude_match_count"]
    return working


def filter_topic_relevant_codomain(
    df: pd.DataFrame,
    include_terms: Iterable[str],
    exclude_terms: Iterable[str] = (),
    min_include_matches: int = 1,
    max_exclude_matches: int | None = 0,
    min_relevance_score: int = 1,
) -> pd.DataFrame:
    scored = score_topic_relevance_codomain(df, include_terms=include_terms, exclude_terms=exclude_terms)
    mask = scored["include_match_count"].ge(int(min_include_matches)) & scored["topic_relevance_score"].ge(int(min_relevance_score))
    if max_exclude_matches is not None:
        mask &= scored["exclude_match_count"].le(int(max_exclude_matches))
    return scored.loc[mask].copy().reset_index(drop=True)


def annotate_codomain_post_noise(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    content = working["Content"].fillna("").astype(str)
    working["url_count"] = content.str.count(r"https?://|www\.")
    working["cashtag_count"] = content.str.count(r"(?<!\w)\$[A-Za-z]{1,6}\b")
    working["alpha_word_count"] = content.str.findall(r"[A-Za-z]{2,}").map(len)
    return working


def filter_codomain_noise_posts(
    df: pd.DataFrame,
    max_urls: int = 1,
    drop_cashtags: bool = True,
    min_alpha_words_with_url: int = 6,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    annotated = annotate_codomain_post_noise(df)
    mask = annotated["url_count"].le(int(max_urls))
    if drop_cashtags:
        mask &= annotated["cashtag_count"].eq(0)
    mask &= ~(
        annotated["url_count"].gt(0)
        & annotated["alpha_word_count"].lt(int(min_alpha_words_with_url))
    )
    filtered = annotated.loc[mask].copy().reset_index(drop=True)
    return filtered, {
        "max_urls": int(max_urls),
        "drop_cashtags": bool(drop_cashtags),
        "min_alpha_words_with_url": int(min_alpha_words_with_url),
    }


def drop_direct_subject_posts(
    df: pd.DataFrame,
    min_include_matches: int = 1,
) -> pd.DataFrame:
    if "include_match_count" not in df.columns:
        return df.copy()
    mask = df["include_match_count"].lt(int(min_include_matches))
    return df.loc[mask].copy().reset_index(drop=True)


def _codomain_author_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "Author ID" not in df.columns:
        return pd.DataFrame()

    working = df.copy()
    if "include_match_count" not in working.columns:
        working["include_match_count"] = 0
    if "exclude_match_count" not in working.columns:
        working["exclude_match_count"] = 0
    if "topic_relevance_score" not in working.columns:
        working["topic_relevance_score"] = 0.0
    if "domain_similarity" not in working.columns:
        working["domain_similarity"] = np.nan
    if "Author Username" not in working.columns:
        working["Author Username"] = ""
    if "url_count" not in working.columns:
        working = annotate_codomain_post_noise(working)
    if "Author Blue Verified (T/F)" not in working.columns:
        working["Author Blue Verified (T/F)"] = ""

    summary = (
        working.groupby("Author ID", dropna=False)
        .agg(
            posts=("Content", "size"),
            subject_posts=("include_match_count", lambda s: int(s.gt(0).sum())),
            include_hits=("include_match_count", "sum"),
            exclude_hits=("exclude_match_count", "sum"),
            avg_relevance=("topic_relevance_score", "mean"),
            avg_domain_similarity=("domain_similarity", "mean"),
            max_domain_similarity=("domain_similarity", "max"),
            author_username=("Author Username", "first"),
            url_share=("url_count", lambda s: float(s.gt(0).mean())),
            blue_verified=("Author Blue Verified (T/F)", "first"),
        )
        .reset_index()
    )
    summary["focus_share"] = np.where(
        summary["posts"].gt(0),
        summary["subject_posts"] / summary["posts"],
        0.0,
    )
    return summary.sort_values(
        ["focus_share", "subject_posts", "include_hits", "avg_relevance", "avg_domain_similarity"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)


def filter_codomain_authors_by_focus(
    df: pd.DataFrame,
    min_author_include_hits: int = 2,
    min_author_focus_share: float = 0.20,
    max_author_url_share: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "Author ID" not in df.columns or "include_match_count" not in df.columns:
        return df.copy(), pd.DataFrame()

    author_summary = _codomain_author_summary(df)
    keep_mask = author_summary["subject_posts"].ge(int(min_author_include_hits)) & author_summary["focus_share"].ge(
        float(min_author_focus_share)
    )
    if max_author_url_share is not None:
        keep_mask &= author_summary["url_share"].le(float(max_author_url_share))
    keep_authors = author_summary.loc[keep_mask, "Author ID"]
    filtered = df.loc[df["Author ID"].isin(set(keep_authors))].copy().reset_index(drop=True)
    kept_summary = author_summary.loc[author_summary["Author ID"].isin(set(keep_authors))].reset_index(drop=True)
    return filtered, kept_summary


def score_codomain_domain_similarity(
    subject: str | SubjectConfig,
    df: pd.DataFrame,
    reference_dataset: str = "domain",
    min_df: int = 2,
    max_df: float = 0.85,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    config = get_subject_config(subject)
    scored = df.copy().reset_index(drop=True)
    if scored.empty:
        scored["domain_similarity"] = pd.Series(dtype=float)
        return scored, {"rows_scored": 0}

    reference = load_stage_table(config, "02_preprocess", reference_dataset, "preprocessed", "parquet")
    reference_text = reference["clean_nostop"].fillna("").astype(str).str.strip()
    reference_text = reference_text.loc[reference_text.str.len().gt(0)].reset_index(drop=True)
    if reference_text.empty:
        raise ValueError(f"No usable reference text found for subject={config.slug}, dataset={reference_dataset}.")

    prepared, _, _ = preprocess_corpus(scored, drop_duplicate_content=False)
    candidate_text = prepared["clean_nostop"].fillna("").astype(str).str.strip()
    candidate_text = candidate_text.loc[candidate_text.str.len().gt(0)]
    prepared = prepared.loc[candidate_text.index].reset_index(drop=True)
    candidate_text = candidate_text.reset_index(drop=True)

    scored["domain_similarity"] = np.nan
    if prepared.empty:
        return scored, {"rows_scored": 0}

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=max_df,
        norm="l2",
        stop_words=None,
    )
    matrix = vectorizer.fit_transform(pd.concat([reference_text, candidate_text], ignore_index=True))
    ref_matrix = matrix[: len(reference_text)]
    cand_matrix = matrix[len(reference_text) :]
    centroid = np.asarray(ref_matrix.mean(axis=0))
    similarities = cosine_similarity(cand_matrix, centroid).ravel()

    match_columns = [column for column in ("Tweet ID", "Content") if column in prepared.columns]
    score_rows = prepared.loc[:, match_columns].copy()
    score_rows["domain_similarity"] = similarities
    if "Tweet ID" in score_rows.columns:
        score_rows["Tweet ID"] = score_rows["Tweet ID"].astype("string")
        score_rows = score_rows.drop_duplicates(subset=["Tweet ID"], keep="first")
    else:
        score_rows = score_rows.drop_duplicates(subset=["Content"], keep="first")

    if "Tweet ID" in scored.columns and "Tweet ID" in score_rows.columns:
        scored["Tweet ID"] = scored["Tweet ID"].astype("string")
        scored = scored.merge(score_rows[["Tweet ID", "domain_similarity"]], on="Tweet ID", how="left", suffixes=("", "_new"))
        scored["domain_similarity"] = scored["domain_similarity_new"].combine_first(scored["domain_similarity"])
        scored = scored.drop(columns=["domain_similarity_new"], errors="ignore")
    elif "Content" in scored.columns and "Content" in score_rows.columns:
        scored = scored.merge(score_rows[["Content", "domain_similarity"]], on="Content", how="left", suffixes=("", "_new"))
        scored["domain_similarity"] = scored["domain_similarity_new"].combine_first(scored["domain_similarity"])
        scored = scored.drop(columns=["domain_similarity_new"], errors="ignore")

    valid = scored["domain_similarity"].dropna()
    return scored, {
        "rows_scored": int(valid.shape[0]),
        "mean_domain_similarity": float(valid.mean()) if len(valid) else None,
        "median_domain_similarity": float(valid.median()) if len(valid) else None,
        "p75_domain_similarity": float(valid.quantile(0.75)) if len(valid) else None,
        "max_domain_similarity": float(valid.max()) if len(valid) else None,
    }


def filter_codomain_by_domain_similarity(
    df: pd.DataFrame,
    min_similarity: float | None = None,
    similarity_quantile: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if "domain_similarity" not in df.columns:
        return df.copy(), {"threshold": None}

    working = df.copy()
    valid = working["domain_similarity"].dropna()
    if valid.empty:
        return working.iloc[0:0].copy(), {"threshold": None}

    threshold = None
    if similarity_quantile is not None:
        threshold = float(valid.quantile(float(similarity_quantile)))
    if min_similarity is not None:
        threshold = max(float(min_similarity), threshold) if threshold is not None else float(min_similarity)

    mask = working["domain_similarity"].notna()
    if threshold is not None:
        mask &= working["domain_similarity"].ge(float(threshold))

    filtered = working.loc[mask].copy().reset_index(drop=True)
    return filtered, {
        "threshold": threshold,
        "similarity_quantile": None if similarity_quantile is None else float(similarity_quantile),
        "min_similarity": None if min_similarity is None else float(min_similarity),
    }


def _codomain_stage_summary(stage_name: str, frame: pd.DataFrame) -> dict[str, Any]:
    author_count = frame["Author ID"].nunique() if "Author ID" in frame.columns else None
    summary = {
        "stage": stage_name,
        "rows": int(len(frame)),
        "unique_authors": int(author_count) if author_count is not None and not pd.isna(author_count) else None,
    }
    if "include_match_count" in frame.columns:
        summary["mean_include_matches"] = float(frame["include_match_count"].mean()) if len(frame) else 0.0
        summary["mean_exclude_matches"] = float(frame["exclude_match_count"].mean()) if len(frame) else 0.0
        summary["mean_relevance_score"] = float(frame["topic_relevance_score"].mean()) if len(frame) else 0.0
    if "domain_similarity" in frame.columns:
        valid = frame["domain_similarity"].dropna()
        summary["mean_domain_similarity"] = float(valid.mean()) if len(valid) else None
        summary["median_domain_similarity"] = float(valid.median()) if len(valid) else None
    return summary


def build_codomain_variant(
    subject: str | SubjectConfig,
    df: pd.DataFrame,
    variant_name: str,
    include_terms: Iterable[str],
    exclude_terms: Iterable[str] = (),
    drop_retweets: bool = True,
    drop_replies: bool = False,
    max_posts_per_user: int = 12,
    min_include_matches: int = 1,
    max_exclude_matches: int | None = 0,
    min_relevance_score: int = 1,
    min_author_include_hits: int = 2,
    min_author_focus_share: float = 0.20,
    max_author_url_share: float | None = None,
    drop_direct_subject_posts_from_kept_users: bool = False,
    drop_link_heavy_noise: bool = False,
    max_urls: int = 1,
    drop_cashtags: bool = True,
    min_alpha_words_with_url: int = 6,
    domain_similarity_quantile: float | None = None,
    domain_similarity_min: float | None = None,
    sample_size: int = 12,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    config = get_subject_config(subject)
    variant_slug = re.sub(r"[^a-z0-9]+", "_", str(variant_name).strip().lower()).strip("_") or "variant"

    baseline = df.copy().reset_index(drop=True)
    stage_rows = [_codomain_stage_summary("baseline", baseline)]

    noise_filtered = drop_retweets_and_replies(
        baseline,
        drop_retweets=drop_retweets,
        drop_replies=drop_replies,
    )
    stage_rows.append(_codomain_stage_summary("drop_rt_reply_noise", noise_filtered))

    capped = cap_posts_per_user(noise_filtered, max_posts_per_user=max_posts_per_user)
    stage_rows.append(_codomain_stage_summary("cap_posts_per_user", capped))

    scored = score_topic_relevance_codomain(capped, include_terms=include_terms, exclude_terms=exclude_terms)
    stage_rows.append(_codomain_stage_summary("scored", scored))

    tweet_filtered = filter_topic_relevant_codomain(
        capped,
        include_terms=include_terms,
        exclude_terms=exclude_terms,
        min_include_matches=min_include_matches,
        max_exclude_matches=max_exclude_matches,
        min_relevance_score=min_relevance_score,
    )
    stage_rows.append(_codomain_stage_summary("subject_link_post_filter", tweet_filtered))

    author_scoped, author_summary = filter_codomain_authors_by_focus(
        scored,
        min_author_include_hits=min_author_include_hits,
        min_author_focus_share=min_author_focus_share,
        max_author_url_share=max_author_url_share,
    )
    stage_rows.append(_codomain_stage_summary("author_focus_filter", author_scoped))

    working = author_scoped.copy()

    if drop_direct_subject_posts_from_kept_users:
        working = drop_direct_subject_posts(working, min_include_matches=min_include_matches)
        stage_rows.append(_codomain_stage_summary("drop_direct_subject_posts", working))

    noise_filter_meta = {}
    if drop_link_heavy_noise:
        working, noise_filter_meta = filter_codomain_noise_posts(
            working,
            max_urls=max_urls,
            drop_cashtags=drop_cashtags,
            min_alpha_words_with_url=min_alpha_words_with_url,
        )
        stage_rows.append(_codomain_stage_summary("drop_link_heavy_noise", working))

    similarity_scored = working.copy()
    similarity_summary = {}
    if domain_similarity_quantile is not None or domain_similarity_min is not None:
        similarity_scored, similarity_stats = score_codomain_domain_similarity(config, working)
        stage_rows.append(_codomain_stage_summary("domain_similarity_scored", similarity_scored))
        similarity_scored, similarity_filter_meta = filter_codomain_by_domain_similarity(
            similarity_scored,
            min_similarity=domain_similarity_min,
            similarity_quantile=domain_similarity_quantile,
        )
        similarity_summary = {**similarity_stats, **similarity_filter_meta}
        stage_rows.append(_codomain_stage_summary("domain_similarity_filter", similarity_scored))

    final_df = similarity_scored.copy().reset_index(drop=True)
    stage_rows.append(_codomain_stage_summary("final_variant", final_df))

    audit_df = pd.DataFrame(stage_rows)

    keep_ids = set(final_df["Tweet ID"].astype(str)) if "Tweet ID" in final_df.columns else set()
    drop_source = scored
    tweet_key_series = drop_source["Tweet ID"].astype(str) if "Tweet ID" in drop_source.columns else drop_source.index.astype(str)
    dropped_df = drop_source.loc[~tweet_key_series.isin(keep_ids)].copy().reset_index(drop=True)

    preview_columns = [
        column
        for column in [
            "Tweet ID",
            "Author Username",
            "Original User Sentiment",
            "include_match_count",
            "exclude_match_count",
            "topic_relevance_score",
            "domain_similarity",
            "url_count",
            "cashtag_count",
            "alpha_word_count",
            "include_terms_matched",
            "exclude_terms_matched",
            "Content",
        ]
        if column in final_df.columns or column in dropped_df.columns
    ]
    kept_columns = [column for column in preview_columns if column in final_df.columns]
    dropped_columns = [column for column in preview_columns if column in dropped_df.columns]
    kept_sample = final_df.loc[:, kept_columns].head(sample_size).copy()
    dropped_sample = dropped_df.loc[:, dropped_columns].head(sample_size).copy()

    corpus_csv = artifact_path(config, "98_experiments", variant_slug, "filtered_corpus", "csv")
    corpus_parquet = artifact_path(config, "98_experiments", variant_slug, "filtered_corpus", "parquet")
    audit_csv = artifact_path(config, "98_experiments", variant_slug, "filter_audit", "csv")
    authors_csv = artifact_path(config, "98_experiments", variant_slug, "author_focus_summary", "csv")
    kept_csv = artifact_path(config, "98_experiments", variant_slug, "kept_sample", "csv")
    dropped_csv = artifact_path(config, "98_experiments", variant_slug, "dropped_sample", "csv")
    manifest_json = artifact_path(config, "98_experiments", variant_slug, "variant_manifest", "json")

    final_df.to_csv(corpus_csv, index=False)
    final_df.to_parquet(corpus_parquet, index=False)
    audit_df.to_csv(audit_csv, index=False)
    author_summary.to_csv(authors_csv, index=False)
    kept_sample.to_csv(kept_csv, index=False)
    dropped_sample.to_csv(dropped_csv, index=False)

    figures = []
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(audit_df["stage"], audit_df["rows"], color="#4e79a7")
    ax.set_title(f"{config.display_name} Codomain Filter Audit")
    ax.set_ylabel("Rows")
    ax.tick_params(axis="x", rotation=30)
    figures.append(
        _save_figure(
            fig,
            artifact_path(config, "98_experiments", variant_slug, "filter_audit_rows", "png", subdir="figures"),
        )
    )
    plt.close(fig)

    if len(scored):
        fig, ax = plt.subplots(figsize=(8, 4))
        bins = range(int(scored["topic_relevance_score"].min()), int(scored["topic_relevance_score"].max()) + 2)
        ax.hist(scored["topic_relevance_score"], bins=bins, color="#59a14f", align="left", rwidth=0.85)
        ax.set_title(f"{config.display_name} Codomain Relevance Score Distribution")
        ax.set_xlabel("Topic relevance score")
        ax.set_ylabel("Tweets")
        figures.append(
            _save_figure(
                fig,
                artifact_path(config, "98_experiments", variant_slug, "relevance_score_distribution", "png", subdir="figures"),
            )
        )
        plt.close(fig)

    if "domain_similarity" in similarity_scored.columns and similarity_scored["domain_similarity"].notna().any():
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(similarity_scored["domain_similarity"].dropna(), bins=20, color="#9c755f", edgecolor="white")
        ax.set_title(f"{config.display_name} Codomain Domain-Similarity Distribution")
        ax.set_xlabel("Cosine similarity to domain centroid")
        ax.set_ylabel("Tweets")
        figures.append(
            _save_figure(
                fig,
                artifact_path(config, "98_experiments", variant_slug, "domain_similarity_distribution", "png", subdir="figures"),
            )
        )
        plt.close(fig)

    manifest = {
        "subject": config.slug,
        "variant": variant_slug,
        "filters": {
            "include_terms": list(_normalize_terms(include_terms)),
            "exclude_terms": list(_normalize_terms(exclude_terms)),
            "drop_retweets": bool(drop_retweets),
            "drop_replies": bool(drop_replies),
            "max_posts_per_user": int(max_posts_per_user),
            "min_include_matches": int(min_include_matches),
            "max_exclude_matches": None if max_exclude_matches is None else int(max_exclude_matches),
            "min_relevance_score": int(min_relevance_score),
            "min_author_include_hits": int(min_author_include_hits),
            "min_author_focus_share": float(min_author_focus_share),
            "max_author_url_share": None if max_author_url_share is None else float(max_author_url_share),
            "drop_direct_subject_posts_from_kept_users": bool(drop_direct_subject_posts_from_kept_users),
            "drop_link_heavy_noise": bool(drop_link_heavy_noise),
            "max_urls": int(max_urls),
            "drop_cashtags": bool(drop_cashtags),
            "min_alpha_words_with_url": int(min_alpha_words_with_url),
            "domain_similarity_quantile": None if domain_similarity_quantile is None else float(domain_similarity_quantile),
            "domain_similarity_min": None if domain_similarity_min is None else float(domain_similarity_min),
        },
        "similarity_summary": similarity_summary,
        "noise_filter_meta": noise_filter_meta,
        "corpus_csv_path": str(corpus_csv),
        "corpus_parquet_path": str(corpus_parquet),
        "audit_path": str(audit_csv),
        "author_summary_path": str(authors_csv),
        "kept_sample_path": str(kept_csv),
        "dropped_sample_path": str(dropped_csv),
        "figures": figures,
    }
    write_json(manifest_json, manifest)
    manifest["manifest_path"] = str(manifest_json)
    manifest["audit"] = audit_df
    manifest["author_summary"] = author_summary
    manifest["kept_sample"] = kept_sample
    manifest["dropped_sample"] = dropped_sample
    return manifest


def pipeline_stage_status(subject: str | SubjectConfig) -> pd.DataFrame:
    config = get_subject_config(subject)
    expected = [
        ("01_ingest", "domain", "corpus", "csv"),
        ("01_ingest", "codomain", "corpus", "csv"),
        ("02_preprocess", "domain", "preprocessed", "parquet"),
        ("02_preprocess", "codomain", "preprocessed", "parquet"),
        ("03_sentiment", "domain", "scores", "parquet"),
        ("03_sentiment", "codomain", "scores", "parquet"),
        ("04_eda", "domain", "eda_summary", "json"),
        ("04_eda", "codomain", "eda_summary", "json"),
        ("05_tfidf", "domain", "tfidf_summary", "json"),
        ("05_tfidf", "codomain", "tfidf_summary", "json"),
        ("06_topic_modeling", "domain", "topic_model_summary", "json"),
        ("06_topic_modeling", "codomain", "topic_model_summary", "json"),
        ("07_local_llm", "domain", "prompt", "txt"),
        ("07_local_llm", "codomain", "prompt", "txt"),
    ]
    rows = []
    for stage, dataset, artifact, ext in expected:
        path = artifact_path(config, stage, dataset, artifact, ext)
        rows.append(
            {
                "stage": stage,
                "dataset": dataset,
                "artifact": artifact,
                "path": str(path),
                "exists": bool(path.exists()),
            }
        )
    return pd.DataFrame(rows)


def run_subject_pipeline(
    subject: str | SubjectConfig,
    datasets: Sequence[str] = ("domain", "codomain"),
    preprocess_kwargs: dict[str, Any] | None = None,
    sentiment_kwargs: dict[str, Any] | None = None,
    eda_kwargs: dict[str, Any] | None = None,
    tfidf_kwargs: dict[str, Any] | None = None,
    topic_kwargs: dict[str, Any] | None = None,
    run_llm: bool = False,
    llm_backend: str = "prompt_only",
    llm_model: str | None = None,
    llm_max_tokens: int = 3072,
) -> dict[str, Any]:
    config = get_subject_config(subject)
    preprocess_kwargs = dict(preprocess_kwargs or {})
    sentiment_kwargs = dict(sentiment_kwargs or {})
    eda_kwargs = dict(eda_kwargs or {})
    tfidf_kwargs = dict(tfidf_kwargs or {})
    topic_kwargs = dict(topic_kwargs or {})

    stage_rows = []
    ingest = ingest_legacy_subject_data(config)
    stage_rows.append(
        {
            "stage": "01_ingest",
            "dataset": "all",
            "status": "ok",
            "rows": None,
            "path": ingest.get("manifest_path"),
        }
    )

    for dataset in datasets:
        prep = preprocess_stage(config, dataset, **preprocess_kwargs)
        stage_rows.append(
            {
                "stage": "02_preprocess",
                "dataset": dataset,
                "status": "ok",
                "rows": prep.get("rows"),
                "path": prep.get("preprocessed_path"),
            }
        )

        sent = sentiment_stage(config, dataset, **sentiment_kwargs)
        stage_rows.append(
            {
                "stage": "03_sentiment",
                "dataset": dataset,
                "status": "ok",
                "rows": sent.get("rows"),
                "path": sent.get("scores_path"),
            }
        )

        eda = eda_stage(config, dataset, **eda_kwargs)
        stage_rows.append(
            {
                "stage": "04_eda",
                "dataset": dataset,
                "status": "ok",
                "rows": eda.get("rows"),
                "path": eda.get("summary_path"),
            }
        )

        tfidf = tfidf_stage(config, dataset, **tfidf_kwargs)
        stage_rows.append(
            {
                "stage": "05_tfidf",
                "dataset": dataset,
                "status": "ok",
                "rows": tfidf.get("rows"),
                "path": tfidf.get("summary_path"),
            }
        )

        topic = topic_model_stage(config, dataset, **topic_kwargs)
        stage_rows.append(
            {
                "stage": "06_topic_modeling",
                "dataset": dataset,
                "status": "ok",
                "rows": None,
                "path": topic.get("summary_path"),
            }
        )

        bundle = prepare_local_llm_stage(config, dataset)
        llm_path = bundle["prompt_path"]
        llm_status = "prepared"
        if run_llm:
            llm_result = run_local_llm_stage(
                config,
                dataset,
                backend=llm_backend,
                model=llm_model,
                max_tokens=llm_max_tokens,
            )
            llm_path = llm_result.get("result_path") or llm_result.get("prompt_path")
            llm_status = "ok" if llm_result.get("result_path") else "prepared"
        stage_rows.append(
            {
                "stage": "07_local_llm",
                "dataset": dataset,
                "status": llm_status,
                "rows": None,
                "path": llm_path,
            }
        )

    manifest = {
        "subject": config.slug,
        "datasets": list(datasets),
        "run_llm": bool(run_llm),
        "llm_backend": llm_backend,
        "llm_model": llm_model,
        "stages": stage_rows,
    }
    manifest_path = subject_output_dir(config) / "pipeline_run_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def codomain_improvement_candidates() -> pd.DataFrame:
    return pd.DataFrame(CODOMAIN_IMPROVEMENTS)
