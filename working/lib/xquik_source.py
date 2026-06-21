"""Collect Xquik tweet search results into the working corpus CSV schema."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = "https://xquik.com"
CSV_COLUMNS = (
    "Tweet ID",
    "URL",
    "Content",
    "Author ID",
    "Author Username",
    "Author Blue Verified (T/F)",
    "Followers",
    "Author Location",
    "Likes",
    "Views",
    "Date",
)
CODOMAIN_COLUMNS = (*CSV_COLUMNS, "Original User Sentiment")


def first_value(source: dict[str, Any], *names: str) -> Any:
    """Return the first present, non-empty mapping value for the given aliases."""
    for name in names:
        if name in source and source[name] not in (None, ""):
            return source[name]
    return ""


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return str(value)


def as_number_text(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return as_text(value)


def created_to_date(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, (int, float)):
        return dt.datetime.fromtimestamp(value, tz=dt.timezone.utc).isoformat()
    text = str(value).strip()
    if text.isdigit():
        return dt.datetime.fromtimestamp(int(text), tz=dt.timezone.utc).isoformat()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return dt.datetime.fromisoformat(text).isoformat()
    except ValueError:
        return str(value)


def normalize_username(value: Any) -> str:
    return as_text(value).lstrip("@")


def tweet_url(tweet_id: str, username: str, explicit_url: Any) -> str:
    url = as_text(explicit_url)
    if url:
        return url
    if tweet_id and username:
        return f"https://x.com/{username}/status/{tweet_id}"
    return ""


def tweet_to_row(
    tweet: dict[str, Any],
    original_user_sentiment: str = "",
) -> dict[str, str]:
    author = first_value(tweet, "author", "user")
    if not isinstance(author, dict):
        author = {}

    tweet_id = as_text(first_value(tweet, "id", "tweet_id", "tweetId"))
    username = normalize_username(
        first_value(
            author,
            "username",
            "screen_name",
            "screenName",
            "handle",
        )
        or first_value(tweet, "author_username", "authorUsername", "username")
    )

    row = {
        "Tweet ID": tweet_id,
        "URL": tweet_url(
            tweet_id,
            username,
            first_value(tweet, "url", "tweet_url", "tweetUrl"),
        ),
        "Content": as_text(
            first_value(tweet, "text", "full_text", "fullText", "content")
        ),
        "Author ID": as_text(
            first_value(author, "id", "user_id", "userId")
            or first_value(tweet, "author_id", "authorId", "user_id", "userId")
        ),
        "Author Username": username,
        "Author Blue Verified (T/F)": as_text(
            first_value(author, "is_blue_verified", "isBlueVerified", "verified")
        ),
        "Followers": as_number_text(
            first_value(author, "followers", "followers_count", "followersCount")
        ),
        "Author Location": as_text(
            first_value(author, "location")
            or first_value(tweet, "author_location", "authorLocation")
        ),
        "Likes": as_number_text(
            first_value(
                tweet,
                "like_count",
                "likeCount",
                "favorite_count",
                "favoriteCount",
                "likes",
            )
        ),
        "Views": as_number_text(first_value(tweet, "view_count", "viewCount", "views")),
        "Date": created_to_date(
            first_value(tweet, "created", "created_at", "createdAt", "date")
        ),
    }
    if original_user_sentiment:
        row["Original User Sentiment"] = original_user_sentiment
    return row


def extract_tweets(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        candidates = payload
    elif isinstance(payload, dict):
        candidates = first_value(payload, "tweets", "items", "data")
    else:
        candidates = []

    if not isinstance(candidates, list):
        return []
    return [item for item in candidates if isinstance(item, dict)]


def build_search_url(
    base_url: str,
    query: str,
    query_type: str,
    limit: int,
) -> str:
    params = urllib.parse.urlencode(
        {
            "q": query,
            "queryType": query_type,
            "limit": str(limit),
        }
    )
    return f"{base_url.rstrip('/')}/api/v1/x/tweets/search?{params}"


def fetch_xquik_tweets(
    api_key: str,
    base_url: str,
    query: str,
    query_type: str,
    limit: int,
    timeout: float,
) -> list[dict[str, Any]]:
    request = urllib.request.Request(
        build_search_url(base_url, query, query_type, limit),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"Xquik search failed with HTTP {exc.code}.") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Xquik search failed: {exc.reason}") from exc
    return extract_tweets(payload)


def write_corpus_csv(
    tweets: list[dict[str, Any]],
    output: Path,
    dataset: str,
    original_user_sentiment: str = "",
) -> int:
    columns = CODOMAIN_COLUMNS if dataset == "codomain" else CSV_COLUMNS
    rows = [tweet_to_row(tweet, original_user_sentiment) for tweet in tweets]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect Xquik tweet search results into the working corpus CSV schema."
    )
    parser.add_argument("--query", required=True, help="X search query.")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="CSV output path.",
    )
    parser.add_argument("--dataset", choices=("domain", "codomain"), default="domain")
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum tweets to request.",
    )
    parser.add_argument("--query-type", choices=("Latest", "Top"), default="Latest")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("XQUIK_BASE_URL", DEFAULT_BASE_URL),
    )
    parser.add_argument("--api-key", default=os.environ.get("XQUIK_API_KEY"))
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--original-user-sentiment",
        default="",
        help="Optional codomain seed label copied into the corpus.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.api_key:
        raise SystemExit("Set XQUIK_API_KEY or pass --api-key.")
    if args.limit < 1:
        raise SystemExit("--limit must be positive.")

    tweets = fetch_xquik_tweets(
        api_key=args.api_key,
        base_url=args.base_url,
        query=args.query,
        query_type=args.query_type,
        limit=args.limit,
        timeout=args.timeout,
    )
    row_count = write_corpus_csv(
        tweets=tweets,
        output=args.output,
        dataset=args.dataset,
        original_user_sentiment=args.original_user_sentiment,
    )
    sys.stderr.write(f"Wrote {row_count} rows to {args.output}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
