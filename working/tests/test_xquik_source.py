from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import xquik_source


class XquikSourceTest(unittest.TestCase):
    def test_tweet_to_row_maps_search_result_to_working_schema(self) -> None:
        row = xquik_source.tweet_to_row(
            {
                "id": "12345",
                "text": "Topic modeling from fresh X data",
                "created": 1781438400,
                "like_count": 7,
                "view_count": 99,
                "author": {
                    "id": "42",
                    "username": "researcher",
                    "verified": True,
                    "followers_count": 1200,
                    "location": "London",
                },
            }
        )

        self.assertEqual(row["Tweet ID"], "12345")
        self.assertEqual(row["URL"], "https://x.com/researcher/status/12345")
        self.assertEqual(row["Content"], "Topic modeling from fresh X data")
        self.assertEqual(row["Author ID"], "42")
        self.assertEqual(row["Author Username"], "researcher")
        self.assertEqual(row["Author Blue Verified (T/F)"], "TRUE")
        self.assertEqual(row["Followers"], "1200")
        self.assertEqual(row["Author Location"], "London")
        self.assertEqual(row["Likes"], "7")
        self.assertEqual(row["Views"], "99")
        self.assertEqual(row["Date"], "2026-06-14T12:00:00+00:00")

    def test_extract_tweets_accepts_normalized_response(self) -> None:
        tweets = xquik_source.extract_tweets(
            {
                "tweets": [
                    {"id": "1", "text": "kept"},
                    "ignored",
                    {"id": "2", "text": "kept too"},
                ],
                "has_more": False,
            }
        )

        self.assertEqual(
            tweets,
            [{"id": "1", "text": "kept"}, {"id": "2", "text": "kept too"}],
        )

    def test_write_corpus_csv_adds_codomain_sentiment_column(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "codomain.csv"
            count = xquik_source.write_corpus_csv(
                tweets=[{"id": "1", "text": "codomain text"}],
                output=output,
                dataset="codomain",
                original_user_sentiment="positive",
            )

            self.assertEqual(count, 1)
            with output.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(rows[0]["Tweet ID"], "1")
        self.assertEqual(rows[0]["Content"], "codomain text")
        self.assertEqual(rows[0]["Original User Sentiment"], "positive")


if __name__ == "__main__":
    unittest.main()
