# Domain vs Co-domain on X (Twitter): Reproducible Sentiment + Topic Pipeline

**Goal.** Turn noisy social posts into standardized, comparable snapshots of language and sentiment that you can audit, visualize, and summarize.

**Scope.** Three entities — NFL, UK Prime Minister (Keir Starmer), Costco — each analyzed as domain (on-keyword) and co-domain (same users, off-keyword) corpora in 5-day collection windows (~60K posts total).

## TL;DR
	•	A reproducible workflow collects → cleans → labels sentiment (RoBERTa) → scores terms with complementary TF–IDF views.
	•	Domain-first sampling yields the clearest, action-ready themes; co-domain adds context but can dilute entity-specific signal.
	•	LLMs produce more interpretable summaries when fed wIDF-ranked terms + exemplar posts, not raw counts or raw text.

## What's in this repo
```
data/                      # raw + cleaned text and metadata per entity & slice
  presentation.pdf
  poster.pdf
  n8n/					   # n8n workflows for both domain and user feeds
  entity/                  # entities: costco, keir_starmer, nfl
    domain/                # CSV of on-keyword corpus for this entity
      checkpoints/         # domain checkpoint files (sentiment scoring, outputs, etc.)
      figures/             # exported PNGs for this entity's domain slice
    codomain/              # CSVs of off-keyword corpus and user IDs (same users)
      checkpoints/         # codomain checkpoint files (sentiment scoring, outputs, etc.)
      figures/             # exported PNGs for this entity's codomain slice
```
> Tip: Start by browsing figures/ — each entity has paired views (signed difference and log₂ ratio) for unigrams and n-grams.

## Methods in one page

### Two text streams
	•	Sentiment stream (minimal cleaning): lowercase; strip URLs/mentions/emojis; keep cues useful for polarity.
	•	Term-profiling stream (for features): lowercase → tokenize → lemmatize → remove stopwords → build n-grams (1–2 or 1–3).

### Sentiment labeling
	•	RoBERTa classifier → positive / neutral / negative per post.

### Term scoring (per sentiment slice)
	•	Aggregate TF–IDF (agg): sum of TF–IDF over all docs in the slice → overall prominence.
	•	Sentiment-weighted IDF (wIDF): (coverage in slice) × (global rarity) = (df_s/D_s) \times idf(t) → slice-characteristic usage.
	•	In practice, agg and wIDF rankings align very strongly (ρ≈0.999); we standardize on wIDF downstream for simpler interpretation.

### Comparing sentiments (what the figures show)
	•	Most Polarized Terms — Signed (pos − neg): emphasizes magnitude of lean.
	•	Most Characteristic Terms — log₂(pos / neg): emphasizes contrast; 0 = balanced, ± = skew.
	•	Short n-grams sharpen signals (e.g., parking lot, line costco gas, caramel brownie sundae, FSD v14).

### Topics (optional)
	•	LDA on the term-profiling stream to expose coherent themes.
	•	Choose K via grid search: pick the smallest K near the minimum held-out perplexity (predictive fit of words), then confirm on a coherence plateau and quick stability checks across seeds/subsamples.

## Domain vs co-domain: how to use
	•	Domain (on-keyword): crisp, entity-specific narratives → primary view.
	•	Co-domain (same users, off-keyword): lifestyle/macro context → supplement the domain view, not replace it.

## Interpreting the Costco example (pattern you’ll see across entities)
	•	Value symbols anchor trust (e.g., $1.50 hot dog, gas).
	•	Price/quality transparency—especially for meat/chicken—shapes sentiment and membership ROI.
	•	Family/bulk tradeoffs drive cross-shopping; Sam’s appears as the head-to-head benchmark.
	•	n-grams disambiguate (e.g., costco gold bar vs costco gold → bullion discussion, not membership tiers).

## LLM-ready summaries
	•	Provide each entity/slice with:
	•	wIDF-ranked terms by sentiment
	•	a handful of exemplar posts
	•	simple counts (docs, tokens)
	•	Avoid raw text dumps or pure frequency tables. This structure consistently yields short, faithful summaries.

## Reproducibility notes
	•	Default n-grams: (1,2); try (2,2) or (3,3) for more specificity with a small recall trade-off.

## Limitations & transfer guidance
	•	Short time windows; English-only; one sentiment model.
	•	Beyond tweets (e.g., call logs, support tickets, product reviews) often requires more generalized preprocessing (turn-level segmentation, no hashtag/mention assumptions) and retuned n-gram windows/models. Expect parameters not to transfer 1:1.

## Responsible use
	•	Respect platform terms and privacy norms.
	•	Intended for research and aggregate insight, not individual profiling or automated enforcement.

## Citation
> Linden, T., & Shoemaker, K. (2025). Domain vs Co-domain on X: Reproducible Sentiment + Topic Pipeline. GitHub repository.
