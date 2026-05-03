# NLU Fallback Topic Modeller

Unsupervised NLP pipeline for clustering and ranking voicebot fallback utterances. Built to surface the most common topics buried in no-match traffic — turning raw fallback logs into a ranked, timestamped report consumed by a PowerBI dashboard.

---

## Background

Built to address a recurring gap in voicebot analytics: aggregate no-match rates tell you *how often* the NLU fails, but nothing about *why*. The top topics driving fallbacks are almost never obvious from intent reporting alone, and manually reviewing thousands of utterances per month is not scalable.

This pipeline was run monthly against production fallback exports from enterprise voicebot deployments, automatically clustering utterances into topics and appending results to a master CSV. The append-per-run design means PowerBI can read the single file and trend topic volumes over time — showing whether a spike in a particular topic correlates with a product change, outage, or seasonal event.

The output directly informed NLU improvement decisions: new intents were created for high-volume fallback topics, existing intents were expanded with training phrases drawn from the representative utterances, and persistent low-volume topics were flagged for human review.

---

## How It Works

```
Input CSV (utterance column)
        │
        ▼
  TextProcessor         — filter English, lemmatize, strip stopwords
        │
        ▼
  FeatureExtractor      — TF-IDF (unigrams + bigrams, top 1000 features)
        │
        ▼
  ClusteringModel       — K-Means, K selected by silhouette score
        │
        ▼
  TopicIdentifier       — label each cluster from centroid TF-IDF terms
        │
        ▼
  TopicRanker           — sort clusters by utterance volume
        │
        ▼
  ResultExporter        — append to master CSV (timestamped) → PowerBI
```

---

## Usage

```bash
python top_topics_fallback.py
```

Input is read from `INPUT_CSV` in `.env`. Results are appended to `OUTPUT_CSV`.

**Input CSV format** — one required column:

| utterance |
|---|
| i want to pay my bill |
| my internet is down |
| ... |

**Output CSV format:**

| Rank | Topic | Count | Percentage | Timestamp | Utterance_1 | ... | Utterance_15 |
|---|---|---|---|---|---|---|---|
| 1 | payment and bill, pay, invoice... | 342 | 12.4% | 04-2026 | ... | | |

---

## Setup

```bash
git clone https://github.com/your-username/nlu-fallback-topic-modeller.git
cd nlu-fallback-topic-modeller

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env with your input/output paths
```

Place your fallback CSV in the `data/` folder (gitignored).

---

## Configuration

All parameters are set in `.env` — see `.env.example`. Key options:

| Variable | Default | Description |
|---|---|---|
| `INPUT_CSV` | `data/fallback.csv` | Monthly fallback utterance export |
| `OUTPUT_CSV` | `output/fallback_master_output.csv` | Appended master file for PowerBI |
| `MAX_CLUSTERS` | `30` | Upper bound for K search |
| `TOP_N_CLUSTERS` | `20` | How many topics to rank and export |
| `TOP_N_WORDS` | `15` | TF-IDF terms used to label each topic |

---

## PowerBI Integration

The master output CSV is designed to be a direct PowerBI data source:

- Each run **appends** a new set of rows with the current `Timestamp` (MM-YYYY)
- PowerBI reads the full file and can slice by `Timestamp` to show topic trends over time
- `Rank`, `Count`, and `Percentage` columns are ready-to-use measures
- `Utterance_1` through `Utterance_15` give example context per topic per period

Recommended refresh: run monthly after exporting the latest fallback data, then trigger a PowerBI dataset refresh.

---

## Expanding This Project

The class-based architecture is designed to be extended. Potential next steps:

- **BERTopic** — swap `FeatureExtractor` + `ClusteringModel` for transformer-based topic modelling for richer semantic clustering
- **Scheduled runs** — wrap `main()` in a cron job or Azure Function to automate monthly execution
- **DBSCAN** — replace K-Means for density-based clustering that doesn't require specifying K upfront
- **Intent recommendation** — add a step that maps high-volume topics to suggested new intent names and sample training phrases
- **Language expansion** — extend `TextProcessor` to support multilingual fallback analysis

---

## Stack

Python · pandas · scikit-learn · NLTK · langdetect · matplotlib · python-dotenv
