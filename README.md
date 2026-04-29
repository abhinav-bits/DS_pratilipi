# Chapter Recommendation Submission

## Problem

The goal is to recommend chapters to users from implicit `(user, chapter)` interactions.

- There are no timestamps.
- There is no reliable reading order in the interaction log.

Because of that, I treated this as a top-K recommendation problem over unseen chapters, not as a true next-item prediction task.

---

## Data Observations

- About 1M interactions across about 150K users
- About 50K chapters
- The user-chapter matrix is extremely sparse
- About 99.96% of `(user, book)` pairs contain only one chapter

This means there is not enough real sequential signal in the interaction data to learn in-book reading progression in a reliable way.

---

## Final Task Definition

I framed the final task as top-K chapter recommendation with implicit feedback.

For each user with at least 2 interacted chapters:

1. Hold out one chapter with leave-one-out.
2. Train on the remaining interactions.
3. Rank the held-out chapter against all chapters not seen in that user's training set.

### Metrics

- `Hit@K`
- `MRR`
- `NDCG@K`

All metrics are averaged across users.

---

## Approach

### Files

- `analysis.ipynb`
  Data exploration, sparsity checks, sequence-feasibility check, and final problem framing.

- `model_training.ipynb`
  Leave-one-out split, ALS training, evaluation, and comparison with baselines.

- `main.py`
  Reusable helper functions for splits, ranking, metrics, and baselines.

---

### Models Evaluated

- Implicit ALS
- Global popularity baseline
- Primary-tag popularity baseline

I also tried a few simple metadata-based ideas, like tag cosine and ALS with small content boosts, but they did not improve results, so I left them out of the final comparison.

---

## Evaluation

- Split: leave-one-out per user
- Candidates: full catalog minus that user's training interactions
- Baselines are computed from training data only

### Sanity Check

For large candidate sets of around 50K items:

- Random `Hit@K ~= K / N`
- Random `MRR ~= 2 / (N + 1)`

---

## Results

Across the ranking metrics:

- Global popularity slightly outperforms ALS
- Tag-based approaches perform worse

### Interpretation

Global popularity slightly outperforms ALS on most metrics. That matches the dataset:

- interactions are extremely sparse
- users rarely interact with multiple chapters from the same book
- no temporal signal is available

So collaborative filtering can learn only limited personalization here, while globally popular chapters remain a strong baseline.

---

## Limitations

- No timestamps, so I cannot model temporal behavior
- Extremely sparse interactions, so collaborative signal is weak
- Tags are coarse, so content-based signal is limited
- Evaluation is only an offline proxy, not a true next-item prediction setup

---

## How to Run

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install pandas numpy matplotlib scipy jupyter ipython
pip install -r requirements-training.txt

jupyter lab
```

### Execution Order

1. Run `analysis.ipynb`
2. Run `model_training.ipynb`

Make sure `chapters.csv` and `interactions.csv` are in the same directory.
