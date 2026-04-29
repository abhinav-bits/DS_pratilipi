"""
Offline evaluation helpers aligned with analysis.ipynb:
  - Per-user holdout (leave-one-out or random single holdout)
  - Hit@K, MRR, NDCG@K over ranked candidates (unseen = not in user's train set)
  - Train-only statistics for baselines to avoid leakage

No third-party ML deps beyond numpy/pandas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

UserId = Union[int, str]


@dataclass
class UserSplit:
    """Train/test chapter sets per user (users with <2 unique chapters omitted)."""

    user_id: UserId
    train_chapters: Set[int]
    test_chapter: int


def build_leave_one_out_splits(
    interactions: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "chapter_id",
    seed: int = 42,
) -> Tuple[List[UserSplit], pd.DataFrame]:
    """
    One held-out chapter per user with at least 2 distinct chapters.
    Returns splits and the training interaction table (rows whose chapter is in train set per user).

    The training row mask is aligned to ``interactions`` row order (index-safe); do not build
    masks solely from ``groupby`` iteration order.
    """
    rng = np.random.default_rng(seed)
    splits: List[UserSplit] = []

    tmp = interactions
    nuniq = tmp.groupby(user_col, sort=False)[item_col].transform("nunique")
    held_col = pd.Series(np.nan, index=tmp.index, dtype=float)

    for uid, grp in tmp.loc[nuniq >= 2].groupby(user_col, sort=False):
        ch = grp[item_col].unique()
        held = int(rng.choice(ch))
        held_col.loc[grp.index] = float(held)
        train_chapters = {int(x) for x in ch if int(x) != held}
        splits.append(UserSplit(user_id=uid, train_chapters=train_chapters, test_chapter=held))

    chv = tmp[item_col].to_numpy(dtype=np.int64, copy=False)
    hv = held_col.to_numpy(dtype=np.float64, copy=False)
    nu = nuniq.to_numpy(copy=False)
    keep = (nu < 2) | np.isnan(hv) | (chv != hv)
    train_ix = tmp.loc[keep].copy()
    return splits, train_ix


def all_chapter_ids(chapters: pd.DataFrame, col: str = "chapter_id") -> np.ndarray:
    return chapters[col].to_numpy()


def train_chapter_counts(train_ix: pd.DataFrame, item_col: str = "chapter_id") -> pd.Series:
    return train_ix.groupby(item_col, sort=False).size()


def scores_popularity(
    candidate_chapters: np.ndarray,
    train_counts: pd.Series,
) -> np.ndarray:
    """Score = train split interaction count (0 if unseen in train)."""
    return np.array([train_counts.get(int(c), 0) for c in candidate_chapters], dtype=np.float64)


def scores_random(candidate_chapters: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.random(len(candidate_chapters))


def rank_of_target(
    candidate_chapters: np.ndarray,
    scores: np.ndarray,
    target: int,
) -> Optional[int]:
    """
    1-based rank of target among candidates when sorted by score descending.
    Returns None if target not in candidates.
    """
    if target not in set(int(x) for x in candidate_chapters):
        return None
    # Sort indices by score desc, tie-break by chapter id for reproducibility
    order = np.lexsort((candidate_chapters.astype(np.int64), -scores))
    ranked = candidate_chapters[order]
    pos = np.where(ranked == int(target))[0]
    if len(pos) == 0:
        return None
    return int(pos[0]) + 1


def hit_at_k(rank: Optional[int], k: int) -> float:
    if rank is None:
        return 0.0
    return 1.0 if rank <= k else 0.0


def reciprocal_rank(rank: Optional[int]) -> float:
    if rank is None:
        return 0.0
    return 1.0 / float(rank)


def dcg_at_k(rank: Optional[int], k: int) -> float:
    """Binary relevance: 1 if target in top k, graded by position."""
    if rank is None or rank > k:
        return 0.0
    return 1.0 / np.log2(rank + 1)


def idcg_binary_at_k(k: int) -> float:
    return 1.0 / np.log2(2.0)  # best case rank 1


def ndcg_at_k(rank: Optional[int], k: int) -> float:
    idcg = idcg_binary_at_k(k)
    if idcg <= 0:
        return 0.0
    return dcg_at_k(rank, k) / idcg


def evaluate_splits(
    splits: Sequence[UserSplit],
    all_items: np.ndarray,
    score_fn,
    ks: Sequence[int] = (5, 10, 20),
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """
    score_fn(candidate_chapters, user_split) -> scores array same length as candidates.
    """
    rng = rng or np.random.default_rng(0)
    hits = {k: [] for k in ks}
    rr: List[float] = []
    ndcgs = {k: [] for k in ks}

    for sp in splits:
        train = sp.train_chapters
        candidates = np.array([c for c in all_items if int(c) not in train], dtype=np.int64)
        scores = score_fn(candidates, sp, rng)
        rank = rank_of_target(candidates, scores, sp.test_chapter)
        for k in ks:
            hits[k].append(hit_at_k(rank, k))
            ndcgs[k].append(ndcg_at_k(rank, k))
        rr.append(reciprocal_rank(rank))

    out: Dict[str, float] = {"mrr": float(np.mean(rr)), "n_users": float(len(splits))}
    for k in ks:
        out[f"hit@{k}"] = float(np.mean(hits[k]))
        out[f"ndcg@{k}"] = float(np.mean(ndcgs[k]))
    return out


def popularity_score_fn(train_counts: pd.Series):
    def _fn(candidates: np.ndarray, sp: UserSplit, rng: np.random.Generator) -> np.ndarray:
        return scores_popularity(candidates, train_counts)

    return _fn


def random_score_fn():
    def _fn(candidates: np.ndarray, sp: UserSplit, rng: np.random.Generator) -> np.ndarray:
        return scores_random(candidates, rng)

    return _fn


def primary_tag_popularity_scores(
    candidates: np.ndarray,
    chapters: pd.DataFrame,
    train_ix: pd.DataFrame,
    train_counts: pd.Series,
) -> np.ndarray:
    """
    Heuristic: score = train popularity within the chapter's primary (first) tag;
    fallback to global train count if tag missing.
    """
    ch = chapters.set_index("chapter_id")
    tag_counts: Dict[str, float] = {}
    # chapter -> first tag
    meta = train_ix.merge(chapters[["chapter_id", "tags"]], on="chapter_id", how="left")
    meta["primary_tag"] = meta["tags"].fillna("").str.split("|").str[0].replace("", np.nan)
    for _, row in meta.dropna(subset=["primary_tag"]).iterrows():
        t = row["primary_tag"]
        c = int(row["chapter_id"])
        tag_counts.setdefault(t, 0.0)
        tag_counts[t] += 1.0

    scores = np.zeros(len(candidates), dtype=np.float64)
    for i, cid in enumerate(candidates):
        cid = int(cid)
        g = train_counts.get(cid, 0)
        if cid not in ch.index:
            scores[i] = g
            continue
        tags = ch.loc[cid, "tags"]
        if pd.isna(tags) or str(tags).strip() == "":
            scores[i] = g
            continue
        pt = str(tags).split("|")[0].strip()
        scores[i] = tag_counts.get(pt, 0.0) + 0.001 * g  # tiny tie-break from global
    return scores


def tag_popularity_score_fn(chapters: pd.DataFrame, train_ix: pd.DataFrame, train_counts: pd.Series):
    def _fn(candidates: np.ndarray, sp: UserSplit, rng: np.random.Generator) -> np.ndarray:
        return primary_tag_popularity_scores(candidates, chapters, train_ix, train_counts)

    return _fn


def chapter_index_map(chapter_ids: np.ndarray) -> Dict[int, int]:
    return {int(c): i for i, c in enumerate(chapter_ids)}


def popularity_scores_aligned(chapter_ids: np.ndarray, train_counts: pd.Series) -> np.ndarray:
    """Per-catalog-row score from train-only global counts."""
    return np.array([float(train_counts.get(int(c), 0)) for c in chapter_ids], dtype=np.float64)


def build_primary_tag_train_counts(train_ix: pd.DataFrame, chapters: pd.DataFrame) -> pd.Series:
    meta = train_ix.merge(chapters[["chapter_id", "tags"]], on="chapter_id", how="left")
    meta["primary_tag"] = meta["tags"].fillna("").str.split("|").str[0].replace("", np.nan)
    return meta.dropna(subset=["primary_tag"]).groupby("primary_tag", sort=False).size()


def tag_scores_aligned(
    chapter_ids: np.ndarray,
    chapters: pd.DataFrame,
    tag_train_counts: pd.Series,
    train_counts: pd.Series,
) -> np.ndarray:
    """Vectorized tag-bucket popularity + tiny global tie-break (same idea as primary_tag_popularity_scores)."""
    ch = chapters.drop_duplicates(subset=["chapter_id"]).set_index("chapter_id")
    scores = np.zeros(len(chapter_ids), dtype=np.float64)
    for i, cid in enumerate(chapter_ids):
        cid = int(cid)
        g = float(train_counts.get(cid, 0))
        if cid not in ch.index:
            scores[i] = g
            continue
        tags = ch.loc[cid, "tags"]
        if pd.isna(tags) or str(tags).strip() == "":
            scores[i] = g
            continue
        pt = str(tags).split("|")[0].strip()
        scores[i] = float(tag_train_counts.get(pt, 0.0)) + 0.001 * g
    return scores


def _train_mask(sp: UserSplit, m: int, c2i: Dict[int, int]) -> np.ndarray:
    mask = np.zeros(m, dtype=bool)
    for c in sp.train_chapters:
        mask[c2i[int(c)]] = True
    return mask


def ranks_from_global_scores(
    splits: Sequence[UserSplit],
    chapter_ids: np.ndarray,
    scores: np.ndarray,
    c2i: Dict[int, int],
) -> np.ndarray:
    """
    1-based rank of the held-out chapter among candidates (all catalog chapters not in train),
    sorted by score descending, tie-break by smaller chapter_id first (matches rank_of_target).
    """
    cids = chapter_ids.astype(np.int64)
    scores = scores.astype(np.float64)
    m = len(cids)
    ranks = np.empty(len(splits), dtype=np.int64)
    for i, sp in enumerate(splits):
        ti = c2i[sp.test_chapter]
        s_t = scores[ti]
        c_t = cids[ti]
        cand = ~_train_mask(sp, m, c2i)
        better = cand & ((scores > s_t) | ((scores == s_t) & (cids < c_t)))
        ranks[i] = int(better.sum() + 1)
    return ranks


def ranks_from_user_item_factors(
    splits: Sequence[UserSplit],
    chapter_ids: np.ndarray,
    c2i: Dict[int, int],
    user_row_indices: np.ndarray,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    """
    Same ranking rule as ``ranks_from_global_scores``, but scores come from
    ``scores[c] = dot(user_factors[u], item_factors[c])`` per evaluated user.

    **Training-item exclusion:** for each split, ``_train_mask`` marks every chapter in
    ``sp.train_chapters`` (all chapters the user kept in the train fold). The held-out
    chapter is **not** in that set, so it remains a **candidate** and receives a valid rank
    among ``~len(catalog) - |train_chapters|`` items.

    **Implicit ALS alignment:** ``user_factors`` must have shape ``(n_users_in_matrix, n_factors)``
    and ``item_factors`` shape ``(n_items, n_factors)``, matching ``AlternatingLeastSquares`` after
    ``fit(user_items)`` with CSR rows = users, columns = items (the library transposes internally
    for alternating updates; do **not** pass ``train_matrix.T`` into ``fit``).

    ``user_row_indices[i]`` is the training matrix row for ``splits[i].user_id``.
    ``item_factors`` rows must align with catalog column order (same as ``chapter_ids``).
    """
    if len(splits) != len(user_row_indices):
        raise ValueError("splits and user_row_indices must have the same length")

    cids = chapter_ids.astype(np.int64)
    m = len(cids)
    if item_factors.shape[0] != m:
        raise ValueError(
            f"item_factors rows ({item_factors.shape[0]}) must match len(chapter_ids) ({m})"
        )

    ifactors = item_factors.astype(np.float64)
    ufactors = user_factors.astype(np.float64)
    ranks = np.empty(len(splits), dtype=np.int64)

    for start in range(0, len(splits), batch_size):
        end = min(start + batch_size, len(splits))
        ui = user_row_indices[start:end]
        scores_b = ufactors[ui] @ ifactors.T
        for j, sp in enumerate(splits[start:end]):
            scores = scores_b[j]
            ti = c2i[sp.test_chapter]
            s_t = scores[ti]
            c_t = cids[ti]
            cand = ~_train_mask(sp, m, c2i)
            better = cand & ((scores > s_t) | ((scores == s_t) & (cids < c_t)))
            ranks[start + j] = int(better.sum() + 1)
    return ranks


def catalog_author_tagcode_arrays(
    chapters: pd.DataFrame,
    chapter_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align ``author_id`` and factorized primary genre tag to ``chapter_ids`` order (length m).
    Unknown / empty tag → tag code ``-1``.
    """
    meta = pd.DataFrame({"chapter_id": chapter_ids.astype(np.int64)})
    meta = meta.merge(
        chapters[["chapter_id", "author_id", "tags"]].drop_duplicates("chapter_id"),
        on="chapter_id",
        how="left",
    )
    auth = meta["author_id"].fillna(-1).astype(np.int64).to_numpy()
    prim = meta["tags"].fillna("").str.split("|").str[0].str.strip()
    prim = prim.mask(prim == "", pd.NA)
    tag_codes, _ = pd.factorize(prim, use_na_sentinel=True)
    tag_codes = tag_codes.astype(np.int32)
    return auth, tag_codes


def content_alignment_boost(
    sp: UserSplit,
    c2i: Dict[int, int],
    author_arr: np.ndarray,
    tagcode_arr: np.ndarray,
    m: int,
    genre_weight: float = 0.12,
    author_weight: float = 0.12,
) -> np.ndarray:
    """
    Cheap additive boost: +author_weight on chapters whose author appears in the user's
    train chapters; +genre_weight on chapters whose primary tag code matches any train chapter.
    """
    train_idx = np.array([c2i[int(c)] for c in sp.train_chapters], dtype=np.int32)
    ta = np.unique(author_arr[train_idx])
    ta = ta[ta >= 0]
    tt = np.unique(tagcode_arr[train_idx])
    tt = tt[tt >= 0]
    boost = np.zeros(m, dtype=np.float64)
    if len(ta):
        boost += author_weight * np.isin(author_arr, ta).astype(np.float64)
    if len(tt):
        boost += genre_weight * np.isin(tagcode_arr, tt).astype(np.float64)
    return boost


def ranks_from_als_with_content_boost(
    splits: Sequence[UserSplit],
    chapter_ids: np.ndarray,
    c2i: Dict[int, int],
    user_row_indices: np.ndarray,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    author_arr: np.ndarray,
    tagcode_arr: np.ndarray,
    genre_weight: float = 0.12,
    author_weight: float = 0.12,
    batch_size: int = 512,
) -> np.ndarray:
    """ALS dot-product scores plus ``content_alignment_boost`` before ranking."""
    if len(splits) != len(user_row_indices):
        raise ValueError("splits and user_row_indices must have the same length")

    cids = chapter_ids.astype(np.int64)
    m = len(cids)
    if item_factors.shape[0] != m:
        raise ValueError(
            f"item_factors rows ({item_factors.shape[0]}) must match len(chapter_ids) ({m})"
        )

    ifactors = item_factors.astype(np.float64)
    ufactors = user_factors.astype(np.float64)
    ranks = np.empty(len(splits), dtype=np.int64)

    for start in range(0, len(splits), batch_size):
        end = min(start + batch_size, len(splits))
        ui = user_row_indices[start:end]
        scores_b = ufactors[ui] @ ifactors.T
        for j, sp in enumerate(splits[start:end]):
            boost = content_alignment_boost(
                sp, c2i, author_arr, tagcode_arr, m, genre_weight, author_weight
            )
            scores = scores_b[j] + boost
            ti = c2i[sp.test_chapter]
            s_t = scores[ti]
            c_t = cids[ti]
            cand = ~_train_mask(sp, m, c2i)
            better = cand & ((scores > s_t) | ((scores == s_t) & (cids < c_t)))
            ranks[start + j] = int(better.sum() + 1)
    return ranks


def candidate_counts(splits: Sequence[UserSplit], m: int, c2i: Dict[int, int]) -> np.ndarray:
    out = np.empty(len(splits), dtype=np.int64)
    for i, sp in enumerate(splits):
        out[i] = int((~_train_mask(sp, m, c2i)).sum())
    return out


def harmonic_table(max_n: int) -> np.ndarray:
    """H_n = sum_{j=1}^n 1/j at index n-1."""
    if max_n <= 0:
        return np.zeros(0, dtype=np.float64)
    return np.cumsum(1.0 / np.arange(1, max_n + 1, dtype=np.float64))


def expected_uniform_random_metrics(
    n_candidates: np.ndarray,
    ks: Sequence[int] = (5, 10, 20),
) -> Dict[str, float]:
    """
    Expected Hit@K, MRR, NDCG@K if the held-out item's rank were uniform on {1..n}
    (closed form; interpret as sanity-check baseline, not fitted model).
    """
    max_n = int(np.max(n_candidates)) if len(n_candidates) else 0
    h_tab = harmonic_table(max_n)

    def h_of(n: int) -> float:
        if n <= 0:
            return 0.0
        return float(h_tab[n - 1])

    mrr = np.array([h_of(int(n)) / int(n) if n > 0 else 0.0 for n in n_candidates])
    out: Dict[str, float] = {"mrr": float(np.mean(mrr)), "n_users": float(len(n_candidates))}
    for k in ks:
        hits = np.array([min(1.0, k / int(n)) if n > 0 else 0.0 for n in n_candidates])
        out[f"hit@{k}"] = float(np.mean(hits))
        ndcg_exp = []
        for n in n_candidates:
            nn = int(n)
            if nn <= 0:
                ndcg_exp.append(0.0)
                continue
            dcg_sum = sum(1.0 / np.log2(p + 1) for p in range(1, min(k, nn) + 1))
            ndcg_exp.append((dcg_sum / nn) / idcg_binary_at_k(k))
        out[f"ndcg@{k}"] = float(np.mean(ndcg_exp))
    return out


def metrics_from_ranks(ranks: np.ndarray, ks: Sequence[int] = (5, 10, 20)) -> Dict[str, float]:
    out: Dict[str, float] = {
        "mrr": float(np.mean(1.0 / ranks.astype(np.float64))),
        "n_users": float(len(ranks)),
    }
    for k in ks:
        out[f"hit@{k}"] = float(np.mean((ranks <= k).astype(np.float64)))
        nd = np.array([ndcg_at_k(int(r), k) for r in ranks])
        out[f"ndcg@{k}"] = float(np.mean(nd))
    return out


def build_chapter_tag_multihot(
    chapters: pd.DataFrame,
    chapter_ids: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    seen: Set[str] = set()
    for v in chapters["tags"].fillna(""):
        for p in str(v).strip().split("|"):
            t = p.strip()
            if t:
                seen.add(t)
    tag_list = sorted(seen)
    T = len(tag_list)
    if T == 0:
        return np.zeros((len(chapter_ids), 0), dtype=np.float64), []
    jm = {t: j for j, t in enumerate(tag_list)}
    X = np.zeros((len(chapter_ids), T), dtype=np.float64)
    ch = chapters.drop_duplicates(subset=["chapter_id"]).set_index("chapter_id")
    for r, cid in enumerate(chapter_ids):
        cid = int(cid)
        if cid not in ch.index:
            continue
        tv = ch.at[cid, "tags"]
        if pd.isna(tv) or not str(tv).strip():
            continue
        for p in str(tv).split("|"):
            j = jm.get(p.strip())
            if j is not None:
                X[r, j] = 1.0
    return X, tag_list


def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    out = np.asarray(X, dtype=np.float64, order="C").copy()
    n = np.linalg.norm(out, axis=1, keepdims=True)
    nz = n[:, 0] > 2 * eps
    n = np.maximum(n, eps)
    out[nz] /= n[nz]
    return out


def user_tag_profiles_train_only(
    splits: Sequence[UserSplit],
    X_bin: np.ndarray,
    c2i: Dict[int, int],
) -> np.ndarray:
    U = np.zeros((len(splits), X_bin.shape[1]), dtype=np.float64)
    for i, sp in enumerate(splits):
        ix = [c2i[int(c)] for c in sp.train_chapters]
        if ix:
            U[i] = X_bin[ix].mean(axis=0)
    return l2_normalize_rows(U)


def ranks_from_tag_cosine(
    splits: Sequence[UserSplit],
    chapter_ids: np.ndarray,
    c2i: Dict[int, int],
    X_item_unit: np.ndarray,
    user_profiles_unit: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    if len(splits) != len(user_profiles_unit) or X_item_unit.shape[0] != len(chapter_ids):
        raise ValueError("shape mismatch")
    cids = chapter_ids.astype(np.int64)
    Xi = np.asarray(X_item_unit, dtype=np.float64)
    Pu = np.asarray(user_profiles_unit, dtype=np.float64)
    m, ranks = len(cids), np.empty(len(splits), dtype=np.int64)
    for s in range(0, len(splits), batch_size):
        e = min(s + batch_size, len(splits))
        S = Pu[s:e] @ Xi.T
        for j, sp in enumerate(splits[s:e]):
            sc = S[j]
            ti = c2i[sp.test_chapter]
            st, ct = sc[ti], cids[ti]
            cand = ~_train_mask(sp, m, c2i)
            ranks[s + j] = int((cand & ((sc > st) | ((sc == st) & (cids < ct)))).sum() + 1)
    return ranks


def subsample_splits(splits: List[UserSplit], n: int, seed: int = 42) -> List[UserSplit]:
    if n >= len(splits):
        return splits
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(splits), size=n, replace=False)
    return [splits[i] for i in idx]
