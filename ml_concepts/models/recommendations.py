"""
Recommendations – Recommendation System Demo
─────────────────────────────────────────────
Covers:
  1. Synthetic user-item rating matrix generation
  2. User-Based Collaborative Filtering  (cosine similarity)
  3. Item-Based Collaborative Filtering  (adjusted cosine)
  4. Matrix Factorisation                (Truncated SVD)
  5. Content-Based Filtering             (item feature cosine similarity)
  6. Evaluation – RMSE, MAE, Precision@K, Recall@K, nDCG@K
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler

PALETTE = sns.color_palette("tab10")

# ── Synthetic dataset ─────────────────────────────────────────────────────────
GENRES    = ["Action","Comedy","Drama","Sci-Fi","Romance","Thriller","Horror","Animation"]
N_USERS   = 120
N_ITEMS   = 60
N_FACTORS = 8     # true latent factors


def make_dataset(n_users=N_USERS, n_items=N_ITEMS, sparsity=0.65, seed=42):
    """Generate a realistic sparse rating matrix (1–5 scale)."""
    rng = np.random.default_rng(seed)
    # Latent factor model
    U = rng.standard_normal((n_users, N_FACTORS))
    V = rng.standard_normal((n_items, N_FACTORS))
    base = U @ V.T
    # Scale to [1, 5]
    base = (base - base.min()) / (base.max() - base.min()) * 4 + 1
    # Add noise
    base += 0.3 * rng.standard_normal(base.shape)
    base = np.clip(np.round(base), 1, 5)

    # Enforce sparsity (some users haven't rated some items)
    mask = rng.random((n_users, n_items)) > sparsity
    ratings = np.where(mask, base, np.nan)

    # Item features (genre one-hot)
    n_genres = len(GENRES)
    item_features = np.zeros((n_items, n_genres))
    primary = rng.integers(0, n_genres, n_items)
    secondary = (primary + rng.integers(1, n_genres, n_items)) % n_genres
    for i, (p, s) in enumerate(zip(primary, secondary)):
        item_features[i, p] = 1.0
        item_features[i, s] = 0.5

    items = pd.DataFrame(
        item_features, columns=GENRES,
        index=[f"Movie_{i:03d}" for i in range(n_items)],
    )
    users = [f"User_{i:03d}" for i in range(n_users)]
    R = pd.DataFrame(ratings, index=users, columns=items.index)
    return R, items


# ── User-Based CF ─────────────────────────────────────────────────────────────
class UserBasedCF:
    def __init__(self, k_neighbours=15):
        self.k = k_neighbours
        self.sim  = None
        self.R    = None
        self.mean = None

    def fit(self, R: pd.DataFrame):
        self.R    = R.copy()
        self.mean = R.mean(axis=1)
        R_c       = R.subtract(self.mean, axis=0).fillna(0)
        self.sim  = pd.DataFrame(
            cosine_similarity(R_c.values), index=R.index, columns=R.index)
        return self

    def predict(self, user: str, item: str) -> float:
        if not np.isnan(self.R.loc[user, item]):  # already rated
            return self.R.loc[user, item]
        sims = self.sim[user].drop(user)
        rated_mask = ~self.R[item].isna()
        peers = sims[rated_mask].nlargest(self.k)
        if len(peers) == 0:
            return self.mean[user]
        num = (peers * (self.R.loc[peers.index, item] - self.mean[peers.index])).sum()
        den = peers.abs().sum() + 1e-9
        return np.clip(self.mean[user] + num / den, 1, 5)

    def top_n(self, user: str, n=10):
        unrated = self.R.columns[self.R.loc[user].isna()]
        scores  = {item: self.predict(user, item) for item in unrated}
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]


# ── Item-Based CF ─────────────────────────────────────────────────────────────
class ItemBasedCF:
    def __init__(self, k_items=20):
        self.k   = k_items
        self.sim = None
        self.R   = None

    def fit(self, R: pd.DataFrame):
        self.R  = R.copy()
        R_adj   = R.subtract(R.mean(axis=1), axis=0).fillna(0)   # adjusted cosine
        self.sim = pd.DataFrame(
            cosine_similarity(R_adj.T.values),
            index=R.columns, columns=R.columns)
        return self

    def predict(self, user: str, item: str) -> float:
        user_rated = self.R.loc[user].dropna()
        if item in user_rated:
            return user_rated[item]
        sims   = self.sim[item].drop(item)
        common = sims[user_rated.index].nlargest(self.k)
        if len(common) == 0:
            return self.R[item].mean()
        num = (common * user_rated[common.index]).sum()
        den = common.abs().sum() + 1e-9
        return np.clip(num / den, 1, 5)

    def top_n(self, user: str, n=10):
        unrated = self.R.columns[self.R.loc[user].isna()]
        scores  = {item: self.predict(user, item) for item in unrated}
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]


# ── Matrix Factorisation (SVD) ────────────────────────────────────────────────
class SVDRecommender:
    def __init__(self, n_components=20):
        self.k    = n_components
        self.svd  = TruncatedSVD(n_components=n_components, random_state=42)
        self.mean = None
        self.cols = None
        self.idx  = None
        self.R_hat= None

    def fit(self, R: pd.DataFrame):
        self.mean = R.stack().mean()
        self.cols = R.columns
        self.idx  = R.index
        R_fill    = R.fillna(self.mean).values
        U         = self.svd.fit_transform(R_fill)
        Vt        = self.svd.components_
        R_hat_raw = U @ Vt
        self.R_hat = pd.DataFrame(np.clip(R_hat_raw, 1, 5),
                                  index=self.idx, columns=self.cols)
        return self

    def predict(self, user: str, item: str) -> float:
        return float(self.R_hat.loc[user, item])

    def top_n(self, user: str, n=10, R_orig: pd.DataFrame = None):
        unrated = R_orig.columns[R_orig.loc[user].isna()] if R_orig is not None else self.cols
        scores  = {item: self.predict(user, item) for item in unrated}
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]

    @property
    def explained_variance(self):
        return self.svd.explained_variance_ratio_.sum()


# ── Content-Based Filtering ───────────────────────────────────────────────────
class ContentBasedFilter:
    def __init__(self):
        self.item_sim = None
        self.features = None
        self.R        = None

    def fit(self, R: pd.DataFrame, item_features: pd.DataFrame):
        self.R        = R.copy()
        self.features = item_features
        self.item_sim = pd.DataFrame(
            cosine_similarity(item_features.values),
            index=item_features.index, columns=item_features.index)
        return self

    def user_profile(self, user: str) -> pd.Series:
        rated = self.R.loc[user].dropna()
        if len(rated) == 0:
            return self.features.mean()
        return (self.features.loc[rated.index].T * rated.values).T.mean()

    def predict(self, user: str, item: str) -> float:
        profile = self.user_profile(user)
        item_vec= self.features.loc[item]
        num = (profile * item_vec).sum()
        den = (np.linalg.norm(profile) * np.linalg.norm(item_vec)) + 1e-9
        sim = num / den
        return np.clip(sim * 4 + 1, 1, 5)

    def top_n(self, user: str, n=10):
        unrated = self.R.columns[self.R.loc[user].isna()]
        scores  = {item: self.predict(user, item) for item in unrated}
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]


# ── Evaluation ────────────────────────────────────────────────────────────────
def train_test_split_ratings(R: pd.DataFrame, test_frac=0.2, seed=42):
    rng  = np.random.default_rng(seed)
    R_tr = R.copy(); R_te = {}
    known = [(u, i) for u in R.index for i in R.columns if not np.isnan(R.loc[u, i])]
    rng.shuffle(known)
    te_pairs = known[:int(len(known) * test_frac)]
    for u, i in te_pairs:
        R_te[(u, i)] = R.loc[u, i]
        R_tr.loc[u, i] = np.nan
    return R_tr, R_te


def rmse(true_dict, model_predict_fn):
    errors = [(v - model_predict_fn(u, i))**2 for (u, i), v in true_dict.items()]
    return np.sqrt(np.mean(errors))


def precision_recall_at_k(R_true, model_top_n_fn, k=10, threshold=3.5):
    precs, recs = [], []
    for user in R_true.index:
        relevant = set(R_true.columns[R_true.loc[user] >= threshold].tolist())
        if not relevant:
            continue
        recs_list = [i for i, _ in model_top_n_fn(user, n=k)]
        hits = len(set(recs_list) & relevant)
        precs.append(hits / k)
        recs.append(hits / len(relevant))
    return np.mean(precs), np.mean(recs)


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_rating_matrix(R: pd.DataFrame, title="Rating Matrix", path=None):
    fig, ax = plt.subplots(figsize=(14, 6))
    R_vis   = R.fillna(0).values[:30, :30]
    im = ax.imshow(R_vis, cmap="YlOrRd", aspect="auto", vmin=0, vmax=5)
    plt.colorbar(im, ax=ax, label="Rating (0 = unrated)")
    ax.set_xlabel("Items (first 30)"); ax.set_ylabel("Users (first 30)")
    ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight"); print(f"  Saved → {path}")
    else:
        plt.show()
    plt.close(fig)


def plot_model_comparison(names, rmses, prec_k, rec_k, path=None):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, vals, ylabel, title in zip(
        axes,
        [rmses, prec_k, rec_k],
        ["RMSE (↓)", "Precision@10 (↑)", "Recall@10 (↑)"],
        ["RMSE Comparison", "Precision@10", "Recall@10"],
    ):
        bars = ax.bar(names, vals, color=[PALETTE[i % 10] for i in range(len(names))])
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.set_ylabel(ylabel); ax.set_title(title, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.2 + 0.1)
        ax.set_xticklabels(names, rotation=20, ha="right")
    plt.suptitle("Recommendation Model Comparison", fontweight="bold", fontsize=13)
    plt.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight"); print(f"  Saved → {path}")
    else:
        plt.show()
    plt.close(fig)


# ── run ───────────────────────────────────────────────────────────────────────
def run(save_plots=False, plot_dir="."):
    print("\n" + "=" * 60)
    print("  RECOMMENDATION SYSTEMS")
    print("=" * 60)

    R, items = make_dataset()
    rated    = R.notna().sum().sum()
    total    = R.shape[0] * R.shape[1]
    print(f"\n  Dataset : {R.shape[0]} users × {R.shape[1]} items")
    print(f"  Ratings : {rated} / {total}  ({rated/total*100:.1f}% dense)")
    print(f"  Rating range : {R.stack().min():.0f} – {R.stack().max():.0f}")

    plot_rating_matrix(R, path=f"{plot_dir}/rec_matrix.png" if save_plots else None)

    R_tr, R_te = train_test_split_ratings(R)

    # ── Train models ──────────────────────────────────────────────────────────
    ub = UserBasedCF(k_neighbours=15).fit(R_tr)
    ib = ItemBasedCF(k_items=20).fit(R_tr)
    sv = SVDRecommender(n_components=20).fit(R_tr)
    cb = ContentBasedFilter().fit(R_tr, items)

    models = {"User-CF": ub, "Item-CF": ib, "SVD": sv, "Content-Based": cb}

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\n  Evaluation on held-out ratings:")
    names_e, rmses_e, precs_e, recs_e = [], [], [], []
    for name, model in models.items():
        r  = rmse(R_te, model.predict)
        p, rc = precision_recall_at_k(R, model.top_n, k=10)
        print(f"  {name:<16}  RMSE={r:.4f}  P@10={p:.4f}  R@10={rc:.4f}")
        names_e.append(name); rmses_e.append(r)
        precs_e.append(p); recs_e.append(rc)

    plot_model_comparison(names_e, rmses_e, precs_e, recs_e,
                          path=f"{plot_dir}/rec_comparison.png" if save_plots else None)

    # ── Sample recommendations ─────────────────────────────────────────────────
    user = R.index[0]
    print(f"\n  Top-5 recommendations for {user} (SVD):")
    for item, score in sv.top_n(user, n=5, R_orig=R_tr):
        genres = items.loc[item].nlargest(2).index.tolist()
        print(f"    {item}  score={score:.2f}  genres={genres}")

    # ── SVD explained variance ────────────────────────────────────────────────
    print(f"\n  SVD explained variance @ k=20: {sv.explained_variance*100:.1f}%")

    print("\n  Key takeaways:")
    print("    • User-CF: 'users like you also liked…' — needs enough user overlap")
    print("    • Item-CF: 'because you liked X, try Y' — more stable, less cold-start sensitive")
    print("    • SVD: decomposes rating matrix into latent user/item factors")
    print("    • Content-Based: uses item attributes — works for new items (cold start)")
