"""
Streamlit Interactive UI – ML Concepts Demo
============================================
Run:   streamlit run app.py
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.datasets import make_classification, make_regression, make_blobs, load_iris, load_wine
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.ensemble import (
    BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier,
    AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier,
    VotingClassifier, StackingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, davies_bouldin_score, confusion_matrix, ConfusionMatrixDisplay,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Concepts Interactive Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = sns.color_palette("tab10")
plt.rcParams.update({"figure.dpi": 100, "font.size": 10})

# ── Shared helpers ────────────────────────────────────────────────────────────
def show_metrics(metrics: dict):
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        col.metric(label, value)


def fig_to_st(fig):
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def load_dataset_clf(name):
    """Return (X_train, X_test, y_train, y_test), feat_names, target_names."""
    if name == "Wine":
        d = load_wine()
        X = StandardScaler().fit_transform(d.data)
        splits = train_test_split(X, d.target, test_size=0.2, random_state=42)
        return splits, list(d.feature_names), list(d.target_names)
    elif name == "Iris":
        d = load_iris()
        splits = train_test_split(d.data, d.target, test_size=0.2, random_state=42)
        return splits, list(d.feature_names), list(d.target_names)
    else:
        X, y = make_classification(
            500, n_features=10, n_informative=6, n_redundant=2,
            n_classes=3, n_clusters_per_class=1, random_state=42,
        )
        splits = train_test_split(X, y, test_size=0.2, random_state=42)
        return splits, [f"feature_{i}" for i in range(10)], ["Class 0", "Class 1", "Class 2"]


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🤖 ML Concepts")
    st.caption("Interactive Demonstration")
    st.divider()
    page = st.radio(
        "Navigate",
        [
            "🏠  Home",
            "📈  Linear Regression",
            "🌳  Decision Tree",
            "🔵  K-Means Clustering",
            "🌲  Bagging & Random Forest",
            "⚡  Boosting",
            "🎯  Ensemble Methods",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("scikit-learn · Streamlit · matplotlib")


# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
def page_home():
    st.title("🤖 Machine Learning Concepts — Interactive Demo")
    st.markdown(
        "Explore six core ML techniques. Adjust inputs in the **sidebar** and "
        "watch results update instantly."
    )
    st.divider()

    cards = [
        ("📈", "Linear Regression",       "OLS · Ridge · Lasso · Polynomial expansion"),
        ("🌳", "Decision Tree",            "Gini/Entropy · CCP pruning · depth sweep"),
        ("🔵", "K-Means Clustering",       "Elbow · Silhouette · Agglomerative comparison"),
        ("🌲", "Bagging & Random Forest",  "Bootstrap · OOB · ExtraTrees · n_estimators sweep"),
        ("⚡", "Boosting",                 "AdaBoost · GBM · HistGBM · stagewise curves"),
        ("🎯", "Ensemble Methods",         "Hard/Soft voting · Weighted · Stacking · Diversity"),
    ]
    cols = st.columns(3)
    for i, (icon, name, desc) in enumerate(cards):
        with cols[i % 3]:
            st.info(f"**{icon} {name}**\n\n{desc}")

    st.divider()
    st.markdown("""
    ### How to use
    1. Pick a concept from the **left sidebar**
    2. Tune hyperparameters with sliders and dropdowns
    3. Metrics and charts refresh automatically
    """)


# ══════════════════════════════════════════════════════════════════════════════
# LINEAR REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
def page_linear_regression():
    st.title("📈 Linear Regression")
    st.markdown("Compare OLS, Ridge, Lasso, and Polynomial regression on a synthetic dataset.")

    # ── Sidebar inputs ────────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("Dataset")
        n_samples  = st.slider("Samples",       100, 2000, 500, 50)
        n_features = st.slider("Features",        2,   20,   8,  1)
        noise      = st.slider("Noise level",    1.0, 100.0, 20.0, 1.0)
        test_size  = st.slider("Test split %",   10,   50,  20,  5) / 100

        st.subheader("Model")
        model_type = st.selectbox("Variant", ["OLS", "Ridge", "Lasso", "Polynomial"])
        alpha, degree = 1.0, 2
        if model_type in ("Ridge", "Lasso"):
            alpha = st.slider("Alpha (regularisation)", 0.001, 10.0, 1.0, 0.001, format="%.3f")
        if model_type == "Polynomial":
            degree = st.slider("Polynomial degree", 2, 5, 2)

    # ── Data ──────────────────────────────────────────────────────────────────
    @st.cache_data
    def _reg_data(n_samples, n_features, noise, test_size):
        X, y = make_regression(
            n_samples=n_samples, n_features=n_features,
            n_informative=max(2, n_features // 2),
            noise=noise, random_state=42,
        )
        return train_test_split(X, y, test_size=test_size, random_state=42)

    X_train, X_test, y_train, y_test = _reg_data(n_samples, n_features, noise, test_size)
    scaler = StandardScaler()
    Xtr_s  = scaler.fit_transform(X_train)
    Xte_s  = scaler.transform(X_test)

    # ── Model ─────────────────────────────────────────────────────────────────
    if model_type == "OLS":
        model, Xtr, Xte = LinearRegression(), Xtr_s, Xte_s
    elif model_type == "Ridge":
        model, Xtr, Xte = Ridge(alpha=alpha), Xtr_s, Xte_s
    elif model_type == "Lasso":
        model, Xtr, Xte = Lasso(alpha=alpha, max_iter=5000), Xtr_s, Xte_s
    else:
        model = Pipeline([
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("lr",   LinearRegression()),
        ])
        Xtr, Xte = X_train, X_test

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)

    # ── Metrics ───────────────────────────────────────────────────────────────
    mse = mean_squared_error(y_test, y_pred)
    show_metrics({
        "R²":   f"{r2_score(y_test, y_pred):.4f}",
        "MSE":  f"{mse:.2f}",
        "RMSE": f"{np.sqrt(mse):.2f}",
        "MAE":  f"{mean_absolute_error(y_test, y_pred):.2f}",
    })

    # ── Plots ─────────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Predicted vs Actual")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_test, y_pred, alpha=0.55, color=PALETTE[0], s=22)
        lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=2, label="Perfect fit")
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        ax.set_title(f"{model_type} – Fit"); ax.legend()
        fig_to_st(fig)

    with col2:
        st.subheader("Residual Plot")
        residuals = y_test - y_pred
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_pred, residuals, alpha=0.55, color=PALETTE[1], s=22)
        ax.axhline(0, color="red", lw=2, ls="--")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Residual")
        ax.set_title("Residual Distribution")
        fig_to_st(fig)

    # ── Learning curve ────────────────────────────────────────────────────────
    st.subheader("Learning Curve")
    train_sz, tr_sc, val_sc = learning_curve(
        model, Xtr, y_train, cv=5,
        train_sizes=np.linspace(0.1, 1.0, 8), scoring="r2",
    )
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.fill_between(train_sz, tr_sc.mean(1)-tr_sc.std(1), tr_sc.mean(1)+tr_sc.std(1), alpha=0.2, color=PALETTE[0])
    ax.fill_between(train_sz, val_sc.mean(1)-val_sc.std(1), val_sc.mean(1)+val_sc.std(1), alpha=0.2, color=PALETTE[1])
    ax.plot(train_sz, tr_sc.mean(1),  "o-", color=PALETTE[0], label="Train R²")
    ax.plot(train_sz, val_sc.mean(1), "o-", color=PALETTE[1], label="Validation R²")
    ax.set_xlabel("Training examples"); ax.set_ylabel("R²")
    ax.set_title(f"{model_type} – Learning Curve"); ax.legend()
    fig_to_st(fig)

    # ── Coefficients table ────────────────────────────────────────────────────
    if model_type in ("OLS", "Ridge", "Lasso"):
        with st.expander("Coefficients"):
            coefs = model.coef_
            df = pd.DataFrame({
                "Feature":     [f"x{i}" for i in range(len(coefs))],
                "Coefficient": np.round(coefs, 4),
                "Abs Weight":  np.round(np.abs(coefs), 4),
            }).sort_values("Abs Weight", ascending=False).reset_index(drop=True)
            st.dataframe(df, use_container_width=True)

    # ── All-variant comparison ────────────────────────────────────────────────
    with st.expander("Compare All Variants"):
        rows = []
        for vname, vm, vXtr, vXte in [
            ("OLS",          LinearRegression(),                         Xtr_s,   Xte_s),
            ("Ridge (α=1)",  Ridge(1.0),                                 Xtr_s,   Xte_s),
            ("Lasso (α=0.1)",Lasso(0.1, max_iter=5000),                  Xtr_s,   Xte_s),
            ("Polynomial 2", Pipeline([("p", PolynomialFeatures(2, include_bias=False)), ("lr", LinearRegression())]), X_train, X_test),
            ("Polynomial 3", Pipeline([("p", PolynomialFeatures(3, include_bias=False)), ("lr", LinearRegression())]), X_train, X_test),
        ]:
            vm.fit(vXtr, y_train)
            yp = vm.predict(vXte)
            mse_v = mean_squared_error(y_test, yp)
            rows.append({"Model": vname, "R²": round(r2_score(y_test, yp), 4),
                         "RMSE": round(np.sqrt(mse_v), 2), "MAE": round(mean_absolute_error(y_test, yp), 2)})
        df_cmp = pd.DataFrame(rows).sort_values("R²", ascending=False).reset_index(drop=True)
        st.dataframe(df_cmp, use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 3))
        bars = ax.barh(df_cmp["Model"], df_cmp["R²"],
                       color=[PALETTE[i % 10] for i in range(len(df_cmp))])
        ax.bar_label(bars, fmt="%.4f", padding=4)
        ax.set_xlim(0, 1.1); ax.set_xlabel("R²")
        ax.set_title("Variant Comparison"); ax.invert_yaxis()
        fig_to_st(fig)

    st.info("**Key takeaway:** Ridge shrinks all coefficients (handles multicollinearity). Lasso zeros out weak features (automatic selection). Polynomial captures non-linear patterns but risks overfitting at high degrees.")


# ══════════════════════════════════════════════════════════════════════════════
# DECISION TREE
# ══════════════════════════════════════════════════════════════════════════════
def page_decision_tree():
    st.title("🌳 Decision Tree")
    st.markdown("Visualise tree structure, compare pruning strategies, and switch between classification and regression.")

    with st.sidebar:
        st.subheader("Dataset & Task")
        dataset  = st.selectbox("Dataset", ["Iris", "Wine", "Synthetic"])
        task     = st.selectbox("Task",    ["Classification", "Regression"])

        st.subheader("Hyperparameters")
        criterion = st.selectbox("Criterion", ["gini", "entropy"]) if task == "Classification" else "squared_error"
        max_depth_v = st.slider("Max depth  (0 = unlimited)", 0, 20, 4)
        max_depth   = None if max_depth_v == 0 else max_depth_v
        min_split   = st.slider("Min samples split", 2, 50, 2)
        ccp_alpha   = st.slider("CCP alpha (post-pruning)", 0.0, 0.1, 0.0, 0.001, format="%.3f")

    # ── Data ──────────────────────────────────────────────────────────────────
    @st.cache_data
    def _dt_data(name, task):
        if name == "Iris":
            d = load_iris()
            splits = train_test_split(d.data, d.target, test_size=0.2, random_state=42)
            return splits, list(d.feature_names), list(d.target_names)
        elif name == "Wine":
            d = load_wine()
            X = StandardScaler().fit_transform(d.data)
            splits = train_test_split(X, d.target, test_size=0.2, random_state=42)
            return splits, list(d.feature_names), list(d.target_names)
        else:
            if task == "Classification":
                X, y = make_classification(500, n_informative=5, n_classes=3,
                                           n_clusters_per_class=1, random_state=42)
            else:
                X, y = make_regression(500, n_informative=5, noise=20, random_state=42)
            fn = [f"feature_{i}" for i in range(X.shape[1])]
            return train_test_split(X, y, test_size=0.2, random_state=42), fn, ["C0", "C1", "C2"]

    (X_train, X_test, y_train, y_test), feat_names, class_names = _dt_data(dataset, task)

    # ── Train ─────────────────────────────────────────────────────────────────
    if task == "Classification":
        model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,
                                       min_samples_split=min_split, ccp_alpha=ccp_alpha,
                                       random_state=42)
    else:
        model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_split,
                                      ccp_alpha=ccp_alpha, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ── Metrics ───────────────────────────────────────────────────────────────
    if task == "Classification":
        acc = accuracy_score(y_test, y_pred)
        cv  = cross_val_score(model, X_train, y_train, cv=5).mean()
        show_metrics({"Accuracy": f"{acc:.4f}", "CV Mean (5-fold)": f"{cv:.4f}",
                      "Tree Depth": str(model.get_depth()), "Leaves": str(model.get_n_leaves())})
    else:
        mse = mean_squared_error(y_test, y_pred)
        show_metrics({"R²": f"{r2_score(y_test, y_pred):.4f}", "RMSE": f"{np.sqrt(mse):.2f}",
                      "Tree Depth": str(model.get_depth()), "Leaves": str(model.get_n_leaves())})

    # ── Tree visualisation ────────────────────────────────────────────────────
    st.subheader("Tree Structure  (rendered up to depth 4)")
    fig, ax = plt.subplots(figsize=(22, max(5, min((model.get_depth() or 4), 4) * 2)))
    plot_tree(model, feature_names=feat_names,
              class_names=class_names if task == "Classification" else None,
              filled=True, rounded=True, fontsize=8, ax=ax, max_depth=4)
    ax.set_title(f"{dataset} – Decision Tree", fontweight="bold")
    fig_to_st(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Feature Importances")
        imp = model.feature_importances_
        idx = np.argsort(imp)[::-1]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(range(len(imp)), imp[idx], color=PALETTE[0])
        ax.set_xticks(range(len(imp)))
        ax.set_xticklabels([feat_names[i] for i in idx], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Importance")
        fig_to_st(fig)

    with col2:
        if task == "Classification":
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 4))
            ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, colorbar=False, cmap="Blues")
            ax.set_title("Confusion Matrix")
            fig_to_st(fig)
        else:
            st.subheader("Predicted vs Actual")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(y_test, y_pred, alpha=0.5, s=20, color=PALETTE[2])
            lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
            ax.plot([lo, hi], [lo, hi], "r--")
            ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
            fig_to_st(fig)

    # ── Depth sweep ───────────────────────────────────────────────────────────
    with st.expander("Max Depth → Performance Sweep"):
        depths = list(range(1, 16))
        scores = []
        for d in depths:
            m = (DecisionTreeClassifier(max_depth=d, random_state=42)
                 if task == "Classification"
                 else DecisionTreeRegressor(max_depth=d, random_state=42))
            m.fit(X_train, y_train)
            yp = m.predict(X_test)
            scores.append(accuracy_score(y_test, yp) if task == "Classification" else r2_score(y_test, yp))
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(depths, scores, "o-", color=PALETTE[3])
        ax.axvline(model.get_depth(), color="red", ls="--", label=f"Current depth={model.get_depth()}")
        ax.set_xlabel("max_depth")
        ax.set_ylabel("Test Accuracy" if task == "Classification" else "Test R²")
        ax.set_title("Effect of Tree Depth"); ax.legend()
        fig_to_st(fig)
        df_d = pd.DataFrame({"max_depth": depths, ("Accuracy" if task=="Classification" else "R²"): [round(s,4) for s in scores]})
        st.dataframe(df_d.T, use_container_width=True)

    # ── CCP pruning ───────────────────────────────────────────────────────────
    with st.expander("Cost-Complexity Pruning (CCP α sweep)"):
        ccp_path = DecisionTreeClassifier(random_state=42).cost_complexity_pruning_path(X_train, y_train) \
            if task == "Classification" else \
            DecisionTreeRegressor(random_state=42).cost_complexity_pruning_path(X_train, y_train)
        alphas = ccp_path.ccp_alphas
        alphas = alphas[::max(1, len(alphas)//12)]
        ccp_scores = []
        for a in alphas:
            m = (DecisionTreeClassifier(ccp_alpha=a, random_state=42)
                 if task == "Classification"
                 else DecisionTreeRegressor(ccp_alpha=a, random_state=42))
            m.fit(X_train, y_train)
            yp = m.predict(X_test)
            ccp_scores.append(accuracy_score(y_test, yp) if task=="Classification" else r2_score(y_test, yp))
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(alphas, ccp_scores, "o-", color=PALETTE[4])
        ax.set_xlabel("ccp_alpha")
        ax.set_ylabel("Test Accuracy" if task=="Classification" else "Test R²")
        ax.set_title("CCP Pruning – Effect of Alpha")
        fig_to_st(fig)

    st.info("**Key takeaway:** Shallow trees underfit; deep trees overfit. CCP alpha removes branches that add little information gain — it's a principled alternative to setting max_depth manually.")


# ══════════════════════════════════════════════════════════════════════════════
# K-MEANS
# ══════════════════════════════════════════════════════════════════════════════
def page_kmeans():
    st.title("🔵 K-Means Clustering")
    st.markdown("Discover natural groupings. Use the elbow and silhouette to pick the optimal k.")

    with st.sidebar:
        st.subheader("Data Generation")
        n_samples   = st.slider("Samples",             100, 2000, 400, 50)
        n_true      = st.slider("True clusters",         2,    8,   4,  1)
        cluster_std = st.slider("Cluster spread (std)", 0.3,  3.0, 1.2, 0.1)

        st.subheader("K-Means Settings")
        k_final = st.slider("Clusters to fit  (k)", 2, 12, 4)
        k_max   = st.slider("Sweep k up to",        5, 15, 10)
        init    = st.selectbox("Initialisation", ["k-means++", "random"])
        n_init  = st.slider("Restarts (n_init)", 1, 30, 10)

        st.subheader("Agglomerative Comparison")
        agg_linkage = st.selectbox("Linkage", ["ward", "complete", "average", "single"])

    @st.cache_data
    def _blob_data(n_samples, n_true, cluster_std):
        X, labels = make_blobs(n_samples=n_samples, centers=n_true,
                               cluster_std=cluster_std, random_state=42)
        return StandardScaler().fit_transform(X), labels

    X, true_labels = _blob_data(n_samples, n_true, cluster_std)

    # ── Sweep ─────────────────────────────────────────────────────────────────
    k_range = list(range(1, k_max + 1))
    inertias, sil_scores = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, init=init, n_init=n_init, random_state=42)
        km.fit(X)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X, km.labels_) if k >= 2 else None)

    # ── Final model ───────────────────────────────────────────────────────────
    km_final = KMeans(n_clusters=k_final, init=init, n_init=n_init, random_state=42)
    labels_pred = km_final.fit_predict(X)
    sil  = silhouette_score(X, labels_pred)
    db   = davies_bouldin_score(X, labels_pred)

    show_metrics({
        "Inertia":         f"{km_final.inertia_:.2f}",
        "Silhouette":      f"{sil:.4f}",
        "Davies-Bouldin":  f"{db:.4f}",
        "True k":          str(n_true),
        "Fitted k":        str(k_final),
    })

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Elbow Curve")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(k_range, inertias, "o-", color=PALETTE[1])
        ax.axvline(k_final, color="red", ls="--", label=f"k={k_final}")
        ax.set_xlabel("k"); ax.set_ylabel("Inertia (SSE)")
        ax.set_title("Elbow Method"); ax.legend()
        fig_to_st(fig)

    with col2:
        st.subheader("Silhouette Scores by k")
        sil_ks    = [k for k in k_range if k >= 2]
        sil_vals  = [s for s in sil_scores if s is not None]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(sil_ks, sil_vals, color=[
            "#e74c3c" if k == k_final else PALETTE[2] for k in sil_ks
        ])
        ax.set_xlabel("k"); ax.set_ylabel("Silhouette Score")
        ax.set_title("Silhouette Analysis  (higher = better, red = selected)")
        fig_to_st(fig)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader(f"K-Means Clusters  (k={k_final})")
        fig, ax = plt.subplots(figsize=(6, 5))
        for k in range(k_final):
            m = labels_pred == k
            ax.scatter(X[m, 0], X[m, 1], s=28, alpha=0.7, color=PALETTE[k % 10], label=f"Cluster {k}")
        ax.scatter(km_final.cluster_centers_[:, 0], km_final.cluster_centers_[:, 1],
                   s=200, c="black", marker="X", zorder=5, label="Centroids")
        ax.legend(fontsize=8)
        fig_to_st(fig)

    with col4:
        st.subheader("Ground Truth Labels")
        fig, ax = plt.subplots(figsize=(6, 5))
        for k in range(n_true):
            m = true_labels == k
            ax.scatter(X[m, 0], X[m, 1], s=28, alpha=0.7, color=PALETTE[k % 10], label=f"True {k}")
        ax.legend(fontsize=8)
        fig_to_st(fig)

    # ── Agglomerative comparison ───────────────────────────────────────────────
    with st.expander(f"Agglomerative Clustering  (linkage={agg_linkage}, k={k_final})"):
        agg = AgglomerativeClustering(n_clusters=k_final, linkage=agg_linkage)
        agg_labels = agg.fit_predict(X)
        agg_sil = silhouette_score(X, agg_labels)
        agg_db  = davies_bouldin_score(X, agg_labels)

        show_metrics({"Agglomerative Silhouette": f"{agg_sil:.4f}",
                      "Agglomerative DB": f"{agg_db:.4f}",
                      "K-Means Silhouette": f"{sil:.4f}"})

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, lbl, title in [
            (axes[0], labels_pred, f"K-Means (k={k_final})"),
            (axes[1], agg_labels,  f"Agglomerative ({agg_linkage})"),
        ]:
            unique = np.unique(lbl)
            for k in unique:
                mk = lbl == k
                ax.scatter(X[mk, 0], X[mk, 1], s=28, alpha=0.7, color=PALETTE[k % 10])
            ax.set_title(title)
        fig_to_st(fig)

    # ── Raw cluster data table ─────────────────────────────────────────────────
    with st.expander("Sweep Results Table"):
        df_sw = pd.DataFrame({
            "k":         sil_ks,
            "Inertia":   [round(inertias[k-1], 2) for k in sil_ks],
            "Silhouette":[round(s, 4) for s in sil_vals],
        })
        st.dataframe(df_sw, use_container_width=True)

    st.info("**Key takeaway:** Always standardise before clustering — K-Means is distance-based. Use Elbow + Silhouette together: elbow shows diminishing returns, silhouette confirms cluster cohesion.")


# ══════════════════════════════════════════════════════════════════════════════
# BAGGING & RANDOM FOREST
# ══════════════════════════════════════════════════════════════════════════════
def page_random_forest():
    st.title("🌲 Bagging & Random Forest")
    st.markdown("See how aggregating many decorrelated trees reduces variance and improves generalisation.")

    with st.sidebar:
        st.subheader("Dataset")
        dataset = st.selectbox("Dataset", ["Wine", "Iris", "Synthetic"])

        st.subheader("Random Forest")
        n_est    = st.slider("n_estimators",           1, 500, 100,  5)
        md_v     = st.slider("max_depth  (0=unlimited)", 0,  20,   0,  1)
        max_depth_rf = None if md_v == 0 else md_v
        mf_label = st.selectbox("max_features", ["sqrt", "log2", "All features", "30 %", "60 %"])
        mf_map   = {"sqrt": "sqrt", "log2": "log2", "All features": None, "30 %": 0.3, "60 %": 0.6}
        mf       = mf_map[mf_label]
        oob_on   = st.checkbox("Compute OOB score", True)

        st.subheader("Compare against")
        cmp_dt   = st.checkbox("Single Decision Tree", True)
        cmp_bag  = st.checkbox("Bagging (100 trees)",  True)
        cmp_et   = st.checkbox("ExtraTrees",           True)

    (X_train, X_test, y_train, y_test), feat_names, target_names = load_dataset_clf(dataset)

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf = RandomForestClassifier(n_estimators=n_est, max_depth=max_depth_rf,
                                max_features=mf, oob_score=oob_on,
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_acc  = accuracy_score(y_test, rf.predict(X_test))
    metrics = {"RF Accuracy": f"{rf_acc:.4f}", "Trees": str(n_est)}
    if oob_on:
        metrics["OOB Score"] = f"{rf.oob_score_:.4f}"

    # ── Optional comparators ──────────────────────────────────────────────────
    cmp_names = ["Random Forest"]
    cmp_accs  = [rf_acc]

    if cmp_dt:
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train, y_train)
        dt_acc = accuracy_score(y_test, dt.predict(X_test))
        cmp_names.append("Single DT"); cmp_accs.append(dt_acc)
        metrics["Single DT"] = f"{dt_acc:.4f}"

    if cmp_bag:
        bag = BaggingClassifier(n_estimators=n_est, random_state=42, n_jobs=-1)
        bag.fit(X_train, y_train)
        bag_acc = accuracy_score(y_test, bag.predict(X_test))
        cmp_names.append("Bagging"); cmp_accs.append(bag_acc)
        metrics["Bagging"] = f"{bag_acc:.4f}"

    if cmp_et:
        et = ExtraTreesClassifier(n_estimators=n_est, random_state=42, n_jobs=-1)
        et.fit(X_train, y_train)
        et_acc = accuracy_score(y_test, et.predict(X_test))
        cmp_names.append("ExtraTrees"); cmp_accs.append(et_acc)
        metrics["ExtraTrees"] = f"{et_acc:.4f}"

    show_metrics(metrics)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Feature Importances (RF)")
        imp = rf.feature_importances_
        idx = np.argsort(imp)[::-1][:20]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(range(len(idx)), imp[idx], color=PALETTE[0])
        ax.set_xticks(range(len(idx)))
        ax.set_xticklabels([feat_names[i] for i in idx], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Importance")
        fig_to_st(fig)

    with col2:
        st.subheader("Confusion Matrix (RF)")
        cm = confusion_matrix(y_test, rf.predict(X_test))
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(cm, display_labels=target_names).plot(ax=ax, colorbar=False, cmap="Blues")
        fig_to_st(fig)

    # ── Model comparison ──────────────────────────────────────────────────────
    if len(cmp_names) > 1:
        st.subheader("Model Comparison")
        fig, ax = plt.subplots(figsize=(10, max(3, len(cmp_names) * 0.7)))
        colors = [PALETTE[i % 10] for i in range(len(cmp_names))]
        bars = ax.barh(cmp_names, cmp_accs, color=colors)
        ax.bar_label(bars, fmt="%.4f", padding=4)
        ax.set_xlim(0, 1.12); ax.set_xlabel("Test Accuracy")
        ax.set_title("Bagging / Forest – Model Comparison"); ax.invert_yaxis()
        fig_to_st(fig)

    # ── n_estimators sweep ────────────────────────────────────────────────────
    with st.expander("n_estimators Sweep"):
        ns = [1, 5, 10, 25, 50, 100, 200, 300, 500]
        tr_accs, te_accs = [], []
        for n in ns:
            m = RandomForestClassifier(n_estimators=n, max_features=mf, random_state=42, n_jobs=-1)
            m.fit(X_train, y_train)
            tr_accs.append(accuracy_score(y_train, m.predict(X_train)))
            te_accs.append(accuracy_score(y_test,  m.predict(X_test)))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ns, tr_accs, "o-", color=PALETTE[0], label="Train")
        ax.plot(ns, te_accs, "o-", color=PALETTE[1], label="Test")
        ax.axvline(n_est, color="red", ls="--", label=f"Selected n={n_est}")
        ax.set_xlabel("n_estimators"); ax.set_ylabel("Accuracy")
        ax.set_title("RF – n_estimators Sweep"); ax.legend()
        fig_to_st(fig)

        df_sw = pd.DataFrame({"n_estimators": ns, "Train Acc": [round(a,4) for a in tr_accs],
                               "Test Acc": [round(a,4) for a in te_accs]})
        st.dataframe(df_sw, use_container_width=True)

    # ── max_features sweep ────────────────────────────────────────────────────
    with st.expander("max_features Sweep"):
        mf_opts = [("sqrt", "sqrt"), ("log2", "log2"), ("None (all)", None), ("0.3", 0.3), ("0.6", 0.6)]
        mf_labels_sw = [l for l, _ in mf_opts]
        mf_accs_sw   = []
        for _, mf_val in mf_opts:
            m = RandomForestClassifier(n_estimators=100, max_features=mf_val, random_state=42, n_jobs=-1)
            m.fit(X_train, y_train)
            mf_accs_sw.append(accuracy_score(y_test, m.predict(X_test)))
        fig, ax = plt.subplots(figsize=(8, 3.5))
        bars = ax.bar(mf_labels_sw, mf_accs_sw, color=[PALETTE[i % 10] for i in range(len(mf_labels_sw))])
        ax.bar_label(bars, fmt="%.4f")
        ax.set_ylim(0, 1.1); ax.set_ylabel("Accuracy")
        ax.set_title("max_features Sweep (100 trees)")
        fig_to_st(fig)

    st.info("**Key takeaway:** Bagging reduces variance by averaging many high-variance models. Random Forest additionally randomises feature selection at each split, decorrelating trees for even better results. OOB score gives a free, unbiased accuracy estimate.")


# ══════════════════════════════════════════════════════════════════════════════
# BOOSTING
# ══════════════════════════════════════════════════════════════════════════════
def page_boosting():
    st.title("⚡ Boosting")
    st.markdown("Sequential learners that focus on the previous model's mistakes — reducing bias stage by stage.")

    with st.sidebar:
        st.subheader("Dataset")
        dataset_b = st.selectbox("Dataset", ["Wine", "Iris", "Synthetic"])

        st.subheader("AdaBoost")
        ada_n     = st.slider("n_estimators",     10, 500, 200, 10)
        ada_lr    = st.slider("Learning rate",  0.01, 2.0, 0.5, 0.01)
        ada_depth = st.slider("Base learner depth", 1, 5, 1)

        st.subheader("Gradient Boosting (GBM)")
        gbm_n     = st.slider("GBM: n_estimators",  10, 500, 100, 10)
        gbm_lr    = st.slider("GBM: learning rate", 0.001, 1.0, 0.05, 0.001, format="%.3f")
        gbm_depth = st.slider("GBM: max_depth",      1,  10,   3,   1)
        gbm_sub   = st.slider("GBM: subsample",     0.3,  1.0, 0.8, 0.05)

    (X_train, X_test, y_train, y_test), feat_names, target_names = load_dataset_clf(dataset_b)

    # ── Train ─────────────────────────────────────────────────────────────────
    with st.spinner("Training boosting models…"):
        ada = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=ada_depth),
            n_estimators=ada_n, learning_rate=ada_lr, random_state=42,
        )
        ada.fit(X_train, y_train)
        ada_acc = accuracy_score(y_test, ada.predict(X_test))

        gbm = GradientBoostingClassifier(
            n_estimators=gbm_n, learning_rate=gbm_lr,
            max_depth=gbm_depth, subsample=gbm_sub, random_state=42,
        )
        gbm.fit(X_train, y_train)
        gbm_acc = accuracy_score(y_test, gbm.predict(X_test))

        hgbm = HistGradientBoostingClassifier(
            max_iter=gbm_n, learning_rate=gbm_lr, max_depth=gbm_depth, random_state=42,
        )
        hgbm.fit(X_train, y_train)
        hgbm_acc = accuracy_score(y_test, hgbm.predict(X_test))

    show_metrics({
        "AdaBoost Accuracy":    f"{ada_acc:.4f}",
        "GBM Accuracy":         f"{gbm_acc:.4f}",
        "HistGBM Accuracy":     f"{hgbm_acc:.4f}",
    })

    # ── Stagewise curves ──────────────────────────────────────────────────────
    st.subheader("Stagewise Error Curves")
    col1, col2 = st.columns(2)

    with col1:
        ada_tr = [1. - accuracy_score(y_train, p) for p in ada.staged_predict(X_train)]
        ada_te = [1. - accuracy_score(y_test,  p) for p in ada.staged_predict(X_test)]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ada_tr, label="Train", color=PALETTE[0])
        ax.plot(ada_te, label="Test",  color=PALETTE[1])
        ax.set_xlabel("Estimator #"); ax.set_ylabel("Error Rate")
        ax.set_title("AdaBoost – Stagewise Error"); ax.legend()
        fig_to_st(fig)

    with col2:
        gbm_tr = [1. - accuracy_score(y_train, p) for p in gbm.staged_predict(X_train)]
        gbm_te = [1. - accuracy_score(y_test,  p) for p in gbm.staged_predict(X_test)]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(gbm_tr, label="Train", color=PALETTE[0])
        ax.plot(gbm_te, label="Test",  color=PALETTE[1])
        ax.set_xlabel("Estimator #"); ax.set_ylabel("Error Rate")
        ax.set_title("GBM – Stagewise Error"); ax.legend()
        fig_to_st(fig)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("GBM Feature Importances")
        imp = gbm.feature_importances_
        idx = np.argsort(imp)[::-1][:15]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(range(len(idx)), imp[idx], color=PALETTE[3])
        ax.set_xticks(range(len(idx)))
        ax.set_xticklabels([feat_names[i] for i in idx], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Importance")
        fig_to_st(fig)

    with col4:
        st.subheader("Best Model – Confusion Matrix")
        best_m = max([(ada, ada_acc), (gbm, gbm_acc), (hgbm, hgbm_acc)], key=lambda x: x[1])[0]
        cm = confusion_matrix(y_test, best_m.predict(X_test))
        fig, ax = plt.subplots(figsize=(6, 4))
        ConfusionMatrixDisplay(cm, display_labels=target_names).plot(ax=ax, colorbar=False, cmap="Oranges")
        fig_to_st(fig)

    # ── Algorithm comparison bar ───────────────────────────────────────────────
    with st.expander("Algorithm Comparison"):
        fig, ax = plt.subplots(figsize=(8, 3))
        names_c = ["AdaBoost", "GradientBoosting", "HistGBM"]
        accs_c  = [ada_acc, gbm_acc, hgbm_acc]
        bars = ax.barh(names_c, accs_c, color=[PALETTE[0], PALETTE[1], PALETTE[2]])
        ax.bar_label(bars, fmt="%.4f", padding=4)
        ax.set_xlim(0, 1.1); ax.invert_yaxis()
        ax.set_title("Boosting Algorithms – Accuracy Comparison")
        fig_to_st(fig)

    # ── Learning rate sweep ────────────────────────────────────────────────────
    with st.expander("Learning Rate Sweep (GBM, 100 trees)"):
        lrs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
        lr_accs = []
        for lr in lrs:
            m = GradientBoostingClassifier(n_estimators=100, learning_rate=lr,
                                           max_depth=3, random_state=42)
            m.fit(X_train, y_train)
            lr_accs.append(accuracy_score(y_test, m.predict(X_test)))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.semilogx(lrs, lr_accs, "o-", color=PALETTE[4])
        ax.axvline(gbm_lr, color="red", ls="--", label=f"Selected lr={gbm_lr}")
        ax.set_xlabel("Learning Rate (log scale)"); ax.set_ylabel("Test Accuracy")
        ax.set_title("GBM – Learning Rate Sweep"); ax.legend()
        fig_to_st(fig)
        df_lr = pd.DataFrame({"Learning Rate": lrs, "Accuracy": [round(a, 4) for a in lr_accs]})
        st.dataframe(df_lr, use_container_width=True)

    st.info("**Key takeaway:** Lower learning rate + more trees generalises better but costs more compute. HistGBM bins features into histograms — 10–100× faster and handles missing values natively.")


# ══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE METHODS
# ══════════════════════════════════════════════════════════════════════════════
def page_ensemble():
    st.title("🎯 Ensemble Methods")
    st.markdown("Combine diverse learners — voting, weighted averaging, and stacking — to surpass any single model.")

    with st.sidebar:
        st.subheader("Dataset")
        dataset_e = st.selectbox("Dataset", ["Wine", "Iris", "Synthetic"])

        st.subheader("Base Learners")
        use_dt  = st.checkbox("Decision Tree",      True)
        use_rf  = st.checkbox("Random Forest",      True)
        use_gbm = st.checkbox("Gradient Boosting",  True)
        use_knn = st.checkbox("K-Nearest Neighbors",True)
        use_nb  = st.checkbox("Naive Bayes",        True)
        use_svm = st.checkbox("SVM (RBF kernel)",   True)

        st.subheader("Ensemble Strategies")
        run_hard     = st.checkbox("Hard Voting",       True)
        run_soft     = st.checkbox("Soft Voting",       True)
        run_weighted = st.checkbox("Weighted Voting",   True)
        run_stack_lr = st.checkbox("Stacking  (LR)",   True)
        run_stack_rf = st.checkbox("Stacking  (RF)",   False)

    (X_train, X_test, y_train, y_test), _, target_names = load_dataset_clf(dataset_e)

    # ── Build base learner list ───────────────────────────────────────────────
    candidate = [
        ("dt",  DecisionTreeClassifier(max_depth=4, random_state=42),                      use_dt),
        ("rf",  RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),       use_rf),
        ("gbm", GradientBoostingClassifier(n_estimators=100, random_state=42),              use_gbm),
        ("knn", KNeighborsClassifier(n_neighbors=7),                                        use_knn),
        ("nb",  GaussianNB(),                                                               use_nb),
        ("svm", SVC(probability=True, random_state=42),                                     use_svm),
    ]
    base = [(n, m) for n, m, flag in candidate if flag]

    if not base:
        st.warning("Please select at least one base learner in the sidebar.")
        return

    # ── Train base learners ───────────────────────────────────────────────────
    with st.spinner("Training base learners…"):
        ind_accs   = {}
        fitted     = {}
        for name, model in base:
            model.fit(X_train, y_train)
            ind_accs[name] = accuracy_score(y_test, model.predict(X_test))
            fitted[name]   = model

    st.subheader("Individual Base Learner Accuracy")
    show_metrics({n.upper(): f"{a:.4f}" for n, a in ind_accs.items()})

    # ── Ensemble models ───────────────────────────────────────────────────────
    ens_results = {}
    with st.spinner("Building ensembles…"):
        if run_hard and len(base) >= 2:
            hv = VotingClassifier(base, voting="hard", n_jobs=-1)
            hv.fit(X_train, y_train)
            ens_results["Hard Voting"] = (hv, accuracy_score(y_test, hv.predict(X_test)))

        if run_soft and len(base) >= 2:
            sv = VotingClassifier(base, voting="soft", n_jobs=-1)
            sv.fit(X_train, y_train)
            ens_results["Soft Voting"] = (sv, accuracy_score(y_test, sv.predict(X_test)))

        if run_weighted and len(base) >= 2:
            wts = [ind_accs[n] for n, _ in base]
            wv  = VotingClassifier(base, voting="soft", weights=wts, n_jobs=-1)
            wv.fit(X_train, y_train)
            ens_results["Weighted Voting"] = (wv, accuracy_score(y_test, wv.predict(X_test)))

        if run_stack_lr and len(base) >= 2:
            stk = StackingClassifier(
                base,
                final_estimator=LogisticRegression(max_iter=1000, random_state=42),
                cv=5, stack_method="predict_proba", n_jobs=-1,
            )
            stk.fit(X_train, y_train)
            ens_results["Stacking (LR)"] = (stk, accuracy_score(y_test, stk.predict(X_test)))

        if run_stack_rf and len(base) >= 2:
            stk_rf = StackingClassifier(
                base,
                final_estimator=RandomForestClassifier(n_estimators=50, random_state=42),
                cv=5, stack_method="predict_proba", n_jobs=-1,
            )
            stk_rf.fit(X_train, y_train)
            ens_results["Stacking (RF)"] = (stk_rf, accuracy_score(y_test, stk_rf.predict(X_test)))

    if ens_results:
        st.subheader("Ensemble Accuracy")
        show_metrics({k: f"{v:.4f}" for k, (_, v) in ens_results.items()})

    # ── Full comparison chart ─────────────────────────────────────────────────
    all_names = list(ind_accs.keys()) + list(ens_results.keys())
    all_accs  = list(ind_accs.values()) + [v for _, v in ens_results.values()]
    all_types = (["Base"] * len(ind_accs)) + (["Ensemble"] * len(ens_results))

    st.subheader("Full Comparison")
    fig, ax = plt.subplots(figsize=(12, max(4, len(all_names) * 0.55 + 1)))
    bar_colors = [PALETTE[2] if t == "Ensemble" else PALETTE[0] for t in all_types]
    bars = ax.barh(all_names, all_accs, color=bar_colors)
    ax.bar_label(bars, fmt="%.4f", padding=4)
    ax.set_xlim(0, 1.12); ax.set_xlabel("Test Accuracy")
    ax.set_title("Ensemble Methods – Full Comparison  (blue=base, green=ensemble)")
    ax.invert_yaxis()
    fig_to_st(fig)

    # ── Best model confusion matrix ───────────────────────────────────────────
    if ens_results:
        best_name = max(ens_results, key=lambda k: ens_results[k][1])
        best_model, best_acc = ens_results[best_name]
        st.subheader(f"Best Ensemble – Confusion Matrix  ({best_name}, acc={best_acc:.4f})")
        cm = confusion_matrix(y_test, best_model.predict(X_test))
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(cm, display_labels=target_names).plot(ax=ax, colorbar=False, cmap="Greens")
        fig_to_st(fig)

    # ── Diversity heatmap ─────────────────────────────────────────────────────
    with st.expander("Base Learner Diversity  (Pairwise Prediction Agreement)"):
        ns  = list(fitted.keys())
        prs = np.array([m.predict(X_test) for m in fitted.values()])
        n   = len(ns)
        mat = np.array([[((prs[i] == prs[j]).mean()) for j in range(n)] for i in range(n)])
        fig, ax = plt.subplots(figsize=(max(5, n), max(4, n-1)))
        sns.heatmap(mat, annot=True, fmt=".2f", xticklabels=ns, yticklabels=ns,
                    cmap="YlOrRd", ax=ax, vmin=0, vmax=1)
        ax.set_title("Pairwise Agreement  (low off-diagonal = high diversity = better ensemble)")
        fig_to_st(fig)

    # ── Results table ─────────────────────────────────────────────────────────
    with st.expander("Full Results Table"):
        df = pd.DataFrame({
            "Model":    all_names,
            "Type":     all_types,
            "Accuracy": [round(a, 4) for a in all_accs],
        }).sort_values("Accuracy", ascending=False).reset_index(drop=True)
        st.dataframe(df, use_container_width=True)

    st.info("**Key takeaway:** Ensemble power comes from **diversity**, not just individual accuracy. Low off-diagonal values in the diversity heatmap mean models disagree on different examples — perfect for combining.")


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════
PAGES = {
    "🏠  Home":                    page_home,
    "📈  Linear Regression":       page_linear_regression,
    "🌳  Decision Tree":           page_decision_tree,
    "🔵  K-Means Clustering":      page_kmeans,
    "🌲  Bagging & Random Forest": page_random_forest,
    "⚡  Boosting":                page_boosting,
    "🎯  Ensemble Methods":        page_ensemble,
}

PAGES[page]()
