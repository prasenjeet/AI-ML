"""
Streamlit Interactive UI – ML Concepts Demo
============================================
Run:   streamlit run app.py
"""

import os as _os_top
_os_top.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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
            "🧠  Neural Networks",
            "📝  NLP",
            "🖼️  Computer Vision",
            "🎬  Recommendations",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("scikit-learn · TensorFlow · Streamlit")


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
        ("🔵", "ANN",                      "Feedforward MLP · activations · optimizers"),
        ("🟣", "DNN",                      "Deep layers · BatchNorm · Dropout · depth sweep"),
        ("🖼️", "CNN",                      "Conv2D · Pooling · MNIST · filter sweep"),
        ("🔁", "RNN",                      "SimpleRNN · seq length · stacked layers"),
        ("⏳", "LSTM",                     "Gates · GRU · Bidirectional · long-range memory"),
        ("📝", "NLP",                      "TF-IDF · Classification · Sentiment · Topics"),
        ("🖼️", "Computer Vision",          "Augmentation · HOG · CNN CIFAR-10 · Transfer"),
        ("🎬", "Recommendations",          "User-CF · Item-CF · SVD · Content-Based"),
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
# ══════════════════════════════════════════════════════════════════════════════
# NEURAL NETWORKS  (ANN · DNN · CNN · RNN · LSTM)
# ══════════════════════════════════════════════════════════════════════════════
import os as _os
import subprocess as _subprocess
import sys as _sys

_os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def _check_tf_available() -> bool:
    """Run TF import in a subprocess; returns False if it crashes (e.g. no AVX)."""
    try:
        r = _subprocess.run(
            [_sys.executable, "-c", "import tensorflow"],
            capture_output=True, timeout=20,
        )
        return r.returncode == 0
    except Exception:
        return False

_TF_AVAILABLE: bool = _check_tf_available()

def _tf():
    """Lazy-import TensorFlow; returns (None,None,None,None) when unavailable."""
    if not _TF_AVAILABLE:
        return None, None, None, None
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    return tf, keras, layers, callbacks

_TF_UNAVAILABLE_MSG = (
    "⚠️ **TensorFlow is not available on this machine.**  \n"
    "This CPU does not support the AVX instructions required by the installed "
    "TensorFlow build.  \n\n"
    "To enable neural-network sections, install a non-AVX TensorFlow build or "
    "run the app on a machine with AVX support."
)


# ── Shared NN helpers ─────────────────────────────────────────────────────────
def _plot_history_st(history, title="Training History"):
    """Render loss + accuracy curves inside Streamlit."""
    keys = list(history.history.keys())
    has_acc = any("acc" in k for k in keys)
    n = 2 if has_acc else 1
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]

    axes[0].plot(history.history["loss"],     label="Train",      color=PALETTE[0])
    if "val_loss" in history.history:
        axes[0].plot(history.history["val_loss"], label="Validation", color=PALETTE[1])
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} – Loss"); axes[0].legend()

    if has_acc:
        ak  = "accuracy"     if "accuracy"     in keys else "acc"
        vak = "val_accuracy" if "val_accuracy" in keys else "val_acc"
        axes[1].plot(history.history[ak],  label="Train",      color=PALETTE[2])
        if vak in history.history:
            axes[1].plot(history.history[vak], label="Validation", color=PALETTE[3])
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
        axes[1].set_title(f"{title} – Accuracy"); axes[1].legend()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _plot_ts_st(y_true, y_pred, title="Prediction vs Actual"):
    n_show = min(300, len(y_true))
    fig, axes = plt.subplots(2, 1, figsize=(11, 5))
    axes[0].plot(y_true[:n_show], label="Actual",    color=PALETTE[0], lw=1.2)
    axes[0].plot(y_pred[:n_show], label="Predicted", color=PALETTE[1], lw=1.2, alpha=0.85)
    axes[0].set_title(title); axes[0].legend()
    err = np.abs(y_true[:n_show] - y_pred[:n_show])
    axes[1].fill_between(range(n_show), err, alpha=0.5, color=PALETTE[2])
    axes[1].set_title("Absolute Error"); axes[1].set_xlabel("Time step")
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)


def _arch_md(model) -> str:
    lines = ["```", f"{'Layer':<24} {'Output shape':<22} {'Params':>10}",
             "-" * 58]
    total = 0
    for layer in model.layers:
        name = layer.__class__.__name__
        try:
            shape = str(layer.output.shape)
        except Exception:
            shape = "?"
        p = layer.count_params()
        total += p
        lines.append(f"{name:<24} {shape:<22} {p:>10,}")
    lines += ["-" * 58, f"{'Total params':<46} {total:>10,}", "```"]
    return "\n".join(lines)


def _make_time_series():
    rng = np.random.default_rng(42)
    t = np.linspace(0, 6 * np.pi * 10, 3000)
    return (np.sin(0.3 * t) + 0.5 * np.sin(0.7 * t)
            + 0.3 * np.sin(1.3 * t)
            + 0.08 * rng.standard_normal(3000)).astype(np.float32)


def _make_sequences(signal, seq_len):
    X, y = [], []
    for i in range(seq_len, len(signal)):
        X.append(signal[i - seq_len: i])
        y.append(signal[i])
    return np.array(X, dtype=np.float32)[..., np.newaxis], np.array(y, dtype=np.float32)


def _tabular(name):
    from sklearn.model_selection import train_test_split
    if name == "Iris":
        from sklearn.datasets import load_iris
        d = load_iris(); X, y = d.data, d.target
        cnames = list(d.target_names)
    elif name == "Wine":
        from sklearn.datasets import load_wine
        d = load_wine(); X, y = d.data, d.target
        cnames = list(d.target_names)
    else:
        from sklearn.datasets import make_classification
        X, y = make_classification(600, n_features=15, n_informative=8,
                                   n_classes=3, n_clusters_per_class=1, random_state=42)
        cnames = ["Class 0", "Class 1", "Class 2"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    return sc.fit_transform(Xtr).astype(np.float32), sc.transform(Xte).astype(np.float32), ytr, yte, cnames


# ── Progress callback ─────────────────────────────────────────────────────────
def _keras_progress_cb(n_epochs, bar, status):
    _, keras, _, callbacks = _tf()
    if keras is None:
        return None

    class _CB(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            bar.progress((epoch + 1) / n_epochs)
            parts = [f"epoch {epoch+1}/{n_epochs}"]
            for k, v in logs.items():
                parts.append(f"{k}={v:.4f}")
            status.text("  ".join(parts))

    return _CB()


# ══════════════════════════════════════════════════════════════════════════════
# TAB: ANN
# ══════════════════════════════════════════════════════════════════════════════
def _tab_ann():
    tf, keras, layers, cbs_mod = _tf()
    if tf is None:
        st.warning(_TF_UNAVAILABLE_MSG)
        return

    st.markdown("### Artificial Neural Network — feedforward MLP for tabular data")
    st.markdown(
        "A fully-connected network where every neuron in one layer connects to every neuron "
        "in the next. The simplest neural architecture — great for structured / tabular data."
    )

    # ── Controls ──────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        dataset   = st.selectbox("Dataset",    ["Wine", "Iris", "Synthetic"], key="ann_ds")
        n_layers  = st.slider("Hidden layers",  1, 6, 2, key="ann_nl")
        n_neurons = st.slider("Neurons / layer", 8, 256, 64, 8, key="ann_nn")
    with c2:
        activation = st.selectbox("Activation", ["relu", "tanh", "sigmoid", "elu"], key="ann_act")
        optimizer  = st.selectbox("Optimizer",  ["adam", "rmsprop", "sgd"], key="ann_opt")
        lr         = st.select_slider("Learning rate", [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
                                      value=1e-3, key="ann_lr")
    with c3:
        epochs   = st.slider("Epochs",          10, 200, 80, 10, key="ann_ep")
        batch_sz = st.select_slider("Batch size", [8, 16, 32, 64, 128], value=32, key="ann_bs")
        dropout  = st.slider("Dropout",         0.0, 0.5, 0.0, 0.05, key="ann_dr")

    if st.button("🚀 Train ANN", key="ann_train"):
        X_train, X_test, y_train, y_test, cls_names = _tabular(dataset)
        n_feat, n_cls = X_train.shape[1], len(cls_names)

        keras.backend.clear_session()
        inp = keras.Input(shape=(n_feat,))
        x   = inp
        for _ in range(n_layers):
            x = layers.Dense(n_neurons, activation=activation)(x)
            if dropout > 0:
                x = layers.Dropout(dropout)(x)
        out = layers.Dense(n_cls, activation="softmax")(x)
        model = keras.Model(inp, out, name="ANN")

        opt_obj = {"adam": keras.optimizers.Adam(lr),
                   "rmsprop": keras.optimizers.RMSprop(lr),
                   "sgd": keras.optimizers.SGD(lr)}[optimizer]
        model.compile(optimizer=opt_obj, loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])

        # Architecture display
        st.markdown("**Architecture**")
        st.markdown(_arch_md(model))

        bar    = st.progress(0)
        status = st.empty()
        history = model.fit(
            X_train, y_train, epochs=epochs, batch_size=batch_sz,
            validation_split=0.2, verbose=0,
            callbacks=[_keras_progress_cb(epochs, bar, status)],
        )
        bar.empty(); status.empty()

        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        acc = accuracy_score(y_test, y_pred)

        show_metrics({"Test Accuracy": f"{acc:.4f}",
                      "Val Accuracy":  f"{history.history['val_accuracy'][-1]:.4f}",
                      "Final Loss":    f"{history.history['loss'][-1]:.4f}",
                      "Total Params":  f"{model.count_params():,}"})

        col1, col2 = st.columns(2)
        with col1:
            _plot_history_st(history, "ANN")
        with col2:
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay(cm, display_labels=cls_names).plot(ax=ax, colorbar=False, cmap="Blues")
            ax.set_title("Confusion Matrix")
            fig_to_st(fig)

        # Activation comparison
        with st.expander("Compare All Activations"):
            act_names, act_accs = [], []
            prog = st.progress(0)
            for i, act in enumerate(["relu", "tanh", "sigmoid", "elu", "selu"]):
                keras.backend.clear_session()
                m_inp = keras.Input(shape=(n_feat,))
                mx = m_inp
                for _ in range(n_layers):
                    mx = layers.Dense(n_neurons, activation=act)(mx)
                m_out = layers.Dense(n_cls, activation="softmax")(mx)
                m = keras.Model(m_inp, m_out)
                m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
                m.fit(X_train, y_train, epochs=50, batch_size=batch_sz,
                      validation_split=0.2, verbose=0)
                yp = np.argmax(m.predict(X_test, verbose=0), axis=1)
                act_names.append(act); act_accs.append(accuracy_score(y_test, yp))
                prog.progress((i + 1) / 5)
            prog.empty()
            fig, ax = plt.subplots(figsize=(8, 3))
            bars = ax.bar(act_names, act_accs, color=[PALETTE[i % 10] for i in range(5)])
            ax.bar_label(bars, fmt="%.4f"); ax.set_ylim(0, 1.1)
            ax.set_title("Activation Function Comparison"); fig_to_st(fig)

    else:
        st.info("👆 Set parameters above and click **Train ANN** to start.")

    st.info("**Key concept:** ANN = universal function approximator. Deeper/wider networks "
            "can model more complex patterns, but need regularisation (Dropout) to generalise.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB: DNN
# ══════════════════════════════════════════════════════════════════════════════
def _tab_dnn():
    tf, keras, layers, cbs_mod = _tf()
    if tf is None:
        st.warning(_TF_UNAVAILABLE_MSG)
        return

    st.markdown("### Deep Neural Network — going deeper with BatchNorm & Dropout")
    st.markdown(
        "A DNN extends the ANN with many more hidden layers. Batch Normalisation and "
        "Dropout are essential to train deep networks stably and without overfitting."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        dataset  = st.selectbox("Dataset",     ["Synthetic", "Wine", "Iris"], key="dnn_ds")
        n_layers = st.slider("Hidden layers",   2, 10, 5, key="dnn_nl")
        width    = st.slider("Layer width",     16, 256, 128, 16, key="dnn_w")
    with c2:
        dropout   = st.slider("Dropout",        0.0, 0.5, 0.3, 0.05, key="dnn_dr")
        batch_norm= st.checkbox("Batch Normalisation", True, key="dnn_bn")
        initializer = st.selectbox("Initialiser",
                                   ["he_normal", "glorot_uniform", "he_uniform", "lecun_normal"],
                                   key="dnn_init")
    with c3:
        epochs   = st.slider("Epochs", 20, 200, 100, 10, key="dnn_ep")
        batch_sz = st.select_slider("Batch size", [16, 32, 64, 128], value=32, key="dnn_bs")
        lr       = st.select_slider("Learning rate", [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
                                    value=1e-3, key="dnn_lr")

    if st.button("🚀 Train DNN", key="dnn_train"):
        X_train, X_test, y_train, y_test, cls_names = _tabular(dataset)
        n_feat, n_cls = X_train.shape[1], len(cls_names)

        keras.backend.clear_session()
        inp = keras.Input(shape=(n_feat,))
        x   = inp
        for i in range(n_layers):
            units = max(width // (2 ** (i // 3)), 16)
            x = layers.Dense(units, activation=None, kernel_initializer=initializer)(x)
            if batch_norm:
                x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            if dropout > 0 and i < n_layers - 1:
                x = layers.Dropout(dropout)(x)
        out = layers.Dense(n_cls, activation="softmax")(x)
        model = keras.Model(inp, out, name="DNN")
        model.compile(
            optimizer=keras.optimizers.Adam(lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        st.markdown("**Architecture**")
        st.markdown(_arch_md(model))

        bar = st.progress(0); status = st.empty()
        es  = keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, verbose=0)
        rlr = keras.callbacks.ReduceLROnPlateau(patience=8, factor=0.5, verbose=0)
        history = model.fit(
            X_train, y_train, epochs=epochs, batch_size=batch_sz,
            validation_split=0.2, verbose=0,
            callbacks=[_keras_progress_cb(epochs, bar, status), es, rlr],
        )
        bar.empty(); status.empty()

        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        acc = accuracy_score(y_test, y_pred)
        actual_epochs = len(history.history["loss"])

        show_metrics({"Test Accuracy": f"{acc:.4f}",
                      "Val Accuracy":  f"{history.history['val_accuracy'][-1]:.4f}",
                      "Epochs run":    str(actual_epochs),
                      "Total Params":  f"{model.count_params():,}"})

        col1, col2 = st.columns(2)
        with col1:
            _plot_history_st(history, "DNN")
        with col2:
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay(cm, display_labels=cls_names).plot(ax=ax, colorbar=False, cmap="Purples")
            ax.set_title("Confusion Matrix"); fig_to_st(fig)

        # Depth sweep
        with st.expander("Depth Sweep (layers 1 → 10)"):
            depths, d_accs = list(range(1, 11)), []
            prog = st.progress(0)
            for i, d in enumerate(depths):
                keras.backend.clear_session()
                m_inp = keras.Input(shape=(n_feat,))
                mx = m_inp
                for _ in range(d):
                    mx = layers.Dense(width, activation=None, kernel_initializer=initializer)(mx)
                    if batch_norm:
                        mx = layers.BatchNormalization()(mx)
                    mx = layers.Activation("relu")(mx)
                    if dropout > 0:
                        mx = layers.Dropout(dropout)(mx)
                m_out = layers.Dense(n_cls, activation="softmax")(mx)
                m = keras.Model(m_inp, m_out)
                m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
                m.fit(X_train, y_train, epochs=60, batch_size=batch_sz,
                      validation_split=0.2, verbose=0,
                      callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
                yp = np.argmax(m.predict(X_test, verbose=0), axis=1)
                d_accs.append(accuracy_score(y_test, yp))
                prog.progress((i + 1) / len(depths))
            prog.empty()
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(depths, d_accs, "o-", color=PALETTE[0])
            ax.axvline(n_layers, color="red", ls="--", label=f"Selected={n_layers}")
            ax.set_xlabel("Number of layers"); ax.set_ylabel("Test Accuracy")
            ax.set_title("DNN – Depth vs Accuracy"); ax.legend(); fig_to_st(fig)

    else:
        st.info("👆 Configure the DNN above and click **Train DNN**.")

    st.info("**Key concept:** BatchNorm normalises layer inputs at each mini-batch, "
            "accelerating training and allowing higher learning rates. Dropout randomly "
            "deactivates neurons during training, acting as an ensemble of thinner networks.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB: CNN
# ══════════════════════════════════════════════════════════════════════════════
def _tab_cnn():
    tf, keras, layers, _ = _tf()
    if tf is None:
        st.warning(_TF_UNAVAILABLE_MSG)
        return

    st.markdown("### Convolutional Neural Network — image recognition on MNIST")
    st.markdown(
        "Conv layers slide small filters over the input, detecting local patterns "
        "(edges, curves …) while sharing weights — massively more efficient than dense "
        "layers for image data."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        n_train   = st.select_slider("Training samples", [2000, 4000, 6000, 8000, 10000], value=6000, key="cnn_ntr")
        filters1  = st.select_slider("Conv block-1 filters", [8, 16, 32, 64], value=32, key="cnn_f1")
        filters2  = st.select_slider("Conv block-2 filters", [16, 32, 64, 128], value=64, key="cnn_f2")
    with c2:
        kernel_sz = st.selectbox("Kernel size", [3, 5], key="cnn_ks")
        dense_u   = st.select_slider("Dense units", [32, 64, 128, 256], value=128, key="cnn_du")
        dropout   = st.slider("Dropout (dense head)", 0.0, 0.5, 0.3, 0.05, key="cnn_dr")
    with c3:
        epochs    = st.slider("Epochs", 3, 20, 8, key="cnn_ep")
        batch_sz  = st.select_slider("Batch size", [32, 64, 128, 256], value=128, key="cnn_bs")
        use_gap   = st.checkbox("GlobalAvgPool instead of Flatten", False, key="cnn_gap")

    if st.button("🚀 Train CNN", key="cnn_train"):
        with st.spinner("Loading MNIST …"):
            (X_tr_all, y_tr_all), (X_te, y_te) = tf.keras.datasets.mnist.load_data()
            X_tr = (X_tr_all[:n_train][..., np.newaxis] / 255.0).astype(np.float32)
            y_tr = y_tr_all[:n_train]
            X_te = (X_te[:2000][..., np.newaxis] / 255.0).astype(np.float32)
            y_te = y_te[:2000]

        DIGITS = [str(i) for i in range(10)]
        keras.backend.clear_session()

        # Build model
        inp = keras.Input(shape=(28, 28, 1))
        x   = inp
        # Block 1
        x = layers.Conv2D(filters1, kernel_sz, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters1, kernel_sz, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.25)(x)
        # Block 2
        x = layers.Conv2D(filters2, kernel_sz, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters2, kernel_sz, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.25)(x)
        # Head
        x = layers.GlobalAveragePooling2D()(x) if use_gap else layers.Flatten()(x)
        x = layers.Dense(dense_u, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        out = layers.Dense(10, activation="softmax")(x)
        model = keras.Model(inp, out, name="CNN")
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])

        st.markdown("**Architecture**")
        st.markdown(_arch_md(model))

        bar = st.progress(0); status = st.empty()
        history = model.fit(
            X_tr, y_tr, epochs=epochs, batch_size=batch_sz,
            validation_split=0.15, verbose=0,
            callbacks=[_keras_progress_cb(epochs, bar, status)],
        )
        bar.empty(); status.empty()

        y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
        acc = accuracy_score(y_te, y_pred)

        show_metrics({"Test Accuracy":  f"{acc:.4f}",
                      "Val Accuracy":   f"{history.history['val_accuracy'][-1]:.4f}",
                      "Total Params":   f"{model.count_params():,}",
                      "Training size":  str(n_train)})

        col1, col2 = st.columns(2)
        with col1:
            _plot_history_st(history, "CNN (MNIST)")
        with col2:
            cm = confusion_matrix(y_te, y_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay(cm, display_labels=DIGITS).plot(ax=ax, colorbar=False, cmap="Oranges")
            ax.set_title("Confusion Matrix"); fig_to_st(fig)

        # Sample predictions
        st.subheader("Sample Predictions  (green = correct, red = wrong)")
        n_show = 12
        fig, axes = plt.subplots(2, n_show // 2, figsize=(n_show * 1.4, 5))
        axes = axes.flatten()
        wrong = np.where(y_te != y_pred)[0]
        right = np.where(y_te == y_pred)[0]
        idxs  = list(right[:n_show // 2]) + list(wrong[:n_show // 2])
        for ax, idx in zip(axes, idxs):
            ax.imshow(X_te[idx].squeeze(), cmap="gray")
            color = "green" if y_te[idx] == y_pred[idx] else "red"
            ax.set_title(f"T:{y_te[idx]} P:{y_pred[idx]}", color=color, fontsize=9)
            ax.axis("off")
        plt.suptitle("Top: Correct   Bottom: Mistakes", fontsize=11)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        # Conv filter visualization (weights of first conv layer)
        with st.expander("Conv Layer 1 – Learned Filters"):
            first_conv = [l for l in model.layers if isinstance(l, layers.Conv2D)][0]
            W = first_conv.get_weights()[0]          # (kH, kW, 1, n_filters)
            n_f = min(W.shape[-1], 16)
            fig, axes = plt.subplots(2, n_f // 2, figsize=(n_f * 0.9, 3.5))
            axes = axes.flatten()
            for i in range(n_f):
                filt = W[:, :, 0, i]
                filt = (filt - filt.min()) / (filt.max() - filt.min() + 1e-8)
                axes[i].imshow(filt, cmap="viridis"); axes[i].axis("off")
                axes[i].set_title(f"f{i}", fontsize=7)
            plt.suptitle("Learned Conv1 Filters", fontsize=10)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    else:
        st.info("👆 Set parameters and click **Train CNN** to start.")

    st.info("**Key concept:** Convolutions detect local patterns with shared weights — "
            "edge detectors, curve detectors, etc. emerge automatically from training. "
            "MaxPooling introduces spatial invariance; BatchNorm stabilises deep conv stacks.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB: RNN
# ══════════════════════════════════════════════════════════════════════════════
def _tab_rnn():
    tf, keras, layers, _ = _tf()
    if tf is None:
        st.warning(_TF_UNAVAILABLE_MSG)
        return

    st.markdown("### Recurrent Neural Network — time-series prediction")
    st.markdown(
        "An RNN maintains a **hidden state** passed from one time step to the next. "
        "This makes it suitable for sequential data, but vanilla RNNs struggle with "
        "long-range dependencies due to vanishing gradients."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        seq_len = st.slider("Sequence length", 10, 150, 50, key="rnn_sl")
        units   = st.select_slider("Hidden units", [16, 32, 64, 128], value=64, key="rnn_u")
        n_layers= st.slider("RNN layers", 1, 4, 1, key="rnn_nl")
    with c2:
        dropout = st.slider("Dropout", 0.0, 0.5, 0.0, 0.05, key="rnn_dr")
        epochs  = st.slider("Epochs", 10, 80, 30, key="rnn_ep")
        batch_sz= st.select_slider("Batch size", [32, 64, 128, 256], value=64, key="rnn_bs")
    with c3:
        noise    = st.slider("Signal noise", 0.01, 0.5, 0.08, 0.01, key="rnn_noise")
        vs_lstm  = st.checkbox("Compare with LSTM", True, key="rnn_vs")
        vs_mlp   = st.checkbox("Compare with MLP baseline", True, key="rnn_vsm")

    if st.button("🚀 Train RNN", key="rnn_train"):
        # Data
        rng = np.random.default_rng(42)
        t   = np.linspace(0, 6 * np.pi * 10, 3000)
        sig = (np.sin(0.3*t) + 0.5*np.sin(0.7*t) + 0.3*np.sin(1.3*t)
               + noise * rng.standard_normal(3000)).astype(np.float32)
        X, y = _make_sequences(sig, seq_len)
        split = int(0.8 * len(X))
        X_tr, X_te, y_tr, y_te = X[:split], X[split:], y[:split], y[split:]

        # Build RNN
        keras.backend.clear_session()
        inp = keras.Input(shape=(seq_len, 1))
        x   = inp
        for i in range(n_layers):
            x = layers.SimpleRNN(units, return_sequences=(i < n_layers - 1),
                                 dropout=dropout)(x)
        out = layers.Dense(1)(x)
        rnn_model = keras.Model(inp, out, name="SimpleRNN")
        rnn_model.compile(optimizer="adam", loss="mse")

        st.markdown("**Architecture**")
        st.markdown(_arch_md(rnn_model))

        bar = st.progress(0); status = st.empty()
        history = rnn_model.fit(
            X_tr, y_tr, epochs=epochs, batch_size=batch_sz,
            validation_split=0.15, verbose=0,
            callbacks=[_keras_progress_cb(epochs, bar, status)],
        )
        bar.empty(); status.empty()

        y_pred = rnn_model.predict(X_te, verbose=0).flatten()
        rmse_rnn = float(np.sqrt(np.mean((y_te - y_pred)**2)))

        metrics = {"RNN RMSE": f"{rmse_rnn:.4f}", "RNN Params": f"{rnn_model.count_params():,}"}

        # Optional comparisons
        rmse_lstm_val, rmse_mlp_val = None, None
        if vs_lstm:
            keras.backend.clear_session()
            li = keras.Input(shape=(seq_len, 1))
            lx = li
            for i in range(n_layers):
                lx = layers.LSTM(units, return_sequences=(i < n_layers - 1), dropout=dropout)(lx)
            lout = layers.Dense(1)(lx)
            lstm_m = keras.Model(li, lout, name="LSTM")
            lstm_m.compile(optimizer="adam", loss="mse")
            lstm_m.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_sz,
                       validation_split=0.15, verbose=0)
            lp = lstm_m.predict(X_te, verbose=0).flatten()
            rmse_lstm_val = float(np.sqrt(np.mean((y_te - lp)**2)))
            metrics["LSTM RMSE"] = f"{rmse_lstm_val:.4f}"

        if vs_mlp:
            keras.backend.clear_session()
            mi = keras.Input(shape=(seq_len, 1))
            mx = layers.Flatten()(mi)
            mx = layers.Dense(units, activation="relu")(mx)
            mx = layers.Dense(units // 2, activation="relu")(mx)
            mo = layers.Dense(1)(mx)
            mlp_m = keras.Model(mi, mo, name="MLP")
            mlp_m.compile(optimizer="adam", loss="mse")
            mlp_m.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_sz,
                      validation_split=0.15, verbose=0)
            mp = mlp_m.predict(X_te, verbose=0).flatten()
            rmse_mlp_val = float(np.sqrt(np.mean((y_te - mp)**2)))
            metrics["MLP RMSE"] = f"{rmse_mlp_val:.4f}"

        show_metrics(metrics)

        col1, col2 = st.columns(2)
        with col1:
            _plot_history_st(history, "RNN Training")
        with col2:
            _plot_ts_st(y_te, y_pred, "RNN – Prediction vs Actual")

        # Model comparison bar
        if vs_lstm or vs_mlp:
            with st.expander("Model RMSE Comparison"):
                cmp_names = ["RNN"]
                cmp_vals  = [rmse_rnn]
                if vs_lstm and rmse_lstm_val: cmp_names.append("LSTM");  cmp_vals.append(rmse_lstm_val)
                if vs_mlp  and rmse_mlp_val:  cmp_names.append("MLP");   cmp_vals.append(rmse_mlp_val)
                fig, ax = plt.subplots(figsize=(8, 3))
                bars = ax.bar(cmp_names, cmp_vals, color=[PALETTE[i % 10] for i in range(len(cmp_names))])
                ax.bar_label(bars, fmt="%.4f"); ax.set_ylabel("RMSE (lower = better)")
                ax.set_title("Model Comparison"); fig_to_st(fig)

    else:
        st.info("👆 Set parameters and click **Train RNN** to start.")

    st.info("**Key concept:** The hidden state *h_t = f(W·x_t + U·h_{t-1})* is the RNN's "
            "memory. Vanishing gradients make it hard to remember events more than ~20 steps "
            "back — the motivation for LSTM's gating mechanism.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB: LSTM
# ══════════════════════════════════════════════════════════════════════════════
def _tab_lstm():
    tf, keras, layers, _ = _tf()
    if tf is None:
        st.warning(_TF_UNAVAILABLE_MSG)
        return

    st.markdown("### LSTM — Long Short-Term Memory")
    st.markdown(
        "LSTMs add three **gates** (forget, input, output) and a **cell state** — "
        "a separate 'memory highway' that can carry information across many time steps "
        "without vanishing. GRU is a lighter variant with two gates."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        seq_len    = st.slider("Sequence length", 10, 150, 60, key="lstm_sl")
        units      = st.select_slider("LSTM units", [16, 32, 64, 128], value=64, key="lstm_u")
        n_layers   = st.slider("LSTM layers", 1, 4, 2, key="lstm_nl")
    with c2:
        dropout    = st.slider("Dropout", 0.0, 0.5, 0.1, 0.05, key="lstm_dr")
        bidir      = st.checkbox("Bidirectional", False, key="lstm_bidir")
        epochs     = st.slider("Epochs", 10, 80, 30, key="lstm_ep")
    with c3:
        batch_sz   = st.select_slider("Batch size", [32, 64, 128, 256], value=64, key="lstm_bs")
        noise      = st.slider("Signal noise", 0.01, 0.5, 0.08, 0.01, key="lstm_noise")
        cmp_rnn    = st.checkbox("Compare with RNN", True, key="lstm_crnn")
        cmp_gru    = st.checkbox("Compare with GRU", True, key="lstm_cgru")

    if st.button("🚀 Train LSTM", key="lstm_train"):
        rng = np.random.default_rng(42)
        t   = np.linspace(0, 6 * np.pi * 10, 3000)
        sig = (np.sin(0.3*t) + 0.5*np.sin(0.7*t) + 0.3*np.sin(1.3*t)
               + noise * rng.standard_normal(3000)).astype(np.float32)
        X, y = _make_sequences(sig, seq_len)
        split = int(0.8 * len(X))
        X_tr, X_te, y_tr, y_te = X[:split], X[split:], y[:split], y[split:]

        def _build(cell_cls, bidir_flag=False):
            keras.backend.clear_session()
            i = keras.Input(shape=(seq_len, 1))
            x = i
            for j in range(n_layers):
                rs = (j < n_layers - 1)
                cell = cell_cls(units, return_sequences=rs, dropout=dropout)
                x = layers.Bidirectional(cell)(x) if bidir_flag else cell(x)
            return keras.Model(i, layers.Dense(1)(x))

        # ── LSTM ──────────────────────────────────────────────────────────────
        lstm_m = _build(layers.LSTM, bidir)
        lstm_m.compile(optimizer="adam", loss="mse")

        st.markdown("**LSTM Architecture**")
        st.markdown(_arch_md(lstm_m))

        bar = st.progress(0); status = st.empty()
        history = lstm_m.fit(
            X_tr, y_tr, epochs=epochs, batch_size=batch_sz,
            validation_split=0.15, verbose=0,
            callbacks=[_keras_progress_cb(epochs, bar, status)],
        )
        bar.empty(); status.empty()

        y_pred = lstm_m.predict(X_te, verbose=0).flatten()
        rmse_lstm = float(np.sqrt(np.mean((y_te - y_pred)**2)))

        metrics = {"LSTM RMSE": f"{rmse_lstm:.4f}",
                   "LSTM Params": f"{lstm_m.count_params():,}",
                   "Bidirectional": str(bidir)}

        # ── Optional comparisons ───────────────────────────────────────────────
        preds_dict = {"LSTM": (y_pred, rmse_lstm)}

        if cmp_rnn:
            rnn_m = _build(layers.SimpleRNN)
            rnn_m.compile(optimizer="adam", loss="mse")
            rnn_m.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_sz,
                      validation_split=0.15, verbose=0)
            rp = rnn_m.predict(X_te, verbose=0).flatten()
            rmse_rnn = float(np.sqrt(np.mean((y_te - rp)**2)))
            metrics["RNN RMSE"] = f"{rmse_rnn:.4f}"
            preds_dict["RNN"] = (rp, rmse_rnn)

        if cmp_gru:
            gru_m = _build(layers.GRU)
            gru_m.compile(optimizer="adam", loss="mse")
            gru_m.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_sz,
                      validation_split=0.15, verbose=0)
            gp = gru_m.predict(X_te, verbose=0).flatten()
            rmse_gru = float(np.sqrt(np.mean((y_te - gp)**2)))
            metrics["GRU RMSE"] = f"{rmse_gru:.4f}"
            preds_dict["GRU"] = (gp, rmse_gru)

        show_metrics(metrics)

        col1, col2 = st.columns(2)
        with col1:
            _plot_history_st(history, "LSTM Training")
        with col2:
            _plot_ts_st(y_te, y_pred, "LSTM – Prediction vs Actual")

        # Overlay predictions
        if len(preds_dict) > 1:
            with st.expander("Overlay: LSTM vs RNN vs GRU predictions"):
                n_show = 200
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(y_te[:n_show], label="Actual", color="black", lw=1.5)
                for i, (name, (pred, rmse_v)) in enumerate(preds_dict.items()):
                    ax.plot(pred[:n_show], label=f"{name} (RMSE={rmse_v:.4f})",
                            color=PALETTE[i], lw=1.2, alpha=0.85)
                ax.set_title("Model Predictions Overlay"); ax.legend()
                fig_to_st(fig)

            # RMSE comparison bar
            fig, ax = plt.subplots(figsize=(8, 3))
            names_c = list(preds_dict.keys())
            rmse_c  = [v for _, v in preds_dict.values()]
            bars = ax.bar(names_c, rmse_c, color=[PALETTE[i % 10] for i in range(len(names_c))])
            ax.bar_label(bars, fmt="%.4f"); ax.set_ylabel("RMSE (lower = better)")
            ax.set_title("LSTM · GRU · RNN — RMSE Comparison"); fig_to_st(fig)

        # Gate explanation
        with st.expander("📖 LSTM Gate Mechanism"):
            st.markdown("""
| Gate | Formula | Purpose |
|------|---------|---------|
| **Forget** | *f_t = σ(W_f · [h_{t-1}, x_t] + b_f)* | Decides what to throw away from cell state |
| **Input**  | *i_t = σ(W_i · [h_{t-1}, x_t] + b_i)* | Decides which new info to store |
| **Update** | *C̃_t = tanh(W_C · [h_{t-1}, x_t])* | New candidate values for cell state |
| **Output** | *o_t = σ(W_o · [h_{t-1}, x_t] + b_o)* | Decides what to output as hidden state |

Cell state update: **C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t**

**GRU** merges forget + input into a single *update gate* and removes the separate cell state,
cutting parameters by ~25 % with minimal accuracy loss on most tasks.
""")

    else:
        st.info("👆 Configure and click **Train LSTM** to start.")

    st.info("**Key concept:** The LSTM cell state *C_t* acts as a 'conveyor belt' that "
            "can carry gradients across hundreds of time steps without vanishing. "
            "GRU achieves similar results with fewer parameters.")


# ══════════════════════════════════════════════════════════════════════════════
# NEURAL NETWORKS PAGE (tabs router)
# ══════════════════════════════════════════════════════════════════════════════
def page_neural_networks():
    st.title("🧠 Neural Networks")
    st.markdown(
        "Five architectures — from simple feedforward to sequential memory networks — "
        "each with interactive hyperparameters and live training."
    )

    tab_ann, tab_dnn, tab_cnn, tab_rnn, tab_lstm = st.tabs([
        "🔵 ANN", "🟣 DNN", "🖼️ CNN", "🔁 RNN", "⏳ LSTM"
    ])
    with tab_ann:  _tab_ann()
    with tab_dnn:  _tab_dnn()
    with tab_cnn:  _tab_cnn()
    with tab_rnn:  _tab_rnn()
    with tab_lstm: _tab_lstm()


# ══════════════════════════════════════════════════════════════════════════════
# NLP PAGE
# ══════════════════════════════════════════════════════════════════════════════
def page_nlp():
    import re, warnings
    warnings.filterwarnings("ignore")
    import nltk
    for _r in ["punkt","punkt_tab","stopwords","wordnet",
               "averaged_perceptron_tagger","averaged_perceptron_tagger_eng",
               "vader_lexicon","omw-1.4"]:
        nltk.download(_r, quiet=True)
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords as nltk_sw
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.naive_bayes import ComplementNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, classification_report

    STOP_W = set(nltk_sw.words("english"))
    STEMMER = PorterStemmer(); LEMMA = WordNetLemmatizer()

    def clean(text):
        return re.sub(r"\s+", " ", re.sub(r"[^a-z\s']", " ", text.lower())).strip()

    def preprocess(text, stem=False, lemma=True):
        tokens = [t for t in word_tokenize(clean(text)) if t.isalpha() and t not in STOP_W and len(t) > 2]
        if stem:   tokens = [STEMMER.stem(t) for t in tokens]
        elif lemma:tokens = [LEMMA.lemmatize(t) for t in tokens]
        return tokens

    st.title("📝 Natural Language Processing")
    st.markdown("Text preprocessing, classification, sentiment analysis, and topic modelling — all interactive.")

    tab_pre, tab_cls, tab_sent, tab_topic, tab_tfidf = st.tabs([
        "🔤 Preprocessing", "🏷️ Classification", "💭 Sentiment", "📚 Topics", "📊 TF-IDF Explorer"
    ])

    # ── Preprocessing tab ─────────────────────────────────────────────────────
    with tab_pre:
        st.subheader("Text Preprocessing Pipeline")
        default = "Machine learning algorithms automatically discover patterns from large datasets without explicit rules."
        text_in = st.text_area("Enter text", default, height=90, key="nlp_text")
        c1, c2 = st.columns(2)
        rm_stop = c1.checkbox("Remove stopwords", True, key="nlp_rs")
        use_lem = c1.checkbox("Lemmatise",        True, key="nlp_lem")
        use_stem= c2.checkbox("Stem",             False, key="nlp_stem")
        n_gram  = c2.selectbox("N-gram range", ["Unigram (1,1)","Bigram (1,2)","Trigram (1,3)"], key="nlp_ng")
        ng_map  = {"Unigram (1,1)":(1,1),"Bigram (1,2)":(1,2),"Trigram (1,3)":(1,3)}
        ng      = ng_map[n_gram]

        tokens_raw   = [t for t in word_tokenize(clean(text_in)) if t.isalpha()]
        tokens_stop  = [t for t in tokens_raw if t not in STOP_W] if rm_stop else tokens_raw
        tokens_final = ([STEMMER.stem(t) for t in tokens_stop] if use_stem
                        else ([LEMMA.lemmatize(t) for t in tokens_stop] if use_lem else tokens_stop))

        col1, col2, col3 = st.columns(3)
        col1.metric("Original tokens", len(tokens_raw))
        col2.metric("After stopword removal", len(tokens_stop))
        col3.metric("After stem/lemma", len(tokens_final))

        st.markdown("**Processed tokens:**")
        st.code(" · ".join(tokens_final))

        if tokens_final:
            from collections import Counter
            freq = Counter(tokens_final).most_common(15)
            words, counts = zip(*freq)
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.bar(words, counts, color=PALETTE[0])
            ax.set_title("Word Frequency"); ax.set_ylabel("Count")
            plt.xticks(rotation=40, ha="right"); fig_to_st(fig)

        # N-gram TF-IDF on sample corpus
        corpus = [
            "Machine learning uses data to build predictive models.",
            "Deep learning neural networks process complex patterns.",
            "Natural language processing handles text and speech data.",
            "Computer vision algorithms interpret images and video.",
            "Reinforcement learning agents learn through trial and reward.",
        ] + [text_in]
        tv = TfidfVectorizer(ngram_range=ng)
        X_corp = tv.fit_transform(corpus)
        feat_names = tv.get_feature_names_out()
        mean_scores = X_corp.mean(axis=0).A1
        top_idx = mean_scores.argsort()[-15:][::-1]
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.barh(feat_names[top_idx][::-1], mean_scores[top_idx][::-1], color=PALETTE[1])
        ax.set_title(f"Top TF-IDF Terms  (n-gram={ng})"); ax.set_xlabel("Mean TF-IDF")
        fig_to_st(fig)

    # ── Classification tab ────────────────────────────────────────────────────
    with tab_cls:
        st.subheader("Text Classification — 20 Newsgroups")
        ALL_CATS = ["sci.med","sci.space","rec.sport.baseball",
                    "talk.politics.misc","comp.graphics","rec.motorcycles",
                    "sci.electronics","talk.religion.misc"]
        selected_cats = st.multiselect("Categories", ALL_CATS,
                                       default=ALL_CATS[:4], key="nlp_cats")
        c1, c2, c3 = st.columns(3)
        vec_type = c1.selectbox("Vectorizer", ["TF-IDF","Count","Binary"], key="nlp_vec")
        clf_name = c2.selectbox("Classifier",
                                ["Complement NB","Logistic Regression","Linear SVM"], key="nlp_clf")
        max_feat = c3.select_slider("Max features",
                                    [1000,5000,10000,20000,50000], value=10000, key="nlp_mf")
        use_ngram= c1.checkbox("Add bigrams", True, key="nlp_bg")
        rm_hdr   = c2.checkbox("Remove headers/footers", True, key="nlp_hdr")

        if len(selected_cats) < 2:
            st.warning("Select at least 2 categories."); return

        if st.button("🚀 Train Classifier", key="nlp_train"):
            with st.spinner("Fetching data and training …"):
                remove = ("headers","footers","quotes") if rm_hdr else ()
                data = fetch_20newsgroups(subset="all", categories=selected_cats, remove=remove)
                from sklearn.model_selection import train_test_split as tts
                X_tr, X_te, y_tr, y_te = tts(data.data, data.target, test_size=0.25, random_state=42)

                ng_r = (1,2) if use_ngram else (1,1)
                if vec_type == "TF-IDF":
                    vec = TfidfVectorizer(max_features=max_feat, sublinear_tf=True,
                                         stop_words="english", ngram_range=ng_r)
                elif vec_type == "Count":
                    vec = CountVectorizer(max_features=max_feat, stop_words="english", ngram_range=ng_r)
                else:
                    vec = CountVectorizer(max_features=max_feat, stop_words="english",
                                         binary=True, ngram_range=ng_r)

                Xtr = vec.fit_transform(X_tr); Xte = vec.transform(X_te)

                clf_map = {
                    "Complement NB":       ComplementNB(alpha=0.1),
                    "Logistic Regression": LogisticRegression(max_iter=500, C=5, random_state=42),
                    "Linear SVM":          LinearSVC(C=1.0, max_iter=2000),
                }
                clf = clf_map[clf_name]; clf.fit(Xtr, y_tr)
                y_pred = clf.predict(Xte)
                acc = accuracy_score(y_te, y_pred)

            show_metrics({"Accuracy": f"{acc:.4f}", "Categories": str(len(selected_cats)),
                          "Train size": str(len(X_tr)), "Test size": str(len(X_te))})

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Confusion Matrix")
                cm_v = confusion_matrix(y_te, y_pred)
                fig, ax = plt.subplots(figsize=(6, 5))
                ConfusionMatrixDisplay(cm_v, display_labels=selected_cats).plot(
                    ax=ax, colorbar=False, cmap="Blues", xticks_rotation=30)
                fig_to_st(fig)
            with col2:
                st.subheader("Top TF-IDF Terms")
                Xall = vec.transform(data.data)
                mn   = Xall.mean(axis=0).A1
                ti   = mn.argsort()[-15:][::-1]
                fn   = vec.get_feature_names_out()
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.barh(fn[ti][::-1], mn[ti][::-1], color=PALETTE[0])
                ax.set_title("Top 15 TF-IDF Terms"); ax.set_xlabel("Mean score")
                fig_to_st(fig)

            with st.expander("Full Classification Report"):
                st.code(classification_report(y_te, y_pred, target_names=selected_cats))
        else:
            st.info("👆 Choose categories and classifier, then click **Train Classifier**.")

    # ── Sentiment tab ─────────────────────────────────────────────────────────
    with tab_sent:
        st.subheader("Sentiment Analysis — VADER Lexicon")
        sia = SentimentIntensityAnalyzer()
        PRESETS = {
            "😊 Positive":  "This product is absolutely fantastic! Best purchase I've ever made. Highly recommend to everyone!",
            "😞 Negative":  "Terrible quality. Broke after one day. Complete waste of money. Very disappointed with the purchase.",
            "😐 Neutral":   "The package arrived on Thursday. It contains the standard components as described in the manual.",
            "🤔 Mixed":     "The food was amazing but the service was incredibly slow and the price was quite high for the portions.",
            "Custom":       "",
        }
        preset = st.selectbox("Choose a preset or enter custom text", list(PRESETS.keys()), key="nlp_sp")
        default_sent = PRESETS[preset]
        sentiment_text = st.text_area("Text to analyse", default_sent, height=80, key="nlp_stext")

        if sentiment_text.strip():
            scores = sia.polarity_scores(sentiment_text)
            compound = scores["compound"]
            label    = "Positive 😊" if compound > 0.05 else ("Negative 😞" if compound < -0.05 else "Neutral 😐")

            show_metrics({"Compound": f"{compound:+.3f}", "Positive": f"{scores['pos']:.3f}",
                          "Neutral": f"{scores['neu']:.3f}", "Negative": f"{scores['neg']:.3f}",
                          "Verdict": label})

            # Gauge-like bar
            fig, ax = plt.subplots(figsize=(10, 2))
            color = "#27ae60" if compound > 0.05 else ("#e74c3c" if compound < -0.05 else "#95a5a6")
            ax.barh(["Compound"], [compound + 1], color="#ecf0f1", height=0.5)
            ax.barh(["Compound"], [compound + 1 - 1], left=[1], color=color, height=0.5)
            ax.axvline(1, color="gray", lw=1, ls="--")
            ax.set_xlim(0, 2); ax.set_xticks([0, 0.5, 1, 1.5, 2])
            ax.set_xticklabels(["-1 (neg)", "-0.5", "0 (neutral)", "+0.5", "+1 (pos)"])
            ax.set_title(f"Sentiment: {label}  (compound = {compound:+.3f})")
            fig_to_st(fig)

            # Token-level sentiment
            st.subheader("Token-level Sentiment Scores")
            tokens = word_tokenize(sentiment_text)[:30]
            tok_scores = [(t, sia.polarity_scores(t)["compound"]) for t in tokens if t.isalpha()]
            if tok_scores:
                t_names, t_vals = zip(*tok_scores)
                t_cols = ["#27ae60" if v > 0.1 else ("#e74c3c" if v < -0.1 else "#95a5a6") for v in t_vals]
                fig, ax = plt.subplots(figsize=(12, 3))
                ax.bar(t_names, t_vals, color=t_cols)
                ax.axhline(0, color="black", lw=0.8)
                ax.set_title("Per-token Sentiment (VADER)")
                plt.xticks(rotation=45, ha="right"); fig_to_st(fig)

            # Batch analysis of all presets
            with st.expander("Compare All Preset Texts"):
                rows = []
                for name, txt in PRESETS.items():
                    if not txt: continue
                    s = sia.polarity_scores(txt)
                    rows.append({"Text": name, "Compound": round(s["compound"],3),
                                 "Positive": round(s["pos"],3), "Neutral": round(s["neu"],3),
                                 "Negative": round(s["neg"],3)})
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ── Topic Modelling tab ───────────────────────────────────────────────────
    with tab_topic:
        st.subheader("Topic Modelling — Latent Dirichlet Allocation (LDA)")
        c1, c2, c3 = st.columns(3)
        n_topics  = c1.slider("Number of topics", 2, 10, 5, key="nlp_nt")
        top_words = c2.slider("Words per topic",   5, 20, 12, key="nlp_tw")
        max_iter  = c3.slider("LDA iterations",    5, 30,  15, key="nlp_iter")
        cats_lda  = st.multiselect("Categories",
                                   ["sci.med","sci.space","rec.sport.baseball",
                                    "talk.politics.misc","comp.graphics","rec.motorcycles"],
                                   default=["sci.med","sci.space","rec.sport.baseball",
                                            "talk.politics.misc","comp.graphics"],
                                   key="nlp_lda_cats")

        if st.button("🚀 Fit LDA", key="nlp_lda"):
            with st.spinner("Fitting LDA …"):
                data = fetch_20newsgroups(subset="train", categories=cats_lda,
                                         remove=("headers","footers","quotes"))
                cv_lda = CountVectorizer(max_features=8000, stop_words="english", max_df=0.95, min_df=3)
                X_lda  = cv_lda.fit_transform(data.data)
                lda    = LatentDirichletAllocation(n_components=n_topics, max_iter=max_iter,
                                                   learning_method="online", random_state=42)
                lda.fit(X_lda)
                fn = cv_lda.get_feature_names_out()

            show_metrics({"Topics": str(n_topics), "Documents": str(X_lda.shape[0]),
                          "Vocabulary": str(X_lda.shape[1]),
                          "Perplexity": f"{lda.perplexity(X_lda):.1f}"})

            ncols = min(n_topics, 5)
            nrows = (n_topics + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 4))
            axes = np.array(axes).flatten()
            for i, (topic, ax) in enumerate(zip(lda.components_, axes)):
                idx  = topic.argsort()[-top_words:][::-1]
                wds  = fn[idx]; vals = topic[idx] / topic.sum()
                ax.barh(wds[::-1], vals[::-1], color=PALETTE[i % 10])
                ax.set_title(f"Topic {i+1}", fontweight="bold", fontsize=9)
                ax.tick_params(labelsize=7)
            for ax in axes[n_topics:]: ax.axis("off")
            plt.suptitle("LDA Topics — Top Words", fontsize=12, fontweight="bold")
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

            # Topic distribution over corpus
            doc_topics = lda.transform(X_lda)
            dom_topic  = doc_topics.argmax(axis=1)
            fig, ax = plt.subplots(figsize=(8, 3))
            counts = np.bincount(dom_topic, minlength=n_topics)
            ax.bar([f"T{i+1}" for i in range(n_topics)], counts,
                   color=[PALETTE[i % 10] for i in range(n_topics)])
            ax.set_title("Document Count per Dominant Topic"); ax.set_ylabel("Documents")
            fig_to_st(fig)
        else:
            st.info("👆 Select categories and click **Fit LDA** to discover topics.")

    # ── TF-IDF Explorer tab ───────────────────────────────────────────────────
    with tab_tfidf:
        st.subheader("TF-IDF Matrix Explorer")
        st.markdown("Enter 3–6 short documents and explore their TF-IDF representation.")
        docs_default = [
            "deep learning transforms computer vision and image recognition tasks",
            "natural language processing enables machines to understand human text",
            "reinforcement learning agents optimise decisions through environment rewards",
            "generative adversarial networks create realistic synthetic images",
            "transformer architecture revolutionised natural language processing models",
        ]
        docs = []
        for i, d in enumerate(docs_default):
            docs.append(st.text_input(f"Document {i+1}", d, key=f"nlp_doc{i}"))
        docs = [d for d in docs if d.strip()]

        if len(docs) >= 2:
            ng_exp = st.selectbox("N-gram", ["(1,1)","(1,2)"], key="nlp_ng_exp")
            ng_v   = (1,1) if ng_exp == "(1,1)" else (1,2)
            tv_exp = TfidfVectorizer(ngram_range=ng_v, stop_words="english")
            X_exp  = tv_exp.fit_transform(docs).toarray()
            fn_exp = tv_exp.get_feature_names_out()

            # Heatmap
            fig, ax = plt.subplots(figsize=(min(20, len(fn_exp)*0.5+2), len(docs)*0.8+1))
            sns.heatmap(X_exp, xticklabels=fn_exp, yticklabels=[f"Doc {i+1}" for i in range(len(docs))],
                        cmap="YlOrRd", ax=ax, linewidths=0.3)
            ax.set_title("TF-IDF Matrix Heatmap", fontweight="bold")
            plt.xticks(rotation=60, ha="right", fontsize=8)
            fig_to_st(fig)

            # Cosine similarity between documents
            from sklearn.metrics.pairwise import cosine_similarity as cos_sim
            sim = cos_sim(X_exp)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(sim, annot=True, fmt=".2f",
                        xticklabels=[f"D{i+1}" for i in range(len(docs))],
                        yticklabels=[f"D{i+1}" for i in range(len(docs))],
                        cmap="Blues", ax=ax, vmin=0, vmax=1)
            ax.set_title("Document Cosine Similarity")
            fig_to_st(fig)

    st.info("**Key concept:** TF-IDF rewards terms that are frequent in one document but rare "
            "across the corpus — much better than raw counts for distinguishing topics.")


# ══════════════════════════════════════════════════════════════════════════════
# COMPUTER VISION PAGE
# ══════════════════════════════════════════════════════════════════════════════
def page_computer_vision():
    import warnings; warnings.filterwarnings("ignore")
    from skimage.feature import hog as sk_hog
    from skimage.filters import sobel as sk_sobel
    from skimage.color import rgb2gray as sk_rgb2gray
    from skimage.transform import resize as sk_rz
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score
    from scipy.ndimage import rotate as nd_rotate

    CIFAR_NAMES = ["airplane","automobile","bird","cat","deer",
                   "dog","frog","horse","ship","truck"]

    @st.cache_data
    def _load_cifar(n_tr, n_te):
        _, keras2, _, _ = _tf()
        (X_tr, y_tr),(X_te, y_te) = keras2.datasets.cifar10.load_data()
        return (X_tr[:n_tr].astype(np.float32)/255., y_tr[:n_tr].flatten(),
                X_te[:n_te].astype(np.float32)/255., y_te[:n_te].flatten())

    st.title("🖼️ Computer Vision")
    st.markdown("Augmentation, classical features, CNN training, and transfer learning on CIFAR-10.")

    tab_aug, tab_feat, tab_cnn, tab_tl = st.tabs([
        "🎨 Augmentation", "🔍 Feature Extraction", "🧠 CNN Training", "⚡ Transfer Learning"
    ])

    # ── Augmentation tab ──────────────────────────────────────────────────────
    with tab_aug:
        st.subheader("Image Augmentation Effects")
        c1, c2 = st.columns(2)
        cls_sel   = c1.selectbox("Image class", CIFAR_NAMES, key="cv_cls")
        n_samples = c2.slider("Samples to show", 1, 8, 4, key="cv_ns")

        with st.spinner("Loading CIFAR-10 …"):
            X_tr, y_tr, X_te, y_te = _load_cifar(8000, 2000)

        cls_idx = CIFAR_NAMES.index(cls_sel)
        idxs    = np.where(y_tr == cls_idx)[0][:n_samples]
        imgs    = X_tr[idxs]

        rot_deg   = st.slider("Rotation (°)",    0, 45, 20, key="cv_rot")
        bright    = st.slider("Brightness mult", 0.5, 2.0, 1.4, 0.1, key="cv_br")
        flip_h    = st.checkbox("Horizontal flip", True, key="cv_fh")
        flip_v    = st.checkbox("Vertical flip",   False, key="cv_fv")

        def aug_img(img):
            variants = {"Original": img}
            if flip_h:  variants["H-Flip"]  = img[:, ::-1, :]
            if flip_v:  variants["V-Flip"]  = img[::-1, :, :]
            if rot_deg: variants[f"Rot {rot_deg}°"] = np.clip(nd_rotate(img, rot_deg, reshape=False),0,1)
            variants[f"Bright×{bright}"] = np.clip(img * bright, 0, 1)
            variants["Grayscale"]  = np.stack([sk_rgb2gray(img)]*3, axis=-1)
            return variants

        for img in imgs:
            variants = aug_img(img)
            fig, axes = plt.subplots(1, len(variants), figsize=(len(variants)*2, 2.2))
            for ax, (name, v) in zip(axes, variants.items()):
                ax.imshow(np.clip(v, 0, 1)); ax.axis("off"); ax.set_title(name, fontsize=7)
            fig_to_st(fig)

    # ── Feature Extraction tab ────────────────────────────────────────────────
    with tab_feat:
        st.subheader("Classical Feature Extraction")
        with st.spinner("Loading CIFAR-10 …"):
            X_tr, y_tr, X_te, y_te = _load_cifar(8000, 2000)

        c1, c2 = st.columns(2)
        feat_type  = c1.selectbox("Feature type", ["HOG","Color Histogram","HOG + Color"], key="cv_ft")
        svm_c      = c2.select_slider("SVM C", [0.1, 1, 5, 10, 50], value=10, key="cv_c")
        show_vis   = st.checkbox("Show HOG / edge visualisation", True, key="cv_vis")

        # Visualisation
        if show_vis:
            vis_cls = st.selectbox("Visualise class", CIFAR_NAMES, index=1, key="cv_vcls")
            idx_vis = np.where(y_tr == CIFAR_NAMES.index(vis_cls))[0][0]
            img_vis = X_tr[idx_vis]
            gray_vis = sk_rgb2gray(img_vis)
            fd_vis, hog_vis = sk_hog(img_vis, orientations=8, pixels_per_cell=(4,4),
                                     cells_per_block=(2,2), channel_axis=-1, visualize=True)
            edges_vis = sk_sobel(gray_vis)

            fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
            axes[0].imshow(img_vis); axes[0].set_title("Original"); axes[0].axis("off")
            axes[1].imshow(gray_vis, cmap="gray"); axes[1].set_title("Grayscale"); axes[1].axis("off")
            axes[2].imshow(hog_vis, cmap="gray"); axes[2].set_title("HOG"); axes[2].axis("off")
            axes[3].imshow(edges_vis, cmap="gray"); axes[3].set_title("Sobel Edges"); axes[3].axis("off")
            plt.suptitle(f"Feature Visualisation — {vis_cls}", fontweight="bold")
            fig_to_st(fig)

        if st.button("🚀 Extract Features & Train SVM", key="cv_svm"):
            def hog_feats(X):
                return np.array([sk_hog(img, orientations=8, pixels_per_cell=(4,4),
                                        cells_per_block=(2,2), channel_axis=-1) for img in X])
            def col_feats(X, bins=32):
                return np.array([np.concatenate([np.histogram(X[i,:,:,c], bins=bins,
                                                  range=(0,1))[0] for c in range(3)])
                                 for i in range(len(X))], dtype=np.float32)

            with st.spinner("Extracting HOG …"):
                H_tr = hog_feats(X_tr); H_te = hog_feats(X_te)
            with st.spinner("Extracting color histograms …"):
                C_tr = col_feats(X_tr); C_te = col_feats(X_te)

            feat_map = {
                "HOG":             (H_tr, H_te),
                "Color Histogram": (C_tr, C_te),
                "HOG + Color":     (np.hstack([H_tr, C_tr]), np.hstack([H_te, C_te])),
            }
            Ftr, Fte = feat_map[feat_type]

            with st.spinner("Training SVM …"):
                clf = Pipeline([("sc", StandardScaler()),
                                ("svm", SVC(kernel="rbf", C=svm_c, gamma="scale"))])
                clf.fit(Ftr, y_tr)
                acc = accuracy_score(y_te, clf.predict(Fte))

            show_metrics({"Test Accuracy": f"{acc:.4f}", "Feature dim": str(Ftr.shape[1]),
                          "Classifier": f"SVM (C={svm_c})"})

            # Per-class accuracy
            y_pred_svm = clf.predict(Fte)
            per_cls = [(CIFAR_NAMES[c], accuracy_score(y_te[y_te==c], y_pred_svm[y_te==c]))
                       for c in range(10)]
            fig, ax = plt.subplots(figsize=(10, 3))
            names_c, accs_c = zip(*per_cls)
            bars = ax.bar(names_c, accs_c, color=[PALETTE[i%10] for i in range(10)])
            ax.bar_label(bars, fmt="%.3f", fontsize=7)
            ax.set_title(f"{feat_type} + SVM — Per-class Accuracy"); ax.set_ylim(0, 1.15)
            plt.xticks(rotation=30, ha="right"); fig_to_st(fig)

    # ── CNN Training tab ──────────────────────────────────────────────────────
    with tab_cnn:
        if not _TF_AVAILABLE:
            st.warning(_TF_UNAVAILABLE_MSG)
        else:
            tf2, keras2, layers2, _ = _tf()
            st.subheader("CNN Training on CIFAR-10")
            c1, c2, c3 = st.columns(3)
            n_tr    = c1.select_slider("Training samples",[2000,4000,6000,8000,10000],value=6000,key="cv_ntr")
            filters1= c2.select_slider("Conv block-1 filters",[16,32,64],value=32,key="cv_cf1")
            filters2= c3.select_slider("Conv block-2 filters",[32,64,128],value=64,key="cv_cf2")
            dropout = c1.slider("Dropout",0.1,0.5,0.4,0.05,key="cv_drop")
            epochs  = c2.slider("Epochs",3,20,10,key="cv_ep")
            aug_on  = c3.checkbox("Data augmentation",True,key="cv_aug")

            if st.button("🚀 Train CNN", key="cv_cnn"):
                with st.spinner("Loading data …"):
                    X_tr, y_tr, X_te, y_te = _load_cifar(n_tr, 2000)

                keras2.backend.clear_session()
                inp = keras2.Input(shape=(32,32,3))
                x   = inp
                if aug_on:
                    x = layers2.RandomFlip("horizontal")(x)
                    x = layers2.RandomRotation(0.1)(x)
                    x = layers2.RandomZoom(0.1)(x)
                for f in (filters1, filters2):
                    x = layers2.Conv2D(f, 3, padding="same", activation="relu")(x)
                    x = layers2.BatchNormalization()(x)
                    x = layers2.Conv2D(f, 3, padding="same", activation="relu")(x)
                    x = layers2.MaxPooling2D(2)(x)
                    x = layers2.Dropout(0.25)(x)
                x   = layers2.GlobalAveragePooling2D()(x)
                x   = layers2.Dense(256, activation="relu")(x)
                x   = layers2.Dropout(dropout)(x)
                out = layers2.Dense(10, activation="softmax")(x)
                model = keras2.Model(inp, out, name="CNN_CIFAR")
                model.compile(optimizer=keras2.optimizers.Adam(1e-3),
                              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

                st.markdown("**Architecture**"); st.markdown(_arch_md(model))

                bar = st.progress(0); status = st.empty()
                es  = keras2.callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=0)
                rlr = keras2.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=0)
                history = model.fit(X_tr, y_tr, epochs=epochs, batch_size=128,
                                    validation_split=0.15, verbose=0,
                                    callbacks=[_keras_progress_cb(epochs, bar, status), es, rlr])
                bar.empty(); status.empty()

                y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
                acc    = accuracy_score(y_te, y_pred)
                show_metrics({"Test Accuracy": f"{acc:.4f}",
                              "Val Accuracy":  f"{history.history['val_accuracy'][-1]:.4f}",
                              "Params":        f"{model.count_params():,}"})

                col1, col2 = st.columns(2)
                with col1:
                    _plot_history_st(history, "CNN CIFAR-10")
                with col2:
                    cm_v = confusion_matrix(y_te, y_pred)
                    fig, ax = plt.subplots(figsize=(6,5))
                    ConfusionMatrixDisplay(cm_v, display_labels=CIFAR_NAMES).plot(
                        ax=ax, colorbar=False, cmap="Oranges", xticks_rotation=45)
                    fig_to_st(fig)

                # Sample predictions
                st.subheader("Sample Predictions")
                wrong = np.where(y_te != y_pred)[0]
                right = np.where(y_te == y_pred)[0]
                idxs  = list(right[:6]) + list(wrong[:6])
                fig, axes = plt.subplots(2, 6, figsize=(14, 5))
                axes = axes.flatten()
                for ax, idx in zip(axes, idxs):
                    ax.imshow(X_te[idx]); ax.axis("off")
                    color = "green" if y_te[idx]==y_pred[idx] else "red"
                    ax.set_title(f"T:{CIFAR_NAMES[y_te[idx]]}\nP:{CIFAR_NAMES[y_pred[idx]]}",
                                 color=color, fontsize=6)
                plt.suptitle("Top row: Correct  |  Bottom row: Mistakes")
                fig_to_st(fig)
                keras2.backend.clear_session()

            else:
                st.info("👆 Set parameters and click **Train CNN**.")

    # ── Transfer Learning tab ─────────────────────────────────────────────────
    with tab_tl:
        if not _TF_AVAILABLE:
            st.warning(_TF_UNAVAILABLE_MSG)
        else:
            tf2, keras2, layers2, _ = _tf()
            st.subheader("Transfer Learning — MobileNetV2")
            st.markdown(
                "Use a MobileNetV2 backbone pre-trained on ImageNet. "
                "Freeze the backbone, add a small classification head, and train only the head."
            )
            c1, c2 = st.columns(2)
            n_tl   = c1.select_slider("Samples", [1000,2000,3000,4000], value=2000, key="cv_ntl")
            tl_ep  = c2.slider("Head epochs", 3, 15, 6, key="cv_tlep")
            unfreeze= c1.checkbox("Unfreeze top 20 layers (fine-tune)", False, key="cv_uf")

            if st.button("🚀 Run Transfer Learning", key="cv_tl"):
                with st.spinner("Loading CIFAR-10 …"):
                    X_tr, y_tr, X_te, y_te = _load_cifar(n_tl, 1000)
                target = (96, 96)
                with st.spinner("Resizing images to 96×96 …"):
                    X_tr_r = np.array([sk_rz(img, target, anti_aliasing=True) for img in X_tr], dtype=np.float32)
                    X_te_r = np.array([sk_rz(img, target, anti_aliasing=True) for img in X_te], dtype=np.float32)

                keras2.backend.clear_session()
                with st.spinner("Loading MobileNetV2 (ImageNet weights) …"):
                    base = keras2.applications.MobileNetV2(
                        input_shape=(*target,3), include_top=False, weights="imagenet", pooling="avg")
                    base.trainable = False
                    if unfreeze:
                        for layer in base.layers[-20:]:
                            layer.trainable = True

                inp   = keras2.Input(shape=(*target,3))
                x     = keras2.applications.mobilenet_v2.preprocess_input(inp * 255)
                x     = base(x, training=False)
                x     = layers2.Dense(128, activation="relu")(x)
                x     = layers2.Dropout(0.3)(x)
                out   = layers2.Dense(10, activation="softmax")(x)
                model = keras2.Model(inp, out, name="TL_MobileNetV2")
                model.compile(optimizer=keras2.optimizers.Adam(1e-3 if not unfreeze else 5e-5),
                              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

                show_metrics({"Trainable params": f"{sum(np.prod(w.shape) for w in model.trainable_weights):,}",
                              "Frozen params":    f"{sum(np.prod(w.shape) for w in model.non_trainable_weights):,}",
                              "Fine-tune":        str(unfreeze)})

                bar = st.progress(0); status = st.empty()
                history = model.fit(X_tr_r, y_tr, epochs=tl_ep, batch_size=64,
                                    validation_split=0.15, verbose=0,
                                    callbacks=[_keras_progress_cb(tl_ep, bar, status)])
                bar.empty(); status.empty()

                y_pred = np.argmax(model.predict(X_te_r, verbose=0), axis=1)
                acc = accuracy_score(y_te, y_pred)
                show_metrics({"Test Accuracy":  f"{acc:.4f}",
                              "Val Accuracy":   f"{history.history['val_accuracy'][-1]:.4f}"})

                _plot_history_st(history, "Transfer Learning (MobileNetV2)")
                keras2.backend.clear_session()
            else:
                st.info("👆 Click **Run Transfer Learning** to start.")

    st.info("**Key concept:** Transfer learning adapts a model trained on millions of images "
            "to your task — even 1–2k labelled images often outperform a CNN trained from scratch.")


# ══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATIONS PAGE
# ══════════════════════════════════════════════════════════════════════════════
def page_recommendations():
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    from sklearn.decomposition import TruncatedSVD

    st.title("🎬 Recommendation Systems")
    st.markdown("User-Based CF, Item-Based CF, Matrix Factorisation, and Content-Based Filtering "
                "on a synthetic movie-rating dataset.")

    GENRES = ["Action","Comedy","Drama","Sci-Fi","Romance","Thriller","Horror","Animation"]

    @st.cache_data
    def _make_data(n_users, n_items, sparsity, seed):
        rng = np.random.default_rng(seed)
        U = rng.standard_normal((n_users, 8))
        V = rng.standard_normal((n_items, 8))
        base = U @ V.T
        base = (base - base.min()) / (base.max() - base.min()) * 4 + 1
        base += 0.3 * rng.standard_normal(base.shape)
        base = np.clip(np.round(base), 1, 5)
        mask = rng.random((n_users, n_items)) > sparsity
        ratings = np.where(mask, base, np.nan)
        n_genres = len(GENRES)
        feat = np.zeros((n_items, n_genres))
        prim = rng.integers(0, n_genres, n_items)
        sec  = (prim + rng.integers(1, n_genres, n_items)) % n_genres
        for i,(p,s) in enumerate(zip(prim,sec)):
            feat[i,p]=1.; feat[i,s]=0.5
        items_df = pd.DataFrame(feat, columns=GENRES,
                                index=[f"Movie_{i:03d}" for i in range(n_items)])
        users_l  = [f"User_{i:03d}" for i in range(n_users)]
        R = pd.DataFrame(ratings, index=users_l, columns=items_df.index)
        return R, items_df

    # Sidebar-style controls at the top
    with st.expander("⚙️ Dataset Settings", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        n_users   = c1.slider("Users",   50, 300, 120, key="rec_nu")
        n_items   = c2.slider("Items",   20, 150,  60, key="rec_ni")
        sparsity  = c3.slider("Sparsity (fraction missing)", 0.3, 0.85, 0.65, 0.05, key="rec_sp")
        seed      = c4.slider("Random seed", 1, 99, 42, key="rec_seed")

    R, items_df = _make_data(n_users, n_items, sparsity, seed)
    rated        = int(R.notna().sum().sum())
    total        = n_users * n_items
    density      = rated / total

    tab_exp, tab_ub, tab_ib, tab_svd, tab_cb, tab_eval = st.tabs([
        "📊 Dataset", "👥 User-Based CF", "🎬 Item-Based CF",
        "🔢 Matrix Factorisation", "🏷️ Content-Based", "📈 Evaluation"
    ])

    # ── Dataset tab ───────────────────────────────────────────────────────────
    with tab_exp:
        show_metrics({"Users": str(n_users), "Items": str(n_items),
                      "Rated pairs": str(rated),
                      "Density": f"{density*100:.1f}%",
                      "Avg rating": f"{R.stack().mean():.2f}"})

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Rating Matrix (first 25×25)")
            vis = R.fillna(0).values[:25, :25]
            fig, ax = plt.subplots(figsize=(7, 5))
            im = ax.imshow(vis, cmap="YlOrRd", aspect="auto", vmin=0, vmax=5)
            plt.colorbar(im, ax=ax, label="Rating  (0 = unrated)")
            ax.set_xlabel("Items"); ax.set_ylabel("Users")
            ax.set_title("Rating Heatmap"); fig_to_st(fig)

        with col2:
            st.subheader("Rating Distribution")
            vals = R.stack().values
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.hist(vals, bins=[0.5,1.5,2.5,3.5,4.5,5.5],
                    color=PALETTE[0], edgecolor="white", rwidth=0.8)
            ax.set_xlabel("Rating"); ax.set_ylabel("Count")
            ax.set_title("Distribution of Ratings"); fig_to_st(fig)

        # Genre distribution
        dominant_genre = items_df.idxmax(axis=1).value_counts()
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.bar(dominant_genre.index, dominant_genre.values,
               color=[PALETTE[i%10] for i in range(len(dominant_genre))])
        ax.set_title("Item Genre Distribution"); ax.set_ylabel("Count")
        plt.xticks(rotation=30, ha="right"); fig_to_st(fig)

    # ── User-Based CF tab ─────────────────────────────────────────────────────
    with tab_ub:
        st.subheader("User-Based Collaborative Filtering")
        st.markdown("*'Users who rated items similarly to you also liked …'*")
        c1, c2 = st.columns(2)
        k_nb   = c1.slider("k neighbours", 3, 30, 15, key="rec_ub_k")
        sel_u  = c2.selectbox("Select user", R.index.tolist(), key="rec_sel_u")
        top_n_ub = st.slider("Top-N recommendations", 5, 20, 10, key="rec_ub_n")

        # Compute mean-centred matrix & similarity
        mean_u  = R.mean(axis=1)
        R_c     = R.subtract(mean_u, axis=0).fillna(0)
        U_sim   = pd.DataFrame(cos_sim(R_c.values), index=R.index, columns=R.index)

        # Recommend for selected user
        def _ub_predict(user, item):
            sims  = U_sim[user].drop(user)
            rated = ~R[item].isna()
            peers = sims[rated].nlargest(k_nb)
            if len(peers) == 0: return mean_u[user]
            num = (peers * (R.loc[peers.index, item] - mean_u[peers.index])).sum()
            return np.clip(mean_u[user] + num / (peers.abs().sum() + 1e-9), 1, 5)

        unrated  = R.columns[R.loc[sel_u].isna()]
        rec_scores = {item: _ub_predict(sel_u, item) for item in unrated}
        top_recs   = sorted(rec_scores.items(), key=lambda x: x[1], reverse=True)[:top_n_ub]

        col1, col2 = st.columns(2)
        with col1:
            # User similarity heatmap (top 15 users)
            top_sim = U_sim[sel_u].drop(sel_u).nlargest(15)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(top_sim.index[::-1], top_sim.values[::-1], color=PALETTE[0])
            ax.set_title(f"Top 15 Similar Users to {sel_u}"); ax.set_xlabel("Cosine similarity")
            fig_to_st(fig)

        with col2:
            st.subheader(f"Top-{top_n_ub} Recommendations")
            recs_df = pd.DataFrame(top_recs, columns=["Movie", "Predicted Rating"])
            recs_df["Predicted Rating"] = recs_df["Predicted Rating"].round(2)
            recs_df["Genre"] = recs_df["Movie"].apply(
                lambda m: items_df.loc[m].nlargest(1).index[0])
            st.dataframe(recs_df, use_container_width=True)

        # Recommendation bar chart
        items_r, scores_r = zip(*top_recs)
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.barh(items_r[::-1], scores_r[::-1], color=PALETTE[1])
        ax.set_xlabel("Predicted Rating"); ax.set_title(f"User-CF Recommendations for {sel_u}")
        ax.axvline(3.5, color="red", ls="--", label="Good threshold"); ax.legend()
        fig_to_st(fig)

    # ── Item-Based CF tab ─────────────────────────────────────────────────────
    with tab_ib:
        st.subheader("Item-Based Collaborative Filtering")
        st.markdown("*'Because you liked X, you might enjoy Y …'*")
        c1, c2 = st.columns(2)
        k_items = c1.slider("k similar items", 5, 30, 20, key="rec_ib_k")
        sel_itm = c2.selectbox("Anchor item (explore similar)", R.columns.tolist(), key="rec_sel_itm")
        sel_u_ib= st.selectbox("User for personalised recs", R.index.tolist(), key="rec_ib_u")
        top_n_ib= st.slider("Top-N", 5, 20, 10, key="rec_ib_n")

        # Adjusted cosine item similarity
        R_adj   = R.subtract(R.mean(axis=1), axis=0).fillna(0)
        I_sim   = pd.DataFrame(cos_sim(R_adj.T.values), index=R.columns, columns=R.columns)

        sim_items = I_sim[sel_itm].drop(sel_itm).nlargest(10)
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(sim_items.index[::-1], sim_items.values[::-1], color=PALETTE[2])
            ax.set_title(f"Items Most Similar to {sel_itm}")
            ax.set_xlabel("Adjusted Cosine Similarity"); fig_to_st(fig)

        # Personalised recs for user
        def _ib_predict(user, item):
            user_rated = R.loc[user].dropna()
            if item in user_rated: return user_rated[item]
            common = I_sim[item][user_rated.index].nlargest(k_items)
            if len(common) == 0: return R[item].mean()
            return np.clip((common * user_rated[common.index]).sum() /
                           (common.abs().sum() + 1e-9), 1, 5)

        unrated_ib  = R.columns[R.loc[sel_u_ib].isna()]
        recs_ib     = sorted({item: _ib_predict(sel_u_ib, item)
                               for item in unrated_ib}.items(),
                              key=lambda x: x[1], reverse=True)[:top_n_ib]
        with col2:
            st.subheader(f"Top-{top_n_ib} for {sel_u_ib}")
            ib_df = pd.DataFrame(recs_ib, columns=["Movie","Predicted Rating"])
            ib_df["Predicted Rating"] = ib_df["Predicted Rating"].round(2)
            ib_df["Genre"] = ib_df["Movie"].apply(lambda m: items_df.loc[m].nlargest(1).index[0])
            st.dataframe(ib_df, use_container_width=True)

    # ── Matrix Factorisation tab ───────────────────────────────────────────────
    with tab_svd:
        st.subheader("Matrix Factorisation — Truncated SVD")
        st.markdown("Decompose the rating matrix into latent user and item factors.")
        c1, c2 = st.columns(2)
        n_comp  = c1.slider("SVD rank (n_components)", 2, min(40, min(n_users,n_items)-1), 15, key="rec_svd_k")
        svd_u   = c2.selectbox("User for recs", R.index.tolist(), key="rec_svd_u")

        mean_all = float(R.stack().mean())
        R_fill   = R.fillna(mean_all).values
        svd      = TruncatedSVD(n_components=n_comp, random_state=42)
        U_svd    = svd.fit_transform(R_fill)
        R_hat    = pd.DataFrame(np.clip(U_svd @ svd.components_, 1, 5),
                                index=R.index, columns=R.columns)

        ev = svd.explained_variance_ratio_.sum()
        show_metrics({"SVD rank": str(n_comp),
                      "Explained variance": f"{ev*100:.1f}%",
                      "Reconstruction error": f"{np.sqrt(((R_fill - R_hat.values)**2).mean()):.4f}"})

        col1, col2 = st.columns(2)
        with col1:
            # Explained variance by component
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(range(1, n_comp+1), svd.explained_variance_ratio_ * 100, color=PALETTE[0])
            ax.set_xlabel("Component"); ax.set_ylabel("Variance explained (%)")
            ax.set_title("SVD — Variance per Component"); fig_to_st(fig)

        with col2:
            unrated_svd = R.columns[R.loc[svd_u].isna()]
            svd_scores  = {item: float(R_hat.loc[svd_u, item]) for item in unrated_svd}
            top_svd     = sorted(svd_scores.items(), key=lambda x: x[1], reverse=True)[:10]
            st.subheader(f"Top-10 SVD Recs for {svd_u}")
            svd_df = pd.DataFrame(top_svd, columns=["Movie","Predicted Rating"])
            svd_df["Predicted Rating"] = svd_df["Predicted Rating"].round(2)
            svd_df["Genre"] = svd_df["Movie"].apply(lambda m: items_df.loc[m].nlargest(1).index[0])
            st.dataframe(svd_df, use_container_width=True)

        # Rank sweep
        with st.expander("Reconstruction Error vs SVD Rank"):
            ranks = list(range(2, min(41, min(n_users, n_items)-1), 4))
            errors = []
            for r in ranks:
                s = TruncatedSVD(n_components=r, random_state=42)
                Uh = s.fit_transform(R_fill)
                errors.append(np.sqrt(((R_fill - Uh @ s.components_)**2).mean()))
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.plot(ranks, errors, "o-", color=PALETTE[3])
            ax.axvline(n_comp, color="red", ls="--", label=f"Selected k={n_comp}")
            ax.set_xlabel("SVD rank k"); ax.set_ylabel("RMSE (on all cells)")
            ax.set_title("Reconstruction Error vs Rank"); ax.legend()
            fig_to_st(fig)

    # ── Content-Based tab ─────────────────────────────────────────────────────
    with tab_cb:
        st.subheader("Content-Based Filtering")
        st.markdown("Recommend items whose *features* are most similar to what the user has liked.")
        c1, c2 = st.columns(2)
        sel_u_cb = c1.selectbox("User", R.index.tolist(), key="rec_cb_u")
        top_n_cb = c2.slider("Top-N", 5, 20, 10, key="rec_cb_n")

        # Item feature similarity
        feat_sim = pd.DataFrame(cos_sim(items_df.values), index=items_df.index, columns=items_df.index)

        def cb_recommend(user):
            rated_u = R.loc[user].dropna()
            if len(rated_u) == 0: return []
            # User profile = weighted average of item features
            profile = (items_df.loc[rated_u.index].T * rated_u.values).T.mean()
            unrated = R.columns[R.loc[user].isna()]
            scores  = {}
            for item in unrated:
                iv   = items_df.loc[item].values
                num  = (profile.values * iv).sum()
                den  = (np.linalg.norm(profile) * np.linalg.norm(iv)) + 1e-9
                scores[item] = np.clip(num/den * 4 + 1, 1, 5)
            return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n_cb]

        user_profile = (items_df.loc[R.loc[sel_u_cb].dropna().index].T
                        * R.loc[sel_u_cb].dropna().values).T.mean()
        cb_recs = cb_recommend(sel_u_cb)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"User Profile — {sel_u_cb}")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(GENRES, user_profile.values, color=[PALETTE[i%10] for i in range(len(GENRES))])
            ax.set_title("Genre Preference Profile"); ax.set_ylabel("Weight")
            plt.xticks(rotation=30, ha="right"); fig_to_st(fig)

        with col2:
            st.subheader(f"Top-{top_n_cb} Content-Based Recs")
            cb_df = pd.DataFrame(cb_recs, columns=["Movie","Score"])
            cb_df["Score"] = cb_df["Score"].round(3)
            cb_df["Genre"] = cb_df["Movie"].apply(lambda m: items_df.loc[m].nlargest(1).index[0])
            st.dataframe(cb_df, use_container_width=True)

    # ── Evaluation tab ─────────────────────────────────────────────────────────
    with tab_eval:
        st.subheader("Model Comparison — Evaluation Metrics")
        c1, c2 = st.columns(2)
        k_eval     = c1.slider("@K for Precision/Recall", 5, 20, 10, key="rec_k")
        threshold  = c2.slider("'Good' rating threshold", 3.0, 4.5, 3.5, 0.5, key="rec_thr")

        if st.button("🚀 Evaluate All Models", key="rec_eval"):
            # Train/test split
            rng = np.random.default_rng(42)
            known = [(u, i) for u in R.index for i in R.columns if not pd.isna(R.loc[u, i])]
            rng.shuffle(known)
            te_pairs = known[:int(len(known)*0.2)]
            R_tr = R.copy()
            R_te = {(u,i): R.loc[u,i] for u,i in te_pairs}
            for u, i in te_pairs: R_tr.loc[u,i] = np.nan

            # User-CF
            mean_u2 = R_tr.mean(axis=1)
            R_c2    = R_tr.subtract(mean_u2, axis=0).fillna(0)
            U_sim2  = pd.DataFrame(cos_sim(R_c2.values), index=R.index, columns=R.index)
            def ub_pred2(u, item):
                sims = U_sim2[u].drop(u); rated = ~R_tr[item].isna()
                peers = sims[rated].nlargest(15)
                if len(peers)==0: return mean_u2[u]
                num = (peers*(R_tr.loc[peers.index,item]-mean_u2[peers.index])).sum()
                return np.clip(mean_u2[u]+num/(peers.abs().sum()+1e-9),1,5)

            # Item-CF
            R_adj2  = R_tr.subtract(R_tr.mean(axis=1),axis=0).fillna(0)
            I_sim2  = pd.DataFrame(cos_sim(R_adj2.T.values), index=R.columns, columns=R.columns)
            def ib_pred2(u, item):
                ur = R_tr.loc[u].dropna()
                if item in ur: return ur[item]
                common = I_sim2[item][ur.index].nlargest(20)
                if len(common)==0: return R_tr[item].mean()
                return np.clip((common*ur[common.index]).sum()/(common.abs().sum()+1e-9),1,5)

            # SVD
            mean_a2 = float(R_tr.stack().mean())
            R_f2    = R_tr.fillna(mean_a2).values
            svd2    = TruncatedSVD(n_components=20, random_state=42)
            Uh2     = svd2.fit_transform(R_f2)
            R_hat2  = pd.DataFrame(np.clip(Uh2@svd2.components_,1,5), index=R.index, columns=R.columns)
            def svd_pred2(u, item): return float(R_hat2.loc[u, item])

            # Content-based
            def cb_pred2(u, item):
                ur = R_tr.loc[u].dropna()
                if len(ur)==0: return 3.0
                prof = (items_df.loc[ur.index].T * ur.values).T.mean()
                iv   = items_df.loc[item].values
                sim  = (prof.values * iv).sum() / (np.linalg.norm(prof)*np.linalg.norm(iv)+1e-9)
                return np.clip(sim*4+1,1,5)

            models_eval = {"User-CF": ub_pred2, "Item-CF": ib_pred2,
                           "SVD": svd_pred2, "Content-Based": cb_pred2}

            def rmse_fn(pred_fn):
                errs = [(R_te[(u,i)] - pred_fn(u,i))**2 for (u,i) in R_te]
                return float(np.sqrt(np.mean(errs)))

            def prec_rec(top_n_fn):
                precs, recs_l = [], []
                for user in R.index:
                    relevant = set(R.columns[(R.loc[user] >= threshold) & R.loc[user].notna()])
                    if not relevant: continue
                    top = [item for item,_ in top_n_fn(user, n=k_eval)]
                    hits = len(set(top) & relevant)
                    precs.append(hits / k_eval)
                    recs_l.append(hits / len(relevant))
                return float(np.mean(precs)), float(np.mean(recs_l))

            def ub_topn(u, n=10):
                unr = R_tr.columns[R_tr.loc[u].isna()]
                return sorted({i: ub_pred2(u,i) for i in unr}.items(), key=lambda x:-x[1])[:n]
            def ib_topn(u, n=10):
                unr = R_tr.columns[R_tr.loc[u].isna()]
                return sorted({i: ib_pred2(u,i) for i in unr}.items(), key=lambda x:-x[1])[:n]
            def svd_topn(u, n=10):
                unr = R.columns[R_tr.loc[u].isna()]
                return sorted({i: svd_pred2(u,i) for i in unr}.items(), key=lambda x:-x[1])[:n]
            def cb_topn(u, n=10):
                unr = R_tr.columns[R_tr.loc[u].isna()]
                return sorted({i: cb_pred2(u,i) for i in unr}.items(), key=lambda x:-x[1])[:n]

            topn_fns = {"User-CF": ub_topn, "Item-CF": ib_topn,
                        "SVD": svd_topn, "Content-Based": cb_topn}

            rows = []
            with st.spinner("Evaluating …"):
                for name, pred_fn in models_eval.items():
                    r   = rmse_fn(pred_fn)
                    p,rc= prec_rec(topn_fns[name])
                    rows.append({"Model": name, "RMSE": round(r,4),
                                 f"P@{k_eval}": round(p,4), f"R@{k_eval}": round(rc,4)})

            df_eval = pd.DataFrame(rows)
            st.dataframe(df_eval, use_container_width=True)

            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            for ax, col, color, title in zip(
                axes,
                ["RMSE", f"P@{k_eval}", f"R@{k_eval}"],
                [PALETTE[0], PALETTE[1], PALETTE[2]],
                ["RMSE (↓)", f"Precision@{k_eval} (↑)", f"Recall@{k_eval} (↑)"],
            ):
                vals = df_eval[col].values
                bars = ax.bar(df_eval["Model"], vals, color=[PALETTE[i%10] for i in range(4)])
                ax.bar_label(bars, fmt="%.4f", padding=3)
                ax.set_title(title, fontweight="bold")
                ax.set_xticklabels(df_eval["Model"], rotation=20, ha="right")
                ax.set_ylim(0, max(vals)*1.2+0.05)
            plt.suptitle("Recommendation Models — Evaluation", fontweight="bold")
            fig_to_st(fig)
        else:
            st.info("👆 Click **Evaluate All Models** to compare all methods.")

    st.info("**Key concept:** No single method wins — User-CF needs many users, Item-CF is "
            "more stable, SVD captures latent taste profiles, Content-Based solves item cold-start.")


PAGES = {
    "🏠  Home":                    page_home,
    "📈  Linear Regression":       page_linear_regression,
    "🌳  Decision Tree":           page_decision_tree,
    "🔵  K-Means Clustering":      page_kmeans,
    "🌲  Bagging & Random Forest": page_random_forest,
    "⚡  Boosting":                page_boosting,
    "🎯  Ensemble Methods":        page_ensemble,
    "🧠  Neural Networks":         page_neural_networks,
    "📝  NLP":                     page_nlp,
    "🖼️  Computer Vision":         page_computer_vision,
    "🎬  Recommendations":         page_recommendations,
}

PAGES[page]()
