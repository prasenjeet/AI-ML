"""Plotting helpers shared across all ML concept modules."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import plot_tree


# ── colour palette ──────────────────────────────────────────────────────────
PALETTE = sns.color_palette("tab10")
plt.rcParams.update({"figure.dpi": 100, "font.size": 11})


def save_or_show(fig, path=None):
    plt.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved plot → {path}")
    else:
        plt.show()
    plt.close(fig)


# ── Decision Tree ────────────────────────────────────────────────────────────
def plot_decision_tree(clf, feature_names, class_names, title="Decision Tree", path=None):
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(
        clf, feature_names=feature_names, class_names=class_names,
        filled=True, rounded=True, fontsize=9, ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    save_or_show(fig, path)


def plot_feature_importance(importances, feature_names, title="Feature Importances", path=None):
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(importances)), importances[indices], color=PALETTE[0])
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha="right")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Importance")
    save_or_show(fig, path)


# ── K-Means ──────────────────────────────────────────────────────────────────
def plot_clusters(X, labels, centroids=None, title="K-Means Clustering", path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    unique = np.unique(labels)
    for k in unique:
        mask = labels == k
        ax.scatter(X[mask, 0], X[mask, 1], s=40, alpha=0.7,
                   color=PALETTE[k % len(PALETTE)], label=f"Cluster {k}")
    if centroids is not None:
        ax.scatter(centroids[:, 0], centroids[:, 1], s=200, c="black",
                   marker="X", zorder=5, label="Centroids")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    save_or_show(fig, path)


def plot_elbow(ks, inertias, path=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, inertias, "o-", color=PALETTE[1])
    ax.set_xlabel("Number of clusters  k")
    ax.set_ylabel("Inertia (within-cluster SSE)")
    ax.set_title("Elbow Method – Optimal k", fontweight="bold")
    save_or_show(fig, path)


# ── Linear Regression ────────────────────────────────────────────────────────
def plot_regression_fit(y_true, y_pred, title="Predicted vs Actual", path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # scatter
    axes[0].scatter(y_true, y_pred, alpha=0.6, color=PALETTE[2], s=30)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[0].plot([lo, hi], [lo, hi], "r--", lw=2, label="Perfect fit")
    axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
    axes[0].set_title(title, fontweight="bold"); axes[0].legend()

    # residuals
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.6, color=PALETTE[3], s=30)
    axes[1].axhline(0, color="red", lw=2, ls="--")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Residual")
    axes[1].set_title("Residual Plot", fontweight="bold")

    save_or_show(fig, path)


def plot_learning_curve(train_sizes, train_scores, val_scores, title="Learning Curve", path=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(train_sizes, train_scores.mean(1) - train_scores.std(1),
                    train_scores.mean(1) + train_scores.std(1), alpha=0.2, color=PALETTE[0])
    ax.fill_between(train_sizes, val_scores.mean(1) - val_scores.std(1),
                    val_scores.mean(1) + val_scores.std(1), alpha=0.2, color=PALETTE[1])
    ax.plot(train_sizes, train_scores.mean(1), "o-", color=PALETTE[0], label="Training score")
    ax.plot(train_sizes, val_scores.mean(1), "o-", color=PALETTE[1], label="Validation score")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Training examples"); ax.set_ylabel("Score")
    ax.legend()
    save_or_show(fig, path)


# ── Ensemble / Boosting comparison ───────────────────────────────────────────
def plot_model_comparison(names, scores, metric="Accuracy", title="Model Comparison", path=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]
    bars = ax.barh(names, scores, color=colors)
    ax.bar_label(bars, fmt="%.4f", padding=4)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(metric)
    ax.set_title(title, fontweight="bold")
    ax.invert_yaxis()
    save_or_show(fig, path)


def plot_confusion(y_true, y_pred, class_names, title="Confusion Matrix", path=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontweight="bold")
    save_or_show(fig, path)


def plot_boosting_stages(n_estimators_range, train_errors, test_errors,
                         title="Boosting – Error vs Estimators", path=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(n_estimators_range, train_errors, label="Train error", color=PALETTE[0])
    ax.plot(n_estimators_range, test_errors, label="Test error", color=PALETTE[1])
    ax.set_xlabel("Number of estimators"); ax.set_ylabel("Error rate")
    ax.set_title(title, fontweight="bold"); ax.legend()
    save_or_show(fig, path)
