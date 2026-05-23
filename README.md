# Machine Learning Concepts — Sample Project

A self-contained Python project that demonstrates six fundamental machine learning techniques using **scikit-learn**. Available in two modes:

- **Interactive UI** — Streamlit web app with live inputs, dynamic plots, and metric cards
- **CLI runner** — terminal script with printed results and saved PNG charts

---

## Table of Contents

- [Concepts Covered](#concepts-covered)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Interactive UI](#interactive-ui-apppy)
- [CLI Runner](#cli-runner-mainpy)
- [Concept Details](#concept-details)
  - [Linear Regression](#1-linear-regression)
  - [Decision Tree](#2-decision-tree)
  - [K-Means Clustering](#3-k-means-clustering)
  - [Bagging & Random Forest](#4-bagging--random-forest)
  - [Boosting](#5-boosting)
  - [Ensemble Methods](#6-ensemble-methods)
- [Generated Plots](#generated-plots)
- [Dependencies](#dependencies)

---

## Concepts Covered

| # | Concept | Algorithms | Dataset |
|---|---------|------------|---------|
| 1 | Linear Regression | OLS, Ridge, Lasso, Polynomial | Synthetic regression |
| 2 | Decision Tree | Classification & Regression tree, CCP pruning | Iris / Wine / Synthetic |
| 3 | K-Means Clustering | K-Means, Agglomerative | Synthetic blobs |
| 4 | Bagging & Random Forest | BaggingClassifier, RandomForest, ExtraTrees | Wine / Iris / Synthetic |
| 5 | Boosting | AdaBoost, GradientBoosting, HistGradientBoosting | Wine / Iris / Synthetic |
| 6 | Ensemble Methods | Hard/Soft/Weighted Voting, Stacking | Wine / Iris / Synthetic |

---

## Project Structure

```
AI-ML/
├── app.py                           # ★ Streamlit interactive UI
├── main.py                          # CLI runner
├── requirements.txt
├── .gitignore
├── plots/                           # Generated PNG charts (21 files)
└── ml_concepts/
    ├── utils/
    │   ├── data_generator.py        # Dataset creation & splitting
    │   └── visualizer.py           # Shared plotting functions
    └── models/
        ├── linear_regression.py
        ├── decision_tree.py
        ├── kmeans_clustering.py
        ├── bagging_random_forest.py
        ├── boosting.py
        └── ensemble.py
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Option A — Interactive UI (recommended)
streamlit run app.py

# Option B — CLI, run all concepts
python main.py --save-plots
```

---

## Interactive UI (`app.py`)

The Streamlit app provides a fully interactive experience — adjust any input and results update instantly without rerunning anything manually.

```bash
streamlit run app.py
```

### Layout

```
┌─────────────────────────┬─────────────────────────────────────────┐
│  Sidebar                │  Main area                              │
│  ───────────────        │  ─────────────────────────────          │
│  🤖 ML Concepts         │  Metric cards (R², Accuracy, etc.)      │
│                         │                                         │
│  Page navigation        │  Side-by-side interactive plots         │
│  (7 pages)              │                                         │
│                         │  Expandable analysis sections           │
│  ───────────────        │                                         │
│  Concept-specific       │  Key concept takeaway                   │
│  sliders & dropdowns    │                                         │
└─────────────────────────┴─────────────────────────────────────────┘
```

### Pages & Controls

| Page | Sidebar Inputs | Live Outputs |
|------|----------------|--------------|
| **🏠 Home** | — | Overview cards for all 6 concepts |
| **📈 Linear Regression** | Variant (OLS/Ridge/Lasso/Poly), alpha, degree, samples, noise, test split | Predicted vs Actual, Residual plot, Learning curve, Coefficients table, All-variant comparison |
| **🌳 Decision Tree** | Dataset, task (classify/regress), criterion, max_depth, min_samples_split, CCP alpha | Tree diagram, Feature importance, Confusion matrix, Depth sweep chart, CCP alpha sweep |
| **🔵 K-Means Clustering** | Samples, true clusters, spread, k, k sweep range, init, n_init, agglomerative linkage | Elbow curve, Silhouette bar chart, Cluster scatter, Ground truth scatter, Agglomerative comparison, Sweep table |
| **🌲 Bagging & Random Forest** | Dataset, n_estimators, max_depth, max_features, OOB toggle, comparator checkboxes (DT/Bagging/ExtraTrees) | Feature importances, Confusion matrix, Model comparison, n_estimators sweep, max_features sweep |
| **⚡ Boosting** | Dataset, AdaBoost (n, lr, depth) and GBM (n, lr, depth, subsample) controls | Stagewise error curves (AdaBoost + GBM), Feature importances, Confusion matrix, Algorithm comparison, LR sweep |
| **🎯 Ensemble Methods** | Dataset, per-learner checkboxes (DT/RF/GBM/KNN/NB/SVM), per-strategy checkboxes (Hard/Soft/Weighted/Stack-LR/Stack-RF) | Individual accuracy metrics, Ensemble accuracy metrics, Full comparison bar chart, Confusion matrix, Diversity heatmap, Results table |

---

## CLI Runner (`main.py`)

Run concepts from the terminal with printed metrics and optional PNG output.

```
python main.py [--topic TOPIC] [--save-plots]

Options:
  --topic       Run a single concept (default: all)
                Choices: lr | dt | km | rf | boost | ens
  --save-plots  Save charts as PNGs to ./plots/
```

**Examples:**

```bash
python main.py                          # All 6 concepts
python main.py --topic lr               # Linear Regression only
python main.py --topic dt               # Decision Tree only
python main.py --topic km               # K-Means Clustering only
python main.py --topic rf               # Bagging & Random Forest only
python main.py --topic boost            # Boosting only
python main.py --topic ens              # Ensemble Methods only
python main.py --save-plots             # Save all 21 charts to ./plots/
python main.py --topic rf --save-plots  # RF only, save plots
```

---

## Concept Details

### 1. Linear Regression
**CLI module:** `ml_concepts/models/linear_regression.py`

Explores the linear family of regression models on a synthetic dataset.

| Variant | Key parameter | Purpose |
|---------|--------------|---------|
| OLS (Ordinary Least Squares) | — | Baseline; minimises sum of squared residuals |
| Ridge | `alpha` (L2 penalty) | Shrinks all coefficients; handles multicollinearity |
| Lasso | `alpha` (L1 penalty) | Zeros out weak features; built-in variable selection |
| Polynomial (deg 2–5) | `degree` | Captures non-linear relationships via feature expansion |

**Interactive UI inputs:** samples, features, noise, test split %, variant selector, alpha slider, polynomial degree slider

**Output:** MSE · RMSE · MAE · R² · Predicted vs Actual · Residual plot · Learning curve · Coefficients table · All-variant comparison chart

**Key concept:** Regularisation (Ridge/Lasso) trades a small increase in bias for a large reduction in variance, improving generalisation on unseen data.

---

### 2. Decision Tree
**CLI module:** `ml_concepts/models/decision_tree.py`

Demonstrates classification and regression trees with pruning strategies.

**What's explored:**
- **Criterion comparison** — Gini impurity vs Information Gain (Entropy)
- **Depth sweep** — `max_depth` from 1–15, showing underfitting → overfitting transition
- **Cost-complexity pruning (CCP)** — `ccp_alpha` parameter to post-prune the tree
- **Regression tree** — predicts continuous values on a synthetic dataset

**Interactive UI inputs:** dataset (Iris/Wine/Synthetic), task (classify/regress), criterion, max_depth, min_samples_split, CCP alpha

**Output:** Tree visualisation · Feature importance · Confusion matrix · Depth sweep chart · CCP alpha sweep chart

**Key concept:** A single unpruned tree memorises training data. Controlling `max_depth` or `ccp_alpha` is essential to generalise beyond the training set.

---

### 3. K-Means Clustering
**CLI module:** `ml_concepts/models/kmeans_clustering.py`

Unsupervised clustering on synthetic 2D blob data.

**What's explored:**
- **Elbow method** — inertia (within-cluster SSE) vs k
- **Silhouette analysis** — finds optimal k by maximising average silhouette score
- **Initialisation strategies** — `k-means++` vs `random` (convergence quality)
- **Agglomerative hierarchical clustering** — ward, complete, average, single linkage

**Metrics reported:** Inertia · Silhouette Score · Davies-Bouldin Index

**Interactive UI inputs:** samples, true clusters, cluster spread, k, sweep range, init strategy, n_init, agglomerative linkage

**Output:** Elbow curve · Silhouette bar chart · Cluster scatter with centroids · Ground truth comparison · Agglomerative side-by-side · Sweep results table

**Key concept:** K-Means is sensitive to scale and initialisation. Always standardise features and prefer `k-means++` init. Use silhouette score alongside the elbow to confirm k.

---

### 4. Bagging & Random Forest
**CLI module:** `ml_concepts/models/bagging_random_forest.py`

Demonstrates how bagging reduces variance by aggregating many decorrelated trees.

| Model | Core idea |
|-------|-----------|
| Single Decision Tree | High-variance baseline |
| BaggingClassifier | Bootstrap samples + aggregate predictions |
| RandomForestClassifier | Bagging + random feature subset at each split |
| ExtraTreesClassifier | RF + randomised split thresholds |

**What's explored:**
- **OOB (Out-of-Bag) score** — free cross-validation estimate without a held-out set
- **n_estimators sweep** — 1 to 500 trees, showing diminishing returns
- **max_features sweep** — `sqrt`, `log2`, `None`, `0.3`, `0.6`

**Interactive UI inputs:** dataset, n_estimators, max_depth, max_features, OOB toggle, comparator checkboxes

**Output:** Feature importance chart · Confusion matrix · Model comparison bar chart · n_estimators sweep · max_features sweep

**Key concept:** Bagging reduces variance by averaging independent high-variance models. Random Forest decorrelates those models further by limiting which features each tree can consider.

---

### 5. Boosting
**CLI module:** `ml_concepts/models/boosting.py`

Sequential ensemble learning — each new model corrects the errors of the previous ones.

| Algorithm | Mechanism |
|-----------|-----------|
| **AdaBoost** | Re-weights misclassified samples; uses decision stumps |
| **GradientBoosting** | Fits residuals (negative gradient) at each stage |
| **HistGradientBoosting** | Bins continuous features → 10–100× faster for large data |

**What's explored:**
- **Stagewise error curves** — train vs test error at every boosting round
- **Learning-rate sweep** — `lr` from 0.001 to 1.0 (shrinkage vs speed trade-off)
- **Feature importance** from the GBM model

**Interactive UI inputs:** dataset, AdaBoost controls (n, lr, depth), GBM controls (n, lr, depth, subsample)

**Output:** AdaBoost stagewise error · GBM stagewise error · Feature importances · Best-model confusion matrix · Algorithm comparison · Learning rate sweep table

**Key concept:** Boosting reduces bias by sequentially fitting weak learners to the errors. A lower learning rate with more estimators consistently generalises better but requires more computation.

---

### 6. Ensemble Methods
**CLI module:** `ml_concepts/models/ensemble.py`

Combines six diverse base learners into higher-level ensembles.

| Strategy | How it combines predictions |
|----------|-----------------------------|
| **Hard Voting** | Majority class vote |
| **Soft Voting** | Average predicted probabilities |
| **Weighted Soft Voting** | Weighted average (weights ∝ individual accuracy) |
| **Stacking (LR meta)** | Logistic Regression learns from base predictions |
| **Stacking (RF meta)** | Random Forest learns from base predictions |

**What's explored:**
- **Diversity analysis** — pairwise prediction agreement heatmap between all base learners
- **CV evaluation** of stacking configurations

**Interactive UI inputs:** dataset, per-learner checkboxes (DT/RF/GBM/KNN/NB/SVM), per-strategy checkboxes

**Output:** Individual accuracy metrics · Ensemble accuracy metrics · Full comparison bar chart · Best-ensemble confusion matrix · Diversity heatmap · Ranked results table

**Key concept:** Ensemble power comes from **diversity**, not just individual accuracy. Low pairwise agreement means errors don't overlap — ideal for combining.

---

## Generated Plots

Running `python main.py --save-plots` produces 21 charts in `./plots/`:

| Filename | Description |
|----------|-------------|
| `lr_fit.png` | OLS predicted vs actual + residual plot |
| `lr_learning_curve.png` | Training size vs R² score |
| `lr_comparison.png` | R² comparison across regression variants |
| `dt_tree.png` | Full decision tree visualisation |
| `dt_feature_importance.png` | Feature importance bar chart |
| `dt_confusion.png` | Decision tree confusion matrix |
| `dt_depth_comparison.png` | Test accuracy vs max_depth |
| `km_elbow.png` | Elbow curve (inertia vs k) |
| `km_clusters.png` | Final K-Means cluster scatter with centroids |
| `km_agglomerative.png` | Agglomerative clustering scatter |
| `km_true_labels.png` | Ground truth labels scatter |
| `rf_feature_importance.png` | Random Forest feature importances |
| `rf_confusion.png` | Random Forest confusion matrix |
| `rf_comparison.png` | Single DT vs Bagging vs RF vs ExtraTrees |
| `boost_ada_stages.png` | AdaBoost stagewise train/test error |
| `boost_gbm_stages.png` | GBM stagewise train/test error |
| `boost_feature_importance.png` | GBM feature importances |
| `boost_confusion.png` | Best boosting model confusion matrix |
| `boost_comparison.png` | AdaBoost vs GBM vs HistGBM |
| `ens_confusion.png` | Best ensemble confusion matrix |
| `ens_comparison.png` | All base models + all ensembles |

---

## Dependencies

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
streamlit>=1.35.0
```

Install with:
```bash
pip install -r requirements.txt
```
