# Machine Learning Concepts — Sample Project

A self-contained Python project that demonstrates six fundamental machine learning techniques using **scikit-learn**. Each concept has its own module with multiple algorithm variants, hyperparameter sweeps, evaluation metrics, and saved plots.

---

## Table of Contents

- [Concepts Covered](#concepts-covered)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Usage](#usage)
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
| 2 | Decision Tree | Classification & Regression tree, CCP pruning | Iris |
| 3 | K-Means Clustering | K-Means, Agglomerative | Synthetic blobs |
| 4 | Bagging & Random Forest | BaggingClassifier, RandomForest, ExtraTrees | Wine |
| 5 | Boosting | AdaBoost, GradientBoosting, HistGradientBoosting | Wine |
| 6 | Ensemble Methods | Hard/Soft/Weighted Voting, Stacking | Wine |

---

## Project Structure

```
AI-ML/
├── main.py                          # CLI entry point
├── requirements.txt
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
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run all concepts (prints results, shows plots interactively)
python main.py

# 3. Run all concepts and save plots as PNGs to ./plots/
python main.py --save-plots
```

---

## Usage

```
python main.py [--topic TOPIC] [--save-plots]

Options:
  --topic   Run a single concept module (default: all)
            Choices: lr | dt | km | rf | boost | ens
  --save-plots
            Save plots as PNG files to ./plots/ instead of displaying them
```

**Examples:**

```bash
python main.py                      # Run all 6 concepts
python main.py --topic lr           # Linear Regression only
python main.py --topic dt           # Decision Tree only
python main.py --topic km           # K-Means Clustering only
python main.py --topic rf           # Bagging & Random Forest only
python main.py --topic boost        # Boosting only
python main.py --topic ens          # Ensemble Methods only
python main.py --save-plots         # Save all 21 charts to ./plots/
python main.py --topic rf --save-plots  # RF only, save plots
```

---

## Concept Details

### 1. Linear Regression
**File:** `ml_concepts/models/linear_regression.py`

Explores the linear family of regression models on a synthetic dataset with 500 samples and 8 features.

| Variant | Key parameter | Purpose |
|---------|--------------|---------|
| OLS (Ordinary Least Squares) | — | Baseline; minimises sum of squared residuals |
| Ridge | `alpha` (L2 penalty) | Shrinks all coefficients; handles multicollinearity |
| Lasso | `alpha` (L1 penalty) | Zeros out weak features; built-in variable selection |
| Polynomial (deg 2) | `degree=2` | Captures non-linear relationships via feature expansion |

**Output includes:**
- MSE, RMSE, MAE, R² for each variant
- Predicted vs Actual scatter plot with residual plot
- Learning curve (training size vs R² score)
- Model comparison bar chart

**Key concept:** Regularisation (Ridge/Lasso) trades a small increase in bias for a large reduction in variance, improving generalisation on unseen data.

---

### 2. Decision Tree
**File:** `ml_concepts/models/decision_tree.py`

Demonstrates classification and regression trees with pruning strategies on the Iris dataset.

**What's explored:**
- **Criterion comparison** — Gini impurity vs Information Gain (Entropy)
- **Depth sweep** — `max_depth` from 1 to 10, showing underfitting → overfitting
- **Cost-complexity pruning (CCP)** — `ccp_alpha` parameter to post-prune the tree
- **Regression tree** — predicts continuous values on a synthetic dataset

**Output includes:**
- Full tree visualisation (coloured nodes)
- Feature importance bar chart
- Confusion matrix
- Depth vs accuracy comparison chart

**Key concept:** A single unpruned tree memorises training data. Controlling `max_depth` or `ccp_alpha` is essential to generalise beyond the training set.

---

### 3. K-Means Clustering
**File:** `ml_concepts/models/kmeans_clustering.py`

Unsupervised clustering on synthetic 2D blob data with 4 true clusters and 400 samples.

**What's explored:**
- **Elbow method** — inertia (within-cluster SSE) vs k from 1–10
- **Silhouette analysis** — finds optimal k by maximising average silhouette score
- **Initialisation strategies** — `k-means++` vs `random` (convergence quality)
- **Agglomerative hierarchical clustering** — ward, complete, average linkage as alternative

**Metrics reported:** Inertia, Silhouette Score, Davies-Bouldin Index

**Output includes:**
- Elbow curve
- Final cluster scatter plot with centroids
- Agglomerative cluster plot
- Ground truth label plot for comparison

**Key concept:** K-Means is sensitive to scale and initialisation. Always standardise features and prefer `k-means++` init. Use silhouette score alongside the elbow to confirm k.

---

### 4. Bagging & Random Forest
**File:** `ml_concepts/models/bagging_random_forest.py`

Demonstrates how bagging reduces variance on the Wine dataset (178 samples, 13 features, 3 classes).

| Model | Core idea |
|-------|-----------|
| Single Decision Tree | High variance baseline |
| BaggingClassifier | Bootstrap samples + aggregate predictions |
| RandomForestClassifier | Bagging + random feature subset at each split |
| ExtraTreesClassifier | RF + randomised split thresholds |

**What's explored:**
- **OOB (Out-of-Bag) score** — free cross-validation estimate without a held-out set
- **n_estimators sweep** — 1 to 200 trees, showing diminishing returns
- **max_features sweep** — `sqrt`, `log2`, `None`, `0.3`, `0.6`

**Output includes:**
- Feature importance chart (RF)
- Confusion matrix
- Comparison bar chart (single DT → Bagging → RF → ExtraTrees)

**Key concept:** Bagging reduces variance by averaging independent high-variance models. Random Forest decorrelates those models further by limiting which features each tree can consider, leading to stronger ensembles.

---

### 5. Boosting
**File:** `ml_concepts/models/boosting.py`

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

**Output includes:**
- AdaBoost stagewise error chart
- GBM stagewise error chart
- Feature importance bar chart
- Boosting algorithm comparison chart

**Key concept:** Boosting reduces bias by sequentially fitting weak learners to the errors. A lower learning rate with more estimators consistently generalises better but requires more computation.

---

### 6. Ensemble Methods
**File:** `ml_concepts/models/ensemble.py`

Combines six diverse base learners (DT, RF, GBM, KNN, Naive Bayes, SVM) into higher-level ensembles.

| Strategy | How it combines predictions |
|----------|-----------------------------|
| **Hard Voting** | Majority class vote |
| **Soft Voting** | Average predicted probabilities |
| **Weighted Soft Voting** | Weighted average (weights ∝ individual accuracy) |
| **Stacking (LR meta)** | Logistic Regression learns from base predictions |
| **Stacking (RF meta)** | Random Forest learns from base predictions |

**What's explored:**
- **Diversity analysis** — pairwise agreement matrix between all base learners
- **CV evaluation** of the best stacking configuration

**Output includes:**
- Pairwise agreement matrix (printed)
- Best ensemble confusion matrix
- Full comparison bar chart (all base models + all ensembles)

**Key concept:** Ensemble power comes from **diversity**, not just individual accuracy. Models that disagree on hard examples — even if each is imperfect — combine to outperform any single model.

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
| `km_clusters.png` | Final K-Means cluster scatter |
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
```

Install with:
```bash
pip install -r requirements.txt
```
