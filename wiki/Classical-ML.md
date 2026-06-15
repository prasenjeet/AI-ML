# Classical ML Concepts

All classical ML modules live in `ml_concepts/models/` and use **scikit-learn**.

---

## 1. Linear Regression
**File:** `ml_concepts/models/linear_regression.py`  
**CLI topic:** `lr`  
**Streamlit page:** 📈 Linear Regression

### Variants

| Variant | sklearn class | Key param | Purpose |
|---------|--------------|-----------|--------|
| OLS | `LinearRegression` | — | Baseline; minimises sum of squared residuals |
| Ridge | `Ridge` | `alpha` (L2) | Shrinks all coefficients equally |
| Lasso | `Lasso` | `alpha` (L1) | Zeros out weak features; automatic feature selection |
| Polynomial | `PolynomialFeatures` + OLS | `degree` | Captures non-linear relationships |

### Metrics
`MSE` · `RMSE` · `MAE` · `R²`

### Outputs
- Predicted vs Actual scatter with regression line
- Residual plot (should be randomly scattered around 0)
- Learning curve (training size vs R²)
- Coefficient table
- Side-by-side R² comparison across variants

### Key Concept
> Regularisation (Ridge/Lasso) adds a penalty term to the loss:
> - **Ridge:** `loss = MSE + α·Σw²` — shrinks all weights, none to zero
> - **Lasso:** `loss = MSE + α·Σ|w|` — drives small weights to exactly zero
>
> The trade-off: a small increase in bias yields a large reduction in variance.

---

## 2. Decision Tree
**File:** `ml_concepts/models/decision_tree.py`  
**CLI topic:** `dt`  
**Streamlit page:** 🌳 Decision Tree

### What's Explored

| Experiment | Values tested |
|-----------|---------------|
| Splitting criterion | Gini impurity vs Information Gain (Entropy) |
| Depth sweep | max_depth = 1 → 15 |
| CCP alpha pruning | ccp_alpha = 0.0 → 0.05 |
| Task | Classification (Iris) + Regression (synthetic) |

### Outputs
- Full tree visualisation (`sklearn.tree.plot_tree`)
- Feature importance bar chart
- Confusion matrix
- Accuracy vs depth curve
- Accuracy vs CCP alpha curve

### Key Concept
> **Gini impurity** measures the probability of misclassifying a randomly chosen element.  
> **Entropy** measures information content.  
> In practice both produce similar trees. Gini is faster to compute.
>
> **Cost-Complexity Pruning (CCP):** removes sub-trees with low impurity improvement per unit of complexity. `ccp_alpha` controls aggressiveness — higher = simpler tree.

---

## 3. K-Means Clustering
**File:** `ml_concepts/models/kmeans_clustering.py`  
**CLI topic:** `km`  
**Streamlit page:** 🔵 K-Means Clustering

### Algorithms

| Algorithm | sklearn class | Detail |
|-----------|--------------|--------|
| K-Means++ | `KMeans(init='k-means++')` | Smart centroid initialisation |
| Random Init | `KMeans(init='random')` | Baseline init |
| Ward Agglomerative | `AgglomerativeClustering(linkage='ward')` | Minimises within-cluster variance |
| Complete Agglomerative | `AgglomerativeClustering(linkage='complete')` | Max distance between clusters |
| Average Agglomerative | `AgglomerativeClustering(linkage='average')` | Mean distance between clusters |

### Metrics

| Metric | Good value | Interpretation |
|--------|-----------|----------------|
| **Inertia** | Low | Sum of squared distances to nearest centroid |
| **Silhouette Score** | High (max 1.0) | How well each point fits its cluster vs neighbours |
| **Davies-Bouldin Index** | Low (min 0) | Ratio of within-cluster to between-cluster distances |

### Outputs
- Elbow curve (inertia vs k)
- Silhouette score bar chart
- Cluster scatter plot with centroids
- Agglomerative clustering scatter
- Ground truth label comparison

### Key Concept
> Always **standardise features** before clustering — K-Means uses Euclidean distance so scale matters.  
> Use **Elbow + Silhouette together**: the elbow shows diminishing returns; the silhouette confirms the peak cluster quality.

---

## 4. Bagging & Random Forest
**File:** `ml_concepts/models/bagging_random_forest.py`  
**CLI topic:** `rf`  
**Streamlit page:** 🌲 Bagging & Random Forest

### Model Progression

```
Single Decision Tree  →  BaggingClassifier  →  RandomForestClassifier  →  ExtraTreesClassifier
      (high variance)      (lower variance)       (decorrelated trees)       (fastest, randomised)
```

| Model | Bootstrap | Feature subset | Split threshold |
|-------|-----------|----------------|----------------|
| Single DT | ✗ | All | Best |
| Bagging | ✅ | All | Best |
| Random Forest | ✅ | √n_features | Best |
| Extra Trees | ✅ | √n_features | Random |

### Key Parameters

| Parameter | Effect |
|-----------|--------|
| `n_estimators` | More trees → lower variance; diminishing returns after ~100 |
| `max_features` | Fewer → more decorrelated trees; fewer → higher individual tree bias |
| `oob_score=True` | Free validation using out-of-bag samples (no train/test split needed) |

### Key Concept
> Random Forest decorrelates trees via **random feature subsets** at each split.  
> Decorrelated errors cancel out when averaged → variance ↓ without bias ↑.

---

## 5. Boosting
**File:** `ml_concepts/models/boosting.py`  
**CLI topic:** `boost`  
**Streamlit page:** ⚡ Boosting

### Algorithms

| Algorithm | Mechanism | Notes |
|-----------|-----------|-------|
| **AdaBoost** | Reweights misclassified samples | Exponential loss; sensitive to noise |
| **Gradient Boosting (GBM)** | Fits residuals (negative gradient) | Flexible; many loss functions |
| **HistGradientBoosting** | Histogrammed features | 10–100× faster than GBM; handles NaN natively |

### Stagewise Error Analysis
For each boosting round, training and validation error are tracked via `staged_predict()`. This shows:
- When the model starts overfitting
- The optimal number of estimators
- Effect of learning rate on convergence speed

### Key Concept
> **Shrinkage (learning rate):** each new tree is scaled by `learning_rate` before being added.  
> Lower LR + more estimators ≈ same accuracy but better generalisation.  
> Rule of thumb: `learning_rate ≤ 0.1` with `n_estimators ≥ 100`.

---

## 6. Ensemble Methods
**File:** `ml_concepts/models/ensemble.py`  
**CLI topic:** `ens`  
**Streamlit page:** 🎯 Ensemble Methods

### Base Learners
Decision Tree · Random Forest · Gradient Boosting · K-Nearest Neighbours · Naïve Bayes · SVM

### Combination Strategies

| Strategy | How it works | sklearn class |
|----------|-------------|---------------|
| Hard Voting | Majority class vote | `VotingClassifier(voting='hard')` |
| Soft Voting | Average class probabilities | `VotingClassifier(voting='soft')` |
| Weighted Soft Voting | Weighted probability average | `VotingClassifier(voting='soft', weights=...)` |
| Stacking (LR meta) | LR trained on base model predictions | `StackingClassifier(final_estimator=LR)` |
| Stacking (RF meta) | RF trained on base model predictions | `StackingClassifier(final_estimator=RF)` |

### Diversity Matrix
Pairwise prediction disagreement between all base learners is computed and displayed as a heatmap. Higher disagreement → more to gain from ensembling.

### Key Concept
> **Ensemble power comes from diversity.** If all models make the same errors, ensembling doesn't help.  
> Stacking learns *how* to combine models rather than averaging blindly.
