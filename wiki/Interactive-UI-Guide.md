# Interactive UI Guide

The Streamlit app (`app.py`) provides an interactive UI for all 14 ML concepts across **11 pages**.

## Launch

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

---

## General Layout

```
┌──────────────────┬──────────────────────────────────────────────┐
│  Sidebar         │  Main area                                   │
│                  │                                              │
│  🤖 ML Concepts  │  ┌─────────┬─────────┬─────────┬──────────┐ │
│  ──────────────  │  │ Metric  │ Metric  │ Metric  │ Metric   │ │
│  Navigation      │  └─────────┴─────────┴─────────┴──────────┘ │
│  links           │                                              │
│                  │  Interactive plot(s)                         │
│  Page-specific   │                                              │
│  controls        │  ▶ Expand: detailed analysis                 │
│  (sliders,       │                                              │
│   dropdowns,     │  💡 Key concept takeaway                     │
│   checkboxes)    │                                              │
└──────────────────┴──────────────────────────────────────────────┘
```

---

## Page 1 — 🏠 Home

- Project overview card grid (one card per concept)
- Quick start instructions
- Links to each demo page

---

## Page 2 — 📈 Linear Regression

### Sidebar Controls
| Control | Default | Effect |
|---------|---------|--------|
| Variant | Ridge | OLS / Ridge / Lasso / Polynomial |
| Alpha (λ) | 1.0 | Regularisation strength |
| Polynomial degree | 2 | Only active when Polynomial selected |
| Samples | 200 | Training dataset size |
| Noise | 0.2 | Synthetic noise level |
| Test split | 20% | Held-out fraction |

### Outputs
1. Metrics row: MSE · RMSE · MAE · R²
2. Predicted vs Actual scatter
3. Residual plot
4. Learning curve
5. Coefficient heatmap (expandable)
6. Variant comparison bar chart (expandable)

---

## Page 3 — 🌳 Decision Tree

### Sidebar Controls
| Control | Options |
|---------|--------|
| Dataset | Wine / Iris / Synthetic |
| Task | Classification / Regression |
| Criterion | Gini / Entropy |
| Max depth | 1–15 |
| CCP alpha | 0.000–0.050 |

### Outputs
1. Accuracy / R² metrics
2. Tree visualisation (sklearn plot_tree)
3. Feature importance bar chart
4. Confusion matrix (classification) or fit scatter (regression)
5. Depth sweep chart (expandable)
6. CCP alpha sweep chart (expandable)

---

## Page 4 — 🔵 K-Means Clustering

### Sidebar Controls
| Control | Options |
|---------|--------|
| Samples | 100–1000 |
| True clusters | 2–8 |
| Cluster spread | 0.5–3.0 |
| k to fit | 2–10 |
| Init method | k-means++ / random |
| n_init | 1–20 |
| Agglomerative linkage | ward / complete / average |

### Outputs
1. Silhouette, Davies-Bouldin, Inertia metrics
2. Cluster scatter plot with centroids
3. Elbow curve
4. Silhouette bar chart
5. Agglomerative vs K-Means comparison (expandable)

---

## Page 5 — 🌲 Bagging & Random Forest

### Sidebar Controls
| Control | Options |
|---------|--------|
| Dataset | Wine / Iris / Synthetic |
| n_estimators | 10–500 |
| max_features | sqrt / log2 / None |
| OOB score | toggle |
| Include comparators | DT / Bagging / ExtraTrees toggles |

### Outputs
1. Accuracy + OOB score metrics
2. Feature importance chart
3. Confusion matrix
4. n_estimators sweep (expandable)
5. max_features comparison (expandable)

---

## Page 6 — ⚡ Boosting

### Sidebar Controls
| Control | Options |
|---------|--------|
| Dataset | Wine / Iris / Synthetic |
| AdaBoost: n_estimators | 10–300 |
| AdaBoost: learning_rate | 0.01–2.0 |
| GBM: n_estimators | 50–500 |
| GBM: learning_rate | 0.01–0.5 |
| GBM: max_depth | 1–8 |

### Outputs
1. Accuracy metrics (AdaBoost / GBM / HistGBM)
2. AdaBoost stagewise train/test error
3. GBM stagewise train/test error
4. Feature importance chart
5. Learning rate sweep (expandable)
6. Algorithm comparison bar (expandable)

---

## Page 7 — 🎯 Ensemble Methods

### Controls (inline)
- Checkboxes to enable/disable each base learner
- Checkboxes to enable/disable each ensemble strategy

### Outputs
1. Accuracy for each base learner + ensemble
2. Comparison bar chart
3. Best model confusion matrix
4. Pairwise diversity heatmap
5. Full results table

---

## Page 8 — 🧠 Neural Networks

Single page with **5 tabs**. Training is triggered by clicking the **Train Model** button in each tab. A live progress bar updates per epoch.

### Tab: 🔵 ANN
| Control | Options |
|---------|--------|
| Dataset | Wine / Iris / Synthetic |
| Hidden layers | 1–5 |
| Neurons/layer | 16–256 |
| Activation | relu / tanh / sigmoid / elu |
| Optimizer | adam / sgd / rmsprop |
| Learning rate | 0.0001–0.1 |
| Epochs | 10–200 |
| Dropout | 0.0–0.5 |

**Outputs:** Training curves · Confusion matrix · Activation comparison (expandable)

### Tab: 🟣 DNN
| Control | Options |
|---------|--------|
| Dataset | Wine / Iris / Synthetic |
| Depth (layers) | 1–10 |
| Width (neurons) | 32–256 |
| Dropout | 0.0–0.5 |
| Batch Normalisation | toggle |
| Initialiser | he_normal / glorot_uniform / he_uniform / lecun_normal |
| Epochs | 10–200 |

**Outputs:** Training curves · Confusion matrix · Depth sweep (expandable)

### Tab: 🖼️ CNN
| Control | Options |
|---------|--------|
| Training size | 2000–10000 |
| Conv filters 1 | 8–64 |
| Conv filters 2 | 16–128 |
| Kernel size | 3 / 5 |
| Dense units | 64–256 |
| Dropout | 0.0–0.5 |
| Epochs | 5–50 |
| Use GAP | toggle (GlobalAveragePooling vs Flatten) |

**Outputs:** Training curves · Confusion matrix · Prediction samples grid · Filter visualisation

### Tab: 🔁 RNN
| Control | Options |
|---------|--------|
| Sequence length | 10–120 |
| RNN units | 16–128 |
| Number of layers | 1–3 |
| Dropout | 0.0–0.5 |
| Epochs | 10–100 |
| Signal noise | 0.01–0.3 |
| Compare LSTM | toggle |
| Compare MLP | toggle |

**Outputs:** Training curves · Prediction overlay · RMSE comparison bar

### Tab: ⏳ LSTM
| Control | Options |
|---------|--------|
| Sequence length | 10–120 |
| LSTM units | 16–128 |
| Number of layers | 1–3 |
| Dropout | 0.0–0.5 |
| Bidirectional | toggle |
| Epochs | 10–100 |
| Signal noise | 0.01–0.3 |
| Compare RNN | toggle |
| Compare GRU | toggle |

**Outputs:** Training curves · Prediction overlay · Model comparison · Gate reference table

---

## Page 9 — 📝 NLP

Single page with **5 tabs**.

### Tab: 🔤 Preprocessing
- Text input area (editable)
- Click **Analyse Text**
- Outputs: token list · stopword-filtered tokens · stemmed tokens · lemmatised tokens · POS tags

### Tab: 🏷️ Classification
- Train size slider · max TF-IDF features · classifier dropdown
- Click **Train Classifier**
- Outputs: accuracy · confusion matrix · top positive/negative feature weights

### Tab: 😊 Sentiment
- Text input area (editable)
- Click **Analyse Sentiment**
- Outputs: compound score · pos/neg/neu bars · colour-coded label (Positive / Neutral / Negative)

### Tab: 📚 Topics (LDA)
- Number of topics (2–10) · max iterations
- Click **Run LDA**
- Outputs: top-10 words per topic · perplexity score

### Tab: 🔍 TF-IDF Explorer
- Multi-line text area (one document per line)
- Max features slider
- Click **Build TF-IDF**
- Outputs: TF-IDF matrix heatmap · top-term bar chart

---

## Page 10 — 🖼️ Computer Vision

Single page with **4 tabs**.

### Tab: 🎨 Augmentation
- Flip H/V toggles · rotation angle · brightness delta · zoom fraction
- Click **Apply Augmentation**
- Outputs: 3-column before/after comparison grid

### Tab: 🔍 Feature Extraction
- HOG orientations · pixels/cell · SVM kernel (linear/rbf)
- Click **Extract & Classify**
- Outputs: HOG visualisation grid · Sobel edges · SVM accuracy (HOG vs colour hist)

### Tab: 🧠 CNN Training
- Conv filters · dropout · epochs · augmentation toggle
- Click **Train CNN**
- Outputs: training curves · confusion matrix (10 CIFAR classes) · sample prediction grid

### Tab: 🚀 Transfer Learning
- Fine-tune toggle · epochs
- Click **Run Transfer Learning**
- Outputs: linear probe accuracy · fine-tuned accuracy · feature TSNE (expandable)

---

## Page 11 — 🎬 Recommendations

Single page with **6 tabs**.

### Tab: 📊 Dataset
- Users / items / sparsity sliders
- Click **Generate Dataset**
- Outputs: rating matrix heatmap · density stats · genre distribution pie

### Tab: 👥 User-Based CF
- k neighbours · target user dropdown
- Click **Run User-CF**
- Outputs: top-10 recommendations table · similarity heatmap

### Tab: 🎬 Item-Based CF
- k items · target user dropdown
- Click **Run Item-CF**
- Outputs: top-10 recommendations table · item similarity heatmap

### Tab: 🔢 Matrix Factorisation
- SVD components slider
- Click **Run SVD**
- Outputs: explained variance · latent factor heatmap · top-10 recommendations

### Tab: 📝 Content-Based
- Target user dropdown
- Click **Run Content-Based**
- Outputs: user genre profile bar chart · top-10 recommendations

### Tab: 📈 Evaluation
- Click **Evaluate All Models**
- Outputs: RMSE / Precision@K / Recall@K comparison bar charts · full metrics table
