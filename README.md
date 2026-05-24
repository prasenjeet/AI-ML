# Machine Learning Concepts — Sample Project

A self-contained Python project demonstrating **fourteen ML, deep learning, and applied AI techniques** using **scikit-learn**, **TensorFlow/Keras**, **NLTK**, and **scikit-image**. Available in two modes:

- **Interactive UI** — Streamlit web app with live inputs, dynamic plots, and metric cards
- **CLI runner** — terminal script with printed results and saved PNG charts

---

## Table of Contents

- [Concepts Covered](#concepts-covered)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Interactive UI](#interactive-ui-apppy)
- [CLI Runner](#cli-runner-mainpy)
- [Classical ML Concepts](#classical-ml-concepts)
  - [Linear Regression](#1-linear-regression)
  - [Decision Tree](#2-decision-tree)
  - [K-Means Clustering](#3-k-means-clustering)
  - [Bagging & Random Forest](#4-bagging--random-forest)
  - [Boosting](#5-boosting)
  - [Ensemble Methods](#6-ensemble-methods)
- [Neural Network Concepts](#neural-network-concepts)
  - [ANN – Artificial Neural Network](#7-ann--artificial-neural-network)
  - [DNN – Deep Neural Network](#8-dnn--deep-neural-network)
  - [CNN – Convolutional Neural Network](#9-cnn--convolutional-neural-network)
  - [RNN – Recurrent Neural Network](#10-rnn--recurrent-neural-network)
  - [LSTM – Long Short-Term Memory](#11-lstm--long-short-term-memory)
- [Applied AI Concepts](#applied-ai-concepts)
  - [NLP – Natural Language Processing](#12-nlp--natural-language-processing)
  - [Computer Vision](#13-computer-vision)
  - [Recommendation Systems](#14-recommendation-systems)
- [Generated Plots](#generated-plots)
- [Dependencies](#dependencies)

---

## Concepts Covered

### Classical ML
| # | Concept | Algorithms | Dataset |
|---|---------|------------|---------|
| 1 | Linear Regression | OLS, Ridge, Lasso, Polynomial | Synthetic regression |
| 2 | Decision Tree | Classification & Regression tree, CCP pruning | Iris / Wine / Synthetic |
| 3 | K-Means Clustering | K-Means, Agglomerative | Synthetic blobs |
| 4 | Bagging & Random Forest | BaggingClassifier, RandomForest, ExtraTrees | Wine / Iris / Synthetic |
| 5 | Boosting | AdaBoost, GradientBoosting, HistGradientBoosting | Wine / Iris / Synthetic |
| 6 | Ensemble Methods | Hard/Soft/Weighted Voting, Stacking | Wine / Iris / Synthetic |

### Neural Networks
| # | Concept | Architecture | Dataset |
|---|---------|-------------|---------|
| 7  | ANN | Feedforward MLP | Wine / Iris / Synthetic (tabular) |
| 8  | DNN | Deep MLP + BatchNorm + Dropout | Synthetic (tabular) |
| 9  | CNN | Conv2D → MaxPool → Dense | MNIST digits |
| 10 | RNN | SimpleRNN (stacked) | Synthetic time series |
| 11 | LSTM | Stacked LSTM / GRU / Bidirectional | Synthetic time series |

### Applied AI
| # | Concept | Techniques | Dataset |
|---|---------|-----------|---------|
| 12 | NLP | Preprocessing, TF-IDF, Classification, Sentiment, LDA | 20 Newsgroups + custom texts |
| 13 | Computer Vision | HOG, Augmentation, SVM, CNN, Transfer Learning | CIFAR-10 |
| 14 | Recommendation Systems | User-CF, Item-CF, SVD, Content-Based | Synthetic ratings matrix |

---

## Project Structure

```
AI-ML/
├── app.py                           # ★ Streamlit interactive UI
├── main.py                          # CLI runner
├── requirements.txt
├── .gitignore
├── plots/                           # Generated PNG charts
└── ml_concepts/
    ├── utils/
    │   ├── data_generator.py        # sklearn dataset helpers
    │   ├── visualizer.py            # Classical ML plot helpers
    │   └── nn_utils.py              # Neural Network data & plot helpers
    └── models/
        ├── linear_regression.py
        ├── decision_tree.py
        ├── kmeans_clustering.py
        ├── bagging_random_forest.py
        ├── boosting.py
        ├── ensemble.py
        ├── ann.py                   # ANN demo
        ├── dnn.py                   # DNN demo
        ├── cnn.py                   # CNN demo (MNIST)
        ├── rnn.py                   # RNN demo
        ├── lstm.py                  # LSTM / GRU demo
        ├── nlp.py                   # ★ NLP demo
        ├── computer_vision.py       # ★ Computer Vision demo (CIFAR-10)
        └── recommendations.py       # ★ Recommendation Systems demo
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

```bash
streamlit run app.py
```

### Layout

```
┌─────────────────────────┬─────────────────────────────────────────┐
│  Sidebar                │  Main area                              │
│  ───────────────        │  ─────────────────────────────          │
│  🤖 ML Concepts         │  Metric cards                           │
│                         │                                         │
│  Page navigation        │  Interactive plots (live update)        │
│  (11 pages)             │                                         │
│                         │  Expandable analysis sections           │
│  Concept-specific       │                                         │
│  sliders & dropdowns    │  Key concept takeaway                   │
└─────────────────────────┴─────────────────────────────────────────┘
```

### Classical ML Pages

| Page | Sidebar Inputs | Live Outputs |
|------|----------------|--------------|
| **📈 Linear Regression** | Variant, alpha, degree, samples, noise, test split | Predicted vs Actual, Residuals, Learning curve, Coefficients, Comparison |
| **🌳 Decision Tree** | Dataset, task, criterion, max_depth, CCP alpha | Tree diagram, Feature importance, Confusion matrix, Depth & CCP sweeps |
| **🔵 K-Means** | Samples, true k, spread, k, init, n_init, linkage | Elbow, Silhouette bars, Cluster scatter, Agglomerative side-by-side |
| **🌲 Bagging & Random Forest** | Dataset, n_estimators, max_features, OOB, comparators | Feature importance, Confusion matrix, n_estimators & max_features sweeps |
| **⚡ Boosting** | Dataset, AdaBoost & GBM controls | Stagewise error curves, Feature importance, LR sweep |
| **🎯 Ensemble Methods** | Per-learner & per-strategy checkboxes | Comparison bar, Confusion matrix, Diversity heatmap, Results table |

### Neural Network Page (`🧠 Neural Networks`)

A single page with **5 tabs** — one per architecture. Each tab has:
- Inline controls (no sidebar) for full hyperparameter control
- **Train Model** button — training runs with a live progress bar
- Results appear immediately after training

| Tab | Controls | Outputs |
|-----|----------|---------|
| **🔵 ANN** | Dataset, layers, neurons, activation, optimizer, LR, epochs, dropout | Training curves, Confusion matrix, Activation comparison expander |
| **🟣 DNN** | Dataset, depth, width, dropout, BatchNorm, initialiser, epochs | Training curves, Confusion matrix, Depth sweep expander |
| **🖼️ CNN** | Training size, filters (×2), kernel, dense units, dropout, epochs | Training curves, Confusion matrix, Sample predictions grid, Learned filter visualisation |
| **🔁 RNN** | Seq length, units, layers, dropout, epochs, noise; compare LSTM/MLP | Training curves, Prediction vs Actual, Model RMSE comparison |
| **⏳ LSTM** | Seq length, units, layers, dropout, bidir, epochs; compare RNN/GRU | Training curves, Prediction vs Actual, Overlay chart, Gate mechanism reference |

### NLP Page (`📝 NLP`)

A single page with **5 tabs** covering the full text processing pipeline.

| Tab | Controls | Outputs |
|-----|----------|---------|
| **🔤 Preprocessing** | Custom text input | Token list, stopword-filtered tokens, stemmed, lemmatised, POS tags |
| **🏷️ Classification** | Train size, max TF-IDF features, classifier (NB/LR/SVM) | Accuracy, confusion matrix, top feature weights |
| **😊 Sentiment** | Custom text input | VADER compound/pos/neg/neu scores, colour-coded sentiment bars |
| **📚 Topics (LDA)** | Number of topics, max iterations | Top-10 words per topic, perplexity score |
| **🔍 TF-IDF Explorer** | Custom corpus (one doc per line), max features | TF-IDF heatmap, top-terms bar chart |

### Computer Vision Page (`🖼️ Computer Vision`)

A single page with **4 tabs** covering the CV pipeline from raw pixels to transfer learning.

| Tab | Controls | Outputs |
|-----|----------|---------|
| **🎨 Augmentation** | Flip H/V toggles, rotation, brightness, zoom | Before/after 3-column comparison grid |
| **🔍 Feature Extraction** | HOG orientations, pixels/cell, SVM kernel | HOG visualisation, Sobel edges, SVM accuracy (HOG vs colour hist) |
| **🧠 CNN Training** | Filters, dropout, epochs, augmentation toggle | Training curves, confusion matrix (10 classes), sample prediction grid |
| **🚀 Transfer Learning** | MobileNetV2 freeze/unfreeze, fine-tune epochs | Linear probe accuracy, fine-tuned accuracy, feature visualisation |

### Recommendations Page (`🎬 Recommendations`)

A single page with **6 tabs** covering collaborative and content-based filtering.

| Tab | Controls | Outputs |
|-----|----------|---------|
| **📊 Dataset** | Users, items, sparsity slider | Rating matrix heatmap, density stats, genre distribution |
| **👥 User-Based CF** | k neighbours, target user | Top-10 recommendations, similarity heatmap |
| **🎬 Item-Based CF** | k items, target user | Top-10 recommendations, item similarity heatmap |
| **🔢 Matrix Factorisation** | SVD components | Explained variance, latent factor heatmap, Top-10 recommendations |
| **📝 Content-Based** | Target user | User genre profile chart, Top-10 recommendations |
| **📈 Evaluation** | All models | RMSE, Precision@K, Recall@K comparison bar charts |

---

## CLI Runner (`main.py`)

```
python main.py [--topic TOPIC] [--save-plots]

Topics (classical ML):
  lr | dt | km | rf | boost | ens

Topics (neural networks):
  ann | dnn | cnn | rnn | lstm

Topics (applied AI):
  nlp | cv | rec
```

**Examples:**

```bash
python main.py                          # All 14 concepts
python main.py --topic nlp              # NLP only
python main.py --topic cv               # Computer Vision only
python main.py --topic rec              # Recommendations only
python main.py --topic ann              # ANN only
python main.py --topic lr               # Linear Regression only
python main.py --save-plots             # Save all charts to ./plots/
```

---

## Classical ML Concepts

### 1. Linear Regression
**File:** `ml_concepts/models/linear_regression.py`

| Variant | Key parameter | Purpose |
|---------|--------------|---------|
| OLS | — | Baseline; minimises squared residuals |
| Ridge | `alpha` (L2) | Shrinks all coefficients; handles multicollinearity |
| Lasso | `alpha` (L1) | Zeros out weak features; feature selection |
| Polynomial (deg 2–5) | `degree` | Captures non-linear relationships |

**Output:** MSE · RMSE · MAE · R² · Predicted vs Actual · Residual plot · Learning curve

**Key concept:** Regularisation (Ridge/Lasso) trades a small bias increase for a large variance reduction.

---

### 2. Decision Tree
**File:** `ml_concepts/models/decision_tree.py`

**What's explored:** Gini vs Entropy · depth sweep (1–15) · CCP alpha pruning · regression tree

**Output:** Tree visualisation · Feature importance · Confusion matrix · Depth & CCP sweeps

**Key concept:** Unpruned trees memorise training data; pruning (max_depth / ccp_alpha) is essential to generalise.

---

### 3. K-Means Clustering
**File:** `ml_concepts/models/kmeans_clustering.py`

**What's explored:** Elbow method · silhouette analysis · k-means++ vs random init · agglomerative linkages

**Metrics:** Inertia · Silhouette Score · Davies-Bouldin Index

**Key concept:** Always standardise before clustering; use Elbow + Silhouette together to pick k.

---

### 4. Bagging & Random Forest
**File:** `ml_concepts/models/bagging_random_forest.py`

| Model | Core idea |
|-------|-----------|
| Single Decision Tree | High-variance baseline |
| BaggingClassifier | Bootstrap + aggregate |
| RandomForestClassifier | Bagging + random feature subset |
| ExtraTreesClassifier | RF + randomised thresholds |

**What's explored:** OOB score · n_estimators sweep · max_features sweep

**Key concept:** Random Forest decorrelates trees via feature randomness → lower variance than plain bagging.

---

### 5. Boosting
**File:** `ml_concepts/models/boosting.py`

| Algorithm | Mechanism |
|-----------|-----------|
| AdaBoost | Re-weights misclassified samples |
| GradientBoosting | Fits negative gradient (residuals) sequentially |
| HistGradientBoosting | Histogrammed features → 10–100× faster |

**What's explored:** Stagewise error curves · learning-rate sweep · feature importance

**Key concept:** Lower learning rate + more estimators generalises better but costs more compute.

---

### 6. Ensemble Methods
**File:** `ml_concepts/models/ensemble.py`

| Strategy | Combination method |
|----------|--------------------|
| Hard Voting | Majority class vote |
| Soft Voting | Average predicted probabilities |
| Weighted Soft Voting | Weighted average (weights ∝ accuracy) |
| Stacking (LR) | Logistic Regression meta-learner |
| Stacking (RF) | Random Forest meta-learner |

**Key concept:** Ensemble power comes from **diversity** — low pairwise agreement between base models.

---

## Neural Network Concepts

### 7. ANN – Artificial Neural Network
**File:** `ml_concepts/models/ann.py`  
**Dataset:** Tabular classification (Wine / Iris / Synthetic)

A fully-connected feedforward network — the foundational deep learning architecture.

**Architecture:**
```
Input(n_features) → Dense(n, activation) × n_layers → Dropout → Dense(n_classes, softmax)
```

**What's explored:**
- Activation functions: ReLU · tanh · sigmoid · ELU
- Optimizers: Adam · SGD · RMSprop
- Hidden layer width sweep (16 → 256)
- Dropout regularisation

**Key concept:** ANN is a universal function approximator. ReLU avoids vanishing gradients; Adam converges faster than SGD on most problems.

---

### 8. DNN – Deep Neural Network
**File:** `ml_concepts/models/dnn.py`  
**Dataset:** Synthetic tabular classification

Extends ANN to many hidden layers with training stabilisers.

**Architecture:**
```
Input → [Dense → BatchNorm → ReLU → Dropout] × n_layers → Dense(n_classes, softmax)
```

**What's explored:**
- Depth sweep: 1 → 10 layers
- Batch Normalisation ablation
- Dropout rate sweep (0 → 0.5)
- Weight initialisers: He Normal · Glorot Uniform · LeCun Normal
- Learning rate schedulers: ReduceLROnPlateau · EarlyStopping

**Key concept:** BatchNorm normalises layer inputs at each mini-batch — essential for training networks deeper than ~5 layers. He initialisation is preferred for ReLU activations.

---

### 9. CNN – Convolutional Neural Network
**File:** `ml_concepts/models/cnn.py`  
**Dataset:** MNIST handwritten digits (subset)

Convolutions detect spatial patterns with shared weights — far more efficient than Dense layers for images.

**Architecture:**
```
Input(28×28×1)
→ [Conv2D(f1) → BN → Conv2D(f1) → MaxPool → Dropout] ×1
→ [Conv2D(f2) → BN → Conv2D(f2) → MaxPool → Dropout] ×1
→ Flatten / GlobalAvgPool → Dense(units) → Dropout → Dense(10, softmax)
```

**What's explored:**
- Filter count sweep: (8,16) → (64,128)
- Kernel size comparison: 3×3 vs 5×5
- GlobalAveragePooling vs Flatten
- Learned filter visualisation (Conv1 weights)
- Sample prediction grid (correct in green, wrong in red)

**Key concept:** Conv layers slide filters over the image, sharing weights spatially — edge/curve detectors emerge automatically from training. MaxPooling adds translation invariance.

---

### 10. RNN – Recurrent Neural Network
**File:** `ml_concepts/models/rnn.py`  
**Dataset:** Synthetic multi-frequency sine wave (time series)

RNNs maintain a hidden state across time steps — suitable for sequential data.

**Architecture:**
```
Input(seq_len, 1) → SimpleRNN(units) [× n_layers] → Dense(1)
```

**What's explored:**
- Sequence length sweep (10 → 120)
- Layer depth sweep (1 → 3)
- Comparison with MLP baseline (no recurrence)
- Vanishing gradient discussion

**Key concept:** The hidden state *h_t = f(W·x_t + U·h_{t-1})* carries information forward in time. Vanilla RNNs forget events more than ~20 steps back — the motivation for LSTM.

---

### 11. LSTM – Long Short-Term Memory
**File:** `ml_concepts/models/lstm.py`  
**Dataset:** Synthetic multi-frequency sine wave (time series)

LSTMs add forget / input / output gates and a separate cell state to solve vanishing gradients.

**Architecture:**
```
Input(seq_len, 1) → LSTM(units) [× n_layers] → Dense(1)
Optional: Bidirectional wrapper
```

**What's explored:**
- LSTM vs SimpleRNN vs GRU head-to-head (RMSE comparison)
- Bidirectional LSTM (reads sequence forward + backward)
- Dropout & recurrent_dropout sweeps
- Sequence length sweep

**LSTM Gate Mechanism:**

| Gate | Purpose |
|------|---------|
| **Forget** *f_t* | Decides what to discard from cell state |
| **Input** *i_t* | Decides what new information to store |
| **Update** *C̃_t* | New candidate values for cell state |
| **Output** *o_t* | Decides what to expose as hidden state |

Cell state update: **C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t**

**Key concept:** The cell state *C_t* is a 'conveyor belt' that can carry gradients across hundreds of steps. GRU achieves similar results with fewer parameters (2 gates instead of 3).

---

## Applied AI Concepts

### 12. NLP – Natural Language Processing
**File:** `ml_concepts/models/nlp.py`  
**Datasets:** 20 Newsgroups (5 categories), custom text inputs

A complete NLP pipeline from raw text to classification and topic modelling.

**Pipeline stages:**

```
Raw Text
  → Tokenisation (NLTK word_tokenize)
  → Stopword Removal (NLTK stopwords corpus)
  → Stemming (PorterStemmer)  /  Lemmatisation (WordNetLemmatizer)
  → POS Tagging (NLTK averaged_perceptron_tagger)
  → TF-IDF Vectorisation
  → Classification / Sentiment / Topic Modelling
```

**What's explored:**

| Demo | Technique | Detail |
|------|-----------|--------|
| Preprocessing | Tokenise → stem → lemmatise → POS tag | Side-by-side comparison of each step |
| Classification | TF-IDF + ComplementNB / LogReg / LinearSVC | 20 Newsgroups (sci.med, sci.space, baseball, politics, comp) |
| Sentiment | VADER (Valence Aware Dictionary) | compound / positive / negative / neutral scores |
| Topic Modelling | LDA (Latent Dirichlet Allocation) | Top-10 words per topic, perplexity |
| TF-IDF Explorer | Custom corpus | Term importance heatmap, top-terms bar |

**Key concept:** TF-IDF down-weights common words and up-weights distinctive terms — the core of most bag-of-words classifiers. VADER is rule-based and requires no training data (useful for social media text).

---

### 13. Computer Vision
**File:** `ml_concepts/models/computer_vision.py`  
**Dataset:** CIFAR-10 (10 classes: airplane, car, bird, cat, deer, dog, frog, horse, ship, truck)

Covers the full CV pipeline from hand-crafted features to transfer learning.

**Pipeline:**

```
Raw Image (32×32×3)
  → Augmentation (flip, rotate, brightness, zoom)
  → Feature Extraction (HOG / colour histogram)  or  CNN
  → Classification (SVM / Custom CNN / MobileNetV2)
```

**What's explored:**

| Demo | Technique | Detail |
|------|-----------|--------|
| Preprocessing | Channel statistics, sample grid | Mean/std per channel, RGB distribution plots |
| Augmentation | Flip H/V, rotate, brightness, zoom | scikit-image transforms on random samples |
| Feature Extraction | HOG + SVM / Colour Hist + SVM | HOG visualisation, Sobel edges, SVM accuracy |
| CNN Training | Custom dual-Conv CNN + data augmentation | `RandomFlip`, `RandomRotation`, `RandomZoom`, `ReduceLROnPlateau` |
| Transfer Learning | MobileNetV2 (ImageNet pretrained) | Linear probe SVC on frozen features; fine-tuning custom head |

**CNN Architecture (CIFAR-10):**
```
Input(32×32×3)
→ [Conv2D(32) → BN → Conv2D(32) → MaxPool → Dropout(0.25)]
→ [Conv2D(64) → BN → Conv2D(64) → MaxPool → Dropout(0.25)]
→ GlobalAveragePooling
→ Dense(256, ReLU) → Dropout(0.5) → Dense(10, softmax)
```

**Key concept:** Hand-crafted features (HOG, colour histograms) work well for clean images but CNNs learn task-specific features automatically. Transfer learning from a pre-trained backbone typically outperforms training from scratch on small datasets.

---

### 14. Recommendation Systems
**File:** `ml_concepts/models/recommendations.py`  
**Dataset:** Synthetic 120-user × 60-item rating matrix (latent factor model, 1–5 scale, ~35% dense)

Four complementary recommendation strategies with full evaluation.

**Strategies:**

| Method | Core Idea | Cold Start | Needs Features |
|--------|-----------|-----------|----------------|
| **User-Based CF** | "Users like you also liked…" | ❌ | ❌ |
| **Item-Based CF** | "Because you liked X, try Y" | ✓ (item) | ❌ |
| **Matrix Factorisation (SVD)** | Decompose ratings into latent factors | ❌ | ❌ |
| **Content-Based** | Match user profile to item attributes | ✓ (user) | ✅ |

**Mathematical detail:**

*User-Based CF* uses mean-centred cosine similarity:
```
pred(u, i) = μ_u + Σ_{v∈N(u)} sim(u,v) · (r_{v,i} − μ_v) / Σ|sim(u,v)|
```

*SVD* factorises the completed rating matrix:
```
R ≈ U · Σ · V^T    (truncated to k components)
```

*Content-Based* builds a user genre profile:
```
profile(u) = mean of item feature vectors weighted by user ratings
```

**Evaluation metrics:**

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| RMSE | √(Σ(r̂ − r)² / n) | Prediction accuracy on held-out ratings |
| Precision@K | \|relevant ∩ top-K\| / K | Proportion of recommendations that are relevant |
| Recall@K | \|relevant ∩ top-K\| / \|relevant\| | Proportion of relevant items that are recommended |

**Key concept:** No single method wins in all settings. Item-CF is more stable and less cold-start sensitive than User-CF; SVD often achieves the best RMSE; Content-Based handles new items that have no ratings history.

---

## Generated Plots

`python main.py --save-plots` saves charts to `./plots/`:

| File | Description |
|------|-------------|
| `lr_fit.png` | OLS predicted vs actual + residuals |
| `lr_learning_curve.png` | Training size vs R² |
| `lr_comparison.png` | R² across all regression variants |
| `dt_tree.png` | Full tree visualisation |
| `dt_feature_importance.png` | Feature importance bar chart |
| `dt_confusion.png` | Decision tree confusion matrix |
| `dt_depth_comparison.png` | Test accuracy vs max_depth |
| `km_elbow.png` | Elbow curve (inertia vs k) |
| `km_clusters.png` | K-Means cluster scatter + centroids |
| `km_agglomerative.png` | Agglomerative clustering scatter |
| `km_true_labels.png` | Ground truth labels |
| `rf_feature_importance.png` | Random Forest feature importances |
| `rf_confusion.png` | Random Forest confusion matrix |
| `rf_comparison.png` | DT vs Bagging vs RF vs ExtraTrees |
| `boost_ada_stages.png` | AdaBoost stagewise train/test error |
| `boost_gbm_stages.png` | GBM stagewise train/test error |
| `boost_feature_importance.png` | GBM feature importances |
| `boost_confusion.png` | Best boosting model confusion matrix |
| `boost_comparison.png` | AdaBoost vs GBM vs HistGBM |
| `ens_confusion.png` | Best ensemble confusion matrix |
| `ens_comparison.png` | All base models + all ensembles |
| `ann_history.png` | ANN loss & accuracy curves |
| `ann_confusion.png` | ANN confusion matrix |
| `dnn_history.png` | DNN loss & accuracy curves |
| `dnn_confusion.png` | DNN confusion matrix |
| `cnn_history.png` | CNN loss & accuracy curves |
| `cnn_confusion.png` | CNN confusion matrix (10 digits) |
| `cnn_samples.png` | CNN prediction samples grid |
| `rnn_history.png` | RNN training loss |
| `rnn_prediction.png` | RNN prediction vs actual |
| `lstm_history.png` | LSTM training loss |
| `lstm_prediction.png` | LSTM prediction vs actual |
| `nlp_sentiment.png` | VADER sentiment scores for sample texts |
| `nlp_tfidf.png` | TF-IDF term importance heatmap |
| `cv_samples.png` | CIFAR-10 sample image grid |
| `cv_augmentation.png` | Augmented image comparison (before/after) |
| `cv_hog.png` | HOG feature visualisation on sample images |
| `cv_edges.png` | Sobel edge detection |
| `cv_history.png` | CNN training curves on CIFAR-10 |
| `cv_confusion.png` | CNN confusion matrix (10 CIFAR classes) |
| `rec_matrix.png` | User-item rating matrix heatmap |
| `rec_comparison.png` | RMSE / Precision@K / Recall@K comparison |

---

## Dependencies

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scikit-image>=0.21.0
matplotlib>=3.7.0
seaborn>=0.12.0
Pillow>=10.0.0
nltk>=3.8.0
streamlit>=1.35.0
tensorflow-cpu>=2.15.0
```

Install with:
```bash
pip install -r requirements.txt
```
