# Applied AI Concepts

Three applied AI domains built on top of the ML and deep learning foundations.

---

## 12. Natural Language Processing (NLP)
**File:** `ml_concepts/models/nlp.py`  
**CLI topic:** `nlp`  
**Streamlit page:** 📝 NLP  
**Library:** NLTK + scikit-learn

### Full Pipeline

```
Raw Text
  ↓
Tokenisation        → word_tokenize() → ["Natural", "language", "processing", ...]
  ↓
Stopword Removal    → remove ["the", "is", "at", ...] → ["Natural", "language", ...]
  ↓
Stemming            → PorterStemmer   → ["natur", "languag", ...]
Lemmatisation       → WordNetLemmatizer → ["natural", "language", ...]
  ↓
POS Tagging         → [("Natural", "JJ"), ("language", "NN"), ...]
  ↓
Vectorisation       → TF-IDF / BoW / N-grams
  ↓
Classification / Sentiment / Topic Modelling
```

### Demo 1 — Text Preprocessing

Side-by-side comparison of each pipeline stage on sample texts:

| Stage | Example output |
|-------|---------------|
| Tokens | `["machine", "learning", "is", "fascinating"]` |
| Filtered | `["machine", "learning", "fascinating"]` |
| Stemmed | `["machin", "learn", "fascin"]` |
| Lemmatised | `["machine", "learning", "fascinating"]` |
| POS | `[("machine", "NN"), ("learning", "VBG"), ...]` |

### Demo 2 — Text Classification

**Dataset:** 20 Newsgroups (5 categories)
- `sci.med` · `sci.space` · `rec.sport.baseball` · `talk.politics.misc` · `comp.graphics`

**Pipeline:** `TfidfVectorizer` → Classifier

| Classifier | Strength | Notes |
|-----------|---------|-------|
| **ComplementNB** | Fast; good on imbalanced text | Variant of Naïve Bayes for text |
| **LogisticRegression** | Interpretable; feature weights | Good general-purpose text classifier |
| **LinearSVC** | Fastest at inference | Scales well to large vocabularies |

### Demo 3 — Sentiment Analysis

**VADER** (Valence Aware Dictionary and sEntiment Reasoner):
- Rule-based; no training data required
- Outputs `compound` (-1 to +1), `pos`, `neg`, `neu` scores
- Handles punctuation, capitalization, emoticons, and slang

```python
# Example
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
sia.polarity_scores("This movie was GREAT!!!")  
# → {'neg': 0.0, 'neu': 0.267, 'pos': 0.733, 'compound': 0.6588}
```

### Demo 4 — Topic Modelling (LDA)

**Latent Dirichlet Allocation** models each document as a mixture of topics, where each topic is a distribution over words.

```
Document → Topic mixture: 70% Topic-1, 20% Topic-3, 10% Topic-5
Topic-1  → Word distribution: "space" (0.08), "nasa" (0.06), "orbit" (0.05), ...
```

Metric: **Perplexity** — lower is better (model is less surprised by held-out data).

### Key Concept
> **TF-IDF** (Term Frequency–Inverse Document Frequency):  
> `TF-IDF(t, d) = TF(t, d) × log(N / df(t))`  
> Down-weights common words ("the", "is") and up-weights distinctive terms.  
> This is the workhorse of bag-of-words text classification.

---

## 13. Computer Vision
**File:** `ml_concepts/models/computer_vision.py`  
**CLI topic:** `cv`  
**Streamlit page:** 🖼️ Computer Vision  
**Dataset:** CIFAR-10 (10 classes, 32×32 colour images)

### CIFAR-10 Classes
`airplane · automobile · bird · cat · deer · dog · frog · horse · ship · truck`

### Pipeline

```
Raw Image (32×32×3)
  ↓
Augmentation        → flip, rotate, brightness, zoom
  ↓
Feature Extraction  → HOG descriptor  OR  Colour Histogram  OR  CNN features
  ↓
Classification      → SVM  OR  Custom CNN  OR  MobileNetV2 (transfer)
```

### Demo 1 — Image Augmentation

| Transform | Library | Parameter range |
|-----------|---------|----------------|
| Horizontal flip | scikit-image | — |
| Vertical flip | scikit-image | — |
| Rotation | `scipy.ndimage.rotate` | ±30° |
| Brightness | NumPy clip | ±40 pixel values |
| Zoom (crop) | NumPy slice | 80% of original |

### Demo 2 — Hand-crafted Feature Extraction

**HOG (Histogram of Oriented Gradients):**
```
Image → Compute pixel gradients → Bin into orientation histograms → L2 normalise
```
- orientations=8, pixels_per_cell=(4,4), cells_per_block=(2,2)
- Captures local shape and edge information
- Pairs well with linear SVM

**Colour Histograms:**
- 32-bin histogram per R/G/B channel → 96-dimensional vector
- Captures colour distribution but loses spatial information

**Sobel Edge Detection:**
```
Gx = [[-1,0,1],[-2,0,2],[-1,0,1]]   # horizontal edges
Gy = Gx.T                            # vertical edges  
magnitude = √(Gx² + Gy²)
```

### Demo 3 — Custom CNN on CIFAR-10

```
Input(32×32×3)
  → RandomFlip + RandomRotation + RandomZoom      # Keras augmentation layers
  → Conv2D(32) → BN → Conv2D(32) → MaxPool → Dropout(0.25)
  → Conv2D(64) → BN → Conv2D(64) → MaxPool → Dropout(0.25)
  → GlobalAveragePooling
  → Dense(256, relu) → Dropout(0.5) → Dense(10, softmax)
```

Callbacks: `ReduceLROnPlateau` + `EarlyStopping`

### Demo 4 — Transfer Learning (MobileNetV2)

| Stage | What happens |
|-------|-------------|
| **Feature extraction** | Freeze MobileNetV2 backbone; train linear SVC on extracted features |
| **Fine-tuning** | Unfreeze top layers; continue training with low LR |

Images are resized to 96×96 for MobileNetV2 (smaller than the default 224×224 for speed).

### Key Concept
> **Transfer learning** works because the low-level features learned on ImageNet (edges, textures, colour blobs) are universal — they transfer to other visual tasks.  
> The deeper layers (object parts, high-level semantics) are more task-specific and benefit from fine-tuning.

---

## 14. Recommendation Systems
**File:** `ml_concepts/models/recommendations.py`  
**CLI topic:** `rec`  
**Streamlit page:** 🎬 Recommendations

### Synthetic Dataset
- **120 users × 60 movies**, ratings 1–5
- Generated via latent factor model: `R_base = U @ V.T`  (8 latent factors)
- ~35% density (65% sparse — users haven't rated most movies)
- Item features: genre one-hot vectors (8 genres: Action, Comedy, Drama, Sci-Fi, Romance, Thriller, Horror, Animation)

### Strategy Comparison

| Method | Cold Start (user) | Cold Start (item) | Needs features | Explainability |
|--------|:-----------------:|:-----------------:|:--------------:|:--------------:|
| User-Based CF | ❌ | ✅ | ❌ | Medium |
| Item-Based CF | ✅ | ❌ | ❌ | High |
| SVD | ❌ | ❌ | ❌ | Low |
| Content-Based | ✅ | ❌ | ✅ | High |

### Strategy 1 — User-Based Collaborative Filtering

```
1. Mean-centre each user's ratings:  r̃_{u,i} = r_{u,i} − μ_u
2. Compute cosine similarity between all user pairs
3. Find k nearest neighbours of target user
4. Predict: pred(u, i) = μ_u + Σ_{v∈N(u)} sim(u,v)·r̃_{v,i} / Σ|sim(u,v)|
```

### Strategy 2 — Item-Based Collaborative Filtering

```
1. Adjusted cosine: subtract user mean before computing item-item similarity
2. Find k most similar items to target item
3. Predict: pred(u, i) = Σ sim(i,j)·r_{u,j} / Σ|sim(i,j)|  for j rated by u
```

More stable than User-CF — item similarities change more slowly than user preferences.

### Strategy 3 — Matrix Factorisation (SVD)

```
R (120×60)  →  fill NaN with global mean  →  TruncatedSVD  →  R_hat (120×60)

R ≈ U (120×k) · Σ (k×k) · V^T (k×60)
```

- `k = n_components = 20` latent factors
- `R_hat` gives predicted ratings for all user-item pairs
- **Explained variance** shows how much rating variance k components capture

### Strategy 4 — Content-Based Filtering

```
1. Build user genre profile:
   profile(u) = mean of item feature vectors, weighted by user's ratings
   
2. Score unrated items:
   sim(profile, item_vec) = (profile · item_vec) / (||profile|| · ||item_vec||)
   
3. Scale cosine similarity [−1,1] → predicted rating [1,5]
```

### Evaluation Metrics

| Metric | Formula | Computed on |
|--------|---------|-------------|
| **RMSE** | `√(Σ(r̂−r)²/n)` | 20% held-out ratings |
| **Precision@K** | `\|relevant ∩ top-K\| / K` | All users, K=10 |
| **Recall@K** | `\|relevant ∩ top-K\| / \|relevant\|` | All users, K=10 |

*Relevant* = items with true rating ≥ 3.5

### Key Concept
> No single recommendation strategy dominates in all scenarios:  
> - **Item-CF** is most commonly deployed — stable and explainable  
> - **SVD** achieves lowest RMSE but is a black box  
> - **Content-Based** is the only option for brand-new items (zero ratings)  
> - **Hybrid systems** (combine two or more) are used in production (Netflix, Spotify)
