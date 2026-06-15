# Getting Started

## Prerequisites

| Requirement | Version |
|-------------|--------|
| Python | 3.9 – 3.11 |
| pip | ≥ 21 |

> **Note:** `tensorflow-cpu` is used to keep the install lean. GPU support requires replacing it with `tensorflow` and appropriate CUDA drivers.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/prasenjeet/AI-ML.git
cd AI-ML

# 2. (Optional but recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### What gets installed

| Package | Purpose |
|---------|---------|
| `numpy`, `pandas` | Numerical computing and data frames |
| `scikit-learn` | Classical ML algorithms |
| `scikit-image` | Image feature extraction (HOG, edges) |
| `matplotlib`, `seaborn` | Plotting |
| `Pillow` | Image loading and augmentation |
| `nltk` | NLP tokenisation, stemming, lemmatisation, VADER |
| `streamlit` | Interactive web UI |
| `tensorflow-cpu` | ANN / DNN / CNN / RNN / LSTM / Transfer Learning |

---

## First Run

### Option A — Interactive UI (recommended)

```bash
streamlit run app.py
```

Opens at **http://localhost:8501** in your browser.  
Use the sidebar to navigate between the 11 pages.

### Option B — CLI

```bash
# Run all 14 topics and save PNG plots
python main.py --save-plots

# Run a single topic
python main.py --topic lr
```

See the [CLI Reference](CLI-Reference) for the full list of topics.

---

## NLTK Data

The NLP module downloads NLTK corpora on first use (requires an internet connection on first run):

```
nltk_data/
  corpora/stopwords
  corpora/wordnet
  tokenizers/punkt
  sentiment/vader_lexicon
  taggers/averaged_perceptron_tagger
```

After the first download these are cached locally and no further network access is needed.

---

## Dataset Downloads

| Dataset | When | Cached location |
|---------|------|-----------------|
| MNIST | First CNN/ANN run | `~/.keras/datasets/` |
| CIFAR-10 | First Computer Vision run | `~/.keras/datasets/` |
| 20 Newsgroups | First NLP run | `~/scikit_learn_data/` |

---

## Environment Variables

| Variable | Default | Effect |
|----------|---------|--------|
| `TF_CPP_MIN_LOG_LEVEL` | `3` | Suppresses TensorFlow C++ info/warning logs |

Set automatically at the top of `app.py`; no manual configuration needed.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: nltk` | Run `pip install nltk` |
| `LookupError: NLTK punkt` | Run `python -c "import nltk; nltk.download('punkt')"` |
| Streamlit page blank after clicking Train | Training is in progress — watch the progress bar |
| TF log spam in terminal | Already suppressed via `TF_CPP_MIN_LOG_LEVEL=3` |
| `OMP: Error #15` on macOS | `export KMP_DUPLICATE_LIB_OK=TRUE` |
