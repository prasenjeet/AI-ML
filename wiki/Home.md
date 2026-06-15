# ML Concepts — Wiki Home

Welcome to the **ML Concepts** project wiki. This project demonstrates **14 machine learning and applied AI techniques** through an interactive Streamlit UI and a CLI runner.

---

## Navigation

| Page | What's inside |
|------|---------------|
| [Getting Started](Getting-Started) | Installation, Quick Start, Environment |
| [Classical ML](Classical-ML) | Linear Regression · Decision Tree · K-Means · Bagging/RF · Boosting · Ensemble |
| [Neural Networks](Neural-Networks) | ANN · DNN · CNN · RNN · LSTM |
| [Applied AI](Applied-AI) | NLP · Computer Vision · Recommendation Systems |
| [Interactive UI Guide](Interactive-UI-Guide) | Streamlit page-by-page walkthrough |
| [CLI Reference](CLI-Reference) | `main.py` flags, topics, and examples |

---

## Project Overview

```
AI-ML/
├── app.py                      # Streamlit interactive UI  (11 pages)
├── main.py                     # CLI runner               (14 topics)
├── requirements.txt
├── wiki/                       # ← you are here
└── ml_concepts/
    ├── utils/
    │   ├── data_generator.py
    │   ├── visualizer.py
    │   └── nn_utils.py
    └── models/
        ├── linear_regression.py
        ├── decision_tree.py
        ├── kmeans_clustering.py
        ├── bagging_random_forest.py
        ├── boosting.py
        ├── ensemble.py
        ├── ann.py
        ├── dnn.py
        ├── cnn.py
        ├── rnn.py
        ├── lstm.py
        ├── nlp.py
        ├── computer_vision.py
        └── recommendations.py
```

---

## Concept Map

```
ML Concepts
│
├── Classical ML (sklearn)
│   ├── Supervised Learning
│   │   ├── Linear Regression  ──── OLS / Ridge / Lasso / Polynomial
│   │   └── Decision Tree      ──── Gini / Entropy / Pruning
│   ├── Unsupervised Learning
│   │   └── K-Means            ──── Elbow / Silhouette / Agglomerative
│   └── Ensemble Methods
│       ├── Bagging / Random Forest
│       ├── Boosting           ──── AdaBoost / GBM / HistGBM
│       └── Voting / Stacking
│
├── Neural Networks (TensorFlow/Keras)
│   ├── Feedforward
│   │   ├── ANN  ──── MLP, activation comparison
│   │   └── DNN  ──── BatchNorm, dropout, weight init
│   ├── Image
│   │   └── CNN  ──── Conv2D, MaxPool, GAP  (MNIST)
│   └── Sequential
│       ├── RNN  ──── SimpleRNN  (time series)
│       └── LSTM ──── Gated cells, GRU, Bidirectional
│
└── Applied AI
    ├── NLP    ──── Preprocessing / TF-IDF / Sentiment / LDA
    ├── CV     ──── HOG / SVM / CNN / Transfer Learning  (CIFAR-10)
    └── RecSys ──── User-CF / Item-CF / SVD / Content-Based
```

---

## Quick Links

- **Run the UI:** `streamlit run app.py`
- **Run everything:** `python main.py --save-plots`
- **Single topic:** `python main.py --topic nlp`
- **README:** see `README.md` in the repo root
