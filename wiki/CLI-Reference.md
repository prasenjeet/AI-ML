# CLI Reference

`main.py` is a command-line runner that executes any or all of the 14 ML concept demos.

---

## Usage

```
python main.py [--topic TOPIC] [--save-plots]
```

---

## Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--topic` | string | *(none)* | Run a single topic. Omit to run all 14. |
| `--save-plots` | flag | `False` | Save PNG charts to `./plots/` instead of displaying them interactively. |

---

## Topic Keys

### Classical ML

| Key | Topic | File |
|-----|-------|------|
| `lr` | Linear Regression | `ml_concepts/models/linear_regression.py` |
| `dt` | Decision Tree | `ml_concepts/models/decision_tree.py` |
| `km` | K-Means Clustering | `ml_concepts/models/kmeans_clustering.py` |
| `rf` | Bagging & Random Forest | `ml_concepts/models/bagging_random_forest.py` |
| `boost` | Boosting | `ml_concepts/models/boosting.py` |
| `ens` | Ensemble Methods | `ml_concepts/models/ensemble.py` |

### Neural Networks

| Key | Topic | File |
|-----|-------|------|
| `ann` | Artificial Neural Network | `ml_concepts/models/ann.py` |
| `dnn` | Deep Neural Network | `ml_concepts/models/dnn.py` |
| `cnn` | Convolutional Neural Network | `ml_concepts/models/cnn.py` |
| `rnn` | Recurrent Neural Network | `ml_concepts/models/rnn.py` |
| `lstm` | Long Short-Term Memory | `ml_concepts/models/lstm.py` |

### Applied AI

| Key | Topic | File |
|-----|-------|------|
| `nlp` | Natural Language Processing | `ml_concepts/models/nlp.py` |
| `cv` | Computer Vision | `ml_concepts/models/computer_vision.py` |
| `rec` | Recommendation Systems | `ml_concepts/models/recommendations.py` |

---

## Examples

```bash
# Run all 14 topics (display plots interactively)
python main.py

# Run all 14 topics and save plots to ./plots/
python main.py --save-plots

# Run a single classical ML topic
python main.py --topic lr
python main.py --topic dt
python main.py --topic km
python main.py --topic rf
python main.py --topic boost
python main.py --topic ens

# Run a single neural network topic
python main.py --topic ann
python main.py --topic dnn
python main.py --topic cnn
python main.py --topic rnn
python main.py --topic lstm

# Run a single applied AI topic
python main.py --topic nlp
python main.py --topic cv
python main.py --topic rec

# Run and save plots for a single topic
python main.py --topic cnn --save-plots
python main.py --topic nlp --save-plots
```

---

## Output Format

```
╔══════════════════════════════════════════════════════════╗
║              ML CONCEPTS DEMONSTRATION                   ║
╚══════════════════════════════════════════════════════════╝

▶▶▶  Linear Regression  ◀◀◀

  ============================================================
  LINEAR REGRESSION
  ============================================================

  OLS        MSE=0.0412  RMSE=0.2029  MAE=0.1601  R²=0.9588
  Ridge      MSE=0.0418  RMSE=0.2045  MAE=0.1612  R²=0.9581
  ...

  ✓ Completed in 1.3s

════════════════════════════════════════════════════════════
  All done in 42.7s
  38 plots saved in ./plots/
    • ann_confusion.png
    • ann_history.png
    • ...
════════════════════════════════════════════════════════════
```

---

## Saved Plot Files

When `--save-plots` is active, charts are written to `./plots/`:

```
plots/
├── lr_fit.png
├── lr_learning_curve.png
├── lr_comparison.png
├── dt_tree.png
├── dt_feature_importance.png
├── dt_confusion.png
├── dt_depth_comparison.png
├── km_elbow.png
├── km_clusters.png
├── km_agglomerative.png
├── km_true_labels.png
├── rf_feature_importance.png
├── rf_confusion.png
├── rf_comparison.png
├── boost_ada_stages.png
├── boost_gbm_stages.png
├── boost_feature_importance.png
├── boost_confusion.png
├── boost_comparison.png
├── ens_confusion.png
├── ens_comparison.png
├── ann_history.png
├── ann_confusion.png
├── dnn_history.png
├── dnn_confusion.png
├── cnn_history.png
├── cnn_confusion.png
├── cnn_samples.png
├── rnn_history.png
├── rnn_prediction.png
├── lstm_history.png
├── lstm_prediction.png
├── nlp_sentiment.png
├── nlp_tfidf.png
├── cv_samples.png
├── cv_augmentation.png
├── cv_hog.png
├── cv_edges.png
├── cv_history.png
├── cv_confusion.png
├── rec_matrix.png
└── rec_comparison.png
```
