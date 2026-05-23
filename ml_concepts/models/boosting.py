"""
Boosting demo
─────────────
Covers:
  • AdaBoost  – adaptive re-weighting of misclassified samples
  • Gradient Boosting (sklearn GBM) – gradient descent in function space
  • HistGradientBoosting – histogram-based fast GBM
  • Stagewise error curves (n_estimators vs test error)
  • Learning-rate sweep
  • Feature importance from GBM
"""

import numpy as np
from sklearn.ensemble import (
    AdaBoostClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import cross_val_score

from ml_concepts.utils.data_generator import get_classification_data, get_wine_data
from ml_concepts.utils.visualizer import (
    plot_boosting_stages, plot_model_comparison,
    plot_feature_importance, plot_confusion,
)


def _staged_error(model, X_test, y_test):
    """Return per-stage test error for AdaBoost / GBM."""
    errors = []
    for y_pred in model.staged_predict(X_test):
        errors.append(1.0 - accuracy_score(y_test, y_pred))
    return np.array(errors)


def run(save_plots=False, plot_dir="."):
    print("\n" + "=" * 60)
    print("  BOOSTING")
    print("=" * 60)

    (X_train, X_test, y_train, y_test), feat_names, target_names = get_wine_data()
    target_names = list(target_names)

    # ── AdaBoost ─────────────────────────────────────────────────────────
    print("\n  [AdaBoost]")
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),   # stumps
        n_estimators=200, learning_rate=0.5, random_state=42,
    )
    ada.fit(X_train, y_train)
    ada_acc = accuracy_score(y_test, ada.predict(X_test))
    print(f"    Accuracy          : {ada_acc:.4f}")
    cv = cross_val_score(ada, X_train, y_train, cv=5)
    print(f"    CV mean ± std     : {cv.mean():.4f} ± {cv.std():.4f}")

    # staged error
    ada_errors = _staged_error(ada, X_test, y_test)
    ada_train_errors = np.array([1. - accuracy_score(y_train, p)
                                  for p in ada.staged_predict(X_train)])

    # ── Gradient Boosting ────────────────────────────────────────────────
    print("\n  [Gradient Boosting]")
    gbm = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3,
        subsample=0.8, random_state=42,
    )
    gbm.fit(X_train, y_train)
    gbm_acc = accuracy_score(y_test, gbm.predict(X_test))
    print(f"    Accuracy          : {gbm_acc:.4f}")
    print(f"    Train loss        : {gbm.train_score_[-1]:.4f}")

    gbm_errors = _staged_error(gbm, X_test, y_test)
    gbm_train_errors = np.array([1. - accuracy_score(y_train, p)
                                  for p in gbm.staged_predict(X_train)])

    # staged error plots
    stages = np.arange(1, len(ada_errors) + 1)
    path_ada = f"{plot_dir}/boost_ada_stages.png" if save_plots else None
    plot_boosting_stages(stages, ada_train_errors, ada_errors,
                         "AdaBoost – Stagewise Error", path=path_ada)

    stages_gbm = np.arange(1, len(gbm_errors) + 1)
    path_gbm = f"{plot_dir}/boost_gbm_stages.png" if save_plots else None
    plot_boosting_stages(stages_gbm, gbm_train_errors, gbm_errors,
                         "Gradient Boosting – Stagewise Error", path=path_gbm)

    # ── HistGradientBoosting (fast) ──────────────────────────────────────
    print("\n  [HistGradientBoosting]")
    hgbm = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=4, random_state=42,
    )
    hgbm.fit(X_train, y_train)
    hgbm_acc = accuracy_score(y_test, hgbm.predict(X_test))
    print(f"    Accuracy          : {hgbm_acc:.4f}")

    # ── Feature importance (GBM) ─────────────────────────────────────────
    path_fi = f"{plot_dir}/boost_feature_importance.png" if save_plots else None
    plot_feature_importance(
        gbm.feature_importances_, list(feat_names),
        "GBM – Feature Importances (Wine)", path=path_fi,
    )

    # ── Confusion (best model) ───────────────────────────────────────────
    best_model = max([(ada, ada_acc), (gbm, gbm_acc), (hgbm, hgbm_acc)],
                     key=lambda x: x[1])[0]
    path_cm = f"{plot_dir}/boost_confusion.png" if save_plots else None
    plot_confusion(y_test, best_model.predict(X_test), target_names,
                   "Boosting Best Model – Confusion Matrix", path=path_cm)

    # ── Learning-rate sweep ──────────────────────────────────────────────
    print("\n  Learning-rate sweep (GBM, 100 trees):")
    lr_results = {}
    for lr in (0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0):
        m = GradientBoostingClassifier(n_estimators=100, learning_rate=lr,
                                       max_depth=3, random_state=42)
        m.fit(X_train, y_train)
        acc = accuracy_score(y_test, m.predict(X_test))
        lr_results[lr] = acc
        print(f"    lr={lr:<5}  acc={acc:.4f}")

    # ── Comparison ───────────────────────────────────────────────────────
    names_cmp = ["AdaBoost", "GradientBoosting", "HistGradBoost"]
    accs_cmp  = [ada_acc, gbm_acc, hgbm_acc]
    path_cmp  = f"{plot_dir}/boost_comparison.png" if save_plots else None
    plot_model_comparison(names_cmp, accs_cmp, metric="Accuracy",
                          title="Boosting Algorithms – Comparison", path=path_cmp)

    print("\n  Key takeaways:")
    print("    • AdaBoost re-weights hard examples; sensitive to noisy labels")
    print("    • GBM fits residuals (negative gradient) sequentially")
    print("    • Lower learning rate + more trees → better generalisation (with cost)")
    print("    • HistGBM bins continuous features → 10–100× faster on large datasets")
