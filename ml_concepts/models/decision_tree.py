"""
Decision Tree demo
──────────────────
Covers:
  • Classification tree (Gini / Entropy)
  • Regression tree
  • Pruning via max_depth and ccp_alpha (cost-complexity)
  • Feature importance
  • Visualisation of the tree structure
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

from ml_concepts.utils.data_generator import get_classification_data, get_regression_data, get_iris_data
from ml_concepts.utils.visualizer import (
    plot_decision_tree, plot_feature_importance,
    plot_model_comparison, plot_confusion,
)


def _depth_sweep(X_train, y_train, X_test, y_test, depths):
    """Return accuracy for each max_depth."""
    accs = []
    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, random_state=42)
        clf.fit(X_train, y_train)
        accs.append(accuracy_score(y_test, clf.predict(X_test)))
    return accs


def run(save_plots=False, plot_dir="."):
    print("\n" + "=" * 60)
    print("  DECISION TREE")
    print("=" * 60)

    # ── Classification on Iris ───────────────────────────────────────────
    (X_train, X_test, y_train, y_test), feat_names, target_names = get_iris_data()
    target_names = list(target_names)

    print("\n  [Classification – Iris dataset]")
    clf = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"    Accuracy  : {acc:.4f}")
    cv = cross_val_score(clf, X_train, y_train, cv=5)
    print(f"    CV mean   : {cv.mean():.4f}  ±{cv.std():.4f}")

    path_tree = f"{plot_dir}/dt_tree.png" if save_plots else None
    plot_decision_tree(clf, feat_names, target_names,
                       title="Decision Tree (Iris, max_depth=4)", path=path_tree)

    path_fi = f"{plot_dir}/dt_feature_importance.png" if save_plots else None
    plot_feature_importance(clf.feature_importances_, feat_names,
                            "Decision Tree – Feature Importances", path=path_fi)

    path_cm = f"{plot_dir}/dt_confusion.png" if save_plots else None
    plot_confusion(y_test, y_pred, target_names, "Decision Tree – Confusion Matrix", path=path_cm)

    # ── Depth sweep ──────────────────────────────────────────────────────
    depths = list(range(1, 11))
    accs = _depth_sweep(X_train, y_train, X_test, y_test, depths)
    print("\n  Depth sweep (test accuracy):")
    for d, a in zip(depths, accs):
        bar = "█" * int(a * 30)
        print(f"    depth={d:2d}  {a:.4f}  {bar}")

    # ── Criterion comparison ─────────────────────────────────────────────
    results = {}
    for crit in ("gini", "entropy"):
        m = DecisionTreeClassifier(criterion=crit, max_depth=4, random_state=42)
        m.fit(X_train, y_train)
        results[crit] = accuracy_score(y_test, m.predict(X_test))
    print(f"\n  Gini accuracy   : {results['gini']:.4f}")
    print(f"  Entropy accuracy: {results['entropy']:.4f}")

    # ── Cost-complexity pruning (CCP) ────────────────────────────────────
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    alphas = path.ccp_alphas[::max(1, len(path.ccp_alphas) // 8)]
    pruning_accs = []
    for alpha in alphas:
        m = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
        m.fit(X_train, y_train)
        pruning_accs.append(accuracy_score(y_test, m.predict(X_test)))
    print("\n  CCP pruning (alpha → accuracy):")
    for a, acc in zip(alphas, pruning_accs):
        print(f"    α={a:.5f}  acc={acc:.4f}")

    # ── Regression tree ──────────────────────────────────────────────────
    print("\n  [Regression tree]")
    X_tr, X_te, y_tr, y_te = get_regression_data()
    reg = DecisionTreeRegressor(max_depth=5, random_state=42)
    reg.fit(X_tr, y_tr)
    y_pred_r = reg.predict(X_te)
    mse = mean_squared_error(y_te, y_pred_r)
    print(f"    MSE  : {mse:.4f}   RMSE: {np.sqrt(mse):.4f}")
    print(f"    R²   : {r2_score(y_te, y_pred_r):.4f}")

    # ── Summary comparison ───────────────────────────────────────────────
    names_cmp = [f"depth={d}" for d in depths[:6]]
    path_cmp = f"{plot_dir}/dt_depth_comparison.png" if save_plots else None
    plot_model_comparison(names_cmp, accs[:6], metric="Accuracy",
                          title="Decision Tree – Effect of max_depth", path=path_cmp)

    print("\n  Key takeaways:")
    print("    • Deep trees overfit; pruning (max_depth / ccp_alpha) improves generalisation")
    print("    • Gini and Entropy criteria produce similar trees on most datasets")
    print("    • Feature importance reveals which splits carry the most information gain")
