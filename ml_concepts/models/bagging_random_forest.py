"""
Bagging & Random Forest demo
─────────────────────────────
Covers:
  • BaggingClassifier (base: Decision Tree) – shows variance reduction
  • RandomForestClassifier with hyperparameter tuning
  • Out-of-bag (OOB) error
  • Extra-Trees (ExtraTreesClassifier)
  • Feature importance from the forest
  • Bias-variance decomposition illustration (train vs test error curves)
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier
)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from ml_concepts.utils.data_generator import get_classification_data, get_wine_data
from ml_concepts.utils.visualizer import (
    plot_feature_importance, plot_model_comparison, plot_confusion,
)


def _n_estimators_sweep(Model, X_tr, y_tr, X_te, y_te, ns):
    train_accs, test_accs = [], []
    for n in ns:
        m = Model(n_estimators=n, random_state=42, n_jobs=-1)
        m.fit(X_tr, y_tr)
        train_accs.append(accuracy_score(y_tr, m.predict(X_tr)))
        test_accs.append(accuracy_score(y_te, m.predict(X_te)))
    return np.array(train_accs), np.array(test_accs)


def run(save_plots=False, plot_dir="."):
    print("\n" + "=" * 60)
    print("  BAGGING & RANDOM FOREST")
    print("=" * 60)

    # Wine dataset (scaled) – moderate difficulty
    (X_train, X_test, y_train, y_test), feat_names, target_names = get_wine_data()
    target_names = list(target_names)

    # ── Bagging ──────────────────────────────────────────────────────────
    print("\n  [Bagging – base: Decision Tree]")
    single_dt = DecisionTreeClassifier(random_state=42)
    single_dt.fit(X_train, y_train)
    dt_acc = accuracy_score(y_test, single_dt.predict(X_test))
    print(f"    Single DT accuracy    : {dt_acc:.4f}")

    bag = BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=100, max_samples=0.8, max_features=0.8,
        bootstrap=True, oob_score=True, random_state=42, n_jobs=-1,
    )
    bag.fit(X_train, y_train)
    bag_acc = accuracy_score(y_test, bag.predict(X_test))
    print(f"    Bagging accuracy      : {bag_acc:.4f}")
    print(f"    Bagging OOB score     : {bag.oob_score_:.4f}")
    print(f"    Variance reduced by   : {(dt_acc - bag_acc):.4f} (test acc Δ)")

    # ── Random Forest ────────────────────────────────────────────────────
    print("\n  [Random Forest]")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, max_features="sqrt",
        oob_score=True, random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    y_pred_rf = rf.predict(X_test)
    print(f"    RF test accuracy      : {rf_acc:.4f}")
    print(f"    RF OOB score          : {rf.oob_score_:.4f}")
    cv = cross_val_score(rf, X_train, y_train, cv=5)
    print(f"    CV mean ± std         : {cv.mean():.4f} ± {cv.std():.4f}")

    # ── Feature importance ───────────────────────────────────────────────
    path_fi = f"{plot_dir}/rf_feature_importance.png" if save_plots else None
    plot_feature_importance(
        rf.feature_importances_, list(feat_names),
        "Random Forest – Feature Importances (Wine)", path=path_fi,
    )

    # ── Confusion matrix ─────────────────────────────────────────────────
    path_cm = f"{plot_dir}/rf_confusion.png" if save_plots else None
    plot_confusion(y_test, y_pred_rf, target_names,
                   "Random Forest – Confusion Matrix", path=path_cm)

    # ── ExtraTrees ───────────────────────────────────────────────────────
    print("\n  [Extra-Trees]")
    et = ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    et.fit(X_train, y_train)
    et_acc = accuracy_score(y_test, et.predict(X_test))
    print(f"    ExtraTrees accuracy   : {et_acc:.4f}")

    # ── n_estimators sweep ───────────────────────────────────────────────
    ns = [1, 5, 10, 25, 50, 100, 200]
    print(f"\n  n_estimators sweep (RF):")
    train_accs, test_accs = _n_estimators_sweep(
        RandomForestClassifier, X_train, y_train, X_test, y_test, ns
    )
    for n, tr, te in zip(ns, train_accs, test_accs):
        print(f"    n={n:>3}  train={tr:.4f}  test={te:.4f}")

    # ── max_features sweep ───────────────────────────────────────────────
    print("\n  max_features sweep (RF, 100 trees):")
    mf_results = {}
    for mf in ("sqrt", "log2", None, 0.3, 0.6):
        m = RandomForestClassifier(n_estimators=100, max_features=mf,
                                   random_state=42, n_jobs=-1)
        m.fit(X_train, y_train)
        acc = accuracy_score(y_test, m.predict(X_test))
        mf_results[str(mf)] = acc
        print(f"    max_features={str(mf):<6}  acc={acc:.4f}")

    # ── Summary comparison ───────────────────────────────────────────────
    names_cmp = ["Single DT", "Bagging(100)", "RandomForest(200)", "ExtraTrees(200)"]
    accs_cmp  = [dt_acc, bag_acc, rf_acc, et_acc]
    path_cmp  = f"{plot_dir}/rf_comparison.png" if save_plots else None
    plot_model_comparison(names_cmp, accs_cmp, metric="Accuracy",
                          title="Bagging / Forest Comparison", path=path_cmp)

    print("\n  Key takeaways:")
    print("    • Bagging reduces variance by averaging many high-variance models")
    print("    • Random Forest adds feature-level randomness → further decorrelates trees")
    print("    • ExtraTrees randomises split thresholds too → fastest, often comparable")
    print("    • OOB score is a reliable free CV estimate (~66 % of data per tree)")
