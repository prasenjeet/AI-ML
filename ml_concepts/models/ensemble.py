"""
Ensemble Methods demo
──────────────────────
Covers:
  • Hard voting (majority vote)
  • Soft voting (probability averaging)
  • Stacking (meta-learner on base predictions)
  • Weighted voting
  • Diversity analysis (pairwise agreement between base models)
  • Comparison: individual models vs all ensemble strategies
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier,
    AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

from ml_concepts.utils.data_generator import get_classification_data, get_wine_data
from ml_concepts.utils.visualizer import plot_model_comparison, plot_confusion


def _pairwise_agreement(models, X_test):
    """Return n×n matrix of pairwise prediction agreements."""
    preds = np.array([m.predict(X_test) for m in models])
    n = len(models)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i, j] = (preds[i] == preds[j]).mean()
    return mat


def run(save_plots=False, plot_dir="."):
    print("\n" + "=" * 60)
    print("  ENSEMBLE METHODS")
    print("=" * 60)

    (X_train, X_test, y_train, y_test), feat_names, target_names = get_wine_data()
    target_names = list(target_names)

    # ── Define base learners ─────────────────────────────────────────────
    base_learners = [
        ("dt",  DecisionTreeClassifier(max_depth=4, random_state=42)),
        ("rf",  RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ("gbm", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ("knn", KNeighborsClassifier(n_neighbors=7)),
        ("nb",  GaussianNB()),
        ("svm", SVC(probability=True, random_state=42)),
    ]

    print("\n  [Individual base learner accuracies]")
    individual_accs = {}
    fitted_bases = []
    for name, model in base_learners:
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        individual_accs[name] = acc
        fitted_bases.append(model)
        print(f"    {name:<6}  acc={acc:.4f}")

    # ── Pairwise agreement (diversity) ───────────────────────────────────
    print("\n  Pairwise agreement matrix (higher diagonal expected):")
    agreement = _pairwise_agreement(fitted_bases, X_test)
    names_short = [n for n, _ in base_learners]
    header = "       " + "  ".join(f"{n:<6}" for n in names_short)
    print(f"  {header}")
    for i, n in enumerate(names_short):
        row = "  ".join(f"{agreement[i,j]:.3f}" for j in range(len(names_short)))
        print(f"    {n:<6}  {row}")

    # ── Hard voting ──────────────────────────────────────────────────────
    print("\n  [Hard Voting Classifier]")
    hard_vote = VotingClassifier(estimators=base_learners, voting="hard", n_jobs=-1)
    hard_vote.fit(X_train, y_train)
    hard_acc = accuracy_score(y_test, hard_vote.predict(X_test))
    print(f"    Hard voting acc  : {hard_acc:.4f}")

    # ── Soft voting ──────────────────────────────────────────────────────
    print("\n  [Soft Voting Classifier]")
    soft_vote = VotingClassifier(estimators=base_learners, voting="soft", n_jobs=-1)
    soft_vote.fit(X_train, y_train)
    soft_acc = accuracy_score(y_test, soft_vote.predict(X_test))
    print(f"    Soft voting acc  : {soft_acc:.4f}")

    # ── Weighted soft voting ─────────────────────────────────────────────
    print("\n  [Weighted Soft Voting]")
    # weights proportional to individual accuracy
    weights = [individual_accs[n] for n, _ in base_learners]
    weighted_vote = VotingClassifier(
        estimators=base_learners, voting="soft",
        weights=weights, n_jobs=-1,
    )
    weighted_vote.fit(X_train, y_train)
    weighted_acc = accuracy_score(y_test, weighted_vote.predict(X_test))
    print(f"    Weighted soft acc: {weighted_acc:.4f}")

    # ── Stacking ─────────────────────────────────────────────────────────
    print("\n  [Stacking Classifier  –  meta-learner: LogisticRegression]")
    stack = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        stack_method="predict_proba",
        n_jobs=-1,
    )
    stack.fit(X_train, y_train)
    stack_acc = accuracy_score(y_test, stack.predict(X_test))
    print(f"    Stacking acc     : {stack_acc:.4f}")
    cv_stack = cross_val_score(stack, X_train, y_train, cv=5)
    print(f"    CV mean ± std    : {cv_stack.mean():.4f} ± {cv_stack.std():.4f}")

    # ── Stacking with RF meta-learner ─────────────────────────────────────
    print("\n  [Stacking  –  meta-learner: RandomForest]")
    stack_rf = StackingClassifier(
        estimators=base_learners,
        final_estimator=RandomForestClassifier(n_estimators=50, random_state=42),
        cv=5, stack_method="predict_proba", n_jobs=-1,
    )
    stack_rf.fit(X_train, y_train)
    stack_rf_acc = accuracy_score(y_test, stack_rf.predict(X_test))
    print(f"    Stacking(RF) acc : {stack_rf_acc:.4f}")

    # ── Confusion matrix of best ensemble ────────────────────────────────
    best_ensemble = max(
        [(hard_vote, hard_acc), (soft_vote, soft_acc),
         (weighted_vote, weighted_acc), (stack, stack_acc), (stack_rf, stack_rf_acc)],
        key=lambda x: x[1],
    )[0]
    path_cm = f"{plot_dir}/ens_confusion.png" if save_plots else None
    plot_confusion(y_test, best_ensemble.predict(X_test), target_names,
                   "Best Ensemble – Confusion Matrix", path=path_cm)

    # ── Full comparison bar chart ────────────────────────────────────────
    all_names = (
        [f"Base:{n}" for n in names_short]
        + ["Hard Vote", "Soft Vote", "Weighted Vote", "Stack(LR)", "Stack(RF)"]
    )
    all_accs = (
        list(individual_accs.values())
        + [hard_acc, soft_acc, weighted_acc, stack_acc, stack_rf_acc]
    )
    path_cmp = f"{plot_dir}/ens_comparison.png" if save_plots else None
    plot_model_comparison(all_names, all_accs, metric="Accuracy",
                          title="Ensemble Methods – Full Comparison", path=path_cmp)

    print("\n  Key takeaways:")
    print("    • Ensemble strength depends on base-learner DIVERSITY, not just accuracy")
    print("    • Soft voting > hard voting when base models are well-calibrated")
    print("    • Stacking learns optimal combination weights automatically")
    print("    • Adding weak/redundant models rarely hurts but diverse models help a lot")
