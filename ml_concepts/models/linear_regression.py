"""
Linear Regression demo
──────────────────────
Covers:
  • Simple OLS regression with scikit-learn
  • Polynomial feature expansion (degree 2)
  • Ridge & Lasso regularisation
  • Metrics: MSE, RMSE, MAE, R²
  • Learning-curve analysis
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import learning_curve

from ml_concepts.utils.data_generator import get_regression_data, scale_data
from ml_concepts.utils.visualizer import plot_regression_fit, plot_learning_curve, plot_model_comparison


def _metrics(y_true, y_pred, label):
    mse = mean_squared_error(y_true, y_pred)
    print(f"  [{label}]")
    print(f"    MSE  : {mse:.4f}")
    print(f"    RMSE : {np.sqrt(mse):.4f}")
    print(f"    MAE  : {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"    R²   : {r2_score(y_true, y_pred):.4f}")
    return r2_score(y_true, y_pred)


def run(save_plots=False, plot_dir="."):
    print("\n" + "=" * 60)
    print("  LINEAR REGRESSION")
    print("=" * 60)

    X_train, X_test, y_train, y_test = get_regression_data()
    X_train_s, X_test_s = scale_data(X_train, X_test)

    models = {
        "OLS Linear": LinearRegression(),
        "Ridge (α=1)": Ridge(alpha=1.0),
        "Lasso (α=0.1)": Lasso(alpha=0.1, max_iter=5000),
        "Polynomial (deg 2)": Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("lr", LinearRegression()),
        ]),
    }

    names, r2s = [], []
    for name, model in models.items():
        X_tr = X_train_s if name != "Polynomial (deg 2)" else X_train
        X_te = X_test_s if name != "Polynomial (deg 2)" else X_test
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        r2 = _metrics(y_test, y_pred, name)
        names.append(name); r2s.append(max(r2, 0))

    # best model plots
    best = LinearRegression().fit(X_train_s, y_train)
    y_pred_best = best.predict(X_test_s)

    path_fit = f"{plot_dir}/lr_fit.png" if save_plots else None
    plot_regression_fit(y_test, y_pred_best, "OLS – Predicted vs Actual", path=path_fit)

    # learning curve
    train_sz, tr_sc, val_sc = learning_curve(
        LinearRegression(), X_train_s, y_train,
        cv=5, train_sizes=np.linspace(0.1, 1.0, 8), scoring="r2",
    )
    path_lc = f"{plot_dir}/lr_learning_curve.png" if save_plots else None
    plot_learning_curve(train_sz, tr_sc, val_sc, "OLS Learning Curve", path=path_lc)

    # comparison bar
    path_cmp = f"{plot_dir}/lr_comparison.png" if save_plots else None
    plot_model_comparison(names, r2s, metric="R² Score",
                          title="Linear Regression Variants – R²", path=path_cmp)

    print("\n  Coefficients (OLS):", np.round(best.coef_, 3))
    print("  Intercept         :", round(float(best.intercept_), 3))
    print("\n  Key takeaways:")
    print("    • Ridge shrinks all coefficients → handles multicollinearity")
    print("    • Lasso zeros out less-important features → built-in selection")
    print("    • Polynomial expansion captures non-linear relationships")
