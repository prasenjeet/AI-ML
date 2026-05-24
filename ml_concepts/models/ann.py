"""
ANN – Artificial Neural Network demo
──────────────────────────────────────
Covers:
  • Simple feedforward MLP for tabular classification (Iris / Wine / Synthetic)
  • Activation function comparison  (relu, tanh, sigmoid, elu)
  • Optimizer comparison  (adam, sgd, rmsprop)
  • Depth & width sweeps
  • Training history, confusion matrix, classification report
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, classification_report

from ml_concepts.utils.nn_utils import (
    tabular_data, plot_history, plot_confusion,
)


def build_ann(n_input, n_classes, hidden_units=(64, 32),
              activation="relu", dropout=0.0):
    model = keras.Sequential(name="ANN")
    model.add(layers.Input(shape=(n_input,)))
    for units in hidden_units:
        model.add(layers.Dense(units, activation=activation))
        if dropout > 0:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(n_classes, activation="softmax"))
    return model


def run(save_plots=False, plot_dir="."):
    print("\n" + "=" * 60)
    print("  ARTIFICIAL NEURAL NETWORK (ANN)")
    print("=" * 60)

    X_train, X_test, y_train, y_test, n_feat, n_cls, cls_names = tabular_data("wine")

    # ── Baseline ANN ──────────────────────────────────────────────────────────
    print("\n  [Baseline ANN  –  Wine dataset, 2 hidden layers (64, 32)]")
    model = build_ann(n_feat, n_cls, hidden_units=(64, 32), activation="relu")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    print(model.summary())

    history = model.fit(X_train, y_train, epochs=80, batch_size=16,
                        validation_split=0.2, verbose=0)

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=cls_names))

    path_h = f"{plot_dir}/ann_history.png" if save_plots else None
    plot_history(history, "ANN (Wine)", path=path_h)

    path_c = f"{plot_dir}/ann_confusion.png" if save_plots else None
    plot_confusion(y_test, y_pred, cls_names, "ANN – Confusion Matrix", path=path_c)

    # ── Activation comparison ─────────────────────────────────────────────────
    print("\n  Activation function comparison:")
    for act in ("relu", "tanh", "sigmoid", "elu"):
        m = build_ann(n_feat, n_cls, activation=act)
        m.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
        m.fit(X_train, y_train, epochs=60, batch_size=16,
              validation_split=0.2, verbose=0)
        yp = np.argmax(m.predict(X_test, verbose=0), axis=1)
        print(f"    {act:<10}  acc={accuracy_score(y_test, yp):.4f}")
        keras.backend.clear_session()

    # ── Optimizer comparison ──────────────────────────────────────────────────
    print("\n  Optimizer comparison:")
    for opt_name, opt in [("adam", "adam"), ("sgd", keras.optimizers.SGD(0.01)),
                          ("rmsprop", "rmsprop")]:
        m = build_ann(n_feat, n_cls)
        m.compile(optimizer=opt, loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
        m.fit(X_train, y_train, epochs=60, batch_size=16,
              validation_split=0.2, verbose=0)
        yp = np.argmax(m.predict(X_test, verbose=0), axis=1)
        print(f"    {opt_name:<10}  acc={accuracy_score(y_test, yp):.4f}")
        keras.backend.clear_session()

    # ── Architecture sweep ────────────────────────────────────────────────────
    print("\n  Hidden layer width sweep:")
    for width in (16, 32, 64, 128, 256):
        m = build_ann(n_feat, n_cls, hidden_units=(width, width // 2))
        m.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
        m.fit(X_train, y_train, epochs=60, batch_size=16,
              validation_split=0.2, verbose=0)
        yp = np.argmax(m.predict(X_test, verbose=0), axis=1)
        print(f"    width={width:<4}  acc={accuracy_score(y_test, yp):.4f}")
        keras.backend.clear_session()

    print("\n  Key takeaways:")
    print("    • ReLU is the default activation — avoids vanishing gradients")
    print("    • Adam converges faster than plain SGD on most problems")
    print("    • More neurons ≠ always better — regularise with Dropout")
