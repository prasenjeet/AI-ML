"""
DNN – Deep Neural Network demo
───────────────────────────────
Covers:
  • Going deeper: 3 → 8 hidden layers
  • Batch Normalisation — stable training at depth
  • Dropout regularisation — reducing overfitting
  • Learning-rate schedulers (ReduceLROnPlateau, CosineDecay)
  • Depth sweep: accuracy vs number of layers
  • Weight initialisation comparison (glorot_uniform, he_normal)
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.metrics import accuracy_score

from ml_concepts.utils.nn_utils import tabular_data, plot_history, plot_confusion


def build_dnn(n_input, n_classes, n_layers=5, units=128,
              activation="relu", dropout=0.3,
              batch_norm=True, initializer="he_normal"):
    inp = keras.Input(shape=(n_input,))
    x   = inp
    for i in range(n_layers):
        x = layers.Dense(max(units // (2 ** (i // 2)), 16),
                         activation=None,
                         kernel_initializer=initializer)(x)
        if batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        if dropout > 0 and i < n_layers - 1:
            x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs=inp, outputs=out, name="DNN")


def run(save_plots=False, plot_dir="."):
    print("\n" + "=" * 60)
    print("  DEEP NEURAL NETWORK (DNN)")
    print("=" * 60)

    X_train, X_test, y_train, y_test, n_feat, n_cls, cls_names = tabular_data("synthetic")

    # ── Baseline DNN ──────────────────────────────────────────────────────────
    print("\n  [Baseline DNN  –  5 layers, BN + Dropout=0.3]")
    model = build_dnn(n_feat, n_cls, n_layers=5, units=128)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    cbs = [
        callbacks.ReduceLROnPlateau(patience=8, factor=0.5, verbose=0),
        callbacks.EarlyStopping(patience=15, restore_best_weights=True, verbose=0),
    ]
    history = model.fit(X_train, y_train, epochs=150, batch_size=32,
                        validation_split=0.2, callbacks=cbs, verbose=0)

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(f"  Test accuracy: {accuracy_score(y_test, y_pred):.4f}")

    path_h = f"{plot_dir}/dnn_history.png" if save_plots else None
    plot_history(history, "DNN (Synthetic)", path=path_h)
    path_c = f"{plot_dir}/dnn_confusion.png" if save_plots else None
    plot_confusion(y_test, y_pred, cls_names, "DNN – Confusion Matrix", path=path_c)

    # ── Depth sweep ───────────────────────────────────────────────────────────
    print("\n  Depth sweep (n_layers):")
    depth_results = {}
    for n in (1, 2, 3, 5, 7, 10):
        m = build_dnn(n_feat, n_cls, n_layers=n)
        m.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
        m.fit(X_train, y_train, epochs=80, batch_size=32,
              validation_split=0.2, verbose=0,
              callbacks=[callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
        yp = np.argmax(m.predict(X_test, verbose=0), axis=1)
        acc = accuracy_score(y_test, yp)
        depth_results[n] = acc
        bar = "█" * int(acc * 30)
        print(f"    layers={n:>2}  acc={acc:.4f}  {bar}")
        keras.backend.clear_session()

    # ── Batch Norm ablation ───────────────────────────────────────────────────
    print("\n  Batch Normalisation effect (5 layers):")
    for bn in (True, False):
        m = build_dnn(n_feat, n_cls, n_layers=5, batch_norm=bn)
        m.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
        m.fit(X_train, y_train, epochs=80, batch_size=32,
              validation_split=0.2, verbose=0)
        yp = np.argmax(m.predict(X_test, verbose=0), axis=1)
        print(f"    BN={str(bn):<5}  acc={accuracy_score(y_test, yp):.4f}")
        keras.backend.clear_session()

    # ── Dropout sweep ─────────────────────────────────────────────────────────
    print("\n  Dropout rate sweep (5 layers):")
    for dr in (0.0, 0.1, 0.2, 0.3, 0.5):
        m = build_dnn(n_feat, n_cls, n_layers=5, dropout=dr)
        m.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
        m.fit(X_train, y_train, epochs=80, batch_size=32,
              validation_split=0.2, verbose=0)
        yp = np.argmax(m.predict(X_test, verbose=0), axis=1)
        print(f"    dropout={dr}  acc={accuracy_score(y_test, yp):.4f}")
        keras.backend.clear_session()

    # ── Initialiser comparison ────────────────────────────────────────────────
    print("\n  Weight initialiser comparison:")
    for init in ("glorot_uniform", "he_normal", "he_uniform", "lecun_normal"):
        m = build_dnn(n_feat, n_cls, n_layers=5, initializer=init)
        m.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
        m.fit(X_train, y_train, epochs=80, batch_size=32,
              validation_split=0.2, verbose=0)
        yp = np.argmax(m.predict(X_test, verbose=0), axis=1)
        print(f"    {init:<20}  acc={accuracy_score(y_test, yp):.4f}")
        keras.backend.clear_session()

    print("\n  Key takeaways:")
    print("    • Batch Norm stabilises training of deep networks (internal covariate shift)")
    print("    • Dropout is the simplest & most effective regulariser for DNNs")
    print("    • He initialisation suits ReLU; Glorot suits tanh/sigmoid")
    print("    • ReduceLROnPlateau + EarlyStopping prevent wasted training epochs")
