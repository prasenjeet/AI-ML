"""
CNN – Convolutional Neural Network demo
─────────────────────────────────────────
Covers:
  • Conv2D → BN → MaxPooling2D → Flatten → Dense pipeline on MNIST
  • Filter count sweep  (8 → 64 per layer)
  • Kernel size comparison  (3×3 vs 5×5)
  • Global Average Pooling vs Flatten
  • Data augmentation (random flip, rotation)
  • Prediction samples with correct / incorrect labels
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, classification_report

from ml_concepts.utils.nn_utils import (
    load_mnist_subset, plot_history, plot_confusion, plot_cnn_samples,
)

DIGITS = [str(i) for i in range(10)]


def build_cnn(filters1=32, filters2=64, kernel_size=3,
              dense_units=128, dropout=0.3, use_gap=False):
    inp = keras.Input(shape=(28, 28, 1))
    x   = inp

    # Block 1
    x = layers.Conv2D(filters1, kernel_size, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters1, kernel_size, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(filters2, kernel_size, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters2, kernel_size, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Head
    if use_gap:
        x = layers.GlobalAveragePooling2D()(x)
    else:
        x = layers.Flatten()(x)
    x   = layers.Dense(dense_units, activation="relu")(x)
    x   = layers.Dropout(dropout)(x)
    out = layers.Dense(10, activation="softmax")(x)

    return keras.Model(inputs=inp, outputs=out, name="CNN")


def run(save_plots=False, plot_dir="."):
    print("\n" + "=" * 60)
    print("  CONVOLUTIONAL NEURAL NETWORK (CNN)")
    print("=" * 60)

    X_train, y_train, X_test, y_test = load_mnist_subset()
    print(f"  Training: {X_train.shape}   Test: {X_test.shape}")

    # ── Baseline CNN ──────────────────────────────────────────────────────────
    print("\n  [Baseline CNN  –  MNIST subset (8k/2k), filters=(32,64)]")
    model = build_cnn(filters1=32, filters2=64, kernel_size=3, dense_units=128)
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    history = model.fit(X_train, y_train, epochs=10, batch_size=128,
                        validation_split=0.15, verbose=1)

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=DIGITS))

    path_h = f"{plot_dir}/cnn_history.png" if save_plots else None
    plot_history(history, "CNN (MNIST)", path=path_h)

    path_cm = f"{plot_dir}/cnn_confusion.png" if save_plots else None
    plot_confusion(y_test, y_pred, DIGITS, "CNN – Confusion Matrix", path=path_cm)

    path_s = f"{plot_dir}/cnn_samples.png" if save_plots else None
    plot_cnn_samples(X_test, y_test, y_pred, DIGITS, n=10, path=path_s)

    keras.backend.clear_session()

    # ── Filter count sweep ────────────────────────────────────────────────────
    print("\n  Filter count sweep (filters1, filters2):")
    for f1, f2 in [(8, 16), (16, 32), (32, 64), (64, 128)]:
        m = build_cnn(filters1=f1, filters2=f2)
        m.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
        m.fit(X_train, y_train, epochs=6, batch_size=128,
              validation_split=0.15, verbose=0)
        yp = np.argmax(m.predict(X_test, verbose=0), axis=1)
        params = m.count_params()
        print(f"    filters=({f1},{f2})  acc={accuracy_score(y_test,yp):.4f}  params={params:,}")
        keras.backend.clear_session()

    # ── Kernel size comparison ────────────────────────────────────────────────
    print("\n  Kernel size comparison:")
    for ks in (3, 5):
        m = build_cnn(kernel_size=ks)
        m.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
        m.fit(X_train, y_train, epochs=6, batch_size=128,
              validation_split=0.15, verbose=0)
        yp = np.argmax(m.predict(X_test, verbose=0), axis=1)
        print(f"    kernel={ks}x{ks}  acc={accuracy_score(y_test,yp):.4f}")
        keras.backend.clear_session()

    # ── GAP vs Flatten ────────────────────────────────────────────────────────
    print("\n  GlobalAveragePooling vs Flatten:")
    for use_gap, label in [(False, "Flatten"), (True, "GlobalAvgPool")]:
        m = build_cnn(use_gap=use_gap)
        m.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
        m.fit(X_train, y_train, epochs=6, batch_size=128,
              validation_split=0.15, verbose=0)
        yp = np.argmax(m.predict(X_test, verbose=0), axis=1)
        print(f"    {label:<20}  acc={accuracy_score(y_test,yp):.4f}  params={m.count_params():,}")
        keras.backend.clear_session()

    print("\n  Key takeaways:")
    print("    • Conv layers share weights → far fewer params than Dense for images")
    print("    • Stacking two Conv blocks before pooling captures richer features")
    print("    • GlobalAveragePooling reduces params and overfitting vs Flatten")
    print("    • BatchNorm after Conv stabilises training and acts as regulariser")
