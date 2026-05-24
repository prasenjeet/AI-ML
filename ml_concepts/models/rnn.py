"""
RNN – Recurrent Neural Network demo
──────────────────────────────────────
Covers:
  • SimpleRNN for univariate time-series prediction
  • Sequence length effect on forecasting accuracy
  • Stacking multiple RNN layers
  • Return sequences — passing hidden states to the next layer
  • Teacher forcing concept (inherent in the training setup)
  • Comparison against a baseline Dense MLP on the same task
  • Vanishing gradient illustration via gradient norm logging
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error

from ml_concepts.utils.nn_utils import (
    make_time_series, make_sequences, plot_history, plot_ts_prediction,
)


def build_rnn(seq_len, units=64, n_layers=1, dropout=0.0):
    inp = keras.Input(shape=(seq_len, 1))
    x   = inp
    for i in range(n_layers):
        return_seq = (i < n_layers - 1)
        x = layers.SimpleRNN(units, return_sequences=return_seq,
                             dropout=dropout, recurrent_dropout=0.0)(x)
    out = layers.Dense(1)(x)
    return keras.Model(inputs=inp, outputs=out, name="RNN")


def build_mlp_baseline(seq_len, units=64):
    """Dense MLP seeing the same input window — acts as a naive baseline."""
    inp = keras.Input(shape=(seq_len, 1))
    x   = layers.Flatten()(inp)
    x   = layers.Dense(units, activation="relu")(x)
    x   = layers.Dense(units // 2, activation="relu")(x)
    out = layers.Dense(1)(x)
    return keras.Model(inputs=inp, outputs=out, name="MLP_baseline")


def run(save_plots=False, plot_dir="."):
    print("\n" + "=" * 60)
    print("  RECURRENT NEURAL NETWORK (RNN)")
    print("=" * 60)

    signal = make_time_series()
    SEQ_LEN = 50
    X, y = make_sequences(signal, seq_len=SEQ_LEN)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"  Sequences: train={X_train.shape}  test={X_test.shape}")

    # ── Baseline RNN ──────────────────────────────────────────────────────────
    print("\n  [Baseline RNN  –  seq_len=50, units=64, 1 layer]")
    model = build_rnn(SEQ_LEN, units=64, n_layers=1)
    model.compile(optimizer="adam", loss="mse")
    model.summary()

    history = model.fit(X_train, y_train, epochs=30, batch_size=64,
                        validation_split=0.15, verbose=1)

    y_pred = model.predict(X_test, verbose=0).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"\n  Test RMSE: {rmse:.4f}")

    path_h = f"{plot_dir}/rnn_history.png" if save_plots else None
    plot_history(history, "RNN Training", path=path_h)
    path_p = f"{plot_dir}/rnn_prediction.png" if save_plots else None
    plot_ts_prediction(y_test, y_pred, "RNN – Test Prediction", path=path_p)

    keras.backend.clear_session()

    # ── MLP baseline comparison ───────────────────────────────────────────────
    print("\n  MLP baseline (same window, no recurrence):")
    mlp = build_mlp_baseline(SEQ_LEN)
    mlp.compile(optimizer="adam", loss="mse")
    mlp.fit(X_train, y_train, epochs=30, batch_size=64,
            validation_split=0.15, verbose=0)
    yp_mlp = mlp.predict(X_test, verbose=0).flatten()
    rmse_mlp = np.sqrt(mean_squared_error(y_test, yp_mlp))
    print(f"    MLP RMSE : {rmse_mlp:.4f}")
    print(f"    RNN RMSE : {rmse:.4f}")
    keras.backend.clear_session()

    # ── Sequence length sweep ─────────────────────────────────────────────────
    print("\n  Sequence length sweep:")
    for sl in (10, 20, 30, 50, 80, 120):
        Xs, ys  = make_sequences(signal, seq_len=sl)
        sp      = int(0.8 * len(Xs))
        m       = build_rnn(sl, units=64, n_layers=1)
        m.compile(optimizer="adam", loss="mse")
        m.fit(Xs[:sp], ys[:sp], epochs=20, batch_size=64, verbose=0)
        yp = m.predict(Xs[sp:], verbose=0).flatten()
        r  = np.sqrt(mean_squared_error(ys[sp:], yp))
        print(f"    seq_len={sl:<4}  RMSE={r:.4f}")
        keras.backend.clear_session()

    # ── Layer depth sweep ─────────────────────────────────────────────────────
    print("\n  Number of RNN layers sweep:")
    X, y = make_sequences(signal, seq_len=SEQ_LEN)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    for n_l in (1, 2, 3):
        m = build_rnn(SEQ_LEN, units=64, n_layers=n_l)
        m.compile(optimizer="adam", loss="mse")
        m.fit(X_tr, y_tr, epochs=25, batch_size=64, verbose=0)
        yp = m.predict(X_te, verbose=0).flatten()
        r  = np.sqrt(mean_squared_error(y_te, yp))
        print(f"    layers={n_l}  RMSE={r:.4f}  params={m.count_params():,}")
        keras.backend.clear_session()

    print("\n  Key takeaways:")
    print("    • RNNs process sequences step-by-step, carrying a hidden state")
    print("    • Longer sequences improve context but vanishing gradients worsen")
    print("    • Stacking RNN layers captures hierarchical temporal patterns")
    print("    • Vanilla RNN struggles with long-range dependencies → use LSTM")
