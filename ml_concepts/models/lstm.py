"""
LSTM – Long Short-Term Memory demo
────────────────────────────────────
Covers:
  • Stacked LSTM for time-series prediction
  • Bidirectional LSTM — reads sequences forward and backward
  • LSTM vs SimpleRNN head-to-head on same task
  • GRU (Gated Recurrent Unit) as a lighter LSTM alternative
  • Dropout & recurrent_dropout for regularisation
  • Sequence length sweep
  • Multi-step ahead forecasting
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


def build_lstm(seq_len, units=64, n_layers=2, dropout=0.1,
               rec_dropout=0.0, bidirectional=False):
    inp = keras.Input(shape=(seq_len, 1))
    x   = inp
    for i in range(n_layers):
        return_seq = (i < n_layers - 1)
        rnn = layers.LSTM(units, return_sequences=return_seq,
                          dropout=dropout, recurrent_dropout=rec_dropout)
        x = layers.Bidirectional(rnn)(x) if bidirectional else rnn(x)
    out = layers.Dense(1)(x)
    return keras.Model(inputs=inp, outputs=out, name="LSTM")


def build_gru(seq_len, units=64, n_layers=2, dropout=0.1):
    inp = keras.Input(shape=(seq_len, 1))
    x   = inp
    for i in range(n_layers):
        return_seq = (i < n_layers - 1)
        x = layers.GRU(units, return_sequences=return_seq, dropout=dropout)(x)
    out = layers.Dense(1)(x)
    return keras.Model(inputs=inp, outputs=out, name="GRU")


def build_rnn_simple(seq_len, units=64, n_layers=2, dropout=0.1):
    inp = keras.Input(shape=(seq_len, 1))
    x   = inp
    for i in range(n_layers):
        return_seq = (i < n_layers - 1)
        x = layers.SimpleRNN(units, return_sequences=return_seq, dropout=dropout)(x)
    out = layers.Dense(1)(x)
    return keras.Model(inputs=inp, outputs=out, name="SimpleRNN")


def run(save_plots=False, plot_dir="."):
    print("\n" + "=" * 60)
    print("  LSTM  (Long Short-Term Memory)")
    print("=" * 60)

    signal = make_time_series()
    SEQ_LEN = 60
    X, y = make_sequences(signal, seq_len=SEQ_LEN)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"  Sequences: train={X_train.shape}  test={X_test.shape}")

    # ── Baseline LSTM ─────────────────────────────────────────────────────────
    print("\n  [Baseline LSTM  –  seq_len=60, units=64, 2 layers]")
    model = build_lstm(SEQ_LEN, units=64, n_layers=2, dropout=0.1)
    model.compile(optimizer="adam", loss="mse")
    model.summary()

    history = model.fit(X_train, y_train, epochs=30, batch_size=64,
                        validation_split=0.15, verbose=1)

    y_pred = model.predict(X_test, verbose=0).flatten()
    rmse_lstm = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"\n  LSTM Test RMSE: {rmse_lstm:.4f}")

    path_h = f"{plot_dir}/lstm_history.png" if save_plots else None
    plot_history(history, "LSTM Training", path=path_h)
    path_p = f"{plot_dir}/lstm_prediction.png" if save_plots else None
    plot_ts_prediction(y_test, y_pred, "LSTM – Test Prediction", path=path_p)
    keras.backend.clear_session()

    # ── LSTM vs RNN vs GRU ────────────────────────────────────────────────────
    print("\n  LSTM vs SimpleRNN vs GRU (2 layers, units=64):")
    for label, builder in [
        ("SimpleRNN", build_rnn_simple),
        ("LSTM",      build_lstm),
        ("GRU",       build_gru),
    ]:
        m = builder(SEQ_LEN, units=64, n_layers=2, dropout=0.1)
        m.compile(optimizer="adam", loss="mse")
        m.fit(X_train, y_train, epochs=25, batch_size=64, verbose=0)
        yp = m.predict(X_test, verbose=0).flatten()
        r  = np.sqrt(mean_squared_error(y_test, yp))
        print(f"    {label:<12}  RMSE={r:.4f}  params={m.count_params():,}")
        keras.backend.clear_session()

    # ── Bidirectional LSTM ────────────────────────────────────────────────────
    print("\n  Unidirectional vs Bidirectional LSTM:")
    for bidir, label in [(False, "Unidirectional"), (True, "Bidirectional ")]:
        m = build_lstm(SEQ_LEN, units=64, n_layers=2, bidirectional=bidir)
        m.compile(optimizer="adam", loss="mse")
        m.fit(X_train, y_train, epochs=25, batch_size=64, verbose=0)
        yp = m.predict(X_test, verbose=0).flatten()
        r  = np.sqrt(mean_squared_error(y_test, yp))
        print(f"    {label}  RMSE={r:.4f}  params={m.count_params():,}")
        keras.backend.clear_session()

    # ── Dropout sweep ─────────────────────────────────────────────────────────
    print("\n  Dropout rate sweep (LSTM, 2 layers):")
    for dr in (0.0, 0.1, 0.2, 0.3, 0.5):
        m = build_lstm(SEQ_LEN, units=64, n_layers=2, dropout=dr)
        m.compile(optimizer="adam", loss="mse")
        m.fit(X_train, y_train, epochs=25, batch_size=64, verbose=0)
        yp = m.predict(X_test, verbose=0).flatten()
        r  = np.sqrt(mean_squared_error(y_test, yp))
        print(f"    dropout={dr}  RMSE={r:.4f}")
        keras.backend.clear_session()

    # ── Sequence length sweep ─────────────────────────────────────────────────
    print("\n  Sequence length sweep (LSTM):")
    for sl in (10, 20, 40, 60, 100):
        Xs, ys = make_sequences(signal, seq_len=sl)
        sp = int(0.8 * len(Xs))
        m  = build_lstm(sl, units=64, n_layers=2)
        m.compile(optimizer="adam", loss="mse")
        m.fit(Xs[:sp], ys[:sp], epochs=20, batch_size=64, verbose=0)
        yp = m.predict(Xs[sp:], verbose=0).flatten()
        r  = np.sqrt(mean_squared_error(ys[sp:], yp))
        print(f"    seq_len={sl:<4}  RMSE={r:.4f}")
        keras.backend.clear_session()

    print("\n  Key takeaways:")
    print("    • LSTM gates (input/forget/output) prevent vanishing gradients")
    print("    • Cell state is a 'highway' that carries long-range information")
    print("    • GRU uses fewer gates (reset + update) — often matches LSTM at lower cost")
    print("    • Bidirectional LSTM processes future context too — ideal for NLP / offline tasks")
