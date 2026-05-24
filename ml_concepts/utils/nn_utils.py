"""Shared utilities for Neural Network demos (data generation, plotting)."""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

PALETTE = sns.color_palette("tab10")
plt.rcParams.update({"figure.dpi": 100, "font.size": 10})


# ── Data generators ───────────────────────────────────────────────────────────
def make_time_series(n_points: int = 3000, noise: float = 0.08, random_state: int = 42):
    """Multi-frequency sine wave with noise — used for RNN / LSTM."""
    rng = np.random.default_rng(random_state)
    t = np.linspace(0, 6 * np.pi * 10, n_points)
    signal = (np.sin(0.3 * t)
              + 0.5 * np.sin(0.7 * t)
              + 0.3 * np.sin(1.3 * t)
              + noise * rng.standard_normal(n_points))
    return signal.astype(np.float32)


def make_sequences(signal: np.ndarray, seq_len: int = 50):
    """Slide a window over a 1-D signal → (N, seq_len, 1) inputs, (N,) targets."""
    X, y = [], []
    for i in range(seq_len, len(signal)):
        X.append(signal[i - seq_len: i])
        y.append(signal[i])
    X = np.array(X, dtype=np.float32)[..., np.newaxis]  # (N, T, 1)
    y = np.array(y, dtype=np.float32)
    return X, y


def load_mnist_subset(n_train: int = 8000, n_test: int = 2000):
    """Load MNIST, return (X_train, y_train, X_test, y_test) normalised to [0,1]."""
    import tensorflow as tf
    (X_tr, y_tr), (X_te, y_te) = tf.keras.datasets.mnist.load_data()
    X_tr = X_tr[:n_train].astype(np.float32) / 255.0          # (N,28,28)
    X_te = X_te[:n_test].astype(np.float32) / 255.0
    X_tr = X_tr[..., np.newaxis]                               # (N,28,28,1)
    X_te = X_te[..., np.newaxis]
    return X_tr, y_tr[:n_train], X_te, y_te[:n_test]


def tabular_data(name: str = "wine"):
    """Return scaled train/test splits for a tabular classification dataset."""
    from sklearn.model_selection import train_test_split
    if name == "iris":
        from sklearn.datasets import load_iris
        d = load_iris()
    elif name == "wine":
        from sklearn.datasets import load_wine
        d = load_wine()
    else:
        from sklearn.datasets import make_classification
        X, y = make_classification(600, n_features=15, n_informative=8,
                                   n_classes=3, n_clusters_per_class=1, random_state=42)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        return scaler.fit_transform(Xtr), scaler.transform(Xte), ytr, yte, 15, 3, [f"C{i}" for i in range(3)]

    X, y = d.data, d.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    return (scaler.fit_transform(Xtr), scaler.transform(Xte),
            ytr, yte, X.shape[1], len(np.unique(y)), list(getattr(d, "target_names", [str(i) for i in np.unique(y)])))


# ── Plotting helpers ──────────────────────────────────────────────────────────
def plot_history(history, title="Training History", path=None):
    keys = list(history.history.keys())
    has_acc = any("acc" in k for k in keys)

    n_plots = 2 if has_acc else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    axes[0].plot(history.history["loss"],     label="Train loss", color=PALETTE[0])
    if "val_loss" in history.history:
        axes[0].plot(history.history["val_loss"], label="Val loss",   color=PALETTE[1])
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} – Loss"); axes[0].legend()

    if has_acc:
        acc_key     = "accuracy"     if "accuracy"     in keys else "acc"
        val_acc_key = "val_accuracy" if "val_accuracy" in keys else "val_acc"
        axes[1].plot(history.history[acc_key],     label="Train acc", color=PALETTE[2])
        if val_acc_key in history.history:
            axes[1].plot(history.history[val_acc_key], label="Val acc",   color=PALETTE[3])
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
        axes[1].set_title(f"{title} – Accuracy"); axes[1].legend()

    plt.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight"); print(f"  Saved → {path}")
    else:
        plt.show()
    plt.close(fig)


def plot_confusion(y_true, y_pred, class_names, title="Confusion Matrix", path=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(5, len(class_names)), max(4, len(class_names) - 1)))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight"); print(f"  Saved → {path}")
    else:
        plt.show()
    plt.close(fig)


def plot_ts_prediction(y_true, y_pred, title="Prediction vs Actual", n_show=300, path=None):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    axes[0].plot(y_true[:n_show], label="Actual",    color=PALETTE[0], lw=1.2)
    axes[0].plot(y_pred[:n_show], label="Predicted", color=PALETTE[1], lw=1.2, alpha=0.85)
    axes[0].set_title(title); axes[0].legend()

    err = np.abs(y_true[:n_show] - y_pred[:n_show])
    axes[1].fill_between(range(n_show), err, alpha=0.5, color=PALETTE[2])
    axes[1].set_title("Absolute Error"); axes[1].set_xlabel("Time step")

    plt.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight"); print(f"  Saved → {path}")
    else:
        plt.show()
    plt.close(fig)


def plot_cnn_samples(X, y_true, y_pred, class_names, n=10, path=None):
    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 4))
    for i in range(n):
        for row, (img, true, pred) in enumerate([(X[i], y_true[i], y_pred[i])]):
            ax = axes[row, i]
            ax.imshow(img.squeeze(), cmap="gray")
            color = "green" if true == pred else "red"
            ax.set_title(f"{class_names[pred]}", color=color, fontsize=7)
            ax.axis("off")
    axes[0, 0].set_ylabel("Predictions\n(green=✓, red=✗)", fontsize=7)
    plt.suptitle("CNN Sample Predictions", fontweight="bold")
    plt.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight"); print(f"  Saved → {path}")
    else:
        plt.show()
    plt.close(fig)


def arch_string(model) -> str:
    lines = []
    for layer in model.layers:
        cfg = layer.get_config()
        name = layer.__class__.__name__
        params = layer.count_params()
        shape_out = str(layer.output.shape) if hasattr(layer, "output") else "?"
        lines.append(f"  {name:<22} output={shape_out:<20} params={params:,}")
    return "\n".join(lines)
