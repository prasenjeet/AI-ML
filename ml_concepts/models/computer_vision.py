"""
Computer Vision Demo
──────────────────────
Covers:
  1. Image Pre-processing   – normalisation, channel stats, pixel distribution
  2. Augmentation           – flip, rotation, brightness, zoom; effect comparison
  3. Classical Features     – HOG descriptors, edge detection (Canny/Sobel),
                              colour histograms + SVM classifier
  4. CNN on CIFAR-10        – two Conv blocks + data-augmentation layer
  5. Transfer Learning      – MobileNetV2 frozen backbone → custom head
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from skimage.feature import hog
from skimage.filters import sobel
from skimage.color import rgb2gray
from skimage.transform import resize as sk_resize

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

PALETTE = sns.color_palette("tab10")
CIFAR_CLASSES = ["airplane","automobile","bird","cat","deer",
                 "dog","frog","horse","ship","truck"]


# ── Data helpers ──────────────────────────────────────────────────────────────
def load_cifar(n_train=8000, n_test=2000):
    (X_tr, y_tr), (X_te, y_te) = keras.datasets.cifar10.load_data()
    y_tr = y_tr.flatten(); y_te = y_te.flatten()
    return (X_tr[:n_train].astype(np.float32) / 255.0, y_tr[:n_train],
            X_te[:n_test].astype(np.float32) / 255.0,  y_te[:n_test])


# ── 1. Pre-processing demo ────────────────────────────────────────────────────
def demo_preprocessing(X_train, y_train, save_plots=False, plot_dir="."):
    print("\n  [Image Pre-processing Statistics]")
    print(f"  Shape   : {X_train.shape}   dtype: {X_train.dtype}")
    print(f"  Min/Max : {X_train.min():.3f} / {X_train.max():.3f}")
    print(f"  Mean(R,G,B): {X_train[...,0].mean():.3f}  "
          f"{X_train[...,1].mean():.3f}  {X_train[...,2].mean():.3f}")

    # Sample grid
    fig, axes = plt.subplots(2, 10, figsize=(20, 5))
    for cls in range(10):
        idx = np.where(y_train == cls)[0][0]
        axes[0, cls].imshow(X_train[idx]); axes[0, cls].axis("off")
        axes[0, cls].set_title(CIFAR_CLASSES[cls], fontsize=7)
    # Pixel distribution per channel
    for c, (ch, col) in enumerate(zip(range(3), ["#e74c3c","#27ae60","#2980b9"])):
        axes[1, c].hist(X_train[..., ch].flatten(), bins=50, color=col, alpha=0.7)
        axes[1, c].set_title(["R","G","B"][c] + " channel")
    for ax in axes[1, 3:]:
        ax.axis("off")
    plt.suptitle("CIFAR-10 Samples & Channel Distributions", fontweight="bold")
    plt.tight_layout()
    path = f"{plot_dir}/cv_samples.png" if save_plots else None
    if path: fig.savefig(path, bbox_inches="tight"); print(f"  Saved → {path}")
    else:    plt.show()
    plt.close(fig)


# ── 2. Augmentation ───────────────────────────────────────────────────────────
def _apply_augmentations(img):
    """Return dict of augmented versions of a single (32,32,3) image."""
    from scipy.ndimage import rotate as nd_rotate
    aug = {"Original": img}
    # Horizontal flip
    aug["Flipped (H)"]  = img[:, ::-1, :]
    # Vertical flip
    aug["Flipped (V)"]  = img[::-1, :, :]
    # Rotation 30°
    aug["Rotated 30°"]  = np.clip(nd_rotate(img, 30, reshape=False), 0, 1)
    # Brightness +40 %
    aug["Bright +40%"]  = np.clip(img * 1.4, 0, 1)
    # Darkness -40 %
    aug["Dark -40%"]    = np.clip(img * 0.6, 0, 1)
    # Center crop (zoom)
    c = img[4:28, 4:28, :]
    aug["Zoom (crop)"]  = np.array(
        [sk_resize(c[:,:,ch], (32,32)) for ch in range(3)]).transpose(1,2,0)
    return aug


def demo_augmentation(X_train, y_train, save_plots=False, plot_dir="."):
    print("\n  [Image Augmentation Effects]")
    idx  = np.where(y_train == 0)[0][0]       # pick one airplane
    img  = X_train[idx]
    augs = _apply_augmentations(img)

    n    = len(augs)
    fig, axes = plt.subplots(1, n, figsize=(n * 2.2, 2.5))
    for ax, (name, a) in zip(axes, augs.items()):
        ax.imshow(np.clip(a, 0, 1)); ax.axis("off"); ax.set_title(name, fontsize=8)
    plt.suptitle("Augmentation Showcase", fontweight="bold")
    plt.tight_layout()
    path = f"{plot_dir}/cv_augmentation.png" if save_plots else None
    if path: fig.savefig(path, bbox_inches="tight"); print(f"  Saved → {path}")
    else:    plt.show()
    plt.close(fig)
    print(f"  Demonstrated {n} augmentation types on class 'airplane'")


# ── 3. Classical Features + SVM ───────────────────────────────────────────────
def _extract_hog(X):
    feats = []
    for img in X:
        fd = hog(img, orientations=8, pixels_per_cell=(4, 4),
                 cells_per_block=(2, 2), channel_axis=-1)
        feats.append(fd)
    return np.array(feats)


def _extract_color_hist(X, bins=32):
    feats = []
    for img in X:
        h = np.concatenate([np.histogram(img[..., c], bins=bins, range=(0,1))[0]
                            for c in range(3)])
        feats.append(h)
    return np.array(feats, dtype=np.float32)


def demo_features(X_train, y_train, X_test, y_test,
                  save_plots=False, plot_dir="."):
    print("\n  [Classical Feature Extraction + SVM]")

    # HOG
    print("  Extracting HOG features …")
    H_tr = _extract_hog(X_train)
    H_te = _extract_hog(X_test)
    print(f"  HOG feature dim: {H_tr.shape[1]}")

    clf_hog = Pipeline([("sc", StandardScaler()),
                        ("svm", SVC(kernel="rbf", C=10, gamma="scale"))])
    clf_hog.fit(H_tr, y_train)
    yp_hog = clf_hog.predict(H_te)
    acc_hog = accuracy_score(y_test, yp_hog)
    print(f"  HOG + SVM accuracy: {acc_hog:.4f}")

    # Colour histogram
    print("  Extracting colour histograms …")
    C_tr = _extract_color_hist(X_train)
    C_te = _extract_color_hist(X_test)
    clf_col = Pipeline([("sc", StandardScaler()),
                        ("svm", SVC(kernel="rbf", C=5, gamma="scale"))])
    clf_col.fit(C_tr, y_train)
    acc_col = accuracy_score(y_test, clf_col.predict(C_te))
    print(f"  Color hist + SVM accuracy: {acc_col:.4f}")

    # HOG visualisation
    idx  = np.where(y_train == 1)[0][0]   # car
    fd, vis = hog(X_train[idx], orientations=8, pixels_per_cell=(4,4),
                  cells_per_block=(2,2), channel_axis=-1, visualize=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(X_train[idx]); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(vis, cmap="gray"); axes[1].set_title("HOG Visualisation"); axes[1].axis("off")
    plt.suptitle("HOG Feature Descriptor", fontweight="bold")
    plt.tight_layout()
    path = f"{plot_dir}/cv_hog.png" if save_plots else None
    if path: fig.savefig(path, bbox_inches="tight"); print(f"  Saved → {path}")
    else:    plt.show()
    plt.close(fig)

    # Edge detection
    img_gray = rgb2gray(X_train[idx])
    edges_sobel = sobel(img_gray)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(X_train[idx]); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(img_gray, cmap="gray"); axes[1].set_title("Grayscale"); axes[1].axis("off")
    axes[2].imshow(edges_sobel, cmap="gray"); axes[2].set_title("Sobel Edges"); axes[2].axis("off")
    plt.suptitle("Edge Detection", fontweight="bold")
    plt.tight_layout()
    path2 = f"{plot_dir}/cv_edges.png" if save_plots else None
    if path2: fig.savefig(path2, bbox_inches="tight"); print(f"  Saved → {path2}")
    else:     plt.show()
    plt.close(fig)

    return acc_hog, acc_col


# ── 4. CNN on CIFAR-10 ────────────────────────────────────────────────────────
def build_cnn_cifar(input_shape=(32, 32, 3), n_classes=10,
                    filters=(32, 64), dropout=0.3):
    inp = keras.Input(shape=input_shape)
    x   = inp
    for f in filters:
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.25)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="CNN_CIFAR10")


def demo_cnn(X_train, y_train, X_test, y_test,
             epochs=15, save_plots=False, plot_dir="."):
    print("\n  [CNN on CIFAR-10]")
    keras.backend.clear_session()

    # Data augmentation via Keras preprocessing layer
    augment = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    model = build_cnn_cifar(filters=(32, 64), dropout=0.4)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    def aug_gen(X, y, batch=128, training=True):
        ds = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(4000)
        if training:
            ds = ds.map(lambda x, y: (augment(x, training=True), y))
        return ds.batch(batch).prefetch(tf.data.AUTOTUNE)

    cb = [keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, verbose=0),
          keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, verbose=0)]

    history = model.fit(aug_gen(X_train, y_train), epochs=epochs,
                        validation_data=aug_gen(X_test, y_test, training=False),
                        callbacks=cb, verbose=1)

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n  CNN Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=CIFAR_CLASSES))

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"],     label="Train loss",  color=PALETTE[0])
    axes[0].plot(history.history["val_loss"], label="Val loss",    color=PALETTE[1])
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(history.history["accuracy"],     label="Train acc",  color=PALETTE[2])
    axes[1].plot(history.history["val_accuracy"], label="Val acc",    color=PALETTE[3])
    axes[1].set_title("Accuracy"); axes[1].legend()
    plt.suptitle("CNN CIFAR-10 Training", fontweight="bold")
    plt.tight_layout()
    path = f"{plot_dir}/cv_cnn_history.png" if save_plots else None
    if path: fig.savefig(path, bbox_inches="tight"); print(f"  Saved → {path}")
    else:    plt.show()
    plt.close(fig)

    keras.backend.clear_session()
    return acc


# ── 5. Transfer Learning ──────────────────────────────────────────────────────
def demo_transfer_learning(X_train, y_train, X_test, y_test,
                           save_plots=False, plot_dir="."):
    print("\n  [Transfer Learning – MobileNetV2 Feature Extraction]")
    keras.backend.clear_session()

    # Resize CIFAR-32 → 96 for MobileNetV2 (min 32 input, but 96 is faster than 224)
    target = (96, 96)
    print("  Resizing images …")
    X_tr_r = np.array([sk_resize(img, target, anti_aliasing=True)
                       for img in X_train[:3000]], dtype=np.float32)
    X_te_r = np.array([sk_resize(img, target, anti_aliasing=True)
                       for img in X_test[:600]], dtype=np.float32)
    y_tr_r, y_te_r = y_train[:3000], y_test[:600]

    base = keras.applications.MobileNetV2(
        input_shape=(*target, 3), include_top=False, weights="imagenet",
        pooling="avg")
    base.trainable = False

    print("  Extracting MobileNetV2 features …")
    F_tr = base.predict(keras.applications.mobilenet_v2.preprocess_input(X_tr_r * 255),
                        batch_size=64, verbose=0)
    F_te = base.predict(keras.applications.mobilenet_v2.preprocess_input(X_te_r * 255),
                        batch_size=64, verbose=0)
    print(f"  Feature dim: {F_tr.shape[1]}")

    # Linear probe
    clf = SVC(kernel="linear", C=1.0)
    clf.fit(F_tr, y_tr_r)
    acc_probe = accuracy_score(y_te_r, clf.predict(F_te))
    print(f"  Linear probe accuracy: {acc_probe:.4f}")

    # Fine-tune: add custom head and train a few layers
    inp = keras.Input(shape=(*target, 3))
    x   = keras.applications.mobilenet_v2.preprocess_input(inp)
    x   = base(x, training=False)
    x   = layers.Dense(128, activation="relu")(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inp, out, name="MobileNetV2_head")
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    history = model.fit(X_tr_r, y_tr_r, epochs=8, batch_size=64,
                        validation_split=0.15, verbose=1)
    y_pred  = np.argmax(model.predict(X_te_r, verbose=0), axis=1)
    acc_ft  = accuracy_score(y_te_r, y_pred)
    print(f"  Fine-tuned head accuracy: {acc_ft:.4f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history["accuracy"],     label="Train acc", color=PALETTE[0])
    ax.plot(history.history["val_accuracy"], label="Val acc",   color=PALETTE[1])
    ax.set_title("Transfer Learning (MobileNetV2 + custom head)", fontweight="bold")
    ax.legend(); ax.set_xlabel("Epoch")
    plt.tight_layout()
    path = f"{plot_dir}/cv_transfer.png" if save_plots else None
    if path: fig.savefig(path, bbox_inches="tight"); print(f"  Saved → {path}")
    else:    plt.show()
    plt.close(fig)

    keras.backend.clear_session()
    return acc_probe, acc_ft


# ── run ───────────────────────────────────────────────────────────────────────
def run(save_plots=False, plot_dir="."):
    print("\n" + "=" * 60)
    print("  COMPUTER VISION")
    print("=" * 60)
    X_tr, y_tr, X_te, y_te = load_cifar()
    demo_preprocessing(X_tr, y_tr, save_plots, plot_dir)
    demo_augmentation(X_tr, y_tr, save_plots, plot_dir)
    acc_hog, acc_col = demo_features(X_tr, y_tr, X_te, y_te, save_plots, plot_dir)
    acc_cnn  = demo_cnn(X_tr, y_tr, X_te, y_te, epochs=12, save_plots=save_plots, plot_dir=plot_dir)
    acc_p, acc_ft = demo_transfer_learning(X_tr, y_tr, X_te, y_te, save_plots, plot_dir)

    print(f"\n  Summary:")
    print(f"    HOG + SVM             : {acc_hog:.4f}")
    print(f"    Color Hist + SVM      : {acc_col:.4f}")
    print(f"    CNN (trained)         : {acc_cnn:.4f}")
    print(f"    Transfer (linear)     : {acc_p:.4f}")
    print(f"    Transfer (fine-tuned) : {acc_ft:.4f}")
    print("\n  Key takeaways:")
    print("    • HOG captures shape/texture; colour histograms capture colour distribution")
    print("    • CNNs learn features automatically — far better than hand-crafted ones")
    print("    • Transfer learning: even a frozen ImageNet backbone beats a trained-from-scratch CNN")
    print("      on small datasets due to rich pre-learned feature representations")
