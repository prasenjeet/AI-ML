# Neural Network Concepts

All neural network modules live in `ml_concepts/models/` and use **TensorFlow 2 / Keras**.  
Shared utilities are in `ml_concepts/utils/nn_utils.py`.

---

## Shared Utilities (`nn_utils.py`)

| Function | Purpose |
|----------|---------|
| `load_mnist_subset(n_train, n_test)` | Normalised MNIST (0–1), returns train/test splits |
| `tabular_data(name)` | Scaled wine/iris/synthetic + metadata |
| `make_time_series(n_points, noise)` | Multi-frequency sine wave signal |
| `make_sequences(signal, seq_len)` | Sliding window → `(N, T, 1)` for RNN/LSTM |
| `plot_history(history, title)` | Loss + accuracy dual-panel chart |
| `plot_confusion(y_true, y_pred, labels)` | Seaborn confusion matrix |
| `plot_ts_prediction(y_true, y_pred)` | Time-series overlay + error plot |

---

## 7. ANN – Artificial Neural Network
**File:** `ml_concepts/models/ann.py`  
**CLI topic:** `ann`  
**Streamlit tab:** 🔵 ANN

### Architecture
```
Input(n_features)
  → Dense(units, activation)  × n_layers
  → Dropout(rate)
  → Dense(n_classes, softmax)
```

### Experiments

| Experiment | Values |
|-----------|--------|
| Activation functions | `relu` · `tanh` · `sigmoid` · `elu` |
| Optimizers | `adam` · `sgd` · `rmsprop` |
| Hidden width sweep | 16 → 32 → 64 → 128 → 256 neurons |
| Dropout | 0.0 → 0.5 |

### Key Concept
> **Universal Approximation Theorem:** A single hidden layer with enough neurons can approximate any continuous function.  
> - **ReLU** avoids vanishing gradients (gradient = 1 for positive inputs)  
> - **ELU** is smooth and has non-zero gradient for negative inputs  
> - **Adam** adapts the learning rate per parameter — faster convergence than vanilla SGD

---

## 8. DNN – Deep Neural Network
**File:** `ml_concepts/models/dnn.py`  
**CLI topic:** `dnn`  
**Streamlit tab:** 🟣 DNN

### Architecture
```
Input
  → [Dense(units) → BatchNorm → ReLU → Dropout(rate)]  × n_layers
  → Dense(n_classes, softmax)
```

### Training Stabilisers

| Technique | Purpose | When to use |
|-----------|---------|-------------|
| **Batch Normalisation** | Normalises layer inputs per mini-batch; reduces internal covariate shift | Always for networks > 3 layers |
| **Dropout** | Randomly zeroes activations during training; implicit ensemble | When overfitting |
| **He Normal init** | Sets weights for ReLU to maintain variance | Always with ReLU activations |
| **ReduceLROnPlateau** | Halves LR when val_loss plateaus | Long training runs |
| **EarlyStopping** | Stops when val_loss stops improving | Prevents wasted epochs |

### Depth Sweep
Test accuracy vs number of layers (1–10) shows:
- Too shallow: underfitting
- Optimal depth: depends on dataset complexity
- Too deep (without BatchNorm): degradation due to vanishing gradients

### Key Concept
> **BatchNorm** re-centres and re-scales layer inputs:  
> `BN(x) = γ · (x − μ_B) / √(σ²_B + ε) + β`  
> where `γ` and `β` are learnable. This keeps activations in a stable range and allows much higher learning rates.

---

## 9. CNN – Convolutional Neural Network
**File:** `ml_concepts/models/cnn.py`  
**CLI topic:** `cnn`  
**Streamlit tab:** 🖼️ CNN  
**Dataset:** MNIST handwritten digits

### Architecture
```
Input(28 × 28 × 1)
  → Conv2D(f1, k×k) → BatchNorm → Conv2D(f1, k×k) → MaxPool(2×2) → Dropout
  → Conv2D(f2, k×k) → BatchNorm → Conv2D(f2, k×k) → MaxPool(2×2) → Dropout
  → Flatten / GlobalAveragePooling
  → Dense(units, relu) → Dropout
  → Dense(10, softmax)
```

### Experiments

| Experiment | Options |
|-----------|--------|
| Filter count | (8,16) · (16,32) · (32,64) · (64,128) |
| Kernel size | 3×3 vs 5×5 |
| Spatial pooling | GlobalAveragePooling vs Flatten |

### What You'll See
- **Learned filters** (Conv1 weights visualisation): show oriented edge detectors that emerge from training
- **Sample predictions grid**: correct = green border, wrong = red border
- **Confusion matrix**: for all 10 digit classes

### Key Concept
> **Parameter sharing:** a 3×3 filter applied to a 28×28 image uses only 9 weights (+ bias), regardless of image size. This gives CNNs a huge parameter efficiency advantage over Dense layers for images.  
> **MaxPooling** introduces translation invariance: a digit slightly shifted still activates the same feature.

---

## 10. RNN – Recurrent Neural Network
**File:** `ml_concepts/models/rnn.py`  
**CLI topic:** `rnn`  
**Streamlit tab:** 🔁 RNN  
**Dataset:** Synthetic multi-frequency sine wave

### Architecture
```
Input(seq_len, 1)
  → SimpleRNN(units, return_sequences=True)  × (n_layers − 1)
  → SimpleRNN(units)
  → Dense(1)
```

### Hidden State Update
```
h_t = tanh(W_x · x_t + W_h · h_{t-1} + b)
```

### Experiments

| Experiment | Values |
|-----------|--------|
| Sequence length | 10 → 30 → 60 → 90 → 120 time steps |
| Number of layers | 1 · 2 · 3 |
| Comparison | RNN vs LSTM vs MLP baseline |

### Key Concept
> **Vanishing gradient problem:** gradients of the loss w.r.t. early hidden states shrink exponentially through BPTT (backpropagation through time). A vanilla RNN effectively forgets events more than ~20 steps back.  
> This is the core motivation for LSTM's gated cell state.

---

## 11. LSTM – Long Short-Term Memory
**File:** `ml_concepts/models/lstm.py`  
**CLI topic:** `lstm`  
**Streamlit tab:** ⏳ LSTM  
**Dataset:** Synthetic multi-frequency sine wave

### Architecture
```
Input(seq_len, 1)
  → [Bidirectional()] LSTM(units, return_sequences=True)  × (n_layers − 1)
  → [Bidirectional()] LSTM(units)
  → Dense(1)
```

### Gate Mechanism

| Gate | Equation | Purpose |
|------|----------|---------|
| **Forget** *f_t* | σ(W_f · [h_{t-1}, x_t] + b_f) | What to erase from cell state |
| **Input** *i_t* | σ(W_i · [h_{t-1}, x_t] + b_i) | What new info to store |
| **Update** *C̃_t* | tanh(W_C · [h_{t-1}, x_t] + b_C) | New candidate cell values |
| **Output** *o_t* | σ(W_o · [h_{t-1}, x_t] + b_o) | What to expose as hidden state |

**Cell state update:**  
`C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t`

**Hidden state:**  
`h_t = o_t ⊙ tanh(C_t)`

### GRU vs LSTM

| Property | LSTM | GRU |
|----------|------|-----|
| Gates | 3 (forget, input, output) | 2 (reset, update) |
| Parameters | More | Fewer (~25% fewer) |
| Cell state | Separate *C_t* and *h_t* | Single hidden state |
| Performance | Slightly better on long sequences | Faster; often matches LSTM |

### Bidirectional LSTM
```
x_t  →  LSTM_forward  →  h_t^→
x_t  →  LSTM_backward →  h_t^←
output = concat(h_t^→, h_t^←)
```
Doubles the hidden state size; useful when the full sequence context is available at prediction time.
