"""
Reference implementation: a small CNN trained FROM SCRATCH using only NumPy.

Why this file exists:
- This sandbox has no internet access, so PyTorch cannot be installed here.
- To give you REAL, honest training results (not made-up numbers), this script
  implements the same architecture (Conv -> ReLU -> MaxPool -> Dense -> Softmax)
  by hand, with manual forward and backward passes, and trains it on a real
  dataset (sklearn's digits dataset: 1797 real 8x8 handwritten digit images).
- The companion file `pytorch_cnn.py` has the equivalent, idiomatic PyTorch
  code you'll actually put in your portfolio / GitHub. Run that one on your
  own machine or Google Colab (which has PyTorch preinstalled) to reproduce
  and likely improve on these numbers.

Dataset: sklearn.datasets.load_digits
  - 1797 images, each 8x8 grayscale pixels (0-16 intensity)
  - 10 classes (digits 0-9)
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)


# ---------------------------
# Data loading & preprocessing
# ---------------------------
def load_data():
    digits = load_digits()
    X = digits.data.reshape(-1, 1, 8, 8).astype(np.float64)  # (N, C=1, H=8, W=8)
    X = X / 16.0  # normalize pixel range [0, 16] -> [0, 1]
    y = digits.target.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


# ---------------------------
# im2col helpers (standard trick for fast conv forward/backward)
# ---------------------------
def get_im2col_indices(x_shape, field_h, field_w, padding, stride):
    N, C, H, W = x_shape
    out_h = (H + 2 * padding - field_h) // stride + 1
    out_w = (W + 2 * padding - field_w) // stride + 1

    i0 = np.repeat(np.arange(field_h), field_w)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_h), out_w)
    j0 = np.tile(np.arange(field_w), field_h * C)
    j1 = stride * np.tile(np.arange(out_w), out_h)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), field_h * field_w).reshape(-1, 1)
    return k, i, j, out_h, out_w


def im2col(x, field_h, field_w, padding, stride):
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")
    k, i, j, out_h, out_w = get_im2col_indices(x.shape, field_h, field_w, padding, stride)
    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_h * field_w * C, -1)
    return cols, out_h, out_w


def col2im(cols, x_shape, field_h, field_w, padding, stride):
    N, C, H, W = x_shape
    p = padding
    H_padded, W_padded = H + 2 * p, W + 2 * p
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j, out_h, out_w = get_im2col_indices(x_shape, field_h, field_w, padding, stride)
    cols_reshaped = cols.reshape(C * field_h * field_w, -1, N).transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if p == 0:
        return x_padded
    return x_padded[:, :, p:-p, p:-p]


# ---------------------------
# Layers
# ---------------------------
class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        self.stride = stride
        self.padding = padding
        self.k = kernel_size
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = rng.normal(0, scale, size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.zeros((out_channels, 1))
        self.cache = None

    def forward(self, x):
        N, C, H, W = x.shape
        F, _, HH, WW = self.W.shape
        x_cols, out_h, out_w = im2col(x, HH, WW, self.padding, self.stride)
        W_col = self.W.reshape(F, -1)
        out = W_col @ x_cols + self.b
        out = out.reshape(F, out_h, out_w, N).transpose(3, 0, 1, 2)
        self.cache = (x, x_cols, out_h, out_w)
        return out

    def backward(self, dout, lr):
        x, x_cols, out_h, out_w = self.cache
        F = self.W.shape[0]
        db = dout.sum(axis=(0, 2, 3)).reshape(F, 1)
        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(F, -1)
        dW = dout_reshaped @ x_cols.T
        dW = dW.reshape(self.W.shape)
        W_col = self.W.reshape(F, -1)
        dx_cols = W_col.T @ dout_reshaped
        dx = col2im(dx_cols, x.shape, self.k, self.k, self.padding, self.stride)
        self.W -= lr * dW
        self.b -= lr * db
        return dx


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout, lr=None):
        return dout * self.mask


class MaxPool2D:
    def __init__(self, size=2, stride=2):
        self.size = size
        self.stride = stride

    def forward(self, x):
        N, C, H, W = x.shape
        s, st = self.size, self.stride
        out_h, out_w = (H - s) // st + 1, (W - s) // st + 1
        out = np.zeros((N, C, out_h, out_w))
        self.mask = np.zeros_like(x)
        for i in range(out_h):
            for j in range(out_w):
                h0, w0 = i * st, j * st
                window = x[:, :, h0:h0 + s, w0:w0 + s]
                m = window.max(axis=(2, 3), keepdims=True)
                out[:, :, i, j] = m[:, :, 0, 0]
                self.mask[:, :, h0:h0 + s, w0:w0 + s] += (window == m)
        self.x_shape = x.shape
        return out

    def backward(self, dout, lr=None):
        N, C, H, W = self.x_shape
        s, st = self.size, self.stride
        dx = np.zeros(self.x_shape)
        out_h, out_w = dout.shape[2], dout.shape[3]
        for i in range(out_h):
            for j in range(out_w):
                h0, w0 = i * st, j * st
                dx[:, :, h0:h0 + s, w0:w0 + s] += (
                    self.mask[:, :, h0:h0 + s, w0:w0 + s] * dout[:, :, i:i + 1, j:j + 1]
                )
        return dx


class Flatten:
    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout, lr=None):
        return dout.reshape(self.shape)


class Dense:
    def __init__(self, in_dim, out_dim):
        scale = np.sqrt(2.0 / in_dim)
        self.W = rng.normal(0, scale, size=(in_dim, out_dim))
        self.b = np.zeros((1, out_dim))

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout, lr):
        dW = self.x.T @ dout
        db = dout.sum(axis=0, keepdims=True)
        dx = dout @ self.W.T
        self.W -= lr * dW
        self.b -= lr * db
        return dx


def softmax_cross_entropy(logits, y):
    logits_shift = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits_shift)
    probs = exp / exp.sum(axis=1, keepdims=True)
    N = logits.shape[0]
    log_likelihood = -np.log(probs[np.arange(N), y] + 1e-12)
    loss = log_likelihood.mean()
    dlogits = probs.copy()
    dlogits[np.arange(N), y] -= 1
    dlogits /= N
    return loss, dlogits


# ---------------------------
# Model: Conv -> ReLU -> MaxPool -> Flatten -> Dense -> Dense
# ---------------------------
class SimpleCNN:
    def __init__(self):
        self.conv1 = Conv2D(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(size=2, stride=2)
        self.flatten = Flatten()
        self.fc1 = Dense(8 * 4 * 4, 32)
        self.relu2 = ReLU()
        self.fc2 = Dense(32, 10)

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)
        x = self.flatten.forward(x)
        x = self.fc1.forward(x)
        x = self.relu2.forward(x)
        x = self.fc2.forward(x)
        return x

    def backward(self, dlogits, lr):
        d = self.fc2.backward(dlogits, lr)
        d = self.relu2.backward(d)
        d = self.fc1.backward(d, lr)
        d = self.flatten.backward(d)
        d = self.pool1.backward(d)
        d = self.relu1.backward(d)
        d = self.conv1.backward(d, lr)
        return d

    def predict(self, x):
        logits = self.forward(x)
        return logits.argmax(axis=1)


def train():
    X_train, X_test, y_train, y_test = load_data()
    model = SimpleCNN()

    epochs = 15
    batch_size = 32
    lr = 0.05
    n = X_train.shape[0]

    history = {"train_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n)
        X_train, y_train = X_train[perm], y_train[perm]
        epoch_loss, correct = 0.0, 0

        for start in range(0, n, batch_size):
            xb = X_train[start:start + batch_size]
            yb = y_train[start:start + batch_size]

            logits = model.forward(xb)
            loss, dlogits = softmax_cross_entropy(logits, yb)
            model.backward(dlogits, lr)

            epoch_loss += loss * xb.shape[0]
            correct += (logits.argmax(axis=1) == yb).sum()

        train_loss = epoch_loss / n
        train_acc = correct / n
        test_preds = model.predict(X_test)
        test_acc = (test_preds == y_test).mean()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        print(f"Epoch {epoch:2d}/{epochs} | train_loss={train_loss:.4f} | "
              f"train_acc={train_acc:.4f} | test_acc={test_acc:.4f}")

    final_test_preds = model.predict(X_test)
    final_test_acc = (final_test_preds == y_test).mean()
    print(f"\nFinal test accuracy: {final_test_acc:.4f}")
    return model, history, (X_test, y_test, final_test_preds)


if __name__ == "__main__":
    train()
