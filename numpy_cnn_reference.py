import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)


def load_data():
    d = load_digits()
    x = d.data.reshape(-1, 1, 8, 8).astype(np.float64) / 16.0
    y = d.target.astype(np.int64)

    xtr, xte, ytr, yte = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    return xtr, xte, ytr, yte


def get_idx(shape, fh, fw, pad, st):
    n, c, h, w = shape
    oh = (h + 2 * pad - fh) // st + 1
    ow = (w + 2 * pad - fw) // st + 1

    i0 = np.repeat(np.arange(fh), fw)
    i0 = np.tile(i0, c)
    i1 = st * np.repeat(np.arange(oh), ow)

    j0 = np.tile(np.arange(fw), fh * c)
    j1 = st * np.tile(np.arange(ow), oh)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(c), fh * fw).reshape(-1, 1)

    return k, i, j, oh, ow


def im2col(x, fh, fw, pad, st):
    xp = np.pad(
        x,
        ((0, 0), (0, 0), (pad, pad), (pad, pad)),
        mode="constant"
    )

    k, i, j, oh, ow = get_idx(x.shape, fh, fw, pad, st)
    col = xp[:, k, i, j]

    c = x.shape[1]
    col = col.transpose(1, 2, 0).reshape(fh * fw * c, -1)

    return col, oh, ow


def col2im(col, shape, fh, fw, pad, st):
    n, c, h, w = shape
    hp, wp = h + 2 * pad, w + 2 * pad

    xp = np.zeros((n, c, hp, wp), dtype=col.dtype)
    k, i, j, oh, ow = get_idx(shape, fh, fw, pad, st)

    col = col.reshape(c * fh * fw, -1, n).transpose(2, 0, 1)
    np.add.at(xp, (slice(None), k, i, j), col)

    if pad == 0:
        return xp

    return xp[:, :, pad:-pad, pad:-pad]


class Conv:
    def __init__(self, cin, cout, k, st=1, pad=1):
        self.st = st
        self.pad = pad
        self.k = k

        scale = np.sqrt(2.0 / (cin * k * k))
        self.w = rng.normal(
            0, scale, size=(cout, cin, k, k)
        )
        self.b = np.zeros((cout, 1))
        self.cache = None

    def forward(self, x):
        n, c, h, w = x.shape
        f, _, hh, ww = self.w.shape

        cols, oh, ow = im2col(
            x, hh, ww, self.pad, self.st
        )

        wc = self.w.reshape(f, -1)
        out = wc @ cols + self.b

        out = out.reshape(f, oh, ow, n).transpose(3, 0, 1, 2)
        self.cache = (x, cols, oh, ow)

        return out

    def backward(self, dout, lr):
        x, cols, oh, ow = self.cache
        f = self.w.shape[0]

        db = dout.sum(axis=(0, 2, 3)).reshape(f, 1)
        d = dout.transpose(1, 2, 3, 0).reshape(f, -1)

        dw = d @ cols.T
        dw = dw.reshape(self.w.shape)

        wc = self.w.reshape(f, -1)
        dx = wc.T @ d

        dx = col2im(
            dx, x.shape, self.k, self.k, self.pad, self.st
        )

        self.w -= lr * dw
        self.b -= lr * db

        return dx


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout, lr=None):
        return dout * self.mask


class Pool:
    def __init__(self, size=2, st=2):
        self.size = size
        self.st = st

    def forward(self, x):
        n, c, h, w = x.shape
        s = self.size
        st = self.st

        oh = (h - s) // st + 1
        ow = (w - s) // st + 1

        out = np.zeros((n, c, oh, ow))
        self.mask = np.zeros_like(x)

        for i in range(oh):
            for j in range(ow):
                h0, w0 = i * st, j * st
                win = x[:, :, h0:h0 + s, w0:w0 + s]

                m = win.max(axis=(2, 3), keepdims=True)
                out[:, :, i, j] = m[:, :, 0, 0]

                self.mask[:, :, h0:h0 + s, w0:w0 + s] += (
                    win == m
                )

        self.shape = x.shape
        return out

    def backward(self, dout, lr=None):
        s = self.size
        st = self.st
        dx = np.zeros(self.shape)

        oh, ow = dout.shape[2], dout.shape[3]

        for i in range(oh):
            for j in range(ow):
                h0, w0 = i * st, j * st

                dx[:, :, h0:h0 + s, w0:w0 + s] += (
                    self.mask[:, :, h0:h0 + s, w0:w0 + s]
                    * dout[:, :, i:i + 1, j:j + 1]
                )

        return dx


class Flat:
    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout, lr=None):
        return dout.reshape(self.shape)


class Dense:
    def __init__(self, din, dout):
        scale = np.sqrt(2.0 / din)

        self.w = rng.normal(
            0, scale, size=(din, dout)
        )
        self.b = np.zeros((1, dout))

    def forward(self, x):
        self.x = x
        return x @ self.w + self.b

    def backward(self, dout, lr):
        dw = self.x.T @ dout
        db = dout.sum(axis=0, keepdims=True)
        dx = dout @ self.w.T

        self.w -= lr * dw
        self.b -= lr * db

        return dx


def loss_fn(logits, y):
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    p = exp / exp.sum(axis=1, keepdims=True)

    n = logits.shape[0]
    loss = -np.log(p[np.arange(n), y] + 1e-12).mean()

    dp = p.copy()
    dp[np.arange(n), y] -= 1
    dp /= n

    return loss, dp


class CNN:
    def __init__(self):
        self.c1 = Conv(1, 8, 3, 1, 1)
        self.r1 = ReLU()
        self.p1 = Pool(2, 2)
        self.flat = Flat()
        self.f1 = Dense(8 * 4 * 4, 32)
        self.r2 = ReLU()
        self.f2 = Dense(32, 10)

    def forward(self, x):
        x = self.c1.forward(x)
        x = self.r1.forward(x)
        x = self.p1.forward(x)
        x = self.flat.forward(x)
        x = self.f1.forward(x)
        x = self.r2.forward(x)
        x = self.f2.forward(x)

        return x

    def backward(self, grad, lr):
        grad = self.f2.backward(grad, lr)
        grad = self.r2.backward(grad)
        grad = self.f1.backward(grad, lr)
        grad = self.flat.backward(grad)
        grad = self.p1.backward(grad)
        grad = self.r1.backward(grad)
        grad = self.c1.backward(grad, lr)

        return grad

    def predict(self, x):
        return self.forward(x).argmax(axis=1)


def train():
    xtr, xte, ytr, yte = load_data()
    model = CNN()

    epochs = 15
    bs = 32
    lr = 0.05
    n = xtr.shape[0]

    hist = {
        "train_loss": [],
        "train_acc": [],
        "test_acc": []
    }

    for ep in range(1, epochs + 1):
        idx = rng.permutation(n)
        xtr, ytr = xtr[idx], ytr[idx]

        total_loss = 0
        correct = 0

        for start in range(0, n, bs):
            xb = xtr[start:start + bs]
            yb = ytr[start:start + bs]

            out = model.forward(xb)
            loss, grad = loss_fn(out, yb)

            model.backward(grad, lr)

            total_loss += loss * xb.shape[0]
            correct += (out.argmax(axis=1) == yb).sum()

        tr_loss = total_loss / n
        tr_acc = correct / n

        pred = model.predict(xte)
        te_acc = (pred == yte).mean()

        hist["train_loss"].append(tr_loss)
        hist["train_acc"].append(tr_acc)
        hist["test_acc"].append(te_acc)

        print(
            f"Epoch {ep:2d}/{epochs} | "
            f"train_loss={tr_loss:.4f} | "
            f"train_acc={tr_acc:.4f} | "
            f"test_acc={te_acc:.4f}"
        )

    pred = model.predict(xte)
    acc = (pred == yte).mean()

    print(f"\nFinal test accuracy: {acc:.4f}")

    return model, hist, (xte, yte, pred)


if __name__ == "__main__":
    train()
