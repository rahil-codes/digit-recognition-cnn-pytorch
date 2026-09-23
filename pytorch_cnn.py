import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def get_data(bs=32):
    d = load_digits()
    x = d.data.reshape(-1, 1, 8, 8).astype("float32") / 16.0
    y = d.target.astype("int64")

    xtr, xte, ytr, yte = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    tr_ds = TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr))
    te_ds = TensorDataset(torch.from_numpy(xte), torch.from_numpy(yte))

    tr_ld = DataLoader(tr_ds, batch_size=bs, shuffle=True)
    te_ld = DataLoader(te_ds, batch_size=bs, shuffle=False)

    return tr_ld, te_ld


class CNN(nn.Module):
    def __init__(self, n=10):
        super().__init__()

        self.c1 = nn.Conv2d(1, 8, 3, padding=1)
        self.r1 = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.flat = nn.Flatten()
        self.f1 = nn.Linear(8 * 4 * 4, 32)
        self.r2 = nn.ReLU()
        self.f2 = nn.Linear(32, n)

    def forward(self, x):
        x = self.c1(x)
        x = self.r1(x)
        x = self.pool(x)
        x = self.flat(x)
        x = self.f1(x)
        x = self.r2(x)
        x = self.f2(x)
        return x


def test(model, loader, dev):
    model.eval()
    cor = 0
    tot = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            out = model(x)
            pred = out.argmax(1)
            cor += (pred == y).sum().item()
            tot += y.size(0)

    return cor / tot


def train(ep=15, lr=0.05, bs=32):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_ld, te_ld = get_data(bs)
    model = CNN().to(dev)

    loss_fn = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=lr)

    for e in range(1, ep + 1):
        model.train()
        loss_sum = 0
        cor = 0
        tot = 0

        for x, y in tr_ld:
            x, y = x.to(dev), y.to(dev)

            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

            loss_sum += loss.item() * x.size(0)
            cor += (out.argmax(1) == y).sum().item()
            tot += y.size(0)

        loss = loss_sum / tot
        acc = cor / tot
        te_acc = test(model, te_ld, dev)

        print(
            f"Epoch {e:2d}/{ep} | loss={loss:.4f} | "
            f"train_acc={acc:.4f} | test_acc={te_acc:.4f}"
        )

    acc = test(model, te_ld, dev)
    print(f"\nFinal test accuracy: {acc:.4f}")

    return model

if __name__ == "__main__":
    train()
