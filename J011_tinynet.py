"""

**Name: Shashank Goel**
**Roll no: J011**
**SAP ID: 70092300008**
**Batch: B.Tech Data Science**

tinynet — the smallest neural network library that actually works.

This file is the spine of the course. Today it is a black box: you call it,
it trains, you get a number. Week by week we will open it up, and by the end
of the deep-networks week you will have written every line in here yourself.

No TensorFlow, no PyTorch. Just numpy. Nothing in here is magic.

Conventions (same as our notes): X has shape (n_x, m) — each COLUMN is one
example. y has shape (1, m).
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt


def sigmoid(z):
    # Squashes any score into (0, 1) — a probability.
    # The clip only prevents numerical overflow for extreme scores;
    # sigmoid(±30) is already indistinguishable from 1 or 0.
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def relu(z):
    # The "hinge": negative scores are silenced to 0, positive pass through.
    return np.maximum(0.0, z)


class Net:
    """A fully-connected neural network. Net(layers=[12288, 16, 1]) means:
    12288 inputs -> 16 hidden neurons -> 1 output neuron."""

    def __init__(self, layers, seed=1):
        self.layers = layers
        rng = np.random.default_rng(seed)  # fixed seed: same layers, same result — leaderboards stay fair
        # Weights start as small random numbers, biases as zeros.
        # The sqrt(2/n) scaling is a good default — WHY it matters is a story
        # for the "improving deep networks" weeks. For now: trust the defaults.
        self.W = [rng.standard_normal((layers[i + 1], layers[i])) * np.sqrt(2.0 / layers[i])
                  for i in range(len(layers) - 1)]
        self.b = [np.zeros((layers[i + 1], 1)) for i in range(len(layers) - 1)]
        self.history = {"loss": [], "train_acc": [], "test_acc": []}

    def _forward(self, X):
        # Push the data through every layer, remembering what we saw on the
        # way (Z's and A's) — the backward pass needs them to assign blame.
        A, Zs, As = X, [], [X]
        last = len(self.W) - 1
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            Z = W @ A + b                              # every neuron: weighted score + bias
            A = sigmoid(Z) if i == last else relu(Z)   # hidden layers hinge, the output squashes
            Zs.append(Z)
            As.append(A)
        return A, Zs, As

    def _backward(self, y, Zs, As):
        # Walk the network in reverse, computing how much each weight is to
        # blame for the loss. This is backpropagation — the course's centerpiece.
        m = y.shape[1]
        grads_W, grads_b = [None] * len(self.W), [None] * len(self.b)
        dZ = As[-1] - y  # the famous collapse: prediction minus truth
        for i in reversed(range(len(self.W))):
            grads_W[i] = (dZ @ As[i].T) / m
            grads_b[i] = np.sum(dZ, axis=1, keepdims=True) / m
            if i > 0:
                dZ = (self.W[i].T @ dZ) * (Zs[i - 1] > 0)  # pass blame back through the hinge
        return grads_W, grads_b

    def train(self, X, y, X_test=None, y_test=None, epochs=2000, lr=0.05, log_every=None):
        """Full-batch gradient descent: every epoch looks at ALL examples,
        then takes one downhill step. (Smarter stepping comes in later weeks.)"""
        log_every = log_every or max(1, epochs // 10)
        for epoch in range(1, epochs + 1):
            A, Zs, As = self._forward(X)
            # Binary cross-entropy: how embarrassed the network is, on average.
            eps = 1e-12  # keeps log() away from log(0)
            loss = -np.mean(y * np.log(A + eps) + (1 - y) * np.log(1 - A + eps))
            grads_W, grads_b = self._backward(y, Zs, As)
            for i in range(len(self.W)):
                self.W[i] -= lr * grads_W[i]           # one step downhill
                self.b[i] -= lr * grads_b[i]

            self.history["loss"].append(loss)
            self.history["train_acc"].append(self.accuracy(X, y))
            if X_test is not None:
                self.history["test_acc"].append(self.accuracy(X_test, y_test))
            if epoch % log_every == 0 or epoch == 1:
                test_msg = f" | test acc {self.history['test_acc'][-1]:5.1%}" if X_test is not None else ""
                print(f"epoch {epoch:>5} | loss {loss:.4f} | train acc {self.history['train_acc'][-1]:5.1%}{test_msg}")
        return self

    def predict(self, X):
        A, _, _ = self._forward(X)
        return (A > 0.5).astype(int)

    def predict_proba(self, X):
        A, _, _ = self._forward(X)
        return A

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))

    def plot(self):
        """Loss and accuracy curves. When the two accuracy lines drift apart,
        the network is memorizing instead of learning — remember that sight."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))
        ax1.plot(self.history["loss"], color="tab:red")
        ax1.set(title="loss (embarrassment)", xlabel="epoch")
        ax2.plot(self.history["train_acc"], label="train", color="tab:blue")
        if self.history["test_acc"]:
            ax2.plot(self.history["test_acc"], label="test", color="tab:orange")
        ax2.set(title="accuracy", xlabel="epoch", ylim=(0, 1.05))
        ax2.axhline(0.5, color="gray", ls=":", lw=1)
        ax2.legend()
        fig.tight_layout()
        plt.show()


def load_cat_data(data_dir="data"):
    """Load the classic cat/not-cat dataset (209 train, 50 test, 64x64 RGB).

    Returns flattened, 0-1 scaled matrices ready for a Net, plus the raw
    images and class names for plotting."""
    with h5py.File(f"{data_dir}/train_catvnoncat.h5", "r") as f:
        train_imgs = np.array(f["train_set_x"])          # (209, 64, 64, 3) uint8
        y_train = np.array(f["train_set_y"]).reshape(1, -1)
        classes = [c.decode() for c in f["list_classes"]]
    with h5py.File(f"{data_dir}/test_catvnoncat.h5", "r") as f:
        test_imgs = np.array(f["test_set_x"])
        y_test = np.array(f["test_set_y"]).reshape(1, -1)

    # Each 64x64x3 image becomes one COLUMN of 12288 numbers.
    # Dividing by 255 puts every pixel in [0, 1] — gradient descent takes
    # much happier steps when all inputs live on the same scale.
    X_train = train_imgs.reshape(train_imgs.shape[0], -1).T / 255.0
    X_test = test_imgs.reshape(test_imgs.shape[0], -1).T / 255.0
    return X_train, y_train, X_test, y_test, train_imgs, test_imgs, classes
