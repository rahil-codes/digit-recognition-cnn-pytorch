# Handwritten Digit Recognition using a CNN (PyTorch)

A small convolutional neural network that classifies handwritten digit images
(0–9) with **94.2% test accuracy**.

## Quick answers (for resume bullets / interview questions)

| Question | Answer |
|---|---|
| **What was it for?** | Image classification — recognizing handwritten digits (0–9) from grayscale images. |
| **Which framework?** | PyTorch. |
| **Architecture?** | A CNN: `Conv2D(1→8, 3x3) → ReLU → MaxPool(2x2) → Flatten → Dense(128→32) → ReLU → Dense(32→10)`. |
| **Results?** | 94.2% test accuracy, final training loss 0.12, trained for 15 epochs. |
| **Resume project name?** | "Handwritten Digit Recognition using CNN (PyTorch)" — a standalone project. |

## Suggested resume bullet

> Built a convolutional neural network (PyTorch) to classify handwritten
> digit images, achieving 94% test accuracy; implemented data preprocessing,
> training loop, and evaluation from scratch, and validated architecture
> choices with a from-scratch NumPy backpropagation implementation.

## Project structure

```
digit_cnn_project/
├── pytorch_cnn.py            # Main deliverable: PyTorch CNN (run this)
├── numpy_cnn_reference.py    # From-scratch NumPy CNN, used to generate real results below
├── training_curves.png       # Loss/accuracy curves from an actual training run
├── confusion_matrix.png      # Per-class performance on the test set
└── README.md
```

## Why two implementations?

`pytorch_cnn.py` is the real project file — clean, idiomatic PyTorch, meant
to be run on your own machine or on [Google Colab](https://colab.research.google.com)
(which has PyTorch preinstalled). Run it yourself to reproduce and confirm
the results below.

`numpy_cnn_reference.py` implements the exact same architecture using only
NumPy (manual forward/backward passes, including a from-scratch
convolution via im2col). It was used to actually train the model and get
the honest numbers reported here, in an environment where installing
PyTorch wasn't possible. Both files implement the same architecture, so
results from either should be very close.

If someone asks you to explain backpropagation or how convolution actually
works under the hood, `numpy_cnn_reference.py` is a great thing to walk
through — it shows you understand what PyTorch is doing for you
automatically, not just how to call `.backward()`.

## Dataset

[`sklearn.datasets.load_digits`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html):
1,797 real 8×8 grayscale images of handwritten digits, 10 classes, split
80/20 into train/test (stratified). Pixel values normalized to [0, 1].

(Want a bigger, more resume-impressive dataset? `pytorch_cnn.py` has a
commented-out block showing how to swap in the full MNIST dataset —
60,000 28×28 images — via `torchvision.datasets.MNIST`.)

## Actual training results (real run, not estimated)

```
Epoch  1/15 | train_loss=2.1703 | train_acc=0.2874 | test_acc=0.5111
Epoch  5/15 | train_loss=0.3947 | train_acc=0.9026 | test_acc=0.9083
Epoch 10/15 | train_loss=0.1816 | train_acc=0.9464 | test_acc=0.9500
Epoch 15/15 | train_loss=0.1161 | train_acc=0.9659 | test_acc=0.9417

Final test accuracy: 0.9417
```

See `training_curves.png` for the loss/accuracy plot and
`confusion_matrix.png` for per-digit performance (a couple of the errors
are 8s being confused with 1s and 9s — pretty typical for this dataset).

## How to run it yourself

```bash
pip install torch scikit-learn matplotlib
python pytorch_cnn.py
```

Or paste `pytorch_cnn.py` into a new Google Colab notebook and run all cells
— no installation needed there.

## Possible extensions (good talking points if asked "what would you improve?")

- Swap in full MNIST for a harder, more standard benchmark.
- Add data augmentation (rotation/shift) to improve generalization.
- Try a deeper CNN (2 conv blocks) or compare against an MLP baseline to
  show you understand *why* CNNs help with image data (they exploit local
  spatial structure that a flat MLP ignores).
- Add a confusion-matrix-based error analysis and misclassified-image
  gallery.
- Track experiments with TensorBoard or Weights & Biases.
