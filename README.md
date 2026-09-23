# Handwritten Digit Classification with a CNN (PyTorch)

A small convolutional neural network that classifies handwritten digits (0–9) from the scikit-learn `digits` dataset (8x8 grayscale images, 1797 samples).

## What this does

- Loads the `sklearn.datasets.load_digits` dataset and normalizes pixel values to `[0, 1]`
- Splits into train/test sets (80/20, stratified)
- Defines a small CNN: `Conv2D → ReLU → MaxPool → Flatten → Dense → ReLU → Dense`
- Trains with SGD and cross-entropy loss over 15 epochs
- Reports train/test accuracy per epoch and a final test accuracy

## Architecture

```
Input (1x8x8)
  → Conv2d(1 → 8 channels, 3x3 kernel, padding=1)
  → ReLU
  → MaxPool2d(2x2)                    # 8x8 → 4x4
  → Flatten                            # 8*4*4 = 128
  → Linear(128 → 32)
  → ReLU
  → Linear(32 → 10)                    # 10 digit classes
```

## Results

- Final test accuracy: **93.89%**
- Trained for 15 epochs, batch size 32, learning rate 0.05, SGD optimizer
- Accuracy climbed steadily from 25.8% (epoch 1) to 93.89% (epoch 15) as training loss dropped from 2.28 to 0.18

## Why this dataset

`sklearn.digits` is used instead of full MNIST because it's built into scikit-learn (no download/internet dependency), while still being real, non-trivial image data — useful for testing an implementation end-to-end before scaling up.

## Requirements

```
torch>=2.0
scikit-learn>=1.3
matplotlib>=3.7
```

## Running it

```bash
pip install -r requirements.txt
python pytorch_cnn.py
```

## Notes

- GPU is used automatically if available (`torch.cuda.is_available()`), otherwise falls back to CPU.
- A from-scratch NumPy implementation of the same architecture (manual forward/backward pass, no autograd) is included separately as a reference for understanding what PyTorch's `autograd` and `nn.Conv2d` do under the hood.
