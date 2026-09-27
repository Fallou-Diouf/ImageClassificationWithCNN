# Image Classification with CNN — PyTorch

A complete image classification pipeline built with **PyTorch** and a Convolutional Neural Network (CNN) for classifying images from the **CIFAR-10** dataset.

The project was developed step by step to understand the complete deep learning workflow, from dataset preparation to model evaluation.

---

## Project Overview

The objective of this project is to build and train a CNN capable of classifying RGB images into 10 different categories.

The project covers:

- CIFAR-10 dataset preparation
- Train / validation / test split
- PyTorch `Dataset` and `DataLoader`
- CNN architecture design
- Cross-Entropy Loss
- Adam optimizer
- Training loop
- Validation during training
- Accuracy monitoring
- Training on Apple Silicon using **MPS**
- Final evaluation on the test set
- Training and validation accuracy visualization

---

## Dataset

The project uses the **CIFAR-10** dataset.

CIFAR-10 contains:

- 60,000 RGB images
- Image size: `32 × 32`
- 10 classes
- 50,000 training images
- 10,000 test images

The original training set is split into:

| Dataset | Number of images |
|---|---:|
| Training | 45,000 |
| Validation | 5,000 |
| Test | 10,000 |

The test set is kept separate and is only used for the final evaluation.

### CIFAR-10 Classes

The 10 classes are:

```text
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
```

---

## CNN Architecture

The model is a simple CNN designed specifically for CIFAR-10.

```text
Input
[3, 32, 32]
      ↓
Conv2d
3 → 8 channels
Kernel: 3×3
Padding: 1
      ↓
ReLU
      ↓
MaxPool 2×2
      ↓
[8, 16, 16]
      ↓
Conv2d
8 → 16 channels
Kernel: 3×3
Padding: 1
      ↓
ReLU
      ↓
MaxPool 2×2
      ↓
[16, 8, 8]
      ↓
Flatten
      ↓
1024 features
      ↓
Linear
1024 → 10
      ↓
Output
[10]
```

The final layer produces **10 logits**, one for each CIFAR-10 class.

The model does not apply Softmax to the final output because `CrossEntropyLoss` internally handles the required normalization.

---

## Training

The model is trained using:

- **Loss:** Cross-Entropy Loss
- **Optimizer:** Adam
- **Learning rate:** `0.001`
- **Epochs:** `8`
- **Batch size:** `32`

The training process follows the standard PyTorch workflow:

```text
Forward Pass
     ↓
Loss Calculation
     ↓
Backward Pass
     ↓
Gradient Update
```

For each batch:

1. The images are passed through the CNN.
2. The model produces logits.
3. Cross-Entropy Loss is calculated.
4. Gradients are computed using backpropagation.
5. The optimizer updates the model parameters.

---

## Hardware Acceleration

The project supports Apple Silicon acceleration through **Metal Performance Shaders (MPS)**.

The device is automatically selected:

```python
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
```

This allows the model to run on the Mac GPU when MPS is available.

During training, both the model and input tensors are moved to the selected device:

```python
model = CNN().to(device)

images = images.to(device)
labels = labels.to(device)
```

---

## Validation and Overfitting Monitoring

The model is evaluated on the validation set after each training epoch.

Two metrics are monitored:

- Training Accuracy
- Validation Accuracy

This allows us to compare the model's performance on training and unseen validation data.

For example:

```text
Epoch 7/8 | Train Acc: 0.6530 | Val Acc: 0.6202
Epoch 8/8 | Train Acc: 0.6546 | Val Acc: 0.6178
```

The training accuracy continues to increase while the validation accuracy slightly decreases between epochs 7 and 8.

This provides a small indication that the model may be starting to overfit the training data.

---

## Results

After 8 training epochs:

| Metric | Result |
|---|---:|
| Best Validation Accuracy | **62.02%** |
| Final Validation Accuracy | **61.78%** |
| Test Accuracy | **61.75%** |

The best validation accuracy was obtained at **epoch 7**.

The final test accuracy was:

```text
61.75%
```

The training and validation accuracy were monitored during training to observe the model's learning behavior and identify potential overfitting.

### Accuracy Curve

The training and validation accuracy were visualized across epochs.

The validation accuracy reaches its highest value around epoch 7, while the training accuracy continues to increase slightly afterwards.

This difference between training and validation performance illustrates why validation monitoring is important during model training.

---

## Project Structure

```text
ImageClassificationWithCNN/
│
├── data/
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── README.md
└── .gitignore
```

### `dataset.py`

Responsible for:

- Downloading CIFAR-10
- Creating the training dataset
- Splitting the training data into training and validation sets
- Creating the test dataset
- Creating the DataLoaders

### `model.py`

Contains the CNN architecture implemented with `torch.nn`.

The model contains:

- Two convolutional layers
- ReLU activation functions
- Max pooling layers
- A flattening operation
- One fully connected layer

### `train.py`

Contains:

- Device selection
- Model initialization
- Loss function
- Optimizer
- Training loop
- Validation
- Accuracy tracking
- Accuracy visualization
- Final test evaluation

### `evaluate.py`

Contains the evaluation function used to calculate classification accuracy on a given dataset.

---

## Technologies

- Python
- PyTorch
- Torchvision
- Matplotlib
- Apple MPS

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Fallou-Diouf/ImageClassificationWithCNN.git
```

Go to the project directory:

```bash
cd ImageClassificationWithCNN
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install torch torchvision matplotlib
```

---

## How to Run

Run the training script:

```bash
python3 src/train.py
```

The CIFAR-10 dataset will be downloaded automatically into the `data/` directory the first time the project is executed.

The training process will display the training and validation accuracy for each epoch:

```text
Epoch 1/8 | Train Acc: 0.5026 | Val Acc: 0.4934
Epoch 2/8 | Train Acc: 0.5460 | Val Acc: 0.5460
Epoch 3/8 | Train Acc: 0.5924 | Val Acc: 0.5848
Epoch 4/8 | Train Acc: 0.6106 | Val Acc: 0.5980
Epoch 5/8 | Train Acc: 0.6156 | Val Acc: 0.5916
Epoch 6/8 | Train Acc: 0.6252 | Val Acc: 0.6008
Epoch 7/8 | Train Acc: 0.6530 | Val Acc: 0.6202
Epoch 8/8 | Train Acc: 0.6546 | Val Acc: 0.6178

Test Accuracy: 0.6175
```

An accuracy curve comparing training and validation performance is also displayed after training.

---

## Future Improvements

Possible improvements for future versions include:

- Data augmentation
- Batch Normalization
- Dropout
- A deeper CNN architecture
- Learning rate scheduling
- Best-model checkpointing
- Confusion matrix
- Precision, Recall and F1-score
- Per-class accuracy
- Class activation visualization
- Comparison with transfer learning models such as ResNet
- Experiment tracking

---

## Learning Objectives

This project was also designed as a practical learning exercise to understand the fundamental components of image classification with PyTorch.

The main concepts covered are:

- Tensors and tensor shapes
- CNN architecture
- Convolution
- Padding
- Pooling
- ReLU activation
- Flattening
- Fully connected layers
- Logits
- Cross-Entropy Loss
- Backpropagation
- Optimizers
- Training loops
- Dataset and DataLoader
- Train / validation / test split
- Model evaluation
- Accuracy
- Overfitting monitoring
- GPU acceleration with Apple MPS

---

## Author

**Fallou Diouf**

Master's student in **Vision and Machine Intelligence**  
Université Paris Cité

### Interests

- Computer Vision
- Deep Learning
- Machine Learning
- Image Understanding
- Artificial Intelligence
