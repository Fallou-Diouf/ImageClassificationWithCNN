import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from model import CNN
from dataset import train_loader, val_loader, test_loader
from evaluate import evaluate

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr = 0.001
)

history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": []
}

epochs = 8

for epoch in range(epochs):

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        output = model(images)

        loss = criterion(output, labels)

        loss.backward()
        
        optimizer.step()

    train_accuracy = evaluate(
        model,
        train_loader,
        device
    )

    val_accuracy = evaluate(
        model,
        val_loader,
        device
    )

    history["train_acc"].append(train_accuracy)
    history["val_acc"].append(val_accuracy)

    print(
    f"Epoch {epoch + 1}/{epochs} | " f"Train Acc: {train_accuracy:.4f} | " f"Val Acc: {val_accuracy:.4f}"
    )

test_accuracy = evaluate(
        model,
        test_loader,
        device
    )
print(f"Test Accuracy: {test_accuracy:.4f}")

epochs_range = range(1, epochs + 1)

plt.plot(
    epochs_range,
    history["train_acc"],
    label="Train accuracy"
)

plt.plot(
    epochs_range,
    history["val_acc"],
    label="Validation accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("acc")
plt.legend()
plt.show()