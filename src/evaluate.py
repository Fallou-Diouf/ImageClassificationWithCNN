import torch


def evaluate(model, dataloader, device):

    model.eval()

    with torch.no_grad():

        correct = 0
        total = 0

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    accuracy = correct / total

    return accuracy