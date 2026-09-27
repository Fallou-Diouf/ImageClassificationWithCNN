from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

print("Télécharment ou chargement du dataset CIFAR10")

full_train_dataset = datasets.CIFAR10(
    root = "./data",
    train = True,
    transform = transforms.ToTensor(),
    download = True
)

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [45000, 5000]
)

test_dataset = datasets.CIFAR10(
    root = "./data",
    train = False,
    transform = transforms.ToTensor(),
    download = True
)

train_loader = DataLoader(
    train_dataset,
    batch_size = 32,
    shuffle = True
)

val_loader = DataLoader(
    val_dataset,
    batch_size = 32,
    shuffle = False
)

test_loader = DataLoader(
    test_dataset,
    batch_size = 32,
    shuffle = False
)