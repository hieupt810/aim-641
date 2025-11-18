from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


@dataclass
class Config:
    SEED = 42
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    IMAGE_SIZE = 224
    TEST_RATIO = 0.3
    MODEL_NAME = "efficientnet_b0"
    DATA_DIR = "/kaggle/input/brain-tumor-dataset"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print(f"Using device: {Config.DEVICE}")


def fixRandomSeed(seed: int = Config.SEED):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


fixRandomSeed()
mean = torch.tensor([0.2006, 0.1854, 0.2579])
std = torch.tensor([0.1391, 0.1799, 0.1537])

# Data transformations
transform = transforms.Compose(
    [
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)
dataset = ImageFolder(root=Config.DATA_DIR, transform=transform)

# Split the dataset into training and testing sets
test_size = int(len(dataset) * Config.TEST_RATIO)
train_size = len(dataset) - test_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT).to(Config.DEVICE)
optimizer = Adam(model.parameters(), lr=0.001)
criterion = CrossEntropyLoss()

# Freeze all layers except the classifier
for param in model.parameters():
    param.requires_grad = False

for param in model.classifier.parameters():
    param.requires_grad = True

# Training and testing loop
train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []
best_accuracy = 0.0

for epoch in range(1, Config.NUM_EPOCHS + 1):
    for phase in ["train", "test"]:
        if phase == "train":
            model.train()
            dataloader = train_loader
        else:
            model.eval()
            dataloader = test_loader

        running_loss = 0.0
        correct_predictions = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == "train"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            correct_predictions += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_accuracy = correct_predictions.double() / len(dataloader.dataset)
        if phase == "train":
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_accuracy.item())
        else:
            test_losses.append(epoch_loss)
            test_accuracies.append(epoch_accuracy.item())
            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                torch.save(model.state_dict(), f"best_{Config.MODEL_NAME}_{epoch}.pth")

    print(
        f"Epoch {epoch}/{Config.NUM_EPOCHS} - "
        f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f} - "
        f"Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accuracies[-1]:.4f}"
    )
