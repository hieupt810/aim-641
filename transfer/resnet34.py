import logging

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet34_Weights, resnet34

from config import Config, TransferLearningConfig
from utils import calculateNormalizeParameters, configureFileLogging, fixRandomSeed

fixRandomSeed()
configureFileLogging("pretrained_resnet34_training.log")

logger = logging.getLogger(__name__)

# Data transformations
mean, std = calculateNormalizeParameters()
transform = transforms.Compose(
    [
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

# Dataset
dataset = ImageFolder(root=Config.DATA_DIR, transform=transform)
logger.info("Loaded dataset with %d samples.", len(dataset))

# Split dataset into training and testing sets
test_size = int(len(dataset) * TransferLearningConfig.TEST_RATIO)
train_size = len(dataset) - test_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
logger.info("Training samples: %d, Testing samples: %d", train_size, test_size)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

# Model, optimizer, and loss function
model = resnet34(weights=ResNet34_Weights.DEFAULT).to(Config.DEVICE)
optimizer = Adam(model.parameters(), lr=TransferLearningConfig.LEARNING_RATE)
criterion = CrossEntropyLoss()

# Training and testing loop
train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []
best_accuracy = 0.0

for epoch in range(1, TransferLearningConfig.NUM_EPOCHS):
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
                torch.save(model.state_dict(), f"pretrained_resnet34_epoch_{epoch}.pth")

        logger.info(
            "Epoch %d [%s] - Loss: %.4f, Accuracy: %.4f",
            epoch,
            phase,
            epoch_loss,
            epoch_accuracy,
        )

logger.info("Training complete. Best test accuracy: %.4f", best_accuracy)
