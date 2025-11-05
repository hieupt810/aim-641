import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from config import Config


def fixRandomSeed(seed: int = Config.SEED):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def configureFileLogging(filename: str):
    path = os.path.join(Config.BASE_DIR, "logs", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    logging.basicConfig(
        filename=path,
        filemode="a",
        format="[%(asctime)s][%(levelname)s] %(message)s",
        level=logging.INFO,
    )


def calculateNormalizeParameters(
    data_dir=Config.DATA_DIR, image_size=Config.IMAGE_SIZE, batch_size=Config.BATCH_SIZE
):
    transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    )

    # Load dataset
    dataset = ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Initialize parameters
    mean, std = 0.0, 0.0
    total_images = 0
    for images, _ in loader:
        samples = images.size(0)
        images = images.view(samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += samples

    mean /= total_images
    std /= total_images
    return mean, std
