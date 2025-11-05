import os
from dataclasses import dataclass

import torch
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    SEED: int = 42
    BATCH_SIZE: int = 32
    IMAGE_SIZE: int = 224
    DATA_DIR: str = os.getenv("DATA_DIR", "./brain-tumor-dataset")
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class PretrainedConfig(Config):
    NUM_EPOCHS: int = 20
    TEST_RATIO: float = 0.2
    LEARNING_RATE: float = 0.0001
