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
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR: str = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "brain-tumor-dataset"))
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TransferLearningConfig(Config):
    NUM_EPOCHS: int = 20
    TEST_RATIO: float = 0.2
    LEARNING_RATE: float = 0.0001


@dataclass
class TraditionalLearningConfig(Config):
    NUM_EPOCHS: int = 100
    VAL_RATIO: float = 0.1
    TEST_RATIO: float = 0.3
    LEARNING_RATE: float = 0.01
