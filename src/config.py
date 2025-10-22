import torch
from pathlib import Path

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw" / "jpeg"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_CSV = PROCESSED_DIR / "index.csv"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 4
LEARNING_RATE = 3e-4
NUM_EPOCHS = 8
TEST_SIZE = 0.2
RANDOM_STATE = 42

IMAGE_SIZE = (224, 224)
DEVICE = "cpu"
MODEL_SAVE_PATH = MODELS_DIR / "best_model.pt"

CLASS_MAPPING = {"benign": 0, "malignant": 1}