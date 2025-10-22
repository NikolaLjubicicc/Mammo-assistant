import torch
import cv2
import numpy as np
from pathlib import Path

from config import IMAGE_SIZE, DEVICE, MODEL_SAVE_PATH
from model import create_model

def load_model():
    model = create_model(pretrained=False)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def preprocess_image(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    img = cv2.resize(img, IMAGE_SIZE)
    img = np.repeat(img[..., None], 3, axis=2)
    img = np.moveaxis(img, -1, 0) / 255.0
    
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0)

def predict(img_path, model=None):
    if model is None:
        model = load_model()
    
    img = preprocess_image(img_path).to(DEVICE)
    
    with torch.no_grad():
        out = torch.sigmoid(model(img)).item()
    
    return {
        "probability": out,
        "class": "malignant" if out > 0.5 else "benign",
        "suspicious": out > 0.5
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)
    
    img_path = sys.argv[1]
    result = predict(img_path)
    
    print(f"  Prediction for: {img_path}")
    print(f"  Class: {result['class'].upper()}")
    print(f"  Probability: {result['probability']:.2%}")
    print(f"  Suspicious: {'YES' if result['suspicious'] else 'NO'}")