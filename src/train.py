import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np

from config import (
    INDEX_CSV, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS,
    TEST_SIZE, RANDOM_STATE, DEVICE, MODEL_SAVE_PATH
)
from dataset import MammoDataset
from model import create_model

def create_balanced_sampler(df):
    class_counts = df["label"].value_counts().to_dict()
    weights = [1.0 / class_counts[label] for label in df["label"]]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
    
    return total_loss / len(dataloader.dataset)

def validate(model, dataloader, device):
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            out = torch.sigmoid(model(x)).cpu().numpy().ravel()
            y_pred.extend(out)
            y_true.extend(y.numpy())
    
    auc = roc_auc_score(y_true, y_pred)
    f1 = f1_score(y_true, np.round(y_pred))
    
    return auc, f1, y_true, y_pred

def train():
    print(f" Training on device: {DEVICE}")
    
    print(" Loading dataset...")
    df = pd.read_csv(INDEX_CSV)
    print(f"Total images: {len(df)}")
    print(f"Benign: {(df['label'] == 0).sum()} | Malignant: {(df['label'] == 1).sum()}")
    
    df_train, df_val = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["label"], random_state=RANDOM_STATE
    )
    
    sampler = create_balanced_sampler(df_train)
    train_dl = DataLoader(
        MammoDataset(df_train, augment=True),
        batch_size=BATCH_SIZE,
        sampler=sampler
    )
    val_dl = DataLoader(
        MammoDataset(df_val, augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    print(" Creating model...")
    model = create_model(pretrained=True, fine_tune_layers=3)
    model.to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()
    
    print("Starting training...\n")
    best_auc = 0
    
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_dl, optimizer, loss_fn, DEVICE)
        
        auc, f1, _, _ = validate(model, val_dl, DEVICE)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}: "
              f"Loss={train_loss:.4f}, AUROC={auc:.3f}, F1={f1:.3f}")
    
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f" Best model saved (AUROC: {auc:.3f})")
    
    print(f" Training complete! Best AUROC: {best_auc:.3f}")
    print(f" Model saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()