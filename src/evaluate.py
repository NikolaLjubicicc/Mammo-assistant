import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve
)
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from config import INDEX_CSV, BATCH_SIZE, TEST_SIZE, RANDOM_STATE, DEVICE, MODEL_SAVE_PATH
from dataset import MammoDataset
from model import create_model

def evaluate():
    print("Evaluating model...")
    
    df = pd.read_csv(INDEX_CSV)
    _, df_val = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["label"], random_state=RANDOM_STATE
    )
    
    val_dl = DataLoader(
        MammoDataset(df_val, augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    model = create_model(pretrained=False)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in val_dl:
            x = x.to(DEVICE)
            out = torch.sigmoid(model(x)).cpu().numpy().ravel()
            y_pred.extend(out)
            y_true.extend(y.numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    auc = roc_auc_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    cm = confusion_matrix(y_true, y_pred_binary)
    
    print("Validation Metrics:")
    print(f"  AUROC:     {auc:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"Confusion Matrix:")
    print(f"  {cm}")
    
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Mammography Classification')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
    print("ROC curve saved to: roc_curve.png")

if __name__ == "__main__":
    evaluate()