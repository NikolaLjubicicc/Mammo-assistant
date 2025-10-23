import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import time

from config import INDEX_CSV, BATCH_SIZE, TEST_SIZE, RANDOM_STATE, DEVICE
from dataset import MammoDataset
from model import create_model

def evaluate_model(model_name):
    print(f"\nEvaluating {model_name.upper()}...")
    
    df = pd.read_csv(INDEX_CSV)
    _, df_val = train_test_split(df, test_size=TEST_SIZE, stratify=df["label"], random_state=RANDOM_STATE)
    val_dl = DataLoader(MammoDataset(df_val, augment=False), batch_size=BATCH_SIZE, shuffle=False)

    model = create_model(model_name=model_name, pretrained=False)
    model_path = f"models/best_{model_name}.pt"
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    y_true, y_pred = [], []
    start_time = time.time()
    
    with torch.no_grad():
        for x, y in val_dl:
            x = x.to(DEVICE)
            out = torch.sigmoid(model(x)).cpu().numpy().ravel()
            y_pred.extend(out)
            y_true.extend(y.numpy())
    
    inference_time = time.time() - start_time
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    metrics = {
        "Model": model_name.upper(),
        "AUROC": roc_auc_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred_binary),
        "Precision": precision_score(y_true, y_pred_binary),
        "Recall": recall_score(y_true, y_pred_binary),
        "Inference Time (s)": inference_time
    }
    
    return metrics

def compare_models():
    models = ["mobilenet", "resnet18", "densenet"]  
    
    results = []
    for model_name in models:
        try:
            metrics = evaluate_model(model_name)
            results.append(metrics)
        except FileNotFoundError:
            print(f"Model {model_name} not found. Skipping...")
    
    df_results = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(df_results.to_string(index=False))
    print("="*80)
    
    df_results.to_csv("model_comparison.csv", index=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].bar(df_results["Model"], df_results["AUROC"], color='steelblue')
    axes[0, 0].set_title('AUROC Comparison')
    axes[0, 0].set_ylabel('AUROC')
    axes[0, 0].set_ylim(0, 1)
    
    axes[0, 1].bar(df_results["Model"], df_results["F1"], color='coral')
    axes[0, 1].set_title('F1 Score Comparison')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_ylim(0, 1)
    
    axes[1, 0].scatter(df_results["Precision"], df_results["Recall"], s=200, alpha=0.6)
    for i, model in enumerate(df_results["Model"]):
        axes[1, 0].annotate(model, (df_results["Precision"].iloc[i], df_results["Recall"].iloc[i]))
    axes[1, 0].set_xlabel('Precision')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].set_title('Precision vs Recall')
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    
    axes[1, 1].bar(df_results["Model"], df_results["Inference Time (s)"], color='lightgreen')
    axes[1, 1].set_title('Inference Time Comparison')
    axes[1, 1].set_ylabel('Time (seconds)')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    print("\nComparison plot saved to: model_comparison.png")

if __name__ == "__main__":
    compare_models()