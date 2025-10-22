import re
import pandas as pd
from pathlib import Path
from config import RAW_DIR, INDEX_CSV, PROCESSED_DIR

def find_jpeg_for_dcm(dcm_path):

    matches = re.findall(r"1\.3\.6\.1\.4\.1\.[0-9\.]+", str(dcm_path))
    if not matches:
        return None
    
    dicom_id = matches[-1]
    folder = RAW_DIR / dicom_id
    
    if not folder.exists():
        return None
    
    jpgs = list(folder.glob("*.jpg"))
    if not jpgs:
        return None
    
    return str(jpgs[0])

def prepare_dataset(csv_path):

    print("Loading raw CSV...")
    df = pd.read_csv(csv_path)
    
    print("Linking JPEG files...")
    df["img_path"] = df["image file path"].apply(find_jpeg_for_dcm)
    df = df.dropna(subset=["img_path"])
    
    print(f"Connected images: {len(df)}")
    
    df["pathology"] = df["pathology"].astype(str).str.lower()
    df["label"] = df["pathology"].map({"benign": 0, "malignant": 1})
    df = df.dropna(subset=["img_path", "label"])
    
    df = df.drop_duplicates(subset=["img_path"])
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(INDEX_CSV, index=False)
    
    print(f"\nDataset statistics:")
    print(f"Total images: {len(df)}")
    print(f"Benign: {(df['label'] == 0).sum()}")
    print(f"Malignant: {(df['label'] == 1).sum()}")
    print(f"Saved to: {INDEX_CSV}")
    
    return df

if __name__ == "__main__":
    raw_csv = "data/index.csv"
    prepare_dataset(raw_csv)