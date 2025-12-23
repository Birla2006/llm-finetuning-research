"""
Data preparation script for resume classification task.

This script:
1. Loads raw resume data from Kaggle
2. Cleans and preprocesses text
3. Creates train/val/test splits
4. Saves processed data in formats ready for fine-tuning
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple
import re

# Set random seed for reproducibility
np.random.seed(42)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

def clean_text(text: str) -> str:
    """Clean resume text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep periods and commas
    text = re.sub(r'[^\w\s.,]', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    return text.strip()

def load_and_process_resumes(data_path: str) -> pd.DataFrame:
    """Load and preprocess resume dataset."""
    print(f"Loading data from {data_path}...")

    # Assuming CSV format - adjust based on actual dataset
    df = pd.read_csv(data_path)

    # Print dataset info
    print(f"Total samples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    # Clean text (adjust column name based on dataset)
    text_column = 'Resume' if 'Resume' in df.columns else 'resume_text'
    df['text'] = df[text_column].apply(clean_text)

    # Get category column (adjust based on dataset)
    category_column = 'Category' if 'Category' in df.columns else 'job_category'
    df['label'] = df[category_column]

    # Print category distribution
    print("\nCategory distribution:")
    print(df['label'].value_counts())

    return df[['text', 'label']]

def create_splits(df: pd.DataFrame,
                  train_size: float = 0.7,
                  val_size: float = 0.15,
                  test_size: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/val/test splits."""
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Splits must sum to 1.0"

    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_size + test_size),
        stratify=df['label'],
        random_state=42
    )

    # Second split: val vs test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size/(val_size + test_size),
        stratify=temp_df['label'],
        random_state=42
    )

    print(f"\nSplit sizes:")
    print(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

    return train_df, val_df, test_df

def save_datasets(train_df: pd.DataFrame,
                  val_df: pd.DataFrame,
                  test_df: pd.DataFrame,
                  output_dir: Path):
    """Save processed datasets."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as CSV
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    # Save as JSONL for easier loading with Hugging Face datasets
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        with open(output_dir / f"{split_name}.jsonl", "w") as f:
            for _, row in split_df.iterrows():
                json.dump({"text": row['text'], "label": row['label']}, f)
                f.write("\n")

    # Save label mapping
    label_to_id = {label: idx for idx, label in enumerate(sorted(train_df['label'].unique()))}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    with open(output_dir / "label_mapping.json", "w") as f:
        json.dump({
            "label_to_id": label_to_id,
            "id_to_label": id_to_label,
            "num_labels": len(label_to_id)
        }, f, indent=2)

    print(f"\nDatasets saved to {output_dir}")
    print(f"Number of labels: {len(label_to_id)}")
    print(f"Labels: {list(label_to_id.keys())}")

def main():
    """Main data preparation pipeline."""
    # TODO: Update this path to your downloaded Kaggle dataset
    # Example datasets:
    # - "Resume Dataset" by GUANYU HE
    # - "Resume Dataset" by gauravduttakiit
    # - Or any other resume classification dataset

    raw_data_path = RAW_DATA_DIR / "resumes.csv"

    if not raw_data_path.exists():
        print(f"ERROR: Dataset not found at {raw_data_path}")
        print("\nPlease download a resume dataset from Kaggle:")
        print("1. Go to https://www.kaggle.com/datasets")
        print("2. Search for 'resume dataset' or 'resume classification'")
        print("3. Download and extract to data/raw/")
        print("4. Rename to 'resumes.csv' or update the path in this script")
        return

    # Load and process
    df = load_and_process_resumes(raw_data_path)

    # Create splits
    train_df, val_df, test_df = create_splits(df)

    # Save processed datasets
    save_datasets(train_df, val_df, test_df, PROCESSED_DATA_DIR)

    print("\n✅ Data preparation complete!")

if __name__ == "__main__":
    main()
