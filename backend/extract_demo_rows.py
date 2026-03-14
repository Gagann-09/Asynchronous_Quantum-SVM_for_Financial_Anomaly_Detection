"""
Utility Script: Extract Demo Rows
=================================
Selects 2 random Normal rows and 2 random Fraud rows from the Kaggle dataset,
preprocesses them using the saved Scaler and PCA models, and outputs
the resulting 20-dimensional feature arrays as JSON payloads
for live-demo testing via the Next.js frontend or Swagger UI.

Usage:
    cd backend/
    python extract_demo_rows.py
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
CSV_PATH = os.path.join(DATA_DIR, "creditcard.csv")

RANDOM_STATE = 42


def main():
    print("=" * 60)
    print("  DEMO ROW EXTRACTOR")
    print("=" * 60)

    # 1. Load data
    print("Loading creditcard.csv...")
    df = pd.read_csv(CSV_PATH)

    # 2. Select 2 Normal (Class=0) and 2 Fraud (Class=1)
    normal = df[df["Class"] == 0].sample(n=2, random_state=RANDOM_STATE)
    fraud = df[df["Class"] == 1].sample(n=2, random_state=RANDOM_STATE)
    
    # 3. Load Transformers
    print("Loading scaler.pkl and pca.pkl...")
    try:
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        pca = joblib.load(os.path.join(MODEL_DIR, "pca.pkl"))
    except FileNotFoundError:
        print("\nERROR: Models not found. Run train_noisy_model.py first!")
        return

    # Helper function to process and print
    def process_and_print(df_subset, label_name):
        print(f"\n--- {label_name.upper()} PAYLOADS ---")
        
        # Drop Time and Class
        X = df_subset.drop(columns=["Time", "Class"]).copy()
        
        # Scale Amount
        X["Amount"] = scaler.transform(X[["Amount"]])
        
        # PCA Transform
        X_pca = pca.transform(X.values)
        
        # Print as JSON
        for i, row in enumerate(X_pca):
            payload = {"features": row.tolist()}
            print(f"// {label_name} #{i + 1}")
            print(json.dumps(payload, indent=2))

    # Process and output
    process_and_print(normal, "Normal")
    process_and_print(fraud, "Fraud")
    print("\n" + "=" * 60)
    print("Use these payloads in Swagger UI or your frontend application.")
    print("=" * 60)


if __name__ == "__main__":
    main()
