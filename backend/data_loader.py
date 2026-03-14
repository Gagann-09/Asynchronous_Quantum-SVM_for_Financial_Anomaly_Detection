"""
Data Loader: Kaggle Credit Card Fraud Dataset
==============================================
Loads, preprocesses, and subsamples the real-world credit card fraud dataset
for training quantum and classical SVMs.

Pipeline Shape:
    Raw CSV:           (284,807 x 31)  — Time, V1-V28, Amount, Class
    Drop Time:         (284,807 x 30)
    Scale Amount:      (284,807 x 30)
    Subsample:         (10,000 x 30)   — 9,900 Normal + 100 Fraud
    Remap labels:      (10,000 x 30)   — 0 -> -1, 1 -> +1
    Separate X/y:      X=(10,000 x 29), y=(10,000,)
    PCA(20):           X=(10,000 x 20) — compressed to 20-qubit limit
    Train/Test split:  Train=(8,000 x 20), Test=(2,000 x 20)
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_NORMAL = 9_900
N_FRAUD = 100
N_PCA_COMPONENTS = 20
TEST_SIZE = 0.20
RANDOM_STATE = 42

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
CSV_PATH = os.path.join(DATA_DIR, "creditcard.csv")


def load_kaggle_credit_data(verbose: bool = True):
    """
    Loads and preprocesses the Kaggle Credit Card Fraud dataset.

    Returns:
        X_train, X_test, y_train, y_test — all as numpy arrays.
        y labels: -1 = Normal, +1 = Fraud/Anomaly
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load CSV
    # ------------------------------------------------------------------
    if verbose:
        print("  [data_loader] Loading creditcard.csv...")
    df = pd.read_csv(CSV_PATH)
    if verbose:
        print(f"  [data_loader] Raw shape: {df.shape}")

    # ------------------------------------------------------------------
    # 2. Drop Time column (irrelevant for kernel computation)
    # ------------------------------------------------------------------
    df = df.drop(columns=["Time"])
    if verbose:
        print(f"  [data_loader] After dropping Time: {df.shape}")

    # ------------------------------------------------------------------
    # 3. Scale Amount column using StandardScaler
    # ------------------------------------------------------------------
    amount_scaler = StandardScaler()
    df["Amount"] = amount_scaler.fit_transform(df[["Amount"]])
    if verbose:
        print(f"  [data_loader] Amount column scaled.")

    # ------------------------------------------------------------------
    # 4. Subsample: 9,900 Normal + 100 Fraud
    # ------------------------------------------------------------------
    normal = df[df["Class"] == 0].sample(n=N_NORMAL, random_state=RANDOM_STATE)
    fraud = df[df["Class"] == 1].sample(n=N_FRAUD, random_state=RANDOM_STATE)
    df_sub = pd.concat([normal, fraud]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    if verbose:
        print(f"  [data_loader] Subsampled: {df_sub.shape} (Normal={len(normal)}, Fraud={len(fraud)})")

    # ------------------------------------------------------------------
    # 5. Remap labels: 0 -> -1, 1 -> +1
    # ------------------------------------------------------------------
    y = df_sub["Class"].map({0: -1, 1: 1}).values
    X = df_sub.drop(columns=["Class"]).values
    if verbose:
        print(f"  [data_loader] Labels remapped: Normal(-1)={np.sum(y == -1)}, Fraud(+1)={np.sum(y == 1)}")

    # ------------------------------------------------------------------
    # 6. PCA: 29 features -> 20 (qubit limit)
    # ------------------------------------------------------------------
    pca = PCA(n_components=N_PCA_COMPONENTS, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    explained_var = np.sum(pca.explained_variance_ratio_) * 100
    if verbose:
        print(f"  [data_loader] PCA: {X.shape[1]} -> {N_PCA_COMPONENTS} features ({explained_var:.1f}% variance)")

    # Save fitted transformers
    joblib.dump(amount_scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(pca, os.path.join(MODEL_DIR, "pca.pkl"))
    if verbose:
        print(f"  [data_loader] Saved scaler.pkl and pca.pkl to models/")

    # ------------------------------------------------------------------
    # 7. Train / Test Split (80/20, stratified)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    if verbose:
        print(f"  [data_loader] Train: {X_train.shape} (Fraud: {np.sum(y_train == 1)})")
        print(f"  [data_loader] Test:  {X_test.shape} (Fraud: {np.sum(y_test == 1)})")

    # Save training data (needed at inference for kernel computation)
    joblib.dump(X_train, os.path.join(MODEL_DIR, "X_train.pkl"))

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    print("=" * 60)
    print("  DATA LOADER — Kaggle Credit Card Fraud")
    print("=" * 60)
    X_train, X_test, y_train, y_test = load_kaggle_credit_data()
    print("=" * 60)
    print("  Data loading complete.")
    print("=" * 60)
