"""
Training Script: Noisy Quantum SVM on Real Credit Card Fraud Data
==================================================================
Uses data_loader.py to ingest the Kaggle credit card fraud dataset,
trains an SVM with a noisy quantum Nyström kernel, and evaluates
using PR-AUC and F1-Score.

Usage:
    cd backend/
    python train_noisy_model.py
"""

import os
import sys
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    classification_report,
)

# Ensure imports resolve when run from backend/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.quantum.qsvm import build_noisy_kernel_matrix
from data_loader import load_kaggle_credit_data


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_PCA_COMPONENTS = 20       # Must match our 20-qubit ZZFeatureMap
N_ANCHORS = 100             # Nyström anchor points (50 Normal + 50 Anomaly)
RANDOM_STATE = 42
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load Real Kaggle Credit Card Fraud Data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("NOISY QUANTUM SVM — Real Kaggle Credit Card Fraud")
    print("=" * 60)
    print("\nSTEP 1: Loading and preprocessing data")

    X_train, X_test, y_train, y_test = load_kaggle_credit_data(verbose=True)

    # Remap labels for SVM: -1/+1 -> 0/1 for metrics compatibility
    # data_loader uses -1=Normal, +1=Fraud
    # SVM works with any labels, but pos_label must match for metrics
    print(f"\n  Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # ------------------------------------------------------------------
    # 2. Build Noisy Quantum Nyström Kernel
    # ------------------------------------------------------------------
    print(f"\nSTEP 2: Building noisy quantum Nyström kernel matrices")
    print(f"  Anchors: {N_ANCHORS}, Qubits: {N_PCA_COMPONENTS}")

    np.random.seed(RANDOM_STATE)
    K_train = build_noisy_kernel_matrix(
        X_train, num_qubits=N_PCA_COMPONENTS, n_anchors=N_ANCHORS,
        y_labels=y_train
    )
    print(f"  K_train shape: {K_train.shape}")

    K_test = build_noisy_kernel_matrix(
        X_train, Y=X_test, num_qubits=N_PCA_COMPONENTS, n_anchors=N_ANCHORS,
        y_labels=y_train
    )
    print(f"  K_test shape:  {K_test.shape}")

    # ------------------------------------------------------------------
    # 3. Train SVM with Precomputed Kernel
    # ------------------------------------------------------------------
    print("\nSTEP 3: Training SVM (kernel='precomputed', class_weight='balanced')")

    model = SVC(
        kernel="precomputed",
        class_weight="balanced",
        probability=False,
        random_state=RANDOM_STATE,
    )
    model.fit(K_train, y_train)
    print("  Training complete.")

    # ------------------------------------------------------------------
    # 4. Evaluate: PR-AUC and F1-Score
    # ------------------------------------------------------------------
    print("\nSTEP 4: Evaluation on test set")

    y_pred = model.predict(K_test)
    decision_scores = model.decision_function(K_test)

    # pos_label=1 corresponds to Fraud (+1)
    pr_auc = average_precision_score(y_test, decision_scores)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    print("-" * 60)
    print(f"  PR-AUC:   {pr_auc:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print("-" * 60)

    print("\n  Full Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))

    # ------------------------------------------------------------------
    # 5. Save Trained Model
    # ------------------------------------------------------------------
    model_path = os.path.join(MODEL_DIR, "noisy_qsvm.pkl")
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")
    print("=" * 60)
    print("QUANTUM TRAINING COMPLETE")
    print("=" * 60)

    return pr_auc, f1


if __name__ == "__main__":
    main()
