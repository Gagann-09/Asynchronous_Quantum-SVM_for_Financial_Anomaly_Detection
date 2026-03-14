"""
Training Script: Classical RBF SVM Baseline on Real Credit Card Fraud Data
===========================================================================
Uses data_loader.py to ingest the Kaggle credit card fraud dataset,
trains an RBF-kernel SVM, and evaluates using PR-AUC and F1-Score.

Usage:
    cd backend/
    python train_classical_baseline.py
"""

import os
import sys
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    classification_report,
)

# Ensure imports resolve when run from backend/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_kaggle_credit_data


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_STATE = 42


def main():
    # ------------------------------------------------------------------
    # 1. Load Real Kaggle Credit Card Fraud Data
    # ------------------------------------------------------------------
    print("=" * 65)
    print("  CLASSICAL RBF SVM BASELINE — Real Kaggle Credit Card Fraud")
    print("=" * 65)
    print("\nSTEP 1: Loading and preprocessing data")

    X_train, X_test, y_train, y_test = load_kaggle_credit_data(verbose=True)
    print(f"\n  Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # ------------------------------------------------------------------
    # 2. Train Classical RBF SVM
    # ------------------------------------------------------------------
    print("\nSTEP 2: Training SVC(kernel='rbf', class_weight='balanced')")

    model = SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=True,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    print("  Training complete.")

    # ------------------------------------------------------------------
    # 3. Evaluate
    # ------------------------------------------------------------------
    print("\nSTEP 3: Evaluation on test set")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    classical_pr_auc = average_precision_score(y_test, y_prob)
    classical_f1 = f1_score(y_test, y_pred, pos_label=1)

    print("\n  Full Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))

    print("-" * 65)
    print(f"  Classical PR-AUC:   {classical_pr_auc:.4f}")
    print(f"  Classical F1-Score: {classical_f1:.4f}")
    print("-" * 65)

    # 4. Save model
    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "classical_rbf.pkl")
    import joblib
    joblib.dump(model, model_path)
    print(f"  Model saved to: {model_path}")

    print("=" * 65)
    print("  CLASSICAL TRAINING COMPLETE")
    print("=" * 65)

    return classical_pr_auc, classical_f1


if __name__ == "__main__":
    main()
