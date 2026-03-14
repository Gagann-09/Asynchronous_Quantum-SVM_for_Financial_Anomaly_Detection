import os
import joblib
import numpy as np
from app.worker.celery_app import celery_app
from app.quantum.qsvm import build_noisy_kernel_matrix

# ---------------------------------------------------------------------------
# Load trained model artifacts at module level (once, when worker starts)
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models")

_model = None
_scaler = None
_pca = None
_X_train = None

def _load_model():
    """Lazy-load model artifacts from disk. Returns True if model is available."""
    global _model, _scaler, _pca, _X_train
    if _model is not None:
        return True

    model_path = os.path.join(MODEL_DIR, "noisy_qsvm.pkl")
    if not os.path.exists(model_path):
        return False

    _model = joblib.load(model_path)
    _scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    _pca = joblib.load(os.path.join(MODEL_DIR, "pca.pkl"))
    _X_train = joblib.load(os.path.join(MODEL_DIR, "X_train.pkl"))
    return True


@celery_app.task(bind=True, name="predict_anomaly_task")
def predict_anomaly_task(self, features: list):
    """
    Celery task that receives the validated Pydantic data (list of 20 floats),
    runs inference through the trained noisy QSVM model, and returns
    both a categorical prediction and a confidence score.

    If the trained model is not yet available (train_noisy_model.py has not
    been run), falls back to a heuristic-based dummy prediction.
    """
    if _load_model():
        # --- Real model inference ---
        sample = np.array(features).reshape(1, -1)

        # Build kernel row between this sample and the training data
        kernel_row = build_noisy_kernel_matrix(
            _X_train, Y=sample, num_qubits=sample.shape[1], n_anchors=50
        )

        prediction = _model.predict(kernel_row)[0]
        label = "Anomaly" if prediction == 1 else "Normal"

        # Decision function distance — continuous confidence score
        confidence = float(abs(_model.decision_function(kernel_row)[0]))

        return {
            "prediction": label,
            "confidence_score": round(confidence, 4)
        }
    else:
        # --- Fallback: no trained model available yet ---
        import time
        time.sleep(2.0)

        mean_val = np.mean(features)
        label = "Anomaly" if mean_val < 0.3 else "Normal"
        confidence = round(abs(mean_val - 0.5), 4)

        return {
            "prediction": label,
            "confidence_score": confidence
        }
