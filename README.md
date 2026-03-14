# ⚛️ AquaAnomaly: Asynchronous Quantum-Classical Hybrid SVM for Financial Fraud Detection

![AquaAnomaly Architecture Visualization](aqua_anomaly_header.png)

## Overview
**AquaAnomaly** is a research-grade, full-stack application developed for a 5th-semester VTU (Visvesvaraya Technological University) computer science mini-project. It deploys a hybrid Quantum Support Vector Machine (QSVM) targeted specifically at real-world financial anomaly detection in highly imbalanced (99:1) data.

This project goes beyond theoretical simulation by engineering a "zero-failure," decoupled web architecture that bridges heavy, NISQ-era (Noisy Intermediate-Scale Quantum) circuit simulations with a reactive modern web frontend.

---

## The "Hard Way": Mathematics & Optimization

Evaluating a quantum kernel natively requires $O(M^2)$ circuit executions, where $M$ is the number of data samples. This is mathematically intractable for large datasets and crashes current simulators.

We implemented a robust **Nyström Kernel Approximation** pipeline to overcome this limitation, reducing computational scaling from $O(M^2)$ to $O(M \cdot L)$.

### The Mathematical Pipeline
1.  **Ingestion:** Real-world transaction data (29 financial features) loaded from the Kaggle Credit Card Fraud dataset (`creditcard.csv`).
2.  **Compression (Classical):** 29 features compressed to exactly 20 dimensions using Principal Component Analysis (PCA) to align with a 20-qubit topological constraint.
3.  **Feature Map (Quantum):** 20-qubit **ZZFeatureMap** (reps=2, depth=2). Maps compressed data into a high-dimensional Hilbert space, capturing complex, non-linear correlations through native entanglement gates (CNOT).
4.  **Stratified Anchor Selection:** To combat the 99:1 class imbalance, we implemented stratified Nyström landmarks. We strictly anchor 50 Normal and 50 Fraud samples to ensure the quantum kernel "sees" the minority class.
5.  **Noisy Simulation:** Injected realistic physics-based depolarizing noise model (0.1% 1-qubit gate error, 1% 2-qubit (CX) gate error) into the `AerSimulator` to mimic current IBM physical hardware constraints.

---

## Full-Scale Application Architecture

Quantum simulations are computationally heavy. If run synchronously, they freeze the web server. AquaAnomaly utilizes a decoupled architecture to prevent browser timeouts during inference.

| Component | Responsibility | Technical Stack |
| :--- | :--- | :--- |
| **Frontend** | Interactive Dashboard, Real-time telemetry, Dynamic Polling. | Next.js, React Hooks, Tailwind, Shadcn/UI |
| **API Gateway** | Strict Pydantic Data Validation, Asynchronous Dispatch, CORS. | FastAPI |
| **Middleware** | Asynchronous Message Broker & Task Queue. | Redis, Celery |
| **Backend** | Noise-Aware QSVM Training & Inference, PCA Transformation. | Qiskit 1.0.0, Scikit-Learn |

---

## Results & NISQ Reality

We strictly benchmarked the Noisy QSVM against a classical Radial Basis Function (RBF) SVM using class-weight balancing (`class_weight='balanced'`) on the exact same 99:1 imbalanced Kaggle data (`random_state=42`).

| Metric | Classical RBF SVM | Noisy Q-Nyström-SVM |
| :--- | :---: | :---: |
| **F1-Score (Test)** | **0.6222** | 0.0522 |

**Analysis:** The classical RBF kernel outperformed the quantum model. This is the **expected, scientifically honest result** in the NISQ era. The classical algorithm computes analytical distances with zero noise. The Noisy QSVM's performance is mathematically limited by the 1% depolarizing noise and the Nyström matrix approximation loss. This project proves that while structurally sound, true quantum advantage requires either lower physical noise or datasets with a specific topology that directly matches the chosen quantum feature map.

---

## Local Setup & Reproducibility (Free Stack)

First run the data_loader.py to get 20 features of fraud and clean data, then run the train_noisy_model.py to train the noisy qsvm model. Then run the .bat file to start the application.

This whole stack runs locally for $0 costs on standard CPU compute. Redis is required.

### 1. Repository Structure
Ensure your repo looks like this:
```text
backend/
├── app/main.py, api/routers.py, schemas.py, worker/tasks.py, quantum/qsvm.py
├── data/creditcard.csv (MANUAL DOWNLOAD REQUIRED FROM KAGLLE)
├── models/pca.pkl, scaler.pkl, noisy_qsvm.pkl, classical_rbf.pkl (GENERATED)
├── data_loader.py, train_noisy_model.py, extract_demo_rows.py
└── requirements.txt
frontend/ (Next.js Project Files)
docker-compose.yml