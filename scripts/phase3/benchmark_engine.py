import os
import sys
import time
import tracemalloc
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, brier_score_loss, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
import pennylane as qml

def compute_ece(y_true, y_prob, n_bins=10):
    bin_limits = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        bin_mask = (y_prob >= bin_limits[i]) & (y_prob < bin_limits[i+1])
        if i == n_bins - 1:
            bin_mask = (y_prob >= bin_limits[i]) & (y_prob <= bin_limits[i+1])
        bin_count = np.sum(bin_mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[bin_mask])
            bin_conf = np.mean(y_prob[bin_mask])
            ece += (bin_count / n) * np.abs(bin_acc - bin_conf)
    return ece

def evaluate_predictions(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0
    brier = brier_score_loss(y_true, y_prob)
    ece = compute_ece(y_true, y_prob, n_bins=10)
    
    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "balanced_acc": float(bal_acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "f1": float(f1),
        "brier": float(brier),
        "ece": float(ece),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }

# ----------------- PENNYLANE PQK CIRCUITS ----------------- #
def build_pqk_circuit(n_qubits):
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev)
    def circuit(x):
        # Layer 1: Ry Angle Embedding
        for i in range(n_qubits):
            qml.RY(x[i], wires=i)
        # Entanglement 1: Periodic circular CNOT ladder
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
        # Layer 2: Rz Data Re-uploading
        for i in range(n_qubits):
            qml.RZ(x[i], wires=i)
        # Entanglement 2: Linear nearest-neighbor CNOT ladder
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        # Expectation measurements of Pauli Z and Pauli X
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)] + [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
        
    return circuit

circuit_8  = build_pqk_circuit(8)
circuit_12 = build_pqk_circuit(12)

def extract_pqk_features(X_scaled, n_qubits):
    circuit = circuit_8 if n_qubits == 8 else circuit_12
    n_samples = len(X_scaled)
    # Output dimension is 2 * n_qubits
    q_feats = np.zeros((n_samples, 2 * n_qubits), dtype=np.float64)
    for i in range(n_samples):
        res = circuit(X_scaled[i])
        q_feats[i] = np.array(res, dtype=np.float64)
    return q_feats

def compute_effective_rank(K):
    # Normalized singular values / eigenvalues
    evals = np.linalg.eigvalsh(K)
    evals = evals[evals > 1e-12]
    if len(evals) == 0:
        return 0.0
    p = evals / np.sum(evals)
    entropy = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(entropy))

