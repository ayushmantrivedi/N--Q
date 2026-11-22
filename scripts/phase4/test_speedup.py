import time
import numpy as np
import pennylane as qml
from sklearn.linear_model import LogisticRegression

# 1. Test vectorized gradient evaluation vs loop
n_qubits = 12
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def circ(inputs, u3_w, qaf_w):
    is_batched = (hasattr(inputs, "ndim") and inputs.ndim == 2)
    for i in range(n_qubits):
        val = inputs[:, i] if is_batched else inputs[i]
        qml.RY(val, wires=i)
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
    for l in range(2):
        for i in range(n_qubits):
            qml.U3(u3_w[l, i, 0], u3_w[l, i, 1], u3_w[l, i, 2], wires=i)
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

X_sub = np.random.uniform(0, np.pi, size=(8, 12))
u3_w = np.random.uniform(-np.pi, np.pi, size=(2, 12, 3))
qaf_w = np.empty(0)

# Vectorized
t0 = time.time()
for _ in range(10):
    res_vec = np.array(circ(X_sub, u3_w, qaf_w)).T
t1 = time.time()
print(f"10 Vectorized calls of 8 samples took: {t1 - t0:.3f} s ({(t1 - t0)/10:.4f} s/call)")

# 2. Test solver speed: liblinear vs saga
X = np.random.randn(1172, 249)
y = np.random.randint(0, 2, size=1172)

t0 = time.time()
LogisticRegression(penalty="l1", solver="liblinear", max_iter=200, random_state=42).fit(X, y)
t_lib = time.time() - t0
print(f"LogisticRegression L1 (liblinear) took: {t_lib:.3f} s")

