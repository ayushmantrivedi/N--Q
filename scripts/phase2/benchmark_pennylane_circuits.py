import pennylane as qml
import numpy as np
import time

print("PennyLane version:", qml.__version__)

# Test 8-qubit PQK circuit
n_qubits = 8
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def pqk_circuit(x):
    # Layer 1: Angle embedding (Ry)
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)
    # Entanglement: Ring CNOT
    for i in range(n_qubits):
        qml.CNOT(wires=[i, (i + 1) % n_qubits])
    # Layer 2: Data re-uploading (Rz)
    for i in range(n_qubits):
        qml.RZ(x[i], wires=i)
    # Entanglement: All-to-all or nearest-neighbor
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    # Expectation values of Pauli Z and Pauli X on each qubit
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)] + [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]

# Benchmark execution time
x_dummy = np.random.uniform(0, np.pi, size=n_qubits)
t0 = time.time()
res = pqk_circuit(x_dummy)
t1 = time.time()
print(f"8-qubit single evaluation time: {(t1 - t0)*1000:.2f} ms")
print(f"Output vector shape: {len(res)} (expected {2 * n_qubits})")
print(f"Output sample values: {[round(float(v), 3) for v in res[:6]]}")

# Benchmark 100 evaluations
t0 = time.time()
for _ in range(100):
    _ = pqk_circuit(x_dummy)
t1 = time.time()
print(f"100 evaluations time: {t1 - t0:.3f} s ({(t1-t0)/100*1000:.2f} ms/eval)")
print(f"Estimated time for N=1,465 full dataset PQK extraction: {(t1-t0)/100 * 1465:.2f} s (~{(t1-t0)/100 * 1465 / 60:.1f} minutes)")

# Also test 12-qubit
n_qubits_12 = 12
dev_12 = qml.device("default.qubit", wires=n_qubits_12)

@qml.qnode(dev_12)
def pqk_circuit_12(x):
    for i in range(n_qubits_12):
        qml.RY(x[i], wires=i)
    for i in range(n_qubits_12):
        qml.CNOT(wires=[i, (i + 1) % n_qubits_12])
    for i in range(n_qubits_12):
        qml.RZ(x[i], wires=i)
    for i in range(n_qubits_12 - 1):
        qml.CNOT(wires=[i, i + 1])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits_12)] + [qml.expval(qml.PauliX(i)) for i in range(n_qubits_12)]

x_dummy_12 = np.random.uniform(0, np.pi, size=n_qubits_12)
t0 = time.time()
res_12 = pqk_circuit_12(x_dummy_12)
t1 = time.time()
print(f"\n12-qubit single evaluation time: {(t1 - t0)*1000:.2f} ms")
print(f"Output vector shape: {len(res_12)} (expected {2 * n_qubits_12})")

