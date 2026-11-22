import time
import numpy as np
import pennylane as qml

n_qubits = 8
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def circ(x, u3_w, qaf_w):
    for i in range(n_qubits):
        qml.RY(x[:, i], wires=i)
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
    for l in range(2):
        for i in range(n_qubits):
            qml.U3(u3_w[l, i, 0], u3_w[l, i, 1], u3_w[l, i, 2], wires=i)
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
        if len(qaf_w) > 0:
            for s in range(qaf_w.shape[0]):
                stride = 2**s
                for i in range(n_qubits):
                    target = (i + stride) % n_qubits
                    qml.ControlledPhaseShift(qaf_w[s, i], wires=[i, target])
    
    obs = []
    for i in range(n_qubits):
        obs.append(qml.expval(qml.PauliZ(i)))
    for i in range(n_qubits):
        obs.append(qml.expval(qml.PauliX(i)))
    for i in range(n_qubits - 1):
        obs.append(qml.expval(qml.PauliZ(i) @ qml.PauliZ(i + 1)))
    return obs

X = np.random.uniform(0, np.pi, size=(1465, 8))
u3_w = np.random.uniform(-np.pi, np.pi, size=(2, 8, 3))
qaf_w = np.random.uniform(0, np.pi/2, size=(3, 8))

t0 = time.time()
res = circ(X, u3_w, qaf_w)
res_arr = np.array(res).T
t1 = time.time()

print(f"Vectorized execution of 1,465 samples took {t1 - t0:.2f} seconds!")
print(f"Output shape: {res_arr.shape}")
print(f"Sample mean: {np.mean(res_arr):.4f}, min: {np.min(res_arr):.4f}, max: {np.max(res_arr):.4f}")

