import time
import numpy as np
import pennylane as qml
from concurrent.futures import ProcessPoolExecutor

def run_chunk(args):
    n_qubits, chunk_data = args
    dev = qml.device("default.qubit", wires=n_qubits)
    @qml.qnode(dev)
    def circ(x):
        for i in range(n_qubits):
            qml.RY(x[i], wires=i)
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
        for i in range(n_qubits):
            qml.RZ(x[i], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)] + [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
    
    out = []
    for x in chunk_data:
        out.append(circ(x))
    return np.array(out)

if __name__ == '__main__':
    n_samples = 1465
    n_qubits = 8
    X_dummy = np.random.uniform(0, np.pi, size=(n_samples, n_qubits))
    
    # Split into 8 chunks for 8 processes
    n_workers = 8
    chunks = np.array_split(X_dummy, n_workers)
    args_list = [(n_qubits, c) for c in chunks]
    
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(run_chunk, args_list))
    res = np.vstack(results)
    t1 = time.time()
    print(f"Parallel 8-worker PQK extraction for N={n_samples}: {t1 - t0:.2f} seconds!")
    print(f"Result shape: {res.shape}")

