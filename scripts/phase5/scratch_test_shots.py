import numpy as np
from nadiq.models.quantum.qvbe_v2 import QVBEV2Config, QuantumVariationalBiomarkerEncoderV2

cfg = QVBEV2Config(input_dim=8, n_qubits=8, n_layers=2, measurement_basis="full", seed=42, mode="fixed", shots=1024)
qvbe = QuantumVariationalBiomarkerEncoderV2(cfg)

X = np.random.randn(20, 8)
y = np.random.randint(0, 2, size=20)
qvbe.fit(X, y)
qvbe.freeze()
Q = qvbe.transform(X)
print("Q shape:", Q.shape)
