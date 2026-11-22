try:
    import qiskit
    print(f"Qiskit is installed! Version: {qiskit.__version__}")
except ImportError as e:
    print(f"Qiskit is NOT installed: {e}")

try:
    import pennylane as qml
    print(f"PennyLane is installed! Version: {qml.__version__}")
except ImportError as e:
    print(f"PennyLane is NOT installed: {e}")

