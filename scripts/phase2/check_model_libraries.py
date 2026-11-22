import sys

try:
    import tabpfn
    print(f"TabPFN is installed! Version: {tabpfn.__version__}")
    from tabpfn import TabPFNClassifier
    print("TabPFNClassifier successfully imported.")
    print("TabPFNClassifier docstring / methods:")
    clf = TabPFNClassifier()
    methods = [m for m in dir(clf) if not m.startswith('__')]
    print("Methods:", methods)
except ImportError as e:
    print(f"TabPFN is NOT installed: {e}")

try:
    import lightgbm
    print(f"LightGBM is installed! Version: {lightgbm.__version__}")
except ImportError as e:
    print(f"LightGBM is NOT installed: {e}")

try:
    import xgboost
    print(f"XGBoost is installed! Version: {xgboost.__version__}")
except ImportError as e:
    print(f"XGBoost is NOT installed: {e}")

try:
    import pytorch_tabnet
    print("pytorch_tabnet is installed!")
except ImportError as e:
    print(f"pytorch_tabnet is NOT installed: {e}")

