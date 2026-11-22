import os
import sys

def scan_repo(root_dir):
    print("=== REPOSITORY SCAN ===")
    for root, dirs, files in os.walk(root_dir):
        # Ignore .git, .pytest_cache, __pycache__, .venv
        dirs[:] = [d for d in dirs if d not in ['.git', '.pytest_cache', '__pycache__', '.venv', '.idea']]
        rel_root = os.path.relpath(root, root_dir)
        if rel_root == '.':
            rel_root = ''
        for f in files:
            path = os.path.join(rel_root, f) if rel_root else f
            print(path)

scan_repo(r"C:\Users\ayush\OneDrive\Desktop\Nadi-Q")

