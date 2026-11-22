import os
import sys
import glob
import time
import torch
import torch.optim as optim

try:
    from src.nadiq.models.quantum.qvbe import QuantumVariationalBiomarkerEncoder, QVBEConfig
except ImportError:
    pass

RAW_DIR = r"F:\NADI-Q data\PD_OPEN\raw\biomarkers"
CKPT_DIR = r"F:\NADI-Q data\PD_OPEN\checkpoints\pd_biomarkers_v1"
os.makedirs(CKPT_DIR, exist_ok=True)

def main():
    print("=== NADI-Q LOCAL BIOMARKER TRAINING PIPELINE ===")
    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    print(f"Device: {device}")
    
    files = glob.glob(os.path.join(RAW_DIR, "**", "*.csv"), recursive=True)
    if len(files) == 0:
        print("[CRITICAL] No Biomarker CSV found. Real data is missing.")
        sys.exit(1)
        
    print(f"[OK] Found {len(files)} CSV files.")

    try:
        config = QVBEConfig(
            input_dim=8, # 8 biomarker variables 
            n_qubits=4,
            n_layers=2
        )
        model = QuantumVariationalBiomarkerEncoder(config).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
    except Exception as e:
        print(f"[ERROR] Failed to init Biomarker QVBE: {e}")
        sys.exit(1)
        
    print("\n--- PILOT EPOCH START ---")
    t0 = time.time()
    
    try:
        x = torch.randn(1, 8).to(device)
        out = model(x)
        
        # Dummy loss for pilot
        loss = out.embedding.sum()
        loss.backward()
        optimizer.step()
        
        t1 = time.time()
        print(f"[PILOT OK] Loss computed.")
        print(f"[PILOT OK] Gradients computed.")
        print(f"[PILOT OK] Wall-clock time: {t1 - t0:.2f}s")
        
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, "pilot_checkpoint.pt"))
        print("[SUCCESS] Biomarker Pilot checkpoint saved.")
    except Exception as e:
        print(f"[ERROR] Biomarker Pilot epoch failed: {e}")
        sys.exit(1)
        
    print("\n[INFO] Full Biomarker training budget would proceed here.")
    sys.exit(0)

if __name__ == "__main__":
    main()
