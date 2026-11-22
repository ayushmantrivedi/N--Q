import os
import sys
import glob
import time
import torch
import torch.optim as optim

try:
    from src.nadiq.models.quantum.ctclm import ContinualTemporalClinicalLatentModel as CTCLM, CTCLMConfig
except ImportError:
    pass

RAW_DIR = r"F:\NADI-Q data\PD_OPEN\raw\kinematics"
CKPT_DIR = r"F:\NADI-Q data\PD_OPEN\checkpoints\pd_ctclm_v1"
os.makedirs(CKPT_DIR, exist_ok=True)

def main():
    print("=== NADI-Q LOCAL CTCLM TRAINING PIPELINE ===")
    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    print(f"Device: {device}")
    
    files = glob.glob(os.path.join(RAW_DIR, "**", "*.txt"), recursive=True)
    if len(files) == 0:
        print("[CRITICAL] No kinematics text recordings found. Real data is missing.")
        sys.exit(1)
        
    print(f"[OK] Found {len(files)} gait recordings.")

    try:
        config = CTCLMConfig(
            input_dim=16, # 16 force sensors
            hidden_dim=32,
            time_embedding_dim=8,
            num_layers=1,
            dropout=0.1
        )
        model = CTCLM(config).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
    except Exception as e:
        print(f"[ERROR] Failed to init CTCLM: {e}")
        sys.exit(1)
        
    print("\n--- PILOT EPOCH START ---")
    t0 = time.time()
    
    try:
        # Dummy sequence to represent a real temporal sequence
        x = torch.randn(1, 10, 16).to(device)
        time_offsets = torch.arange(10, dtype=torch.float32).unsqueeze(0).to(device)
        padding_mask = torch.ones(1, 10, dtype=torch.bool).to(device)
        
        out = model(x, time_offsets=time_offsets, padding_mask=padding_mask)
        # Dummy loss to check gradients
        loss = out.h_final.sum()
        loss.backward()
        optimizer.step()
        
        t1 = time.time()
        print(f"[PILOT OK] Loss computed.")
        print(f"[PILOT OK] Gradients computed.")
        print(f"[PILOT OK] Wall-clock time: {t1 - t0:.2f}s")
        
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, "pilot_checkpoint.pt"))
        print("[SUCCESS] CTCLM Pilot checkpoint saved.")
    except Exception as e:
        print(f"[ERROR] CTCLM Pilot epoch failed: {e}")
        sys.exit(1)
        
    print("\n[INFO] Full CTCLM training budget would proceed here.")
    sys.exit(0)

if __name__ == "__main__":
    main()
