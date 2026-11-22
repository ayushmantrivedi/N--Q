import numpy as np
import torch
import mne

class EEGAdapter:
    def __init__(self, target_sfreq=100.0, window_sec=2.0, max_windows=50):
        self.target_sfreq = target_sfreq
        self.window_sec = window_sec
        self.max_windows = max_windows
        # EEG feature dim per window (window_sec * target_sfreq = 200 samples)
        # We will use simple time-domain downsampling for compatibility with CTCLM.
        self.samples_per_window = int(target_sfreq * window_sec)
        
    def process_file(self, file_path):
        """
        Reads a raw EEG file (mne compatible) and converts it to a sequence of windows.
        Returns:
            X: Tensor of shape (T, input_dim)
            mask: Tensor of shape (T,)
            dt: Tensor of shape (T,)
        """
        try:
            # mne.io.read_raw_eeglab handles .set files
            raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
            
            # Simple bandpass and downsample
            raw.filter(1.0, 40.0, fir_design='firwin', verbose=False)
            raw.resample(self.target_sfreq, verbose=False)
            
            # Extract data: shape (channels, times)
            data = raw.get_data()
            n_channels, n_times = data.shape
            
            # Calculate windows
            T_total = n_times // self.samples_per_window
            T = min(T_total, self.max_windows)
            
            if T == 0:
                raise ValueError("EEG recording too short for a single window.")
                
            # For simplicity, we just take the first channel or average channels to fit CTCLM input dim, 
            # or flatten (channel * window_samples) if CTCLM can handle it.
            # CTCLM expects (B, T, input_dim). We'll mean-pool across channels to keep input_dim = samples_per_window
            # to avoid enormous dimensions.
            data_mean_pool = np.mean(data, axis=0) # shape: (n_times,)
            
            # Reshape to (T_total, samples_per_window)
            windows = data_mean_pool[:T * self.samples_per_window].reshape(T, self.samples_per_window)
            
            X = torch.tensor(windows, dtype=torch.float32)
            mask = torch.ones(T, dtype=torch.float32)
            # dt is the time between windows, which is window_sec since they are contiguous
            dt = torch.ones(T, dtype=torch.float32) * self.window_sec
            
            return X, mask, dt
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None, None, None
