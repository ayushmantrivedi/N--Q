import os
import glob
import pandas as pd
import numpy as np
import torch
from ...core.clinical import ClinicalSample, FeatureBlock, PatientTimeline, PredictionSnapshot

class PhysioNetGaitAdapter:
    """
    Adapter for the PhysioNet Gait in Parkinson's Disease dataset.
    Downloads to F:\NADI-Q data\PD_OPEN\raw\kinematics.
    """
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.demographics_path = os.path.join(data_dir, 'demographics.txt')
        self.subjects = self._load_demographics()
        
    def _load_demographics(self):
        if not os.path.exists(self.demographics_path):
            return pd.DataFrame()
        # Parse demographics (ID, Group, etc)
        # Format usually: ID Group ...
        try:
            df = pd.read_csv(self.demographics_path, sep='\t')
            return df
        except:
            return pd.DataFrame()

    def load_subject(self, subject_id: str) -> ClinicalSample:
        # Load the raw .txt files for this subject (e.g. Si01.txt)
        pattern = os.path.join(self.data_dir, f"{subject_id}_*.txt")
        files = glob.glob(pattern)
        
        timeline = PatientTimeline(subject_id=subject_id)
        
        for file in files:
            df = pd.read_csv(file, sep='\t', header=None)
            # 16 sensors per foot + total force.
            # Convert to FeatureBlock
            tensor_data = torch.tensor(df.values, dtype=torch.float32)
            fb = FeatureBlock(tensor_data)
            # Add arbitrary time
            timeline.add_snapshot(PredictionSnapshot(
                timestamp=0, 
                modality_blocks={'kinematics': fb},
                clinical_state='PD' # simplistic
            ))
            
        return ClinicalSample(subject_id=subject_id, timeline=timeline)
