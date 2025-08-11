from __future__ import annotations
import os
import pandas as pd
import torch
from omegaconf import OmegaConf

from .base import ProteinOracle
from src.utils.predictors import BaseCNN
from src.utils.tokenize import Encoder


def get_model(predictor_dir, oracle_dir, use_oracle=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if use_oracle:
        oracle_path = os.path.join(oracle_dir, 'cnn_oracle.ckpt')
        oracle_state_dict = torch.load(oracle_path, map_location=device)
        cfg_path = os.path.join(oracle_dir, 'config.yaml')
        with open(cfg_path, 'r') as fp:
            ckpt_cfg = OmegaConf.load(fp.name)
        cnn_oracle = BaseCNN(**ckpt_cfg.model.predictor) #oracle has same architecture as predictor
        cnn_oracle.load_state_dict(
            {k.replace('predictor.', ''): v for k,v in oracle_state_dict['state_dict'].items()})
        cnn_oracle.to(device)
        return cnn_oracle
    else:
        cfg_path = os.path.join(oracle_dir, 'config.yaml')
        with open(cfg_path, 'r') as fp:
            ckpt_cfg = OmegaConf.load(fp.name)
        predictor_path = os.path.join(predictor_dir, 'last.ckpt')
        predictor_state_dict = torch.load(predictor_path, map_location=device)
        predictor = BaseCNN(**ckpt_cfg.model.predictor) 
        predictor.load_state_dict(
            {k.replace('predictor.', ''): v for k,v in predictor_state_dict['state_dict'].items()})
        predictor = predictor.to(device)
        return predictor
    return None


class AAVOracle(ProteinOracle):
    """ML-based oracle for AAV fitness prediction."""

    def __init__(self):
        super().__init__()
        self.name = "aav"
        # Get path relative to this file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        oracle_dir = os.path.join(base_dir, "../utils/ckpt/AAV/mutations_0/percentile_0.0_1.0")
        
        self.oracle = get_model(predictor_dir=None, oracle_dir=oracle_dir)
        self.tokenizer = Encoder()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _evaluate_protein_impl(self, sequence: str) -> float:
        """Predict fitness using the loaded ML model."""
        tokenized = self.tokenizer.encode([sequence]).to(self.device)
        return self.oracle(tokenized).item()

class GFPOracle(ProteinOracle):
    """ML-based oracle for GFP fitness prediction."""
    
    def __init__(self):
        super().__init__()
        self.name = "gfp"
        base_dir = os.path.dirname(os.path.abspath(__file__))
        predictor_dir = os.path.join(base_dir, "../utils/ckpt/GFP/mutations_7/percentile_0.0_0.3/unsmoothed_smoothed/01_03_2025_23_56")
        oracle_dir = os.path.join(base_dir, "../utils/ckpt/GFP/mutations_0/percentile_0.0_1.0")

        self.oracle = get_model(predictor_dir=predictor_dir, oracle_dir=oracle_dir)
        self.tokenizer = Encoder()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _evaluate_protein_impl(self, sequence: str) -> float:
        """Predict fitness using the loaded ML model."""
        tokenized = self.tokenizer.encode([sequence]).to(self.device)
        return self.oracle(tokenized).item() 