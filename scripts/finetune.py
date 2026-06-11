"""Fine-tuning loop on flytech/python-codes-25k."""

import time
import argparse
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath('.'))

import mlx.core as mx
import mlx.nn as nn

from data.tokenizer import SamatNextTokenizer
from data.dataset import PythonCodeDataset
from model.config_120m import Baseline120MConfig, SamatNext120MConfig
from model.baseline_model import BaselineModel
from model.samatnext_model import SamatNextModel
from train.experiment_trainer import ExperimentTrainer

def run_finetune(model_name: str, seed: int = 42):
    print(f"Starting Fine-tuning for {model_name}...")
    
    if model_name == "Baseline-120M":
        config = Baseline120MConfig()
        model = BaselineModel(config)
    else:
        config = SamatNext120MConfig()
        model = SamatNextModel(config)
        
    tokenizer = SamatNextTokenizer.from_file("data/tokenizer.json", config=config)
        
    # Load Pre-trained weights
    pretrain_dir = Path(f"results/pretrain_120m/{model_name}_seed_{seed}")
    checkpoint_path = pretrain_dir / "step_001000.npz"
    if not checkpoint_path.exists():
        raise RuntimeError(f"Pre-trained checkpoint {checkpoint_path} not found! Run pretrain.py first.")
    
    print(f"Loading pre-trained weights from {checkpoint_path}")
    model.load_weights(str(checkpoint_path))
    
    dataset = PythonCodeDataset(
        tokenizer=tokenizer,
        config=config,
        seed=seed
    )
    dataset.load(local_path="data/python_codes_25k.jsonl")
    
    output_dir = f"results/finetune_120m/{model_name}_seed_{seed}"
    
    trainer = ExperimentTrainer(
        model=model,
        config=config,
        output_dir=output_dir,
        seed=seed,
        max_steps=500 # Finetune for 500 steps
    )
    
    trainer.train(dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["Baseline-120M", "SamatNext-120M"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.model:
        run_finetune(args.model, args.seed)
    else:
        print("=====================================================")
        print("Phase 2: Fine-Tuning (Baseline-120M vs SamatNext-120M)")
        print("=====================================================")
        run_finetune("Baseline-120M", args.seed)
        run_finetune("SamatNext-120M", args.seed)
