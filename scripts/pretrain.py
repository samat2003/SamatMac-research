"""Pre-training loop streaming codeparrot/github-code-clean."""

import time
import argparse
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath('.'))

import mlx.core as mx
import mlx.nn as nn
from datasets import load_dataset

from data.tokenizer import SamatNextTokenizer
from data.dataset import PythonCodeDataset, BatchOutput
from model.config_120m import Baseline120MConfig, SamatNext120MConfig
from model.baseline_model import BaselineModel
from model.samatnext_model import SamatNextModel
from train.experiment_trainer import ExperimentTrainer, BATCH_SIZE, GRAD_ACCUM_STEPS

def run_pretrain(model_name: str, seed: int = 42):
    print(f"Starting Pre-training for {model_name}...")
    
    if model_name == "Baseline-120M":
        config = Baseline120MConfig()
        model = BaselineModel(config)
    else:
        config = SamatNext120MConfig()
        model = SamatNextModel(config)
        
    tokenizer = SamatNextTokenizer.from_file("data/tokenizer.json", config=config)
        
    print(f"Params: {model.count_params():,}")
    
    # Initialize the dataset with HuggingFace streaming dataset
    # We load 100k samples to simulate a robust pre-training pass
    dataset = PythonCodeDataset(
        tokenizer=tokenizer,
        config=config,
        max_samples=100000, 
        seed=seed
    )
    # The dataset loader currently expects either local_path or dataset_name.
    # We override it to stream cleanly and format to exactly what it expects.
    dataset.load(dataset_name="huggingface-course/codeparrot-ds-train")
    
    output_dir = f"results/pretrain_120m/{model_name}_seed_{seed}"
    
    trainer = ExperimentTrainer(
        model=model,
        config=config,
        output_dir=output_dir,
        seed=seed,
        max_steps=1000 # Pre-train for 1000 steps
    )
    
    trainer.train(dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["Baseline-120M", "SamatNext-120M"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.model:
        run_pretrain(args.model, args.seed)
    else:
        print("=====================================================")
        print("Phase 1: Pre-training (Baseline-120M vs SamatNext-120M)")
        print("=====================================================")
        run_pretrain("Baseline-120M", args.seed)
        run_pretrain("SamatNext-120M", args.seed)
