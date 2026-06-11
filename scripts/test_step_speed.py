"""Test end-to-end logical step speed for both models."""

import time
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath('.'))

import mlx.core as mx
from data.tokenizer import SamatNextTokenizer
from data.dataset import PythonCodeDataset
from model.config_120m import Baseline120MConfig, SamatNext120MConfig
from model.baseline_model import Baseline20M as BaselineModel
from model.samatnext_20m import SamatNext20M as SamatNextModel
from train.experiment_trainer import ExperimentTrainer

def test_model_speed(model_name: str):
    print(f"\n{'='*60}\nTesting end-to-end step speed for {model_name}\n{'='*60}")
    
    tokenizer = SamatNextTokenizer.from_file("data/tokenizer.json")
    
    if model_name == "Baseline-120M":
        config = Baseline120MConfig()
        model = BaselineModel(config)
    else:
        config = SamatNext120MConfig()
        model = SamatNextModel(config)
        
    print(f"Params: {model.count_params():,}")
    
    dataset = PythonCodeDataset(tokenizer, config, max_samples=200)
    dataset.load(dataset_name="huggingface-course/codeparrot-ds-train")
    
    output_dir = f"results/speed_test/{model_name}"
    
    import train.experiment_trainer as et
    # Force LOG_EVERY=1 so we can see per-step timing
    original_log = et.LOG_EVERY
    et.LOG_EVERY = 1
    
    trainer = et.ExperimentTrainer(
        model=model,
        config=config,
        output_dir=output_dir,
        seed=42,
        max_steps=5
    )
    
    print(f"\nRunning 5 logical gradient steps (BS={et.BATCH_SIZE} × {et.GRAD_ACCUM_STEPS} accum = effective {et.BATCH_SIZE * et.GRAD_ACCUM_STEPS})...")
    print("Step 0 includes MLX graph compilation time.")
    print("Steps 1-4 show the true cached step-to-step speed.")
    print("-" * 60)
    
    t0 = time.time()
    trainer.train(dataset)
    total_time = time.time() - t0
    
    print("-" * 60)
    print(f"Finished {model_name} in {total_time:.1f}s total.")
    
    # Restore
    et.LOG_EVERY = original_log

if __name__ == "__main__":
    test_model_speed("Baseline-120M")
    test_model_speed("SamatNext-120M")
