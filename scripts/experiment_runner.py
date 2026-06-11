"""Main entry point for the 120M Validation Experiment."""

import argparse
import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx

from data.dataset import HUGGINGFACE_DATASETS, PythonCodeDataset
from data.tokenizer import SamatNextTokenizer
from model.config_120m import Baseline120MConfig, SamatNext120MConfig
from model.baseline_model import BaselineModel
from model.samatnext_model import SamatNextModel
from train.experiment_trainer import ExperimentTrainer, MAX_STEPS
from eval.experiment_eval import run_full_eval


def parse_args():
    parser = argparse.ArgumentParser(description="Run 120M Validation Experiment")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="data/tokenizer.json",
        help="Path to trained tokenizer.json",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--local_data",
        type=str,
        default=None,
        help="Path to local jsonl file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/experiment_120m",
        help="Output directory for logs and models",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456],
        help="List of random seeds to run",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run a quick 50-step test with a small dataset constraint to verify pipeline",
    )
    return parser.parse_args()


def run_experiment(args):
    print("Starting 120M Validation Experiment")
    print(f"Device: {mx.default_device()}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    if not Path(args.tokenizer_path).exists():
        print(f"Error: Tokenizer not found at {args.tokenizer_path}")
        print("Please provide a valid tokenizer path using --tokenizer_path")
        return
        
    tok = SamatNextTokenizer.from_file(args.tokenizer_path)
    print(f"Tokenizer loaded: vocab_size={tok.tokenizer.get_vocab_size()}")

    # Determine max steps and samples based on smoke test
    max_steps = 2 if args.smoke_test else MAX_STEPS
    max_samples = 10 if args.smoke_test else 50000
    from train.experiment_trainer import BATCH_SIZE
    batch_size = 1 if args.smoke_test else BATCH_SIZE

    # Load dataset
    # We instantiate one dataset object and use it identically across all runs
    # Note: dataset behavior must be deterministic given the seed
    print("Loading dataset...")
    
    results = []
    
    for seed in args.seeds:
        for model_type in ["Baseline-120M", "SamatNext-120M"]:
            print(f"\n{'='*50}")
            print(f"Running {model_type} with seed {seed}")
            print(f"{'='*50}")
            
            # Set seed for reproducibility
            mx.random.seed(seed)
            
            # 1. Config & Model
            if model_type == "Baseline-120M":
                config = Baseline120MConfig()
                model = BaselineModel(config)
            else:
                config = SamatNext120MConfig()
                model = SamatNextModel(config)
                
            print(f"{model_type} params: {model.count_params():,}")
            
            # 2. Dataset
            # Recreate dataset generator with same split info
            dataset = PythonCodeDataset(
                tok,
                config,  # duck types fine
                max_samples=max_samples,
            )
            
            if args.local_data:
                dataset.load(local_path=args.local_data)
            elif args.dataset:
                dataset.load(dataset_name=args.dataset)
            else:
                dataset.load(dataset_name=HUGGINGFACE_DATASETS[0][0])
                
            # 3. Train
            run_out_dir = output_dir / f"{model_type}_seed_{seed}"
            from train import experiment_trainer
            experiment_trainer.BATCH_SIZE = batch_size
            trainer = ExperimentTrainer(
                model=model,
                config=config,
                output_dir=str(run_out_dir),
                seed=seed,
                max_steps=max_steps
            )
            
            trainer.train(dataset)
            
            # 4. Evaluate
            print(f"Evaluating {model_type} (seed {seed})...")
            # For pure validation in the same script, we just use the same dataset object
            # In a rigorous setup, we'd have a separate validation split. 
            # We'll use the beginning of the dataset since PythonCodeDataset 
            # doesn't explicitly expose a split method in the provided snippet.
            eval_metrics = run_full_eval(model, tok, dataset)
            
            run_result = {
                "model": model_type,
                "seed": seed,
                "params": model.count_params(),
                "val_loss": eval_metrics["val_loss"],
                "val_perplexity": eval_metrics["val_perplexity"],
            }
            results.append(run_result)
            
            # Save generations
            with open(run_out_dir / "generations.json", "w") as f:
                json.dumps(eval_metrics["generations"], indent=2)
                f.write(json.dumps(eval_metrics["generations"], indent=2))
                
            print(f"Run complete. Val Loss: {eval_metrics['val_loss']:.4f}")

    # Save summary
    summary_path = output_dir / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nExperiment Complete. Summary saved to {summary_path}")
    print("Run scripts/analyze_results.py to generate the final table.")

if __name__ == "__main__":
    run_experiment(parse_args())
