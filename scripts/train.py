"""Main training entrypoint for SamatNext-520M per SPEC.md."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx

from data.dataset import HUGGINGFACE_DATASETS, PythonCodeDataset
from data.tokenizer import SamatNextTokenizer
from model.config import DEFAULT_CONFIG, SamatNextConfig
from model.model import SamatNext
from train.sliding_mac import SlidingMac
from train.trainer import BATCH_SIZE, Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train SamatNext-520M")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
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
        "--synthetic_data",
        type=str,
        default=None,
        help="Path to synthetic jsonl file",
    )
    parser.add_argument("--max_samples", type=int, default=50000)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--benchmark_only", action="store_true")
    parser.add_argument(
        "--no_sliding",
        action="store_true",
        help="Skip SlidingMac benchmark",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("SamatNext-520M Training")
    print(f"Device: {mx.default_device()}")

    tok = SamatNextTokenizer.from_file(args.tokenizer_path)
    print(f"Tokenizer loaded: vocab_size={tok.tokenizer.get_vocab_size()}")

    config = DEFAULT_CONFIG
    model = SamatNext(config)
    print(f"Model built: {model.count_params():,} parameters")

    if not args.no_sliding:
        slider = SlidingMac(model, config)
        best_batch = slider.search()
        report = slider.report()
        print(f"SlidingMac report: {report}")
    else:
        best_batch = BATCH_SIZE

    if args.benchmark_only:
        print("Benchmark complete. Exiting.")
        return

    dataset = PythonCodeDataset(
        tok,
        config,
        max_samples=args.max_samples,
    )
    if args.local_data:
        dataset.load(local_path=args.local_data)
    elif args.synthetic_data:
        dataset.load(local_path=args.synthetic_data)
    elif args.dataset:
        dataset.load(dataset_name=args.dataset)
    else:
        dataset.load(dataset_name=HUGGINGFACE_DATASETS[0][0])
    print(f"Dataset loaded: {len(dataset)} samples")

    trainer = Trainer(model, config, output_dir=args.output_dir)
    trainer.train(dataset, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
