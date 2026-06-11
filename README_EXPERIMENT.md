# 20M Validation Experiment

This directory contains the code to rigorously test whether the architectural novelties in `SamatNext` (alternating differential attention, causal memory bus, MTP) provide an advantage over a standard Transformer baseline at the ~20M parameter scale.

## Experiment Design
- **Models**: `SamatNext-20M` and `Baseline-20M`.
- **Parameter Parity**: Both models use `d_model=256`, 6 layers, and tied embeddings to ensure a fair ~30M parameter budget (inclusive of the vocabulary).
- **Control Variables**: Both models see the exact same dataset, use the exact same tokenizer, token budget, context length, optimizer, schedule, and FIM configuration.
- **Robustness**: The experiment loop runs multiple random seeds to rule out initialization luck.

## Running the Experiment

A unified script manages the experiment end-to-end:

### Smoke Test (Quick Validation)
To verify that the code runs, memory usage is within bounds, and the pipeline executes:
```bash
python scripts/experiment_runner.py --tokenizer_path data/tokenizer.json --smoke_test
```
*Note: Make sure your `tokenizer.json` path is correct. If you don't have one, specify the path to a generic huggingface python tokenizer or generate one first.*

### Full Experiment (Multi-seed)
To run the full multi-seed experiment:
```bash
python scripts/experiment_runner.py --tokenizer_path data/tokenizer.json --seeds 42 123 456
```

You can optionally specify a dataset with `--dataset` or a local file with `--local_data`. By default, it will use the default HuggingFace corpus setup in `data/dataset.py`.

## Analyzing Results
Once the experiment completes, metrics and generations will be saved to `results/experiment_20m`. To summarize the findings:

```bash
python scripts/analyze_results.py
```

This will print a comparative table of average validation losses and perplexity across the tested seeds. Examine the `generations.json` files within the individual seed directories to manually inspect code generation quality and FIM capability.
