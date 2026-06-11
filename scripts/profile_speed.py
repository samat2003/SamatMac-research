"""Speed profiling and optimization script."""
import time
import argparse
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

# We reuse the 20M classes but pass them the 120M configs
from model.config_120m import Baseline120MConfig, SamatNext120MConfig
from model.baseline_model import Baseline20M as BaselineModel
from model.samatnext_20m import SamatNext20M as SamatNextModel

def profile_model(model_name: str, batch_size: int, seq_len: int, steps: int = 10, accum_steps: int = 4):
    print(f"\n{'='*50}\nProfiling {model_name} (BS={batch_size}, Seq={seq_len}, Accum={accum_steps})\n{'='*50}")
    
    if model_name == "Baseline":
        config = Baseline120MConfig(max_seq_len=seq_len)
        model = BaselineModel(config)
    else:
        config = SamatNext120MConfig(max_seq_len=seq_len)
        model = SamatNextModel(config)
        
    print(f"Params: {model.count_params():,}")
    
    # Initialize optimizer
    import mlx.optimizers as optim
    opt = optim.AdamW(learning_rate=1e-4)
    
    # Setup step function
    def loss_fn(model, x, y, mask):
        out = model(x, targets=y, loss_mask=mask)
        return out["loss"], out
        
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    
    # Dummy data
    x = mx.random.randint(0, config.vocab_size, (batch_size, seq_len), dtype=mx.int32)
    y = mx.random.randint(0, config.vocab_size, (batch_size, seq_len), dtype=mx.int32)
    mask = mx.ones((batch_size, seq_len), dtype=mx.bfloat16)
    
    # Warmup
    print("Evaluating graph and running warmup...")
    t0 = time.time()
    (loss, out), grads = loss_and_grad_fn(model, x, y, mask)
    mx.eval(loss, grads)
    warmup_time = time.time() - t0
    print(f"Compilation/Warmup took: {warmup_time:.2f}s")
    
    # Profiling loop
    print(f"Running {steps} profiling steps...")
    t0 = time.time()
    for i in range(steps):
        (loss, out), grads = loss_and_grad_fn(model, x, y, mask)
        opt.update(model, grads)
        mx.eval(loss, grads, model.parameters(), opt.state)
    
    total_time = time.time() - t0
    avg_step_time = total_time / steps
    tokens_per_sec = (batch_size * seq_len * steps) / total_time
    
    print(f"Results:")
    print(f"  Total time for {steps} steps: {total_time:.2f}s")
    print(f"  Average time per step: {avg_step_time:.3f}s")
    print(f"  Throughput: {tokens_per_sec:.1f} tokens/sec")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--seq_lens", type=int, nargs="+", default=[512, 1024])
    args = parser.parse_args()
    
    for seq_len in args.seq_lens:
        for bs in args.batch_sizes:
            profile_model("Baseline", bs, seq_len)
            profile_model("SamatNext", bs, seq_len)
