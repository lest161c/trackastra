import torch
import torch.utils.benchmark as benchmark
import argparse
import os
import gc
from trackastra.model.model_parts import RelativePositionalAttention

def measure_memory(func, *args, **kwargs):
    if not torch.cuda.is_available():
        return 0
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _ = func(*args, **kwargs)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)

def benchmark_attention(dtype, mode, batch_size=2, num_nodes=500, embed_dim=256, n_head=8, coord_dim=3, knn_neighbors=12):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configure env variable for the module
    os.environ["TRACKASTRA_DISABLE_KNN"] = "0" if mode == "knn" else "1"
    os.environ["TRACKASTRA_KNN_NEIGHBORS"] = str(knn_neighbors)
    
    # Create module and move to device
    attn = RelativePositionalAttention(
        coord_dim=coord_dim,
        embed_dim=embed_dim,
        n_head=n_head
    ).to(device).to(dtype)
    
    # Create dummy inputs
    features = torch.randn(batch_size, num_nodes, embed_dim, device=device, dtype=dtype)
    coords = torch.randn(batch_size, num_nodes, coord_dim, device=device, dtype=dtype)
    padding_mask = torch.zeros(batch_size, num_nodes, device=device, dtype=torch.bool)
    
    # Generate cached KNN indices for KNN mode to simulate preprocessing
    knn_indices = None
    if mode == "knn":
        yx = coords[..., 1:]
        dist_chunk = torch.cdist(yx, yx)
        _, knn_indices = torch.topk(dist_chunk, k=knn_neighbors, dim=-1, largest=False)
    
    def forward_pass():
        return attn(features, features, features, coords, padding_mask, knn_indices=knn_indices)

    # Measure memory
    mem_mb = measure_memory(forward_pass)

    # Warmup
    for _ in range(5):
        forward_pass()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Benchmark Speed
    stmt = 'attn(features, features, features, coords, padding_mask, knn_indices=knn_indices)'
    globals_dict = {
        'attn': attn, 
        'features': features, 
        'coords': coords, 
        'padding_mask': padding_mask,
        'knn_indices': knn_indices
    }
    
    t0 = benchmark.Timer(
        stmt=stmt,
        globals=globals_dict,
        num_threads=1,
        label='Attention',
        sub_label=f'dtype={dtype}, mode={mode}',
        description=f'b={batch_size}, n={num_nodes}',
    )
    
    # Run measurement
    result = t0.blocked_autorange(min_run_time=2.0)
    return result, mem_mb

def main():
    parser = argparse.ArgumentParser(description="Benchmark Trackastra Attention")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--num-nodes", type=int, default=1024, help="Number of nodes/cells")
    parser.add_argument("--embed-dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--n-head", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--knn-neighbors", type=int, default=12, help="Number of KNN neighbors")
    args = parser.parse_args()

    print("Running benchmarks...")
    print(f"Config: Batch={args.batch_size}, Nodes={args.num_nodes}, Embed={args.embed_dim}, Heads={args.n_head}, KNN={args.knn_neighbors}")
    print("-" * 50)
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available! VRAM measurements will be 0 and speeds will be on CPU.")

    results = []
    
    # 16-bit Only
    print("Testing FP16 - Default (Full Attention)...")
    res_16_def, mem_16_def = benchmark_attention(
        torch.float16, "default",
        batch_size=args.batch_size, num_nodes=args.num_nodes, 
        embed_dim=args.embed_dim, n_head=args.n_head, knn_neighbors=args.knn_neighbors
    )
    results.append(res_16_def)
    print(f"-> VRAM usage: {mem_16_def:.2f} MB")
    
    print("\nTesting FP16 - KNN Attention (Cached Indices)...")
    res_16_knn, mem_16_knn = benchmark_attention(
        torch.float16, "knn",
        batch_size=args.batch_size, num_nodes=args.num_nodes, 
        embed_dim=args.embed_dim, n_head=args.n_head, knn_neighbors=args.knn_neighbors
    )
    results.append(res_16_knn)
    print(f"-> VRAM usage: {mem_16_knn:.2f} MB")

    if mem_16_def > 0:
        print(f"\n=> VRAM Reduction: {mem_16_def / max(1e-5, mem_16_knn):.2f}x (from {mem_16_def:.2f} MB to {mem_16_knn:.2f} MB)")
    
    print("\nExecution Time Results:")
    print("-" * 50)
    compare = benchmark.Compare(results)
    compare.print()

if __name__ == "__main__":
    main()