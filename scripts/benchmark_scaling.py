import torch
import torch.nn as nn
import time
import os
import csv
import gc
from trackastra.model.model_parts import RelativePositionalAttention

def measure_memory_and_time(mode, batch_size, num_nodes, embed_dim=256, coord_dim=3, knn_neighbors=12, steps=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    use_knn = (mode == "knn")
    
    attn = RelativePositionalAttention(
        coord_dim=coord_dim, 
        embed_dim=embed_dim, 
        n_head=8,
        use_knn_attention=use_knn,
        knn_neighbors=knn_neighbors
    ).to(device).half()
    optimizer = torch.optim.Adam(attn.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    feat = torch.randn(batch_size, num_nodes, embed_dim, device=device, dtype=torch.float16)
    coord = torch.randn(batch_size, num_nodes, coord_dim, device=device, dtype=torch.float16)
    pad = torch.zeros(batch_size, num_nodes, device=device, dtype=torch.bool)
    target = torch.randn(batch_size, num_nodes, embed_dim, device=device, dtype=torch.float16)
    
    knn_idx = None
    if mode == "knn":
        yx = coord[..., 1:]
        dist_chunk = torch.cdist(yx.float(), yx.float())  # cdist better in float32
        _, knn_idx = torch.topk(dist_chunk, k=knn_neighbors, dim=-1, largest=False)
    
    # Warmup
    for _ in range(3):
        optimizer.zero_grad()
        out = attn(feat, feat, feat, coord, padding_mask=pad, knn_indices=knn_idx)
        loss = loss_fn(out, target)
        loss.backward()
    
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    torch.cuda.synchronize()
    t0 = time.time()
    
    for _ in range(steps):
        optimizer.zero_grad()
        out = attn(feat, feat, feat, coord, padding_mask=pad, knn_indices=knn_idx)
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()
        
    torch.cuda.synchronize()
    t1 = time.time()
    
    mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
    avg_time_ms = ((t1 - t0) / steps) * 1000
    
    # clear memory
    del attn, optimizer, feat, coord, pad, target, knn_idx, out, loss
    gc.collect()
    torch.cuda.empty_cache()
    
    return avg_time_ms, mem_mb

def main():
    tokens_list = [500, 1000, 2000, 3000, 4000, 5000, 6000]
    batch_size = 1
    csv_file = "scaling_results.csv"
    
    print(f"Running benchmarks for tokens (cells across time window): {tokens_list}")
    
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Tokens", "Mode", "Time_ms", "VRAM_MB"])
        
        for n in tokens_list:
            print(f"Benchmarking N={n}...")
            try:
                t_def, mem_def = measure_memory_and_time("default", batch_size, n)
                writer.writerow([n, "Default", f"{t_def:.2f}", f"{mem_def:.2f}"])
            except Exception as e:
                print(f"  Default OOM or error at N={n}: {e}")
                writer.writerow([n, "Default", "OOM", "OOM"])
                
            try:
                t_knn, mem_knn = measure_memory_and_time("knn", batch_size, n)
                writer.writerow([n, "KNN", f"{t_knn:.2f}", f"{mem_knn:.2f}"])
            except Exception as e:
                print(f"  KNN OOM or error at N={n}: {e}")
                writer.writerow([n, "KNN", "OOM", "OOM"])
                
    print(f"Results saved to {csv_file}")

if __name__ == "__main__":
    main()
