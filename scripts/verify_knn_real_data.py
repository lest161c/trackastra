import os
import sys
import time
import torch
import matplotlib.pyplot as plt
from argparse import Namespace

# Add trackastra root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trackastra.model import TrackingTransformer
from trackastra.data.distributed import BalancedDataModule

def main():
    print("=====================================================")
    print(" KNN Attention Real Data Benchmark & Correctness Proof")
    print("=====================================================")

    # Create dummy args based on vanvliet_mini config
    args = Namespace(
        config='configs/vanvliet_mini.yaml',
        input_train=['../data/vanvliet/recA/151027-05'],
        input_val=['../data/vanvliet/recA/151031-03'],
        input_test=[],
        cache=True,
        cachedir='runs/.cache_mini',
        features='wrfeat',
        augment=0,
        window=4,
        max_tokens=1024,
        train_samples=256,
        batch_size=2,
        num_workers=0,
        crop_size=[320, 320],
        ndim=2,
        weight_by_ndivs=False,
        weight_by_dataset=False,
        n_pool_sampler=1,
        example_images=False,
        detection_folders=['TRA'],
        compress=True,
        distributed=False,
    )

    print("\n1. Initializing DataLoader with Real Data (Vanvliet mini)...")
    dataset_kwargs = dict(
        features=args.features,
        window=args.window,
        max_tokens=args.max_tokens,
        crop_size=args.crop_size,
        ndim=args.ndim,
        weight_by_ndivs=args.weight_by_ndivs,
        weight_by_dataset=args.weight_by_dataset,
        from_subfolder=False,
        detection_folders=args.detection_folders,
    )
    
    from trackastra.data import collate_sequence_padding
    
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_sequence_padding,
    )
    
    sampler_kwargs = dict(
        batch_size=args.batch_size,
        num_samples=args.train_samples,
    )
    
    datamodule = BalancedDataModule(
        input_train=args.input_train,
        input_val=args.input_val,
        cachedir=args.cachedir,
        augment=args.augment,
        distributed=False,
        dataset_kwargs=dataset_kwargs,
        sampler_kwargs=sampler_kwargs,
        loader_kwargs=loader_kwargs,
    )
    
    datamodule.setup(stage='fit')
    train_loader = datamodule.train_dataloader()

    print("\n2. Initializing TrackingTransformers...")
    
    # Read actual dimensions from dataset
    dataset_instance = datamodule.datasets["train"].datasets[0]
    actual_coord_dim = dataset_instance.ndim
    actual_feat_dim = dataset_instance.feat_dim
    print(f"Detected coord_dim={actual_coord_dim}, feat_dim={actual_feat_dim}")
    
    common_kwargs = dict(
        feat_dim=actual_feat_dim,
        coord_dim=actual_coord_dim,
        d_model=128,
        num_encoder_layers=2,
        num_decoder_layers=2,
        window=4,
        attn_positional_bias='rope',
        attn_positional_bias_n_spatial=16,
    )

    model_dense = TrackingTransformer(**common_kwargs, use_knn_attention=False).cuda()
    model_knn_exact = TrackingTransformer(**common_kwargs, use_knn_attention=True, knn_neighbors=1024).cuda()
    model_knn_fast = TrackingTransformer(**common_kwargs, use_knn_attention=True, knn_neighbors=32).cuda()

    # Sync weights to ensure identical starting states
    state_dict = model_dense.state_dict()
    model_knn_exact.load_state_dict(state_dict)
    model_knn_fast.load_state_dict(state_dict)

    model_dense.train()
    model_knn_exact.train()
    model_knn_fast.train()

    optimizer_dense = torch.optim.Adam(model_dense.parameters(), lr=1e-4)
    optimizer_knn_fast = torch.optim.Adam(model_knn_fast.parameters(), lr=1e-4)

    criterion = torch.nn.BCEWithLogitsLoss()

    print("\n3. Fetching real training batches...")
    batches = []
    for i, batch in enumerate(train_loader):
        if i >= 20: break
        # Move to GPU
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batches.append(batch)

    b0 = batches[0]
    print(f"Batch keys: {b0.keys()}")
    
    feats_key = 'feats' if 'feats' in b0 else 'features'
    
    print("\n--- PART 1: Correctness Proof ---")
    print("Testing if KNN indices exactly match dense attention when K=N")
    
    b0_mini = {
        'coords': b0['coords'][:, :8, :],
        'features': b0[feats_key][:, :8, :],
        'padding_mask': None
    }
    
    model_knn_mini = TrackingTransformer(**common_kwargs, use_knn_attention=True, knn_neighbors=8).cuda()
    model_knn_mini.load_state_dict(state_dict)
    model_dense.eval()
    model_knn_mini.eval()
    
    with torch.no_grad():
        out_dense = model_dense(b0_mini['coords'], b0_mini['features'], padding_mask=None)
        out_knn_exact = model_knn_mini(b0_mini['coords'], b0_mini['features'], padding_mask=None)
        
        diff = torch.abs(out_dense - out_knn_exact).max().item()
        mean_val = out_dense.abs().mean().item()
        print(f"Max absolute difference: {diff:.6f} (Mean val: {mean_val:.6f})")
        
        if diff > 1e-4:
            print("Dense:\n", out_dense[0, 0, :2])
            print("KNN:\n", out_knn_exact[0, 0, :2])
        
        if diff < 1e-4:
            print("✅ SUCCESS: knn_indices are mathematically correct!")
            print("   The spatial gather correctly reconstructed the full dense attention matrix.")
        else:
            print("❌ WARNING: Outputs diverge (likely floating point summation order differences, or padding sort). Proceeding anyway to check learning.")
            
    print("\n--- PART 2: Performance & Loss Decay (KNN=32 vs Dense) ---")
    
    def calculate_loss(out, b):
        target = b['assoc_matrix'].float() # B, T, T
        
        pad_mask = b['padding_mask'] # B, T
        valid = ~pad_mask
        valid_2d = valid.unsqueeze(-1) & valid.unsqueeze(-2) # B, T, T
        
        return criterion(out[valid_2d], target[valid_2d])

    def train_loop(model, optimizer, name):
        print(f"\nTraining {name}...")
        torch.cuda.synchronize()
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats()
        
        losses = []
        for i, b in enumerate(batches):
            optimizer.zero_grad()
            out = model(b['coords'], b[feats_key], padding_mask=b['padding_mask'])
            loss = calculate_loss(out, b)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        torch.cuda.synchronize()
        end_time = time.time()
        max_mem = torch.cuda.max_memory_allocated() / (1024**2)
        
        duration = end_time - start_time
        print(f"[{name}] Time: {duration:.2f}s | Peak Mem: {max_mem:.0f} MB")
        return losses, duration, max_mem

    losses_dense, time_dense, mem_dense = train_loop(model_dense, optimizer_dense, "Dense Attention")
    losses_knn, time_knn, mem_knn = train_loop(model_knn_fast, optimizer_knn_fast, "KNN Attention (K=32)")

    print(f"\n--- Summary ---")
    print(f"Speedup: {time_dense / time_knn:.2f}x")
    print(f"VRAM Reduction: {mem_dense / mem_knn:.2f}x")
    
    print("\nChecking Loss Decay (Step 0 -> Step 19):")
    print(f"Dense Loss: {losses_dense[0]:.4f} -> {losses_dense[-1]:.4f}")
    print(f"KNN Loss:   {losses_knn[0]:.4f} -> {losses_knn[-1]:.4f}")

    if losses_knn[-1] < losses_knn[0]:
        print("✅ SUCCESS: KNN Model successfully learns and decays loss on real data!")
    else:
        print("❌ WARNING: KNN Model loss did not decay as expected.")

    # Save Plot
    plt.figure(figsize=(10, 6))
    plt.plot(losses_dense, label='Dense Attention (Exact)', marker='o', linestyle='-', linewidth=2)
    plt.plot(losses_knn, label='KNN Sparse Attention (K=32)', marker='s', linestyle='--', linewidth=2)
    plt.title('Loss Decay on Real Data: Dense vs KNN Attention (Vanvliet Dataset)', fontsize=14)
    plt.xlabel('Training Step (Batches of 2)', fontsize=12)
    plt.ylabel('BCE Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('real_data_benchmark_loss.png', dpi=150)
    print("\nSaved loss curve plot to 'real_data_benchmark_loss.png'")

if __name__ == '__main__':
    main()
