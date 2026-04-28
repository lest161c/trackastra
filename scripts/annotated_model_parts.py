"""
ANNOTATED GUIDE TO THE KNN ATTENTION OPTIMIZATION
-------------------------------------------------

This file serves as a reference for understanding the exact changes made to 
trackastra/model/model_parts.py to enable O(N*K) scaling. 

Terminology Clarification: 
In the Trackastra paper, "nodes" refers to graph elements during tracking. 
During the Transformer phase, these elements are referred to as "tokens". 
A single token represents a detected cell. Therefore, the total number of 
tokens (N) in a single pass is [Cells per Frame] x [Frames in Window].
The parameter 'max_tokens: 2048' directly enforces the limit of N.

Below are the two heavily modified sections with extreme detail on HOW and WHY 
they work mathematically.
"""

# ==============================================================================
# SECTION 1: RelativePositionalBias (Adding learned spatial/temporal bias)
# ==============================================================================
# BEFORE: We calculated the distance between ALL N*N tokens.
# AFTER: If knn_indices are provided, we ONLY calculate distances for the N*K tokens.

import torch

def annotated_positional_bias_forward(self, coords: torch.Tensor, knn_indices: torch.Tensor = None):
    _B, _N, _D = coords.shape
    t = coords[..., 0]   # Temporal coordinates
    yx = coords[..., 1:] # Spatial coordinates
    
    if knn_indices is not None:
        # [WHY] Advanced Indexing to avoid N^2 materialization
        # Instead of calculating all-to-all distances, we use the pre-calculated 
        # KNN indices (shape: B, N, K) to fetch the exact coordinates of the K neighbors.
        B_idx = torch.arange(_B, device=coords.device).view(_B, 1, 1)
        
        # t_knn becomes shape (B, N, K), representing the time of the K neighbors
        t_knn = t[B_idx, knn_indices]
        # yx_knn becomes shape (B, N, K, 2), representing the (y,x) of the K neighbors
        yx_knn = yx[B_idx, knn_indices, :]
        
        # [HOW] Distance calculation
        # t.unsqueeze(-1) is (B, N, 1). We subtract t_knn (B, N, K) -> Result: (B, N, K)
        temporal_dist = t.unsqueeze(-1) - t_knn
        
        # Same for spatial: (B, N, 1, 2) - (B, N, K, 2) -> norm -> (B, N, K)
        spatial_dist = torch.norm(yx.unsqueeze(-2) - yx_knn, dim=-1)
    else:
        # Original N^2 Fallback
        temporal_dist = t.unsqueeze(-1) - t.unsqueeze(-2)
        spatial_dist = torch.cdist(yx, yx)

    # ... (Bucketization into discrete bins happens here identically to original) ...
    spatial_idx = torch.bucketize(spatial_dist, self.spatial_bins) # (B, N, K)
    # ...

    # We fetch the actual learned bias weights from the embedding table
    # bias shape: (B, N, K, n_head)
    bias = self.bias.index_select(0, idx).view((*spatial_idx.shape, self.n_head))
    
    if knn_indices is not None:
        # [CRITICAL SHAPE]
        # PyTorch SDPA expects masks to align with the sequence length. 
        # Our Sequence length will be manipulated to look like K. 
        # We permute to: (Batch, Heads, Queries(N), Keys(K)) -> (B, nH, N, K)
        bias = bias.permute(0, 3, 1, 2)
    else:
        # Original: (B, nH, N, N)
        bias = bias.transpose(-1, 1)
        
    return bias

# ==============================================================================
# SECTION 2: RelativePositionalAttention (The Core Attention Loop)
# ==============================================================================
# BEFORE: We multiplied Q (B, N, D) by K^T (B, D, N) -> Output: (B, N, N)
# AFTER: We fold N into the Batch dimension, and treat K as the sequence length.

def annotated_attention_forward(q, k, v, coords, knn_indices, ...):
    # ... (Q, K, V projections) ...
    
    if self.use_knn_attention and knn_indices is not None:
        # [WHY] We need to gather the Keys and Values for the K neighbors.
        # B_idx and H_idx create coordinate grids for advanced indexing.
        B_idx = torch.arange(B, device=q.device).view(B, 1, 1, 1)
        H_idx = torch.arange(self.n_head, device=q.device).view(1, self.n_head, 1, 1)
        
        # idx_exp is the knn_indices expanded across all attention heads.
        idx_exp = knn_indices.unsqueeze(1).expand(B, self.n_head, N, self.knn_neighbors)

        # [HOW] k_knn shape becomes: (Batch, Heads, Tokens(N), Neighbors(K), Head_Dim)
        k_knn = k[B_idx, H_idx, idx_exp, :]
        v_knn = v[B_idx, H_idx, idx_exp, :]

        # [THE DIMENSION FOLDING TRICK]
        # PyTorch's optimized scaled_dot_product_attention (Flash Attention) 
        # REQUIRES 4D tensors: (Batch, Head, SeqLen_Q, SeqLen_K)
        # To make it compute ONLY the K neighbors, we pretend that every single Token (N) 
        # is its own isolated batch.
        # So we merge B and N -> (B*N).
        # Query sequence length becomes 1. 
        # Key/Value sequence length becomes K.
        
        # q: (B, H, N, D) -> transpose -> reshape -> (B*N, H, 1, D)
        q_reshaped = q.transpose(1, 2).reshape(B * N, self.n_head, 1, q.shape[-1])
        
        # k_knn: (B, H, N, K, D) -> transpose -> reshape -> (B*N, H, K, D)
        k_reshaped = k_knn.transpose(1, 2).reshape(B * N, self.n_head, self.knn_neighbors, k.shape[-1])
        v_reshaped = v_knn.transpose(1, 2).reshape(B * N, self.n_head, self.knn_neighbors, v.shape[-1])

        # ... (We generate the mask identically to original, just constrained to N*K) ...
        # attn_mask_knn final shape before reshape: (B, H, N, K)
        # Reshape to match the folded batch trick: (B*N, H, 1, K)
        attn_mask_knn = attn_mask_knn.permute(0, 2, 1, 3).reshape(B * N, self.n_head, 1, self.knn_neighbors)

        # [FLASH ATTENTION EXECUTION]
        # Because we passed 4D tensors, PyTorch natively applies FlashAttention!
        # It computes (1 x K) instead of (N x N), vastly reducing VRAM and FLOPs.
        y_reshaped = F.scaled_dot_product_attention(
            q_reshaped, k_reshaped, v_reshaped, attn_mask=attn_mask_knn, dropout_p=0
        )

        # [UNFOLDING]
        # y_reshaped is (B*N, H, 1, D). 
        # We view it back into (B, N, H, D) and transpose to (B, N, H*D)
        y = y_reshaped.view(B, N, self.n_head, -1).transpose(1, 2)
