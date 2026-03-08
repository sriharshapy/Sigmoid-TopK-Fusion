"""
Triton kernel: fused sigmoid + top-k for Mixture of Experts router.

Operates on router logits [batch_size, num_experts] (e.g. [batch_size, 128]):
  1. Applies sigmoid to get router probabilities.
  2. Selects top-k values and their indices per row (for expert selection).

Call this right before expert selection in the MoE forward pass.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _sigmoid_topk_kernel(
    logits_ptr,
    topk_vals_ptr,
    topk_idx_ptr,
    batch_size: tl.constexpr,
    num_experts: tl.constexpr,
    k: tl.constexpr,
    BLOCK_N: tl.constexpr,  # must be >= num_experts so one row fits
):
    """
    One program per row. Load row -> sigmoid in registers -> top-k (no full sigmoid output).
    """
    row = tl.program_id(0)
    if row >= batch_size:
        return

    # Offsets for this row
    row_offs = row * num_experts + tl.arange(0, BLOCK_N)
    mask = tl.arange(0, BLOCK_N) < num_experts
    INF = -1e9

    # Load row, sigmoid (kept in registers only for top-k)
    x = tl.load(logits_ptr + row_offs, mask=mask, other=INF)
    s = 1.0 / (1.0 + tl.exp(-x))
    cur = tl.where(mask, s, INF)
    for ki in range(k):
        best_idx = tl.argmax(cur, axis=0)
        best_val = tl.max(cur)
        tl.store(topk_vals_ptr + row * k + ki, best_val)
        tl.store(topk_idx_ptr + row * k + ki, best_idx)
        # Mask out so next argmax won't pick same element
        cur = tl.where(tl.arange(0, BLOCK_N) == best_idx, INF, cur)


def sigmoid_topk(
    logits: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused sigmoid + top-k on router logits [batch_size, num_experts].
    Only top-k values and indices are written; full sigmoid is not stored.

    Args:
        logits: Router logits, shape (batch_size, num_experts), e.g. (B, 128).
        k: Number of top experts to select per token.

    Returns:
        topk_vals: Top-k values per row, shape (batch_size, k).
        topk_idx: Top-k indices per row, shape (batch_size, k), dtype int32.
    """
    batch_size, num_experts = logits.shape

    # for 128 experts 
    # bit length is 7
    # 1 << 7 = 128
    # so BLOCK_N = 128
    # so one block per row with BLOCK_N >= num_experts (power of 2)
    # so one block per row with BLOCK_N >= num_experts (power of 2)

    BLOCK_N = 1 << (num_experts - 1).bit_length()
    device = logits.device
    dtype = logits.dtype

    topk_vals = torch.empty((batch_size, k), device=device, dtype=dtype)
    topk_idx = torch.empty((batch_size, k), device=device, dtype=torch.int32)

    grid = (batch_size,)
    _sigmoid_topk_kernel[grid](
        logits,
        topk_vals,
        topk_idx,
        batch_size=batch_size,
        num_experts=num_experts,
        k=k,
        BLOCK_N=BLOCK_N,
    )
    return topk_vals, topk_idx


if __name__ == "__main__":
    import argparse
    import time
    p = argparse.ArgumentParser()
    p.add_argument("-f", "--file", default="tensor.pt", help="path to saved tensor")
    p.add_argument("-k", type=int, default=2, help="number of top experts")
    p.add_argument("-n", "--iters", type=int, default=100, help="number of timed iterations")
    p.add_argument("--no-warmup", action="store_true", help="skip warmup (for NCU profiling)")
    args = p.parse_args()

    logits = torch.load(args.file)
    if logits.dim() != 2:
        raise SystemExit(f"Expected 2D tensor (batch, experts), got shape {logits.shape}")
    # Triton kernels require CUDA tensors
    if logits.device.type != "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("Triton requires CUDA; no GPU available.")
        logits = logits.to("cuda")
    k = min(args.k, logits.shape[1])
    device = logits.device

    # Warmup: 3 runs (unless --no-warmup)
    if not args.no_warmup:
        for _ in range(3):
            topk_vals, topk_idx = sigmoid_topk(logits, k)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Timed runs
    start = time.perf_counter()
    for _ in range(args.iters):
        topk_vals, topk_idx = sigmoid_topk(logits, k)
        if device.type == "cuda":
            torch.cuda.synchronize()
    end = time.perf_counter()
    avg_ms = (end - start) / args.iters * 1000
    print(f"Average over {args.iters} runs: {avg_ms:.4f} ms")

    print("topk_vals:")
    print(topk_vals)
    print("topk_idx:")
    print(topk_idx)
