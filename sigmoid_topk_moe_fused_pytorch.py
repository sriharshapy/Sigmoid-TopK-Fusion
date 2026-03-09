"""
PyTorch version: sigmoid + top-k for Mixture of Experts router.

Same API as the Triton fused kernel: sigmoid on [batch_size, num_experts] logits,
then return only top-k values and indices (no full sigmoid tensor).
"""

import torch


def sigmoid_topk(
    logits: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sigmoid + top-k on router logits [batch_size, num_experts].
    Only top-k values and indices are returned; full sigmoid is not stored.

    Args:
        logits: Router logits, shape (batch_size, num_experts), e.g. (B, 128).
        k: Number of top experts to select per token.

    Returns:
        topk_vals: Top-k values per row, shape (batch_size, k).
        topk_idx: Top-k indices per row, shape (batch_size, k), dtype int32.
    """
    probs = logits.sigmoid()
    topk_vals, topk_idx = probs.topk(k, dim=-1)
    return topk_vals, topk_idx.to(torch.int32)


if __name__ == "__main__":
    import argparse
    import time
    p = argparse.ArgumentParser()
    p.add_argument("-f", "--file", default="tensor.pt", help="path to saved tensor")
    p.add_argument("-k", type=int, default=2, help="number of top experts")
    p.add_argument("-n", "--iters", type=int, default=100, help="number of timed iterations")
    p.add_argument("--no-warmup", action="store_true", help="skip warmup (for NCU profiling)")
    p.add_argument("--no-print", action="store_true", help="do not print topk_vals/topk_idx (e.g. for NCU)")
    args = p.parse_args()

    logits = torch.load(args.file)
    if logits.dim() != 2:
        raise SystemExit(f"Expected 2D tensor (batch, experts), got shape {logits.shape}")
    # Run on GPU when available (for benchmarking/NCU)
    if logits.device.type != "cuda" and torch.cuda.is_available():
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
    if not args.no_print:
        print(f"Average over {args.iters} runs: {avg_ms:.4f} ms")
        print("topk_vals:")
        print(topk_vals)
        print("topk_idx:")
        print(topk_idx)
