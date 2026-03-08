"""
Create an n×m tensor and save it to a file in the current directory.
"""
import argparse
import torch

def main():
    p = argparse.ArgumentParser(description="Create n×m tensor and save to file")
    p.add_argument("-n", type=int, default=4, help="number of rows")
    p.add_argument("-m", type=int, default=6, help="number of columns")
    p.add_argument("-o", "--out", default="tensor.pt", help="output file path")
    args = p.parse_args()

    x = torch.randn(args.n, args.m)
    torch.save(x, args.out)
    print(f"Saved {args.n}×{args.m} tensor to {args.out}")
    print(x)

if __name__ == "__main__":
    main()
