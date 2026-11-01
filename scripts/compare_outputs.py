"""Compare two output directories of processed CSVs and report which columns changed.

Usage: python scripts/compare_outputs.py <baseline_dir> <new_dir> [--tol 1e-6]
Rows are matched on the first column (ID / Stat / POS) plus POS when present.
"""

import argparse
import os

import pandas as pd


def load(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def compare_file(old_path: str, new_path: str, tol: float) -> list[str]:
    old, new = load(old_path), load(new_path)
    lines = []
    if set(old.columns) != set(new.columns):
        lines.append(f"  columns removed: {sorted(set(old.columns) - set(new.columns))}")
        lines.append(f"  columns added:   {sorted(set(new.columns) - set(old.columns))}")
    if len(old) != len(new):
        lines.append(f"  row count {len(old)} -> {len(new)}")
    key = [c for c in [old.columns[0], "POS"] if c in old.columns and c in new.columns]
    if old.duplicated(key).any() or new.duplicated(key).any():
        key = None
    if key:
        old, new = old.set_index(key), new.set_index(key)
        common_idx = old.index.intersection(new.index)
        old, new = old.loc[common_idx], new.loc[common_idx]
    elif len(old) == len(new):
        old, new = old.reset_index(drop=True), new.reset_index(drop=True)
    else:
        return lines
    for col in [c for c in old.columns if c in new.columns]:
        if pd.api.types.is_numeric_dtype(old[col]) and pd.api.types.is_numeric_dtype(new[col]):
            diff = (old[col] - new[col]).abs()
            nan_mismatch = old[col].isna() != new[col].isna()
            diff = diff.mask(nan_mismatch, float("inf"))
            n = int((diff > tol).sum())
            if n:
                lines.append(
                    f"  {col}: {n} rows differ, max abs diff {diff.max():.6g}, mean {diff[diff > tol].mean():.6g}"
                )
        elif not old[col].astype(str).equals(new[col].astype(str)):
            n = int((old[col].astype(str) != new[col].astype(str)).sum())
            lines.append(f"  {col}: {n} non-numeric rows differ")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("new")
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()
    for name in sorted(os.listdir(args.baseline)):
        new_path = os.path.join(args.new, name)
        if not os.path.exists(new_path):
            print(f"{name}: MISSING in new")
            continue
        lines = compare_file(os.path.join(args.baseline, name), new_path, args.tol)
        print(f"{name}: {'identical' if not lines else 'DIFFERENT'}")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
