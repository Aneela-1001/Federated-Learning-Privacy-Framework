import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from torchvision import datasets

# Optional existing CSVs (if you already have them). If missing, we auto-download MNIST.
TRAIN_CSV = "mnist_train.csv"
TEST_CSV = "mnist_test.csv"

# Non-IID splits by label groups (5 clients)
# Adjust as needed; each client gets a subset of digits.
CLIENT_SPLITS = {
    "client_1": [0, 1],
    "client_2": [2, 3],
    "client_3": [4, 5],
    "client_4": [6, 7],
    "client_5": [8, 9],
}

BASE_OUTPUT_DIR = "client/data"


def split_and_save(df, split_map, split_type):
    for client, labels in split_map.items():
        client_dir = os.path.join(BASE_OUTPUT_DIR, client)
        os.makedirs(client_dir, exist_ok=True)

        client_df = df[df["label"].isin(labels)]
        output_path = os.path.join(client_dir, f"{split_type}.csv")
        client_df.to_csv(output_path, index=False)

        print(f"Saved {split_type} data for {client} -> {output_path}")


def _mnist_to_df(
    train: bool,
    max_per_label: Optional[int] = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Download MNIST (if needed) and convert to a flat tabular DataFrame:
    feature_1..feature_784 + label

    max_per_label:
      - None: use full dataset (60k train / 10k test)
      - int: cap rows per label (faster, good for quick FL testing)
    """
    root = Path(__file__).parent / "client" / "data" / "_mnist_raw"
    ds = datasets.MNIST(root=str(root), train=train, download=True)

    # Convert to numpy
    x = ds.data.numpy().astype(np.float32) / 255.0  # (N, 28, 28)
    y = ds.targets.numpy().astype(np.int64)  # (N,)

    if max_per_label is not None:
        rng = np.random.default_rng(seed)
        keep_idx: List[int] = []
        for label in range(10):
            idx = np.where(y == label)[0]
            if len(idx) == 0:
                continue
            k = min(max_per_label, len(idx))
            keep_idx.extend(rng.choice(idx, size=k, replace=False).tolist())
        keep_idx = sorted(keep_idx)
        x = x[keep_idx]
        y = y[keep_idx]

    x = x.reshape(x.shape[0], -1)  # (N, 784)

    cols = [f"feature_{i+1}" for i in range(x.shape[1])] + ["label"]
    df = pd.DataFrame(np.concatenate([x, y.reshape(-1, 1)], axis=1), columns=cols)

    # Ensure label is integer (pandas may cast to float due to concat)
    df["label"] = df["label"].astype(int)
    return df


def main():
    # Resolve paths relative to this file (so it works no matter where you run it from)
    base_dir = Path(__file__).parent
    train_csv_path = base_dir / TRAIN_CSV
    test_csv_path = base_dir / TEST_CSV

    print("Loading MNIST data...")

    if train_csv_path.exists() and test_csv_path.exists():
        print(f"Found existing CSVs: {train_csv_path.name}, {test_csv_path.name}")
        train_df = pd.read_csv(train_csv_path)
        test_df = pd.read_csv(test_csv_path)
    else:
        print("MNIST CSVs not found, downloading MNIST via torchvision and generating DataFrames...")
        # Keep this small by default so it runs quickly; set max_per_label=None for full MNIST
        train_df = _mnist_to_df(train=True, max_per_label=2000, seed=42)
        test_df = _mnist_to_df(train=False, max_per_label=500, seed=43)

    print("Performing non-IID split...")
    split_and_save(train_df, CLIENT_SPLITS, "train")
    split_and_save(test_df, CLIENT_SPLITS, "test")

    print("Done. Client datasets created successfully.")


if __name__ == "__main__":
    main()
