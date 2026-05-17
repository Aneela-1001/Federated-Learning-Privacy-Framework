"""
Check non-IID distribution across clients.
"""
import pandas as pd
import os
from pathlib import Path

def check_client_distribution(client_num: int):
    """Check label distribution for a specific client."""
    base_dir = Path("client/data")
    client_dir = base_dir / f"client_{client_num}"
    train_csv = client_dir / "train.csv"
    test_csv = client_dir / "test.csv"
    
    if not train_csv.exists():
        print(f"[ERROR] Client {client_num}: train.csv not found")
        return
    
    # Load data
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv) if test_csv.exists() else None
    
    print(f"\n{'='*60}")
    print(f"Client {client_num} Distribution")
    print(f"{'='*60}")
    
    # Training data
    print(f"\nTraining Data:")
    print(f"  Total samples: {len(train_df)}")
    print(f"  Label distribution:")
    label_counts = train_df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        percentage = (count / len(train_df)) * 100
        print(f"    Label {label}: {count:5d} samples ({percentage:5.1f}%)")
    
    # Test data
    if test_df is not None:
        print(f"\nTest Data:")
        print(f"  Total samples: {len(test_df)}")
        print(f"  Label distribution:")
        test_label_counts = test_df['label'].value_counts().sort_index()
        for label, count in test_label_counts.items():
            percentage = (count / len(test_df)) * 100
            print(f"    Label {label}: {count:5d} samples ({percentage:5.1f}%)")
    
    # Check if non-IID (should have specific labels only)
    unique_labels = sorted(train_df['label'].unique())
    print(f"\n  Unique labels: {unique_labels}")
    
    # Expected labels based on partition_mnist.py
    expected_labels = {
        1: [0, 1],
        2: [2, 3],
        3: [4, 5],
        4: [6, 7],
        5: [8, 9]
    }
    
    if client_num in expected_labels:
        expected = expected_labels[client_num]
        if set(unique_labels) == set(expected):
            print(f"  [OK] Non-IID: Contains only labels {expected} (as expected)")
        else:
            print(f"  [WARN] Non-IID check: Expected {expected}, got {unique_labels}")


def check_all_clients():
    """Check distribution for all clients."""
    print("="*60)
    print("Non-IID Distribution Check")
    print("="*60)
    
    for client_num in range(1, 6):
        check_client_distribution(client_num)
    
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print("\nNon-IID Split (as configured in partition_mnist.py):")
    print("  client_1: labels [0, 1]")
    print("  client_2: labels [2, 3]")
    print("  client_3: labels [4, 5]")
    print("  client_4: labels [6, 7]")
    print("  client_5: labels [8, 9]")
    print("\n[OK] This is a non-IID distribution (each client has different labels)")


if __name__ == "__main__":
    check_all_clients()

