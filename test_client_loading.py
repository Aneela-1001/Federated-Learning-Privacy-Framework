"""
Test script to verify client data loading works correctly.
Tests loading data for a single client.
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "client"))
sys.path.insert(0, str(project_root / "shared"))

# Change to project directory
os.chdir(project_root)

# Import after setting up paths
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# Import client's load_data function
import importlib.util
spec = importlib.util.spec_from_file_location("client_module", project_root / "client" / "client.py")
client_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client_module)
load_data = client_module.load_data

def test_client_loading(client_id: str = "client-1"):
    """Test loading data for a specific client."""
    print("="*60)
    print(f"Testing Data Loading for {client_id}")
    print("="*60)
    
    try:
        # Load data
        print(f"\nLoading data for {client_id}...")
        trainloader, testloader = load_data(client_id, batch_size=32)
        
        print(f"\n[OK] Data loaded successfully!")
        print(f"\nTraining Data:")
        print(f"  - Number of batches: {len(trainloader)}")
        print(f"  - Total samples: {len(trainloader.dataset)}")
        
        # Get a sample batch
        sample_batch = next(iter(trainloader))
        features, labels = sample_batch
        print(f"  - Batch shape: {features.shape}")
        print(f"  - Label shape: {labels.shape}")
        print(f"  - Feature range: [{features.min():.3f}, {features.max():.3f}]")
        print(f"  - Unique labels in batch: {sorted(labels.unique().tolist())}")
        
        print(f"\nTest Data:")
        print(f"  - Number of batches: {len(testloader)}")
        print(f"  - Total samples: {len(testloader.dataset)}")
        
        # Get a sample batch from test
        sample_test_batch = next(iter(testloader))
        test_features, test_labels = sample_test_batch
        print(f"  - Batch shape: {test_features.shape}")
        print(f"  - Label shape: {test_labels.shape}")
        print(f"  - Unique labels in batch: {sorted(test_labels.unique().tolist())}")
        
        # Check label distribution
        print(f"\nLabel Distribution (first 100 training samples):")
        all_labels = []
        for i, (_, labels_batch) in enumerate(trainloader):
            all_labels.extend(labels_batch.tolist())
            if i >= 3:  # Check first 4 batches (~128 samples)
                break
        
        from collections import Counter
        label_counts = Counter(all_labels)
        for label in sorted(label_counts.keys()):
            count = label_counts[label]
            percentage = (count / len(all_labels)) * 100
            print(f"  Label {label}: {count} samples ({percentage:.1f}%)")
        
        print(f"\n[OK] Data loading test passed!")
        return True
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {e}")
        print(f"\nMake sure you've run: python partition_mnist.py")
        return False
    except Exception as e:
        print(f"\n[ERROR] Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test client data loading")
    parser.add_argument(
        "--client-id",
        type=str,
        default="client-1",
        help="Client ID to test (default: client-1)"
    )
    
    args = parser.parse_args()
    
    success = test_client_loading(args.client_id)
    sys.exit(0 if success else 1)

