"""
Test script for federated learning system with 5 clients.
Tests the hybrid privacy model integration.
"""
import subprocess
import time
import sys
import os
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "client"))
sys.path.insert(0, str(Path(__file__).parent / "server"))
sys.path.insert(0, str(Path(__file__).parent / "shared"))

def test_system(num_clients: int = 5, num_rounds: int = 3):
    """
    Test the federated learning system with specified number of clients.
    
    Args:
        num_clients: Number of clients to test with
        num_rounds: Number of federated learning rounds
    """
    print(f"\n{'='*60}")
    print(f"Testing Federated Learning System with {num_clients} Clients")
    print(f"{'='*60}\n")
    
    # Change to project directory
    project_dir = Path(__file__).parent / "federated-learning-system"
    os.chdir(project_dir)
    
    # Start server in background
    print("Starting server...")
    server_process = subprocess.Popen(
        [
            sys.executable,
            "server/server.py",
            "--port", "8080",
            "--min-clients", str(num_clients),
            "--num-rounds", str(num_rounds),
            "--num-clients", str(num_clients),
            "--use-secagg"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=project_dir
    )
    
    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(5)
    
    # Start clients
    client_processes = []
    print(f"Starting {num_clients} clients...")
    
    for i in range(1, num_clients + 1):
        client_id = f"client-{i}"
        print(f"  Starting {client_id}...")
        
        client_process = subprocess.Popen(
            [
                sys.executable,
                "client/client.py",
                "--server-address", "localhost:8080",
                "--client-id", client_id,
                "--num-samples", "100",
                "--num-clients", str(num_clients),
                "--use-dp",
                "--use-secagg",
                "--dp-noise-multiplier", "1.0",
                "--dp-l2-norm-clip", "1.0"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=project_dir
        )
        client_processes.append((client_id, client_process))
        time.sleep(1)  # Stagger client starts
    
    print(f"\nAll {num_clients} clients started. Waiting for training to complete...")
    print("This may take a few minutes...\n")
    
    # Wait for clients to complete
    for client_id, process in client_processes:
        try:
            stdout, stderr = process.communicate(timeout=300)  # 5 minute timeout
            if process.returncode == 0:
                print(f"✓ {client_id} completed successfully")
            else:
                print(f"✗ {client_id} failed with return code {process.returncode}")
                if stderr:
                    print(f"  Error: {stderr.decode()[:200]}")
        except subprocess.TimeoutExpired:
            print(f"✗ {client_id} timed out")
            process.kill()
    
    # Wait for server to complete
    print("\nWaiting for server to complete...")
    try:
        server_stdout, server_stderr = server_process.communicate(timeout=60)
        if server_process.returncode == 0:
            print("✓ Server completed successfully")
        else:
            print(f"✗ Server failed with return code {server_process.returncode}")
            if server_stderr:
                print(f"  Error: {server_stderr.decode()[:200]}")
    except subprocess.TimeoutExpired:
        print("✗ Server timed out")
        server_process.kill()
    
    print(f"\n{'='*60}")
    print("Test completed!")
    print(f"{'='*60}\n")
    
    # Cleanup
    server_process.terminate()
    for _, process in client_processes:
        process.terminate()


def test_with_simulation(num_clients: int = 5, num_rounds: int = 3):
    """
    Test using Flower's simulation mode (faster for testing).
    
    Args:
        num_clients: Number of clients
        num_rounds: Number of rounds
    """
    # Check if Ray is available
    try:
        import ray
    except ImportError:
        print("\n" + "="*60)
        print("ERROR: Ray library not found!")
        print("="*60)
        print("\nSimulation mode requires Ray. Install it with:")
        print("  pip install -U 'flwr[simulation]'")
        print("\nAlternatively, use manual testing:")
        print("  python test_manual.py")
        print("  OR")
        print("  python test_system.py --num-clients 5  # (without --simulation flag)")
        return
    
    print(f"\n{'='*60}")
    print(f"Testing with Flower Simulation ({num_clients} clients)")
    print(f"{'='*60}\n")
    
    project_dir = Path(__file__).parent / "federated-learning-system"
    os.chdir(project_dir)
    
    # Create simulation script
    sim_script = f"""
import flwr as fl
from flwr.server.strategy import FedAvg
from server.server import HybridPrivacyFedAvg
from client.client import FlowerClient, load_data
from shared.model import Net
import torch

def client_fn(cid):
    client_id = f"client-{{cid}}"
    trainloader, valloader = load_data(client_id, num_samples=100)
    model = Net(num_features=784, num_classes=10)
    device = torch.device("cpu")
    
    return FlowerClient(
        model=model,
        trainloader=trainloader,
        valloader=valloader,
        device=device,
        client_id=client_id,
        use_dp=True,
        use_he=False,
        use_secagg=True,
        num_clients={num_clients},
        dp_noise_multiplier=1.0,
        dp_l2_norm_clip=1.0
    )

strategy = HybridPrivacyFedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients={num_clients},
    min_evaluate_clients={num_clients},
    min_available_clients={num_clients},
    use_secagg=True,
    num_clients={num_clients}
)

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients={num_clients},
    config=fl.server.ServerConfig(num_rounds={num_rounds}),
    strategy=strategy,
)
"""
    
    # Write and run simulation script
    sim_file = project_dir / "test_simulation.py"
    with open(sim_file, "w") as f:
        f.write(sim_script)
    
    try:
        result = subprocess.run(
            [sys.executable, "test_simulation.py"],
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        
        if result.returncode == 0:
            print("\n✓ Simulation completed successfully!")
        else:
            print(f"\n✗ Simulation failed with return code {result.returncode}")
    except subprocess.TimeoutExpired:
        print("\n✗ Simulation timed out")
    finally:
        # Cleanup
        if sim_file.exists():
            sim_file.unlink()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test federated learning system")
    parser.add_argument(
        "--num-clients",
        type=int,
        default=5,
        help="Number of clients (default: 5)"
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=3,
        help="Number of federated learning rounds (default: 3)"
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Use Flower simulation mode (faster)"
    )
    
    args = parser.parse_args()
    
    if args.simulation:
        test_with_simulation(args.num_clients, args.num_rounds)
    else:
        test_system(args.num_clients, args.num_rounds)

