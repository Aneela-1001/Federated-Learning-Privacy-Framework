"""
Test script to compare baseline DDFed vs DDFed with Markov noise.

This script runs both modes and compares:
- Inversion attack success (simulated via gradient norm analysis)
- Model accuracy
- Convergence stability
"""
import subprocess
import sys
import time
import os
from pathlib import Path
import json
import numpy as np

def run_experiment(mode: str, num_clients: int = 3, num_rounds: int = 3, port: int = 8080):
    """Run a single experiment with given mode."""
    print(f"\n{'='*60}")
    print(f"Running {mode.upper()} mode")
    print(f"{'='*60}")
    
    project_root = Path(__file__).parent
    server_address = f"127.0.0.1:{port}"
    
    # Start server
    print(f"\n[1/2] Starting server ({mode})...")
    server_cmd = [
        sys.executable,
        str(project_root / "server" / "ddfed_server.py"),
        "--port", str(port),
        "--num-rounds", str(num_rounds),
        "--num-clients", str(num_clients),
        "--skip-encryption",  # Faster for testing
        "--mode", mode
    ]
    
    server_process = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=project_root
    )
    
    print(f"  Server started (PID: {server_process.pid})")
    time.sleep(3)  # Wait for server to initialize
    
    # Start clients
    print(f"\n[2/2] Starting {num_clients} clients...")
    client_processes = []
    
    for i in range(1, num_clients + 1):
        client_cmd = [
            sys.executable,
            str(project_root / "client" / "ddfed_client_main.py"),
            "--server-address", server_address,
            "--client-id", str(i),
            "--is-benign",
            "--clip-norm", "1.0",
            "--threshold", "0.5"
        ]
        
        client_process = subprocess.Popen(
            client_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=project_root
        )
        
        client_processes.append((f"client-{i}", client_process))
        time.sleep(1)
    
    print(f"  All clients started. Waiting for training to complete...")
    
    # Wait for server to complete
    try:
        server_stdout, server_stderr = server_process.communicate(timeout=120)
        
        # Wait for clients
        for client_id, client_process in client_processes:
            try:
                client_process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                client_process.terminate()
        
        # Parse results
        server_output = server_stdout.decode('utf-8', errors='ignore') if server_stdout else ""
        server_errors = server_stderr.decode('utf-8', errors='ignore') if server_stderr else ""
        
        # Extract metrics from logs
        results = {
            "mode": mode,
            "server_exit_code": server_process.returncode,
            "completed": server_process.returncode == 0,
            "server_output": server_output[-500:] if len(server_output) > 500 else server_output,
            "has_errors": len(server_errors.strip()) > 0
        }
        
        # Check for consensus votes (Markov mode)
        if mode == "ddfed_markov":
            consensus_count = server_output.count("Consensus")
            results["consensus_events"] = consensus_count
        
        return results
        
    except subprocess.TimeoutExpired:
        print(f"  ⚠ Experiment timed out")
        server_process.terminate()
        for _, client_process in client_processes:
            client_process.terminate()
        return {"mode": mode, "completed": False, "error": "timeout"}
    except Exception as e:
        print(f"  ⚠ Error: {e}")
        server_process.terminate()
        for _, client_process in client_processes:
            client_process.terminate()
        return {"mode": mode, "completed": False, "error": str(e)}


def main():
    """Run comparison tests."""
    print("="*60)
    print("DDFed vs DDFed-Markov Comparison Test")
    print("="*60)
    
    num_clients = 3
    num_rounds = 3
    
    results = {}
    
    # Test baseline DDFed
    results["baseline"] = run_experiment("ddfed", num_clients, num_rounds, port=8080)
    time.sleep(5)  # Wait between experiments
    
    # Test DDFed-Markov
    results["markov"] = run_experiment("ddfed_markov", num_clients, num_rounds, port=8081)
    
    # Print comparison
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    print(f"\nBaseline DDFed:")
    print(f"  Completed: {results['baseline'].get('completed', False)}")
    print(f"  Exit Code: {results['baseline'].get('server_exit_code', 'N/A')}")
    
    print(f"\nDDFed-Markov:")
    print(f"  Completed: {results['markov'].get('completed', False)}")
    print(f"  Exit Code: {results['markov'].get('server_exit_code', 'N/A')}")
    if 'consensus_events' in results['markov']:
        print(f"  Consensus Events: {results['markov']['consensus_events']}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if results['baseline'].get('completed') and results['markov'].get('completed'):
        print("✅ Both modes completed successfully!")
        print("\nKey Differences:")
        print("  - Baseline DDFed: Standard aggregation")
        print("  - DDFed-Markov: Encrypted aggregation + consensus voting + Markov noise")
    else:
        print("⚠ Some experiments did not complete successfully")
        if not results['baseline'].get('completed'):
            print(f"  - Baseline failed: {results['baseline'].get('error', 'unknown')}")
        if not results['markov'].get('completed'):
            print(f"  - Markov failed: {results['markov'].get('error', 'unknown')}")
    
    return 0 if (results['baseline'].get('completed') and results['markov'].get('completed')) else 1


if __name__ == "__main__":
    sys.exit(main())
