"""
Enhanced DDFedTraining runner with FHE backend support and improved client management.

This script:
1. Starts clients FIRST using helper script
2. Waits for clients to initialize
3. Starts server
4. Monitors training progress
5. Supports both baseline and Markov modes
"""
import subprocess
import sys
import time
import os
from pathlib import Path
import platform

def main():
    """Run DDFedTraining with enhanced client management."""
    print("=" * 60)
    print("DDFedTraining Enhanced Runner")
    print("=" * 60)
    
    # Configuration
    num_clients = 3
    num_rounds = 3
    server_port = 8080
    server_address = f"127.0.0.1:{server_port}"
    mode = "ddfed"  # "ddfed" or "ddfed_markov"
    use_fhe = False  # Set to True to use FHE (slower)
    
    print(f"\nConfiguration:")
    print(f"  - Number of clients: {num_clients}")
    print(f"  - Number of rounds: {num_rounds}")
    print(f"  - Server address: {server_address}")
    print(f"  - Mode: {mode}")
    print(f"  - FHE encryption: {'ENABLED' if use_fhe else 'DISABLED (faster)'}")
    
    project_root = Path(__file__).parent
    
    # Step 1: Start clients FIRST using helper script
    print(f"\n[1/3] Starting {num_clients} clients...")
    if platform.system() == "Windows":
        helper_script = project_root / "start_clients.ps1"
        if helper_script.exists():
            print("  Using PowerShell helper script...")
            ps_cmd = [
                "powershell", "-ExecutionPolicy", "Bypass", "-File",
                str(helper_script),
                "-NumClients", str(num_clients),
                "-ServerAddress", server_address
            ]
            subprocess.Popen(ps_cmd, cwd=project_root)
            print(f"  Clients starting in separate windows...")
            print(f"  Waiting 10 seconds for clients to initialize...")
            time.sleep(10)
        else:
            print("  Helper script not found, starting clients directly...")
            _start_clients_direct(project_root, num_clients, server_address)
            time.sleep(8)
    else:
        print("  Starting clients directly (non-Windows)...")
        _start_clients_direct(project_root, num_clients, server_address)
        time.sleep(8)
    
    # Step 2: Start server
    print(f"\n[2/3] Starting DDFedTraining server...")
    server_cmd = [
        sys.executable,
        str(project_root / "server" / "ddfed_server.py"),
        "--port", str(server_port),
        "--num-rounds", str(num_rounds),
        "--num-clients", str(num_clients),
        "--mode", mode,
        "--clip-norm", "1.0",
        "--perturbation-scale", "0.01",
        "--threshold", "0.5",
        "--fusion-method", "similarity"
    ]
    
    if not use_fhe:
        server_cmd.append("--skip-encryption")
    
    server_process = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=project_root
    )
    
    print(f"  Server started (PID: {server_process.pid})")
    print(f"\n[3/3] Monitoring training progress...")
    print(f"  (Press Ctrl+C to stop all processes)\n")
    
    try:
        # Wait for server to complete
        server_stdout, server_stderr = server_process.communicate()
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        
        if server_stdout:
            output = server_stdout.decode('utf-8', errors='ignore')
            # Extract key metrics
            if "Round" in output:
                print("\nTraining Summary:")
                lines = output.split('\n')
                for line in lines[-20:]:  # Last 20 lines
                    if "Round" in line or "complete" in line.lower():
                        print(f"  {line}")
        
        if server_stderr:
            errors = server_stderr.decode('utf-8', errors='ignore')
            if errors.strip() and "WARNING" not in errors:
                print("\nServer Errors:")
                print(errors[-500:])
        
        print("\n" + "=" * 60)
        print("✅ DDFedTraining completed successfully!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user. Stopping server...")
        if server_process.poll() is None:
            server_process.terminate()
            server_process.wait()
        print("Server stopped.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if server_process.poll() is None:
            server_process.terminate()
        sys.exit(1)


def _start_clients_direct(project_root: Path, num_clients: int, server_address: str):
    """Start clients directly (fallback)."""
    processes = []
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
        proc = subprocess.Popen(
            client_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=project_root
        )
        processes.append(proc)
        time.sleep(1)
    return processes


if __name__ == "__main__":
    main()
