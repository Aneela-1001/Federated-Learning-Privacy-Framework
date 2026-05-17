"""
Runner script for DDFedTraining.
This script demonstrates how to run DDFedTraining with multiple clients.
"""
import subprocess
import sys
import time
import os
from pathlib import Path

def main():
    """Run DDFedTraining with server and clients."""
    print("=" * 60)
    print("DDFedTraining Runner")
    print("=" * 60)
    
    # Configuration
    num_clients = 3
    num_rounds = 3
    server_port = 8080
    server_address = f"127.0.0.1:{server_port}"
    
    print(f"\nConfiguration:")
    print(f"  - Number of clients: {num_clients}")
    print(f"  - Number of rounds: {num_rounds}")
    print(f"  - Server address: {server_address}")
    
    # Get project root
    project_root = Path(__file__).parent
    
    # Start clients FIRST (so they're ready when server starts)
    # Option 1: Use PowerShell helper script (Windows)
    import platform
    if platform.system() == "Windows":
        print(f"\n[1/2] Starting {num_clients} clients using helper script...")
        try:
            helper_script = project_root / "start_clients.ps1"
            if helper_script.exists():
                # Use PowerShell script to start clients
                import subprocess as sp
                ps_cmd = [
                    "powershell", "-ExecutionPolicy", "Bypass", "-File",
                    str(helper_script),
                    "-NumClients", str(num_clients),
                    "-ServerAddress", server_address
                ]
                sp.Popen(ps_cmd, cwd=project_root)
                print(f"  Helper script launched. Clients starting in separate windows...")
                time.sleep(10)  # Give clients time to start
                client_processes = []  # Processes managed by PowerShell script
            else:
                raise FileNotFoundError("Helper script not found")
        except Exception as e:
            print(f"  Warning: Could not use helper script ({e}), starting clients directly...")
            # Fallback to direct start
            client_processes = _start_clients_direct(project_root, num_clients, server_address)
    else:
        # Non-Windows: start clients directly
        client_processes = _start_clients_direct(project_root, num_clients, server_address)
    
    print(f"\n  All {num_clients} clients started. Waiting for clients to initialize...")
    time.sleep(10)  # Increased wait time for clients to fully initialize and connect
    
    # Start server AFTER clients are ready
    print(f"\n[2/2] Starting DDFedTraining server...")
    server_cmd = [
        sys.executable,
        str(project_root / "server" / "ddfed_server.py"),
        "--port", str(server_port),
        "--num-rounds", str(num_rounds),
        "--num-clients", str(num_clients),
        "--clip-norm", "1.0",
        "--perturbation-scale", "0.01",
        "--threshold", "0.5",
        "--fusion-method", "similarity",
        "--skip-encryption"  # Skip FHE for faster testing
    ]
    
    server_process = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=project_root
    )
    
    print(f"  Server started (PID: {server_process.pid})")
    print(f"\nAll processes started. Waiting for training to complete...")
    print(f"  (This may take a while depending on your system)")
    print(f"  (Press Ctrl+C to stop all processes)\n")
    
    try:
        # Wait for server to complete
        server_stdout, server_stderr = server_process.communicate()
        
        print("\n" + "=" * 60)
        print("Server Output:")
        print("=" * 60)
        if server_stdout:
            print(server_stdout.decode('utf-8', errors='ignore'))
        if server_stderr:
            print("Server Errors:")
            print(server_stderr.decode('utf-8', errors='ignore'))
        
        # Wait for clients to complete (if we have process handles)
        if client_processes:
            for client_id, client_process in client_processes:
                try:
                    client_stdout, client_stderr = client_process.communicate(timeout=5)
                    print(f"\n{client_id} Output:")
                    if client_stdout:
                        print(client_stdout.decode('utf-8', errors='ignore')[:500])
                    if client_stderr:
                        print(f"{client_id} Errors:")
                        print(client_stderr.decode('utf-8', errors='ignore')[:500])
                except subprocess.TimeoutExpired:
                    print(f"{client_id}: Still running (may have been started by helper script)")
                except Exception as e:
                    print(f"{client_id}: Could not read output ({e})")
        else:
            print("\nClients were started by helper script - check their windows for output")
        
        print("\n" + "=" * 60)
        print("DDFedTraining completed successfully!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Stopping all processes...")
        
        # Terminate server
        if server_process.poll() is None:
            server_process.terminate()
            server_process.wait()
        
        # Terminate clients
        for client_id, client_process in client_processes:
            if client_process.poll() is None:
                client_process.terminate()
                client_process.wait()
        
        print("All processes stopped.")
        sys.exit(0)
    
    except Exception as e:
        print(f"\nError: {e}")
        
        # Clean up
        if server_process.poll() is None:
            server_process.terminate()
        
        for _, client_process in client_processes:
            if client_process.poll() is None:
                client_process.terminate()
        
        sys.exit(1)


def _start_clients_direct(project_root: Path, num_clients: int, server_address: str):
    """Start clients directly (fallback method)."""
    client_processes = []
    for i in range(1, num_clients + 1):
        client_id = f"client-{i}"
        print(f"  Starting {client_id}...")
        
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
        
        client_processes.append((client_id, client_process))
        time.sleep(1)
    
    return client_processes


if __name__ == "__main__":
    main()
