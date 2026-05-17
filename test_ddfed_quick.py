"""
Quick test script to verify DDFedTraining components work.
Tests with 1 round and 2 clients for faster execution.
"""
import subprocess
import sys
import time
import os
from pathlib import Path

def main():
    """Quick test of DDFedTraining."""
    print("=" * 60)
    print("DDFedTraining Quick Test (1 round, 2 clients)")
    print("=" * 60)
    
    project_root = Path(__file__).parent
    server_port = 8080
    server_address = f"127.0.0.1:{server_port}"
    
    # Start server FIRST so clients can connect immediately
    print("\n[1/3] Starting server...")
    server_cmd = [
        sys.executable,
        str(project_root / "server" / "ddfed_server.py"),
        "--port", str(server_port),
        "--num-rounds", "1",  # Just 1 round for quick test
        "--num-clients", "2",
        "--clip-norm", "1.0",
        "--perturbation-scale", "0.01",
        "--threshold", "0.5",
        "--fusion-method", "similarity",
        "--skip-encryption"
    ]
    
    server_process = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=project_root
    )
    
    print(f"  Server started (PID: {server_process.pid})")
    print("  Waiting for server to initialize...")
    time.sleep(3)  # Give server time to start listening
    
    # Start 2 clients AFTER server is ready
    print("\n[2/3] Starting 2 clients...")
    client_processes = []
    
    for i in range(1, 3):
        client_id = f"client-{i}"
        print(f"  Starting {client_id}...")
        
        client_cmd = [
            sys.executable,
            str(project_root / "client" / "ddfed_client_main.py"),
            "--server-address", server_address,
            "--client-id", str(i),  # Integer ID (will be converted to string in client)
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
        time.sleep(1)  # Stagger client starts
    
    print("  Clients started. Waiting for training to complete...")
    print("\n[3/3] Waiting for training to complete (max 120 seconds)...")
    
    try:
        # Wait with longer timeout
        server_stdout, server_stderr = server_process.communicate(timeout=120)
        
        print("\n" + "=" * 60)
        print("Server Output:")
        print("=" * 60)
        if server_stdout:
            output = server_stdout.decode('utf-8', errors='ignore')
            print(output[-1000:] if len(output) > 1000 else output)  # Last 1000 chars
        if server_stderr:
            errors = server_stderr.decode('utf-8', errors='ignore')
            if errors.strip():
                print("\nServer Errors:")
                print(errors[-500:] if len(errors) > 500 else errors)
        
        # Check clients
        print("\n" + "=" * 60)
        print("Client Status:")
        print("=" * 60)
        for client_id, client_process in client_processes:
            try:
                client_stdout, client_stderr = client_process.communicate(timeout=5)
                status = "[OK] Completed" if client_process.returncode == 0 else f"[FAILED] Exit code: {client_process.returncode}"
                print(f"{client_id}: {status}")
                if client_stdout:
                    output = client_stdout.decode('utf-8', errors='ignore')
                    if output.strip():
                        print(f"  Output (last 200 chars): {output[-200:]}")
                if client_stderr:
                    errors = client_stderr.decode('utf-8', errors='ignore')
                    if errors.strip():
                        print(f"  Errors: {errors[-300:]}")
            except subprocess.TimeoutExpired:
                print(f"{client_id}: Still running (terminating...)")
                client_process.terminate()
                time.sleep(0.5)
                client_process.kill()
        
        # Check for success indicators
        success = True
        if server_process.returncode != 0:
            print("\n[WARNING] Server did not exit cleanly")
            success = False
        
        server_output = server_stdout.decode('utf-8', errors='ignore') if server_stdout else ""
        if "Round" in server_output or "aggregat" in server_output.lower():
            print("\n[SUCCESS] Training rounds executed")
        else:
            print("\n[WARNING] No training rounds detected in server output")
            success = False
        
        print("\n" + "=" * 60)
        if success:
            print("[SUCCESS] DDFedTraining Quick Test: PASSED")
        else:
            print("[WARNING] DDFedTraining Quick Test: ISSUES DETECTED")
        print("=" * 60)
        
        return 0 if success else 1
        
    except subprocess.TimeoutExpired:
        print("\n⚠ Test timed out after 120 seconds")
        print("Gathering diagnostic information...")
        
        # Check server status
        if server_process.poll() is None:
            print("  Server is still running")
        else:
            print(f"  Server exited with code: {server_process.returncode}")
        
        # Check client status and capture output
        for client_id, client_process in client_processes:
            if client_process.poll() is None:
                print(f"  {client_id}: Still running")
            else:
                print(f"  {client_id}: Exited with code: {client_process.returncode}")
                try:
                    client_stdout, client_stderr = client_process.communicate(timeout=2)
                    if client_stdout:
                        output = client_stdout.decode('utf-8', errors='ignore')
                        if output.strip():
                            print(f"    Output: {output[-500:]}")
                    if client_stderr:
                        errors = client_stderr.decode('utf-8', errors='ignore')
                        if errors.strip():
                            print(f"    Errors: {errors[-500:]}")
                except subprocess.TimeoutExpired:
                    print(f"    Could not read output (process still running)")
                except Exception as e:
                    print(f"    Error reading output: {e}")
        
        print("\nTerminating processes...")
        server_process.terminate()
        time.sleep(1)
        server_process.kill()
        for _, client_process in client_processes:
            client_process.terminate()
            time.sleep(0.5)
            client_process.kill()
        
        return 1
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        server_process.terminate()
        for _, client_process in client_processes:
            client_process.terminate()
        return 1


if __name__ == "__main__":
    sys.exit(main())
