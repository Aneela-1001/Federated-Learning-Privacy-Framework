"""
Simple test runner for federated learning system.
Tests with 5 clients by default using manual mode (no Ray required).
This matches the Docker setup which also doesn't require Ray.
"""
import argparse
import subprocess
import sys
import os
import time
from pathlib import Path

def check_and_free_port(port: int):
    """Check if port is in use and free it if possible."""
    import socket
    try:
        # Try to bind to the port to see if it's free
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            test_socket.bind(('127.0.0.1', port))
            test_socket.close()
            return True  # Port is free
        except OSError:
            # Port is in use
            test_socket.close()
            print(f"⚠ Port {port} is already in use. Attempting to free it...")
            # Try to find and kill the process using the port (Windows)
            if sys.platform == 'win32':
                try:
                    # Find process using the port
                    result = subprocess.run(
                        ['netstat', '-ano'], 
                        capture_output=True, 
                        text=True
                    )
                    for line in result.stdout.split('\n'):
                        if f':{port}' in line and 'LISTENING' in line:
                            parts = line.split()
                            if len(parts) >= 5:
                                pid = parts[-1]
                                try:
                                    subprocess.run(['taskkill', '/F', '/PID', pid], 
                                                 capture_output=True, check=False)
                                    print(f"✓ Freed port {port} (killed PID {pid})")
                                    time.sleep(2)  # Wait for port to be released
                                    # Verify port is now free
                                    test_socket2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                                    test_socket2.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                                    try:
                                        test_socket2.bind(('127.0.0.1', port))
                                        test_socket2.close()
                                        return True
                                    except OSError:
                                        test_socket2.close()
                                        pass
                                except:
                                    pass
                except:
                    pass
            print(f"⚠ Could not automatically free port {port}.")
            print(f"  Please manually stop any process using port {port} and try again.")
            return False
    except Exception as e:
        print(f"⚠ Error checking port {port}: {e}")
        return True  # Continue anyway


def main():
    """Run the federated learning test using manual mode (no Ray)."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test federated learning system")
    parser.add_argument(
        "--use-fe",
        action="store_true",
        default=False,
        help="Enable Functional Encryption (default: False, slow)"
    )
    parser.add_argument(
        "--use-ldp",
        action="store_true",
        default=False,
        help="Enable Local Differential Privacy (default: False)"
    )
    parser.add_argument(
        "--use-adaptive-dp",
        action="store_true",
        default=False,
        help="Enable Adaptive DP (default: False)"
    )
    parser.add_argument(
        "--use-robust-agg",
        action="store_true",
        default=False,
        help="Enable Robust Aggregation (default: False)"
    )
    parser.add_argument(
        "--use-he",
        action="store_true",
        default=False,
        help="Enable Homomorphic Encryption (default: False, very slow)"
    )
    args = parser.parse_args()
    
    print("="*60)
    print("Federated Learning System Test")
    print("="*60)
    print("\nThis will test the system with 5 clients using manual mode.")
    print("No Ray required - matches Docker setup.")
    print("\nPrivacy Mechanisms:")
    print("  - DP (Differential Privacy): Enabled")
    print("  - SecAgg (Secure Aggregation): Enabled")
    if args.use_fe:
        print("  - FE (Functional Encryption): Enabled (WARNING: Very slow)")
    if args.use_ldp:
        print("  - LDP (Local DP): Enabled")
    if args.use_adaptive_dp:
        print("  - Adaptive DP: Enabled")
    if args.use_robust_agg:
        print("  - Robust Aggregation: Enabled")
    if args.use_he:
        print("  - HE (Homomorphic Encryption): Enabled (WARNING: Very slow)")
    print("\nNote: FE and HE encryption are computationally expensive.")
    print("      For faster testing, use DP + SecAgg only.\n")
    
    project_dir = Path(__file__).parent
    
    # Check if we're in the right directory
    if not (project_dir / "server" / "server.py").exists():
        print("Error: Please run this script from the federated-learning-system directory")
        sys.exit(1)
    
    # Check and free port 8080 if needed
    if not check_and_free_port(8080):
        print("\nPlease free port 8080 and try again.")
        sys.exit(1)
    
    # Ensure log directories exist
    (project_dir / "server" / "logs").mkdir(parents=True, exist_ok=True)
    (project_dir / "client" / "logs").mkdir(parents=True, exist_ok=True)
    
    processes = []
    
    # Set PYTHONPATH to include project directory so shared module can be found
    env = os.environ.copy()
    pythonpath = str(project_dir)
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = pythonpath + os.pathsep + env['PYTHONPATH']
    else:
        env['PYTHONPATH'] = pythonpath
    
    try:
        # Start server
        print("Starting server...")
        server_process = subprocess.Popen(
            [
                sys.executable,
                "server/server.py",
                "--port", "8080",
                "--min-clients", "5",
                "--num-rounds", "1",  # Reduced to 1 round for faster testing
                "--num-clients", "5",
                "--use-secagg",
                # Note: HE is disabled by default in run_test.py due to performance
                # Uncomment the line below to enable HE (will be much slower)
                # "--use-he"
            ],
            cwd=project_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        processes.append(("server", server_process))
        print(f"✓ Server started (PID: {server_process.pid})")
        
        # Wait for server to start
        print("Waiting for server to initialize...")
        time.sleep(5)  # Increased wait time for server to fully start
        
        # Start clients
        print("\nStarting 5 clients...")
        for i in range(1, 6):
            client_id = f"client-{i}"
            print(f"  Starting {client_id}...")
            
            client_cmd = [
                sys.executable,
                "client/client.py",
                "--server-address", "localhost:8080",
                "--client-id", client_id,
                "--num-samples", "100",
                "--num-clients", "5",
                "--use-dp",
                "--use-secagg",
                "--dp-noise-multiplier", "1.0",
                "--dp-l2-norm-clip", "1.0"
            ]
            
            # Add privacy mechanism flags
            if args.use_fe:
                client_cmd.append("--use-fe")
            if args.use_ldp:
                client_cmd.append("--use-ldp")
                client_cmd.extend(["--ldp-epsilon", "1.0"])
            if args.use_adaptive_dp:
                client_cmd.append("--use-adaptive-dp")
            if args.use_he:
                client_cmd.append("--use-he")
            
            # HE is disabled by default in client.py now
            
            client_process = subprocess.Popen(
                client_cmd,
                cwd=project_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            processes.append((client_id, client_process))
            time.sleep(1)  # Stagger client starts
        
        print("\n✓ All processes started!")
        print("\nProcesses running:")
        for name, proc in processes:
            print(f"  - {name} (PID: {proc.pid})")
        
        print("\n" + "="*60)
        print("Training in progress...")
        print("Check logs in server/logs/ and client/logs/ for details")
        print("Press Ctrl+C to stop all processes")
        print("="*60 + "\n")
        
        # Wait for processes to complete
        print("Waiting for training to complete...")
        # Adjust timeout based on enabled mechanisms
        if args.use_fe or args.use_he:
            max_wait_time = 1800  # 30 minutes for encryption
            print("⚠ Extended timeout to 30 minutes due to encryption overhead")
        else:
            max_wait_time = 600  # 10 minutes for normal operation
        client_wait_after_server = 180  # 3 minutes for clients after server completes
        start_time = time.time()
        server_completed = False
        last_progress_time = 0
        
        while True:
            time.sleep(2)
            
            # Check server status
            server_proc = processes[0][1]  # First process is server
            if not server_completed and server_proc.poll() is not None:
                server_completed = True
                server_returncode = server_proc.returncode
                if server_returncode == 0:
                    print("\n✓ Server completed successfully. Waiting for clients to finish...")
                    # Reset timeout to give clients time after server completes
                    start_time = time.time()
                    max_wait_time = client_wait_after_server
                    last_progress_time = 0
                else:
                    print(f"\n⚠ Server exited with code {server_returncode}")
            
            # Check if all processes are still running
            running = [name for name, proc in processes if proc.poll() is None]
            if not running:
                print("\n✓ All processes completed!")
                break
            
            # Show progress every 30 seconds
            elapsed = time.time() - start_time
            if elapsed - last_progress_time >= 30:
                running_count = len(running)
                print(f"  [Progress] {running_count} process(es) still running... ({int(elapsed)}s elapsed)")
                last_progress_time = elapsed
            
            # Check timeout
            if elapsed > max_wait_time:
                if server_completed:
                    print(f"\n⚠ Timeout: Server completed but clients didn't finish within {max_wait_time} seconds.")
                    print("This may indicate clients are stuck or FE encryption is taking longer than expected.")
                    if args.use_fe:
                        print("FE encryption is computationally expensive - check client logs for progress.")
                    print("Stopping remaining clients...")
                else:
                    print(f"\n⚠ Timeout after {max_wait_time} seconds. Stopping all processes...")
                    if args.use_fe:
                        print("Note: FE encryption is computationally expensive and may require more time.")
                        print("      Check client logs in client/logs/ for encryption progress.")
                
                # Only terminate processes that are still running
                for name, proc in processes:
                    if proc.poll() is None:
                        print(f"  Terminating {name}...")
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                break
            
    except KeyboardInterrupt:
        print("\n\nStopping all processes...")
        for name, proc in processes:
            if proc.poll() is None:
                print(f"  Stopping {name}...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        print("All processes stopped.")
        sys.exit(0)
    
    # Check results
    print("\n" + "="*60)
    print("Test Results:")
    print("="*60)
    
    all_success = True
    for name, proc in processes:
        returncode = proc.returncode
        if returncode == 0:
            print(f"✓ {name} completed successfully")
        elif returncode is None:
            print(f"⚠ {name} was interrupted")
            all_success = False
        else:
            # Check if it's a normal disconnection (common for Flower clients)
            is_normal_disconnect = False
            error_msg = None
            if proc.stderr:
                try:
                    stderr_data = proc.stderr.read()
                    if stderr_data:
                        stderr_text = stderr_data.decode('utf-8', errors='ignore')
                        stderr_lower = stderr_text.lower()
                        # Check for normal disconnection messages
                        if any(keyword in stderr_lower for keyword in [
                            'disconnect', 'shut down', 'connection closed',
                            'server closed', 'normal shutdown'
                        ]):
                            is_normal_disconnect = True
                        else:
                            # Extract error message
                            stderr_lines = stderr_text.split('\n')
                            error_lines = [line.strip() for line in stderr_lines if line.strip()]
                            if error_lines:
                                error_msg = error_lines[-1]
                                if len(error_msg) > 200:
                                    error_msg = error_msg[:200] + "..."
                except:
                    pass
            
            if is_normal_disconnect:
                print(f"✓ {name} disconnected normally (server completed)")
            else:
                print(f"✗ {name} failed (return code: {returncode})")
                all_success = False
                if error_msg:
                    print(f"  Error: {error_msg}")
    
    print("\n" + "="*60)
    if all_success:
        print("✓ Test completed successfully!")
    else:
        print("✗ Test completed with errors")
    print("="*60)
    
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main())

