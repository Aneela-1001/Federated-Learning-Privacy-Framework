import argparse
import subprocess
import sys
import os
import time
from pathlib import Path


def check_and_free_port(port: int):
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port))
        s.close()
        return True
    except OSError:
        print(f"⚠ Port {port} is already in use.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run FL with poisoning attacks (5 clients)")
    parser.add_argument("--attack", choices=["scaling", "ipm", "alie"], required=True)
    parser.add_argument("--malicious-clients", type=int, default=2)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--num-rounds", type=int, default=3)
    parser.add_argument("--scaling-factor", type=float, default=20.0)
    parser.add_argument("--ipm-strength", type=float, default=2.0)
    parser.add_argument("--ipm-l2-bound", type=float, default=None)
    parser.add_argument("--alie-k", type=float, default=2.0)
    parser.add_argument("--use-dp", action="store_true")
    parser.add_argument("--use-secagg", action="store_true")

    args = parser.parse_args()

    if args.malicious_clients < 1 or args.malicious_clients > 5:
        print("Error: malicious-clients must be between 1 and 5")
        sys.exit(1)

    project_dir = Path(__file__).parent
    os.chdir(project_dir)

    if not check_and_free_port(args.port):
        sys.exit(1)

    print("=" * 70)
    print(f"Starting FL with {args.attack.upper()} attack")
    print(f"Clients: 5 | Malicious: {args.malicious_clients} | Benign: {5 - args.malicious_clients}")
    print("=" * 70)

    # ---------------- SERVER ----------------
    server_cmd = [
        sys.executable,
        "server/server.py",
        "--port", str(args.port),
        "--min-clients", "5",
        "--num-rounds", str(args.num_rounds),
        "--num-clients", "5"
    ]
    if args.use_secagg:
        server_cmd.append("--use-secagg")

    print("\nStarting server...")
    server_process = subprocess.Popen(server_cmd)
    time.sleep(5)

    processes = []

    # ---------------- MALICIOUS CLIENTS ----------------
    for i in range(1, args.malicious_clients + 1):
        cmd = [
            sys.executable,
            "client/client.py",
            "--server-address", f"127.0.0.1:{args.port}",
            "--client-id", f"client-{i}",
            "--num-clients", "5"
        ]

        if args.attack == "scaling":
            cmd += ["--use-scaling-attack", "--scaling-factor", str(args.scaling_factor)]
        elif args.attack == "ipm":
            cmd += ["--use-ipm-attack", "--ipm-attack-strength", str(args.ipm_strength)]
            if args.ipm_l2_bound:
                cmd += ["--ipm-l2-bound", str(args.ipm_l2_bound)]
        elif args.attack == "alie":
            cmd += ["--use-alie-attack", "--alie-k", str(args.alie_k)]

        if args.use_dp:
            cmd += ["--use-dp", "--dp-noise-multiplier", "1.0", "--dp-l2-norm-clip", "1.0"]
        if args.use_secagg:
            cmd.append("--use-secagg")

        print(f"Starting malicious client-{i}")
        processes.append(subprocess.Popen(cmd))
        time.sleep(1)

    # ---------------- BENIGN CLIENTS ----------------
    for i in range(args.malicious_clients + 1, 6):
        cmd = [
            sys.executable,
            "client/client.py",
            "--server-address", f"127.0.0.1:{args.port}",
            "--client-id", f"client-{i}",
            "--num-clients", "5"
        ]

        if args.use_dp:
            cmd += ["--use-dp", "--dp-noise-multiplier", "1.0", "--dp-l2-norm-clip", "1.0"]
        if args.use_secagg:
            cmd.append("--use-secagg")

        print(f"Starting benign client-{i}")
        processes.append(subprocess.Popen(cmd))
        time.sleep(1)

    # ---------------- WAIT FOR TRAINING ----------------
    print("\nTraining running...")
    wait_time = args.num_rounds * 60
    print(f"Waiting ~{wait_time} seconds for completion")
    time.sleep(wait_time)

    # ---------------- SHUTDOWN ----------------
    print("\nStopping all processes...")
    server_process.terminate()
    for p in processes:
        p.terminate()

    print("✓ Experiment finished cleanly")


if __name__ == "__main__":
    main()

