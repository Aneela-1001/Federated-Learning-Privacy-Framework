"""
Verification script to check if all components are properly set up.
"""
import sys
from pathlib import Path

def check_imports():
    """Check if all required modules can be imported."""
    print("Checking imports...")
    
    errors = []
    
    # Check basic imports
    try:
        import torch
        print("✓ torch")
    except ImportError as e:
        errors.append(f"✗ torch: {e}")
    
    try:
        import flwr
        print("✓ flwr")
    except ImportError as e:
        errors.append(f"✗ flwr: {e}")
    
    try:
        import numpy
        print("✓ numpy")
    except ImportError as e:
        errors.append(f"✗ numpy: {e}")
    
    # Check privacy libraries
    try:
        import phe
        print("✓ phe (Homomorphic Encryption)")
    except ImportError:
        print("⚠ phe (Homomorphic Encryption) - optional, install with: pip install phe")
    
    # Check shared modules
    try:
        sys.path.insert(0, str(Path(__file__).parent / "shared"))
        from model import Net
        print("✓ shared.model")
    except ImportError as e:
        errors.append(f"✗ shared.model: {e}")
    
    try:
        from utils import setup_logger
        print("✓ shared.utils")
    except ImportError as e:
        errors.append(f"✗ shared.utils: {e}")
    
    try:
        from secagg import SimplifiedSecureAggregation
        print("✓ shared.secagg")
    except ImportError as e:
        errors.append(f"✗ shared.secagg: {e}")
    
    try:
        from he_utils import HomomorphicEncryption
        print("✓ shared.he_utils")
    except ImportError as e:
        errors.append(f"✗ shared.he_utils: {e}")
    
    # Check client modules
    try:
        sys.path.insert(0, str(Path(__file__).parent / "client"))
        from dp_utils import DifferentialPrivacy
        print("✓ client.dp_utils")
    except ImportError as e:
        errors.append(f"✗ client.dp_utils: {e}")
    
    # Check server modules
    try:
        sys.path.insert(0, str(Path(__file__).parent / "server"))
        import server
        print("✓ server.server")
    except ImportError as e:
        errors.append(f"✗ server.server: {e}")
    
    if errors:
        print("\nErrors found:")
        for error in errors:
            print(f"  {error}")
        return False
    else:
        print("\n✓ All imports successful!")
        return True

def check_files():
    """Check if required files exist."""
    print("\nChecking files...")
    
    base_path = Path(__file__).parent
    required_files = [
        "client/client.py",
        "client/dp_utils.py",
        "server/server.py",
        "shared/model.py",
        "shared/utils.py",
        "shared/secagg.py",
        "shared/he_utils.py",
        "shared/data_utils.py",
        "docker-compose.yml",
    ]
    
    missing = []
    for file in required_files:
        if (base_path / file).exists():
            print(f"✓ {file}")
        else:
            missing.append(file)
            print(f"✗ {file} - MISSING")
    
    if missing:
        print(f"\nMissing files: {', '.join(missing)}")
        return False
    else:
        print("\n✓ All required files present!")
        return True

def check_directories():
    """Check if required directories exist."""
    print("\nChecking directories...")
    
    base_path = Path(__file__).parent
    required_dirs = [
        "client/logs",
        "server/logs",
        "client/data",
    ]
    
    missing = []
    for dir_path in required_dirs:
        full_path = base_path / dir_path
        if full_path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"⚠ {dir_path} - will be created automatically")
    
    return True

def main():
    """Run all checks."""
    print("="*60)
    print("Federated Learning System - Setup Verification")
    print("="*60)
    
    # Create necessary directories first to avoid import errors
    base_path = Path(__file__).parent
    required_dirs = [
        base_path / "client" / "logs",
        base_path / "server" / "logs",
        base_path / "client" / "data",
    ]
    for dir_path in required_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    checks = [
        ("File Check", check_files),
        ("Directory Check", check_directories),
        ("Import Check", check_imports),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n{name}:")
        print("-" * 60)
        result = check_func()
        results.append((name, result))
    
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n✓ All checks passed! System is ready to use.")
        print("\nNext steps:")
        print("  1. Run: python run_test.py")
        print("  2. Or: docker compose up --build")
        print("  3. See QUICKSTART.md for detailed instructions")
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

