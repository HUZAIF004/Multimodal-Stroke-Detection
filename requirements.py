"""
Environment Verification and Dependency Installer for Multimodal Stroke Detection System.
Run this script to verify your Python environment or install missing dependencies:
    python requirements.py
"""

import sys
import subprocess

REQUIRED_PACKAGES = [
    ("tensorflow", "tensorflow>=2.15.0"),
    ("torch", "torch>=2.0.0"),
    ("monai", "monai>=1.3.0"),
    ("xgboost", "xgboost>=2.0.0"),
    ("numpy", "numpy>=1.24.0"),
    ("pandas", "pandas>=2.0.0"),
    ("scipy", "scipy>=1.11.0"),
    ("sklearn", "scikit-learn>=1.3.0"),
    ("joblib", "joblib>=1.3.0"),
    ("nibabel", "nibabel>=5.1.0"),
    ("pydicom", "pydicom>=2.4.0"),
    ("flask", "flask>=3.0.0"),
    ("flask_cors", "flask-cors>=4.0.0"),
]


def check_and_install_packages():
    print("=" * 65)
    print("  MULTIMODAL STROKE DETECTION - DEPENDENCY CHECK")
    print(f"  Python Version: {sys.version.split()[0]}")
    print(f"  Interpreter: {sys.executable}")
    print("=" * 65)

    missing = []
    installed = []

    for import_name, pkg_spec in REQUIRED_PACKAGES:
        try:
            __import__(import_name)
            installed.append(pkg_spec)
            print(f"  [+] {pkg_spec:<25} (Installed)")
        except ImportError:
            missing.append(pkg_spec)
            print(f"  [-] {pkg_spec:<25} (MISSING)")

    print("-" * 65)

    if not missing:
        print("  All dependencies are satisfied! You are ready to run app.py.")
        print("=" * 65)
        return

    print(f"  Warning: Found {len(missing)} missing package(s): {', '.join(missing)}")
    user_input = input("  Would you like to install them now? (y/n): ").strip().lower()

    if user_input in ["y", "yes", ""]:
        print("\n  Installing missing packages...")
        cmd = [sys.executable, "-m", "pip", "install"] + missing
        try:
            subprocess.check_call(cmd)
            print("\n  [OK] All missing dependencies have been successfully installed!")
        except subprocess.CalledProcessError as e:
            print(f"\n  [ERROR] Installation failed with code {e.returncode}. Try running: pip install -r requirements.txt")
    else:
        print("\n  Skipping installation. You can install manually using:")
        print("      pip install -r requirements.txt")
    print("=" * 65)


if __name__ == "__main__":
    check_and_install_packages()
