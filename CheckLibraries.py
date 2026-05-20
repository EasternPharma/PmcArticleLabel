"""
Checks that all required libraries are installed and imports are resolvable.
Safe to run in Google Colab — installs missing packages without killing the kernel.
"""

import importlib
import subprocess
import sys

REQUIRED_PACKAGES = {
    "openai": "openai>=1.0.0",
    "requests": "requests>=2.31.0",
    "pydantic": "pydantic>=2.0.0",
    "tqdm": "tqdm>=4.66.0",
}


def check_and_install(import_name: str, install_spec: str) -> bool:
    """Check if a package is importable; install it via pip if missing, then re-check. Returns True if available."""
    if importlib.util.find_spec(import_name) is not None:
        print(f"  [OK]      {import_name}")
        return True

    print(f"  [MISSING] {import_name} — installing {install_spec} ...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", install_spec],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  [ERROR]   Failed to install {import_name}:\n{result.stderr.strip()}")
        return False

    # Invalidate the import cache so the newly installed package is discoverable.
    importlib.invalidate_caches()
    if importlib.util.find_spec(import_name) is not None:
        print(f"  [OK]      {import_name} installed successfully.")
        return True

    print(f"  [ERROR]   {import_name} installed but still not importable. Restart the runtime.")
    return False


def main() -> bool:
    """
    Check all required libraries and install any that are missing.
    Returns True if all packages are available, False otherwise.
    Does NOT call sys.exit() — safe to use inside Google Colab notebooks.
    """
    print("Checking required libraries...\n")
    results = {
        name: check_and_install(name, spec)
        for name, spec in REQUIRED_PACKAGES.items()
    }
    print()

    all_ok = all(results.values())
    if all_ok:
        print("All libraries are available. Ready to run.")
    else:
        failed = [name for name, ok in results.items() if not ok]
        print(f"The following libraries could not be installed: {failed}")
        print("Please install them manually or restart the runtime.")

    return all_ok


if __name__ == "__main__":
    if not main():
        sys.exit(1)
