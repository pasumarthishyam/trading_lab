"""Convenience wrapper — delegates to the test suite runner."""
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "-m", "strategies.VCF.tests.run_all"] + sys.argv[1:],
        cwd=str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent),
    )
