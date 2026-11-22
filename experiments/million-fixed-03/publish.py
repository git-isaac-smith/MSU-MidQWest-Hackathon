"""Refresh the results sheet while jobs run and audit the final million-shot export."""
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    build = [sys.executable,str(ROOT/"scripts/build_million_error_sheet.py")]
    while True:
        try:
            rendered = subprocess.run(build, cwd=ROOT, stdout=subprocess.PIPE, text=True)
            print(rendered.stdout, end="", flush=True)
            if rendered.returncode == 0 and rendered.stdout.strip() == "Complete":
                return
            if rendered.returncode:
                print(f"Renderer exited with status {rendered.returncode}. Retrying in 180 seconds.", flush=True)
        except OSError as error:
            print(error,flush=True)
        time.sleep(180)


if __name__=="__main__":
    main()
