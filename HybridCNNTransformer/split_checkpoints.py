#!/usr/bin/env python
"""For every immediate subfolder X of DIR:
  1. rename X to X_1
  2. create a new, empty X
  3. move everything except *.pt files from X_1 into X (subfolders are kept)
The *.pt files stay in X_1.

Subfolders whose name already ends in _1, or whose X_1 already exists, are skipped, so a second run does
not nest folders. Existing files are never overwritten.

    python split_checkpoints.py DIR
    python split_checkpoints.py DIR --dry-run     # print what would happen, change nothing
"""
import argparse
import shutil
import sys
from pathlib import Path


def split_dir(root: Path, dry_run: bool = False) -> int:
    """Process every subfolder of `root`. Returns the number of folders split."""
    done = 0
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        old = folder.with_name(folder.name + "_1")
        if folder.name.endswith("_1"):
            print(f"skip {folder.name} (name ends in _1)")
            continue
        if old.exists():
            print(f"skip {folder.name} ({old.name} already exists)")
            continue

        print(f"{folder.name} -> {old.name}")
        files = [f for f in folder.rglob("*") if f.is_file() and f.suffix != ".pt"]
        rels = sorted(f.relative_to(folder) for f in files)
        if dry_run:
            for rel in rels:
                print(f"  move {old.name}/{rel.as_posix()} -> {folder.name}/{rel.as_posix()}")
            done += 1
            continue

        folder.rename(old)
        folder.mkdir()
        for rel in rels:
            dest = folder / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                print(f"  keep {old.name}/{rel.as_posix()} ({dest} exists)")
                continue
            shutil.move(str(old / rel), str(dest))
        done += 1
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, help="directory whose subfolders are split")
    ap.add_argument("-n", "--dry-run", action="store_true", help="print what would happen, change nothing")
    args = ap.parse_args(argv)
    if not args.dir.is_dir():
        sys.exit(f"split_checkpoints.py: {args.dir} is not a directory")
    n = split_dir(args.dir, args.dry_run)
    print(f"{n} folder(s) {'would be ' if args.dry_run else ''}split")


if __name__ == "__main__":
    main()
