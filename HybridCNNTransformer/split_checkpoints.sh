#!/usr/bin/env bash
# For every immediate subfolder X of DIR:
#   1. rename X to X_1
#   2. create a new, empty X
#   3. move everything except *.pt files from X_1 into X (subfolders are kept)
# The *.pt files stay in X_1.
#
# Subfolders whose name already ends in _1, or whose X_1 already exists, are skipped,
# so a second run does not nest folders. Existing files are never overwritten.
#
#   ./split_checkpoints.sh DIR
#   ./split_checkpoints.sh --dry-run DIR     # print what would happen, change nothing
set -euo pipefail

dry=0
if [[ "${1:-}" == "-n" || "${1:-}" == "--dry-run" ]]; then
    dry=1
    shift
fi
root="${1:-}"
if [[ -z "$root" || ! -d "$root" ]]; then
    echo "usage: $0 [--dry-run] DIR" >&2
    exit 1
fi

run() {
    if (( dry )); then echo "+ $*"; else "$@"; fi
}

for dir in "$root"/*/; do
    [[ -d "$dir" ]] || continue
    dir="${dir%/}"
    name="${dir##*/}"
    old="${dir}_1"

    if [[ "$name" == *_1 ]]; then
        echo "skip $name (name ends in _1)"
        continue
    fi
    if [[ -e "$old" ]]; then
        echo "skip $name (${name}_1 already exists)"
        continue
    fi

    echo "$name -> ${name}_1"
    run mv -- "$dir" "$old"
    run mkdir -- "$dir"

    src="$old"
    (( dry )) && src="$dir"      # nothing was renamed in a dry run
    while IFS= read -r -d '' f; do
        rel="${f#"$src"/}"
        from="$f"
        (( dry )) && from="$old/$rel"    # show the path the file will have after the rename
        run mkdir -p -- "$dir/$(dirname -- "$rel")"
        run mv -n -- "$from" "$dir/$rel"
    done < <(find "$src" -type f ! -name '*.pt' -print0)
done
