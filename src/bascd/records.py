import csv
import json
import os
import tempfile
from contextlib import contextmanager
from itertools import product
from pathlib import Path

from .decoders import DISTANCES
from .measure import PROBABILITIES, Scenario, environment


def scenarios(distances=DISTANCES, probabilities=PROBABILITIES, bases=("x", "z")):
    return [Scenario(*values) for values in product(distances, probabilities, bases)]


def manifest(command, seed, shots, repeats):
    return {
        "format_version": 1, "command": command, "root_seed": seed, "shots_per_scenario": shots,
        "timing_repeats": repeats, "rounds": "distance", "environment": environment(),
        "noise": {"name": "simple_uniform", "status": "provisional", "generator": "Stim rotated memory",
                  "after_clifford_depolarization": "p", "after_reset_flip_probability": "p",
                  "before_measure_flip_probability": "p", "before_round_data_depolarization": "p"},
    }


@contextmanager
def atomic_text(path: Path):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as out:
        name = out.name
        try:
            yield out
            out.close()
            os.replace(name, path)
        finally:
            out.close()
            Path(name).unlink(missing_ok=True)


def save_json(path, value):
    with atomic_text(path) as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write("\n")


def save_csv(path, rows, fields=None):
    with atomic_text(path) as out:
        writer = csv.DictWriter(out, fieldnames=fields if fields is not None else list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})
