"""Write a replay manifest for all fixed-noise million-shot workloads."""
import hashlib
import importlib.metadata
import platform
from pathlib import Path

from run import HERE, EXP, GRID, PROFILE, RECIPE, TARGET, plan, save

root = EXP.parent
sources = [*sorted((root/"src/bascd").glob("*.py")), HERE/"run.py",HERE/"large_queue.py",
           EXP/"high-noise-categorization-01/fault_sources.py",EXP/"qubit-scaling-01/high-noise/run.py",
           PROFILE,RECIPE,EXP/"iteration-02/sweep/manifest.json",
           EXP/"circuit-frontier-01/protocol.json",EXP/"qubit-scaling-01/high-noise/measured/protocol.json"]
versions = {s:importlib.metadata.version(s) for s in ["numpy","stim","pymatching","stabilizer-qec","scipy","bascd"]}
save(HERE/"protocol.json",dict(
    format_version=1, python=platform.python_version(),platform=platform.platform(),cpu=platform.processor(),
    versions=versions, target=TARGET, sample_seed_namespace="midqwest-million-fixed-03",
    seed_formula="First 8 SHA-256 bytes little-endian of pipe-separated namespace, group, distance, lowercase basis, Python float probability, chunk index; append fault name for each fault component.",
    noise="Provisional simple_uniform. All four noise arguments use p, rotated memory, rounds equal distance.",
    pooling="One million fresh shared records per pooled decoder, balanced across 24 pre-existing scenarios. Other rows use one million records per physical circuit.",
    timing="Wall-clock time inside each batch decoder call. Includes argument conversion inside callable. Excludes circuit generation, Stim sampling, graph construction and scoring. Host shared by concurrent jobs.",
    checksum="SHA-256 of packed detector bytes concatenated with packed observable bytes, shot-major; bit order little.",
    physical_fault_sources="Independent Stim component records, combined by XOR; verify full joint parity signatures against native circuit before matched removal.",
    jobs={g:plan(g) for g in ("grid","pooled","fault","large")},
    file_sha256={path.relative_to(root).as_posix():hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}))
