"""Paired scaling of circuit-level decoders on geometrically larger rotated memories."""

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from importlib.metadata import version

import numpy as np
import stim

from .circuits import memory_circuit
from .measure import Scenario
from .stats import wilson95

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "experiments/iteration-03/bath-calibration/bath-profile.json"
RECIPE = ROOT / "experiments/iteration-04/accuracy/frozen-policy.json"
P = .005
ROOT_SEED = 76241329
SHOTS = 10000
DISTANCES = [3, 5, 7, 15, 31]
LABELS = {
    "mwpm": "Ordinary MWPM", "correlated": "Correlated MWPM",
    "expanded_corr": "Equal-axis correlated MWPM", "clock_corr": "Bath clock · correlated MWPM",
    "selected_corr": "Bath selected-time prior · correlated MWPM",
    "expanded_ord": "Equal-axis ordinary MWPM", "clock_ord": "Bath clock · ordinary MWPM",
    "selected_ord": "Bath selected-time prior · ordinary MWPM", "weighted": "Temporal-weight MWPM",
    "union_find": "Weighted Union-Find", "union_find_handwritten": "Team Union-Find",
    "projected_ordinary": "Projected ordinary MWPM", "css_routed": "Activity routing",
    "capped_gap": "Capped logical-gap routing", "logical_gap": "Logical-gap routing",
    "exact_small": "Exact posterior", "portable": "Portable posterior lookup",
    "fast_hybrid": "Posterior cache + matching", "temporal_lowpass": "Temporal low-pass + matching",
    "local_peel": "Local-pair predecoder", "belief": "Belief matching",
    "bath_belief": "Bath-weighted belief matching", "ensemble3": "Three-member bath ensemble",
    "ensemble5": "Five-member bath ensemble", "ensemble3_jitter": "Jittered matching ensemble",
    "hypergraph_search": "Bounded hypergraph search", "search_belief": "Search with belief fallback",
    "dispute": "Belief / ensemble with search referee", "belief_pair": "Two belief decoders with search referee",
    "disagreement_rescue": "Matching disagreement rescue", "ambiguity_dispatch": "Coincidence routing",
    "adaptive_count": "Calibrated count routing", "adaptive_time": "Calibrated space-time routing",
    "learned": "Learned routing", "learned_rescue": "Learned routing with rescue",
}
HEAVY = {"belief", "bath_belief", "hypergraph_search", "search_belief", "belief_pair", "dispute"}
ENSEMBLES = {"ensemble3", "ensemble5", "ensemble3_jitter"}
PRIMARY = {"mwpm", "correlated", "expanded_corr", "clock_corr"}
CASE_FIELDS = ["distance", "basis", "p", "data_qubits", "physical_qubits", "rounds"]
METRIC_FIELDS = CASE_FIELDS + ["method", "shots", "failures", "ler_pct", "ci_low_pct", "ci_high_pct",
    "batch_us_per_shot", "setup_s", "batch_size", "recovered", "lost", "status", "reason", "planned_shots"]


def planned_shots(name, distance):
    if name in HEAVY:
        return {3: 4096, 5: 4096, 7: 2048, 15: 128, 31: 16}[distance]
    if distance == 31 and name not in PRIMARY:
        return 128 if name in ENSEMBLES | {"temporal_lowpass", "local_peel", "disagreement_rescue",
                                           "learned_rescue", "union_find_handwritten"} else 1024
    if distance == 15 and name in ENSEMBLES | {"temporal_lowpass", "local_peel", "union_find_handwritten"}:
        return 2048
    return SHOTS


def save(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    # On Windows a brief read by the supervisor or OneDrive can hold the old file.
    for attempt in range(20):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.05)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def prepare(out):
    if (out / "protocol.json").exists():
        raise FileExistsError("Choose an output directory without a scaling protocol")
    out.mkdir(parents=True, exist_ok=True)
    cases = []
    for distance in DISTANCES:
        for basis in ("z", "x"):
            folder = out / f"d{distance}-{basis}"
            folder.mkdir()
            c = memory_circuit(basis, distance, distance, P)
            total = len(c.get_final_qubit_coordinates())
            if total != 2 * distance**2 - 1:
                raise ValueError("Unexpected count of active circuit qubits")
            c.to_file(folder / "circuit.stim")
            case = Scenario(distance, P, basis)
            seed = case.seed(ROOT_SEED, "final_test")
            det, obs = c.compile_detector_sampler(seed=seed).sample(SHOTS, separate_observables=True, bit_packed=True)
            np.savez_compressed(folder / "records.npz", detectors=det, truth=obs)
            meta = dict(distance=distance, basis=basis, p=P, rounds=distance, data_qubits=distance**2,
                        physical_qubits=total, detectors=c.num_detectors, seed=seed,
                        training_seed=case.seed(ROOT_SEED, "train"), folder=folder.name,
                        circuit_sha256=digest(str(c).encode()), sample_sha256=digest(det.tobytes()+obs.tobytes()))
            save(folder / "case.json", meta)
            cases.append(meta)
    save(out / "protocol.json", dict(description="Paired fixed-circuit rotated memory at increasing active qubit counts",
         distances=DISTANCES, bases=["x","z"], p=P, rounds="distance", seed=ROOT_SEED, shots_target=SHOTS,
         noise_model="simple_uniform (provisional)", cases=cases, method_labels=LABELS,
         versions={p: version(p) for p in ("numpy","stim","pymatching","stabilizer-qec","psutil","numba","llvmlite")},
         bath_profile_sha256=digest(PROFILE.read_bytes()), recipe_sha256=digest(RECIPE.read_bytes()),
         planned_prefixes={name:[planned_shots(name,d) for d in DISTANCES] for name in LABELS},
         resource_limits="One sequential decoder at a time. A worker is stopped above 2.5 GiB RSS or after 180 s per method (480 s for the four full-budget distance-31 references).",
         timing="Median wall time per shot over full-sized, warmed inference calls on unpacked boolean detector input. Standard batch 128; fixed 16-shot pilots use batch 16. File IO, sampling, evaluating predictions and setup excluded. Required input packing included.",
         calibration="Learned routing loads a matching archived Z policy at distances 5/7 and otherwise trains on 4096 independent records (1024 at d31). Count and space-time routing train on the same independent calibration prefix at cap 50%. Single-thread belief and search; ordinary weighted matching uses temporal scale 0.75 and boundary scale 1."))
    write_coverage(out)


def write_coverage(out):
    coverage = [dict(method=key, scope="Fixed-circuit study", reason=label) for key,label in LABELS.items()]
    coverage.extend([
        dict(method="FFT / CNN / five-round posterior", scope="Separate model", reason="See the five-round FFT study. These decoders use raw data/readout fault histories and a different noise model."),
        dict(method="Location-aware bath erasure", scope="Separate model", reason="See the bath erasure scaling study. Compares public site flags hidden and visible on the same shots in an ideal single-opportunity Z memory. The fixed circuit supplies no flags."),
        dict(method="Iteration 04 distance policy", scope="Constituent methods tested", reason="Exact posterior at d3 and belief/ensemble dispute at d5/7. Larger dispute rows recompile the same recipe for the new circuit."),
        dict(method="Random / AI configuration search", scope="Offline selection", reason="Search proposes decoder settings; it has no separate inference algorithm. Weighted MWPM is included."),
        dict(method="Single-record / worker-pool / unique-row wrappers", scope="Alternative execution modes", reason="This page times serial batch inference. These wrappers change scheduling, caching or call format."),
        dict(method="Spatial bursts and spectral diagnostics", scope="Noise model / analysis", reason="These generate or characterize different shots; they do not predict the logical observable."),
    ])
    write_csv(out / "coverage.csv", coverage, ["method","scope","reason"])


def make_decoder(name, circuit, meta, folder):
    from .accuracy_decoders import compile_candidate
    from .adaptive import AdaptiveDecoder, SyndromeFeatures, fit_policy
    from .bath_prior import BathPrior, bath_prior_dem, load_bath_prior
    from .decoders import DecoderSpec, compile_decoder
    from .exact_small import compile_exact_small
    from .fast_hybrid import FastHybridDecoder
    from .learned_fault_router import LearnedFaultRouter
    from .portable_lookup import PortableLookup
    from . import scale_methods
    d, basis, p = (meta[k] for k in ("distance","basis","p"))
    params = dict(distance=d, rounds=d, basis=basis, p=p)
    if name in scale_methods.METHODS:
        return scale_methods.compile_method(name,circuit,d,basis,p,calibration_seed=meta["training_seed"]).decode_batch
    if name in {"exact_small","portable"}:
        exact = compile_exact_small(circuit.detector_error_model(decompose_errors=False))
        if name == "exact_small":
            return exact.decode_batch
        lut = PortableLookup.from_exact(exact)
        return lambda x: lut.predict_batch(np.packbits(x, axis=1, bitorder="little"))
    model = circuit.detector_error_model(decompose_errors=True)
    if name in {"mwpm","correlated","weighted"}:
        spec = DecoderSpec("weighted_mwpm", .75, 1) if name == "weighted" else DecoderSpec(
            "mwpm" if name == "mwpm" else "correlated_mwpm")
        return compile_decoder(model,spec).decode_batch
    if name.endswith(("_ord","_corr")):
        channel, kind = name.split("_")
        expand = channel == "expanded"
        prior = BathPrior("clock",(1/3,)*3) if expand else load_bath_prior(PROFILE,channel=channel)
        dem = bath_prior_dem(circuit, **params, prior=prior, expand_isotropic=expand)
        return compile_decoder(dem,DecoderSpec("mwpm" if kind=="ord" else "correlated_mwpm")).decode_batch
    if name == "fast_hybrid":
        options = dict(correlated_fallback=True,correlated_min_detectors=21) if d == 5 else {}
        return FastHybridDecoder(model, **options).decode_batch
    n = 1024 if d == 31 else 4096
    if name.startswith("adaptive_"):
        det,truth = circuit.compile_detector_sampler(seed=meta["training_seed"]).sample(n,separate_observables=True)
        fails=[np.any(compile_decoder(model,DecoderSpec(s)).decode_batch(det)!=truth,axis=1)
               for s in ("mwpm","correlated_mwpm")]
        feature = "count" if name == "adaptive_count" else "spacetime"
        extractor = SyndromeFeatures(model,feature)
        policy = fit_policy(model,extractor,extractor.buckets(det),*fails,.50,feature)
        return AdaptiveDecoder(model,policy).decode_batch
    if name in {"learned","learned_rescue"}:
        from .compute_campaign import SpecialistCascade
        archived = ROOT / f"experiments/compute-frontier-02/d{d}/policy.npz"
        local = folder / "trained-policy.npz"
        if basis == "z" and d in (5,7):
            policy = LearnedFaultRouter.load(archived)
        elif local.exists():
            policy = LearnedFaultRouter.load(local)
        else:
            policy = LearnedFaultRouter.fit_sampled(circuit,model,seeds=[meta["training_seed"]],shots_per_seed=n)
            policy.save(local)
        if policy.model_digest != digest(str(model).encode()):
            raise ValueError("Learned policy belongs to a different circuit")
        dec = SpecialistCascade(circuit,model,policy,rescue=name=="learned_rescue")
        return lambda x: dec.decode_batch(np.packbits(x,axis=1,bitorder="little"))
    recipe = load(RECIPE)["recipe"]
    recipe["belief"]["threads"] = 1
    recipe["search"]["threads"] = 1
    return compile_candidate(circuit, params, recipe, name).decode_batch


def evaluate(folder, name, circuit, meta, packed, *, followup_shots=None):
    d = meta["distance"]
    target = followup_shots or planned_shots(name,d)
    if name in {"exact_small","portable"} and circuit.num_detectors > 24:
        return dict(status="unsupported",reason=f"Exact table supports 24 detector bits; this circuit has {circuit.num_detectors}.")
    if name == "fast_hybrid" and d == 31:
        # The implementation forms a dense array for every pair among 1,400 mechanisms.
        pair_bytes = (1400 * 1399 // 2) * packed.shape[1]
        if pair_bytes > 2.5 * 1024**3:
            return dict(status="resource_limit",reason=f"One dense fault-pair array alone needs {pair_bytes/1024**3:.2f} GiB; worker limit 2.5 GiB.")
    t0=time.perf_counter()
    decode=make_decoder(name,circuit,meta,folder)
    warm = circuit.compile_detector_sampler(seed=meta["training_seed"] ^ 0x1D37).sample(3)
    decode(warm)
    setup=time.perf_counter()-t0
    batch=min(128,target)
    predictions=[]
    times=[]
    for start in range(0,target,batch):
        raw=packed[start:min(start+batch,target)]
        bits=np.unpackbits(raw,axis=1,count=circuit.num_detectors,bitorder="little").view(bool)
        begin=time.perf_counter_ns()
        pred=decode(bits)
        elapsed=time.perf_counter_ns()-begin
        if pred.shape!=(len(bits),1) or not np.all((pred==0)|(pred==1)):
            raise ValueError("Expected one binary observable prediction per shot")
        predictions.append(np.asarray(pred[:,0],dtype=np.uint8).copy())
        if len(bits)==batch:
            times.append(elapsed)
    result=np.concatenate(predictions)
    destination = folder if followup_shots is None else folder/"short-probes"
    destination.mkdir(exist_ok=True)
    np.save(destination / f"{name}.npy",result,allow_pickle=False)
    reason = "Fixed-size follow-up after a time limit" if followup_shots else "Prespecified paired prefix"
    return dict(status="ok" if target==SHOTS else "pilot", reason="" if target==SHOTS else reason,
                setup_s=setup,shots=target,batch_size=batch,batch_ns=times,
                batch_us_per_shot=float(np.median(times))/batch/1000,prediction_sha256=digest(result.tobytes()))


def worker(folder, methods):
    meta=load(folder/"case.json")
    circuit=stim.Circuit.from_file(folder/"circuit.stim")
    with np.load(folder/"records.npz") as f:
        packed=f["detectors"]
    for name in methods:
        marker=folder/f"{name}.json"
        if marker.exists():
            continue
        save(folder/"active.json",dict(method=name))
        try:
            row=evaluate(folder,name,circuit,meta,packed)
        except Exception as error:
            row=dict(status="error",reason=f"{type(error).__name__}: {error}")
        save(marker,row)
        print(f"{folder.name} {name} {row['status']} {row.get('reason','')}",flush=True)
        gc.collect()


def short_probes(out):
    """Try a fixed 16-shot prefix after a measured wall-time limit, retaining both results."""
    for meta in load(out/"protocol.json")["cases"]:
        folder=out/meta["folder"]
        for name in LABELS:
            record=load(folder/f"{name}.json")
            if (record["status"]!="resource_limit" or " s including setup and scoring" not in record["reason"]
                    or planned_shots(name, meta["distance"]) <= 16):
                continue
            limit_record=folder/"short-probes"/f"{name}.json"
            if limit_record.exists():
                continue
            circuit=stim.Circuit.from_file(folder/"circuit.stim")
            with np.load(folder/"records.npz") as data:
                packed,truth=data["detectors"],data["truth"][:16,0]
            row=evaluate(folder,name,circuit,meta,packed,followup_shots=16)
            pred=np.load(folder/"short-probes"/f"{name}.npy")
            reference=np.load(folder/"mwpm.npy")[:16]
            wrong,base_wrong=pred!=truth,reference!=truth
            row.update(original_attempt=record, fail_count=int(wrong.sum()),
                       recovered=int((base_wrong&~wrong).sum()),lost=int((wrong&~base_wrong).sum()))
            save(limit_record,row)
            print(f"{folder.name} {name} probe: {row['fail_count']}/16, {row['batch_us_per_shot']:.1f} us/shot",flush=True)


def supervise(folder,methods):
    import psutil
    env=dict(os.environ,OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",OMP_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
    while todo:=[m for m in methods if not (folder/f"{m}.json").exists()]:
        (folder/"active.json").unlink(missing_ok=True)
        cmd=[sys.executable,"-m","bascd.scale_study","worker","--out",str(folder),"--methods",*todo]
        with subprocess.Popen(cmd,env=env) as proc:
            process=psutil.Process(proc.pid)
            active=None
            start=time.monotonic()
            reason=""
            while proc.poll() is None:
                marker=folder/"active.json"
                if marker.exists():
                    method=load(marker)["method"]
                    if method!=active:
                        active=method
                        start=time.monotonic()
                limit=480 if folder.name.startswith("d31-") and active in PRIMARY else 180
                try:
                    # A Windows venv launcher can delegate to a separate Python process.
                    descendants=process.children(recursive=True)
                    resident=sum(p.memory_info().rss for p in [process,*descendants] if p.is_running())
                    if resident > 2.5*1024**3:
                        reason="Worker exceeded 2.5 GiB resident memory"
                    if time.monotonic()-start>limit:
                        reason=f"Worker exceeded {limit} s including setup and scoring"
                except psutil.NoSuchProcess:
                    break
                if reason:
                    for child in reversed(descendants):
                        try:
                            child.kill()
                        except psutil.NoSuchProcess:
                            pass
                    proc.kill()
                    break
                time.sleep(.20)
            code=proc.wait()
            if code!=0:
                active=load(folder/"active.json")["method"] if (folder/"active.json").exists() else todo[0]
                marker=folder/f"{active}.json"
                if not marker.exists():
                    save(marker,dict(status="resource_limit" if reason else "error",
                                     reason=reason or f"Worker exited with code {code}"))
                    print(f"{folder.name} {active}: {reason or code}",flush=True)
                elif not reason:
                    raise RuntimeError(f"Worker failed between methods at {folder}: {code}")
    (folder/"active.json").unlink(missing_ok=True)


def collect(out):
    rows=[]
    protocol=load(out/"protocol.json")
    total=protocol.get("shots_target", SHOTS)
    for meta in protocol["cases"]:
        folder=out/meta["folder"]
        with np.load(folder/"records.npz") as data:
            packed, observable=data["detectors"],data["truth"]
        if (observable.shape != (total,1) or packed.shape != (total,(meta["detectors"]+7)//8)
                or np.any(observable>1) or digest(packed.tobytes()+observable.tobytes())!=meta["sample_sha256"]):
            raise ValueError(f"Sample shape or fingerprint mismatch at {folder}")
        truth=observable[:,0]
        baseline=folder/"mwpm.npy"
        ref=np.load(baseline) if baseline.exists() else None
        for name in LABELS:
            path=folder/f"{name}.json"
            if not path.exists():
                continue
            row=load(path)
            probe=folder/"short-probes"/f"{name}.json"
            if row["status"]=="resource_limit" and probe.exists():
                original=row["reason"]
                row=load(probe)
                row["reason"]=f"Follow-up on 16 shots after the original attempt: {original}"
            n=row.get("shots",0)
            item={k:meta[k] for k in CASE_FIELDS}
            item.update(row,method=name,shots=n,planned_shots=planned_shots(name,meta["distance"]))
            if n:
                source=probe.parent if probe.exists() and "original_attempt" in row else folder
                pred=np.load(source/f"{name}.npy",allow_pickle=False)
                if pred.shape!=(n,) or digest(pred.tobytes())!=row["prediction_sha256"] or ref is None or len(ref)<n:
                    raise ValueError(f"Invalid predictions or missing paired baseline at {folder}/{name}")
                errors=pred!=truth[:n]
                base=ref[:n]!=truth[:n]
                count=int(errors.sum())
                lo,hi=wilson95(count,n)
                item.update(failures=count,ler_pct=100*count/n,ci_low_pct=100*lo,ci_high_pct=100*hi,
                            recovered=int((base&~errors).sum()),lost=int((errors&~base).sum()))
            rows.append(item)
    write_csv(out/"metrics.csv",rows,METRIC_FIELDS)
    package=Path(__file__).parent
    files=sorted(package.glob("*.py"))+[PROFILE,RECIPE]
    archived=[ROOT/f"experiments/compute-frontier-02/d{d}/policy.npz" for d in (5,7)]
    save(out/"implementation.json",dict(
        python=platform.python_version(), platform=platform.platform(), processor=platform.processor(),
        versions={p:version(p) for p in ("numpy","stim","pymatching","stabilizer-qec","numba","llvmlite","psutil")},
        project_sha256={p.relative_to(ROOT).as_posix():digest(p.read_bytes()) for p in files+archived},
        fitted_policy_sha256={p.relative_to(out).as_posix():digest(p.read_bytes()) for p in out.glob("d*-*/trained-policy.npz")},
    ))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["prepare","run","worker","collect","short-probes"])
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--methods",nargs="+",choices=LABELS)
    parser.add_argument("--cases",nargs="+")
    args=parser.parse_args()
    if args.action=="prepare":
        prepare(args.out)
    elif args.action=="worker":
        worker(args.out,args.methods or list(LABELS))
    elif args.action=="collect":
        collect(args.out)
    elif args.action=="short-probes":
        short_probes(args.out)
        collect(args.out)
    else:
        for meta in load(args.out/"protocol.json")["cases"]:
            if not args.cases or meta["folder"] in args.cases:
                supervise(args.out/meta["folder"],args.methods or list(LABELS))
        collect(args.out)


if __name__=="__main__":
    main()
