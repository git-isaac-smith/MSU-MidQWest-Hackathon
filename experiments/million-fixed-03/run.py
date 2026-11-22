"""Repeat the published fixed-circuit comparisons on fresh deterministic Stim records."""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from functools import partial
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

for var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[var] = "1"

import numpy as np
import stim
from bascd import circuit_frontier
from bascd.bath_prior import BathPrior, bath_prior_dem, load_bath_prior
from bascd.circuits import memory_circuit
from bascd.accuracy_decoders import compile_candidate, load_frozen_decoder
from bascd.decoders import DecoderSpec, compile_decoder
from bascd.stats import wilson95

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
RECIPE = EXP / "iteration-04/accuracy/frozen-policy.json"
PROFILE = EXP / "iteration-03/bath-calibration/bath-profile.json"
TARGET = 1_000_000
GRID = [(d,b,p) for d in (3,5,7) for b in ("x","z") for p in (.001,.003,.005,.01)]
HIGH = ("mwpm","correlated","expanded_corr","clock_corr","union_find","css_routed")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value,allow_nan=False), encoding="utf-8")
    for attempt in range(20):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.1)


def study_module(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def seed(*parts):
    raw = "|".join(map(str, ("midqwest-million-fixed-03",*parts)))
    return int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8],"little")


def unpacked(fn, width, records):
    return fn(np.unpackbits(records,axis=1,count=width,bitorder="little"))


def fixed_methods(d,b,p,c,kind):
    model = c.detector_error_model(decompose_errors=True)
    if kind == "pooled":
        recipe = read(RECIPE)["recipe"]
        params = dict(distance=d, basis=b, p=p, rounds=d)
        methods = {key:compile_candidate(c,params,recipe,key).decode_batch for key in ("native","bath_clock","belief")}
        methods["hybrid"] = load_frozen_decoder(RECIPE,c).decode_batch
        eq = bath_prior_dem(c,**params,prior=BathPrior("clock",(1/3,)*3),expand_isotropic=True)
        methods["equal"] = compile_decoder(eq,DecoderSpec("correlated_mwpm")).decode_batch
        return {k:partial(unpacked,fn,c.num_detectors) for k,fn in methods.items()}
    case = {}
    names = ["mwpm","correlated"]
    if b == "z" and p == .005 and d <= 7:
        stage = "primary" if d == 3 else "confirmation"
        case = read(EXP/f"circuit-frontier-01/cases/{stage}-d{d}-z-p0.005-ordinary/manifest.json")
        names.append("fast_hybrid" if d == 3 else "css_routed")
    return {n:circuit_frontier.compile_method(n,c,model,case) for n in names}


def plan(group):
    if group == "grid":
        return [(group,d,b,p,"together",TARGET) for d,b,p in GRID+[(7,b,p) for b in ("x","z") for p in (.015,.02,.04)]]
    if group == "pooled":
        return [(group,d,b,p,"together",TARGET//len(GRID)+(i<TARGET%len(GRID))) for i,(d,b,p) in enumerate(GRID)]
    if group == "fault":
        return [(group,7,"z",p,"together",TARGET) for p in (.005,.01,.02)]
    if group == "large":
        return [(group,d,b,p,method,TARGET)
                for d,p,bases,methods in [(d,.01,("z",),HIGH) for d in (7,15,31)]
                    + [(d,.005,("x","z"),HIGH[:4]) for d in (15,31)]
                for b in bases for method in methods]
    raise ValueError(group)


def file_for(job):
    group,d,b,p,method,_ = job
    return HERE/"checkpoints"/group/f"d{d}-{b}-p{p:g}-{method}.json"


def jobs_for(j,s):
    group,d,b,p,method,n = j
    # Fixed 512-shot seeds bound memory even for 29,760-detector d31 records.
    batch = 512 if group == "large" and d >= 15 else 10_000
    return [(i,min(batch,n-i*batch)) for i in range((n+batch-1)//batch)][s:]


def run_job(args):
    j,root,limit,*bounds = args
    group,d,b,p,method,n = j
    path = root/file_for(j).relative_to(HERE)
    if bounds:
        first,last = bounds
        path = root/"parts"/path.stem/f"{first:04d}-{last:04d}.json"
    state = read(path) if path.exists() else dict(job=list(j), chunks=[],versions=dict(
        stim=stim.__version__,numpy=np.__version__,pymatching=circuit_frontier.pymatching.__version__))
    assert state["job"] == list(j)
    done = {chunk["index"] for chunk in state["chunks"]}
    todo = [(i,count) for i,count in jobs_for(j,0) if i not in done
            and (not bounds or first <= i < last)]
    if not todo:
        return
    if limit:
        todo = todo[:limit]
    c = memory_circuit(b,d,d,p)
    state["circuit_sha256"] = hashlib.sha256(str(c).encode()).hexdigest()
    if group == "large":
        mod = study_module("high_noise",EXP/"qubit-scaling-01/high-noise/run.py")
        if p == .01:
            case = read(EXP/f"qubit-scaling-01/high-noise/measured/d{d}-z/case.json")
            assert case["circuit_sha256"] == state["circuit_sha256"]
            methods = {method:mod.make_decoder(c,case,method)}
        elif method in ("mwpm","correlated"):
            methods = {method:circuit_frontier.compile_method(method,c,c.detector_error_model(decompose_errors=True),{})}
        else:
            prior = BathPrior("clock",(1/3,)*3) if method=="expanded_corr" else load_bath_prior(PROFILE,channel="clock")
            model = bath_prior_dem(c,basis=b,distance=d,rounds=d,p=p,prior=prior,expand_isotropic=method=="expanded_corr")
            dec = compile_decoder(model,DecoderSpec("correlated_mwpm"))
            methods = {method:partial(unpacked,dec.decode_batch,c.num_detectors)}
    elif group == "fault":
        mod = study_module("fault_sources",EXP/"high-noise-categorization-01/fault_sources.py")
        dec = compile_decoder(c.detector_error_model(decompose_errors=True),DecoderSpec("correlated_mwpm"))
        circuits = {f:stim.Circuit.generated(f"surface_code:rotated_memory_{b}",distance=d,rounds=d,**{arg:p})
                    for f,arg in mod.FAMILIES.items()}
        assert all(cc.without_noise()==c.without_noise() for cc in circuits.values())
        native,composed = mod.signature_log_weights([c]),mod.signature_log_weights(circuits.values())
        assert native.keys()==composed.keys() and max(abs(native[k]-composed[k]) for k in native)<1e-12
        methods = {k:partial(unpacked,dec.decode_batch,c.num_detectors) for k in ["full",*circuits]}
    else:
        methods = fixed_methods(d,b,p,c,group)
    for i,count in todo:
        # All methods at a large-code point replay precisely the same sampler calls.
        s = seed(group,d,b,p,i)
        if group == "fault":
            effects = {k:cc.compile_detector_sampler(seed=seed(group,d,b,p,i,k)).sample(
                count,separate_observables=True,bit_packed=True) for k,cc in circuits.items()}
            det,obs = [np.bitwise_xor.reduce([e[k] for e in effects.values()]) for k in (0,1)]
        else:
            det,obs = c.compile_detector_sampler(seed=s).sample(count,separate_observables=True,bit_packed=True)
        h = hashlib.sha256(det.tobytes())
        h.update(obs.tobytes())
        r = dict(index=i,seed=s,shots=count,sample_sha256=h.hexdigest(),methods={})
        baseline = None
        for key,fn in methods.items():
            x,y = det,obs
            if group == "fault" and key!="full":
                x,y = det^effects[key][0],obs^effects[key][1]
            start = time.perf_counter_ns()
            pred = np.asarray(fn(x))
            ns = time.perf_counter_ns()-start
            if pred.shape!=y.shape or not np.all((pred==0)|(pred==1)):
                raise ValueError(f"{key} decoder output does not match observables")
            failed = np.any(pred!=y,axis=1)
            if baseline is None:
                baseline = failed
            r["methods"][key] = dict(wrong=int(failed.sum()),ns=ns,recovered=int((baseline & ~failed).sum()),
                                      introduced=int((~baseline & failed).sum()))
            if group == "large":
                r["methods"][key]["flags"] = np.packbits(failed,bitorder="little").tobytes().hex()
        state["chunks"].append(r)
        save(path,state)
    print(group,d,b,p,method,len(state["chunks"]),flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("group",choices=("grid","pooled","fault","large"))
    ap.add_argument("--workers",type=int,default=2)
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--job",type=int)
    args = ap.parse_args()
    root = HERE/("smoke" if args.smoke else "")
    jobs = plan(args.group)
    if args.job is not None:
        jobs = [jobs[args.job]]
    if args.smoke:
        jobs = [(*j[:-1],64) for j in jobs]
    args_list = [(j,root,None) for j in jobs]
    if args.workers==1:
        for item in args_list:
            run_job(item)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(run_job,args_list,chunksize=1))


if __name__ == "__main__":
    main()
