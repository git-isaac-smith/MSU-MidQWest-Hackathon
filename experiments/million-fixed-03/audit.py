"""Check denominators, fresh seeds, paired counts and public portable lookup replay."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import stim
from bascd.portable_lookup import PortableLookup
from bascd.circuits import memory_circuit
from bascd.bath_prior import load_bath_prior
from run import EXP, HERE, PROFILE, RECIPE, file_for, jobs_for, plan, read, save, seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete",action="store_true")
    parser.add_argument("--replay-lookup",action="store_true")
    args = parser.parse_args()
    grouped = {}
    paired_hashes = {}
    checks, paired_rows = [], []
    baseline_bits = {}
    for group in ("grid","fault","pooled","large"):
        total,finished = 0,0
        for job in plan(group):
            group,d,b,p,method,target=job
            path = file_for(job)
            if not path.exists():
                if args.require_complete:
                    raise ValueError(f"Missing {job}")
                continue
            state = read(path)
            assert list(job)==state["job"]
            planned = dict(jobs_for(job,0))
            seen = set()
            n = 0
            for row in state["chunks"]:
                i = row["index"]
                assert i not in seen and row["shots"]==planned[i] and row["seed"]==seed(group,d,b,p,i)
                seen.add(i)
                count = row["shots"]
                n+=count
                baseline = next(iter(row["methods"].values()))["wrong"]
                for result in row["methods"].values():
                    wrong,wins,losses = (result[k] for k in ("wrong","recovered","introduced"))
                    assert 0<=wins<=baseline<=count and 0<=losses<=count-baseline
                    assert baseline-wins+losses==wrong and result["ns"]>0
                    if group=="large":
                        bits=np.unpackbits(np.frombuffer(bytes.fromhex(result["flags"]),dtype=np.uint8),
                                           count=count,bitorder="little")
                        assert int(bits.sum())==wrong
                if group=="large":
                    key=(d,b,p,i)
                    assert paired_hashes.setdefault(key,row["sample_sha256"])==row["sample_sha256"]
                    mask = bytes.fromhex(row["methods"][method]["flags"])
                    if method=="mwpm":
                        baseline_bits[key]=mask
            assert 0<=n<=target
            if args.require_complete and n!=target:
                raise ValueError(f"Incomplete {job} with {n} shots")
            total+=n
            finished+=int(n==target)
        grouped[group]=dict(jobs_complete=finished,jobs_target=len(plan(group)),scored_job_shots=total)
    for job in plan("large"):
        _,d,b,p,method,_=job
        path=file_for(job)
        if not path.exists():
            continue
        n=wins=losses=0
        for r in read(path)["chunks"]:
            key=(d,b,p,r["index"])
            if key not in baseline_bits:
                continue
            assert paired_hashes[key]==r["sample_sha256"]
            ordinary=int.from_bytes(baseline_bits[key],"little")
            candidate=int.from_bytes(bytes.fromhex(r["methods"][method]["flags"]),"little")
            wins+=(ordinary & ~candidate).bit_count()
            losses+=(~ordinary & candidate).bit_count()
            n+=r["shots"]
        paired_rows.append(dict(distance=d,basis=b,p=p,method=method,paired_shots=n,
                                ordinary_only_failed=wins,candidate_only_failed=losses))
    if paired_rows:
        with (HERE/"large-paired.csv").open("w",newline="",encoding="utf-8") as out:
            writer=csv.DictWriter(out,fieldnames=list(paired_rows[0]))
            writer.writeheader()
            writer.writerows(paired_rows)
    clock = load_bath_prior(PROFILE,channel="clock").ratios
    assert np.allclose(clock,read(RECIPE)["recipe"]["bath_xyz"]["clock"],atol=1e-15,rtol=0)
    if args.replay_lookup:
        export = EXP/"circuit-frontier-01/portable-d3"
        table=PortableLookup.load(export)
        circuit=stim.Circuit.from_file(export/"circuit.stim")
        assert circuit==memory_circuit("z",3,3,.005)
        state=read(file_for(("grid",3,"z",.005,"together",1_000_000)))
        total=wrong=0
        for row in state["chunks"]:
            det,obs=circuit.compile_detector_sampler(seed=row["seed"]).sample(
                row["shots"],separate_observables=True,bit_packed=True)
            assert hashlib.sha256(det.tobytes()+obs.tobytes()).hexdigest()==row["sample_sha256"]
            errors=int((table.predict_batch(det)!=obs).sum())
            assert errors==row["methods"]["fast_hybrid"]["wrong"]
            total+=row["shots"]
            wrong+=errors
        checks.append(dict(test="Archived portable lookup versus new posterior implementation",shots=total,wrong=wrong))
    result=dict(groups=grouped,large_paired_hashes=len(paired_hashes),replays=checks)
    print(json.dumps(result,indent=2))
    save(HERE/"audit.json",result)


if __name__=="__main__":
    main()
