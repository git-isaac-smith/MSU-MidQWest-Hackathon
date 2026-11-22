"""Checkpointed parallel ranges for the published distance-scaling comparisons."""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

from run import HERE, file_for, jobs_for, plan, read, run_job, save


def collect(job):
    """Merge scored ranges by seed index. Never count repeated shots twice."""
    output = file_for(job)
    current = read(output) if output.exists() else None
    records = {r["index"]: r for r in current["chunks"]} if current else {}
    before = len(records)
    for source in (HERE/"parts"/output.stem).glob("*.json"):
        part = read(source)
        if current is None:
            current = {**part,"chunks":[]}
        if part["job"] != list(job) or part["circuit_sha256"] != current["circuit_sha256"]:
            raise ValueError(f"Mixed experiment definitions in {source}")
        for r in part["chunks"]:
            existing = records.get(r["index"])
            if existing is not None and existing!=r:
                raise ValueError(f"Overlapping nonidentical records in {source}")
            records[r["index"]] = r
    if len(records)>before:
        current["chunks"] = [records[i] for i in sorted(records)]
        save(output,current)
    return set(records)


def work_items(jobs):
    by_job = []
    for index,job in enumerate(jobs):
        complete = collect(job)
        indices = [i for i,_ in jobs_for(job,0) if i not in complete]
        spans = []
        # Earlier monolithic checkpoints can end inside a block. Resume only missing indexes.
        if job[1] == 31:
            block = 40
        elif job[1] == 15:
            block = 180
        else:
            block = 25
        for i in indices:
            if spans and i == spans[-1][1] and spans[-1][1]-spans[-1][0]<block:
                spans[-1] = (spans[-1][0],i+1)
            else:
                spans.append((i,i+1))
        by_job.append([(index,a,b) for a,b in spans])
    # Rotate through conditions. Keep several independent ranges available per method.
    return [items[k] for k in range(max((len(x) for x in by_job),default=0))
            for items in by_job if k<len(items)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker",nargs=3,type=int,metavar=("JOB","FIRST","LAST"))
    ap.add_argument("--workers",type=int,default=10)
    ap.add_argument("--collect",action="store_true")
    args = ap.parse_args()
    jobs = plan("large")
    if args.worker:
        idx,first,last = args.worker
        run_job((jobs[idx],HERE,None,first,last))
        return
    if args.collect:
        for job in jobs:
            collect(job)
        return
    pending = work_items(jobs)
    active,retries = {},{}
    logs = HERE/"queue-logs"
    logs.mkdir(exist_ok=True)
    last_collect = 0
    while pending or active:
        for proc,(item,log) in list(active.items()):
            rc = proc.poll()
            if rc is None:
                continue
            log.close()
            del active[proc]
            if rc:
                retries[item] = retries.get(item,0)+1
                if retries[item]>3:
                    raise RuntimeError(f"Worker {item} failed four times; see {logs}.")
                pending.append(item)
        if time.monotonic()-last_collect>90:
            for job in jobs:
                collect(job)
            save(HERE/"queue-status.json",dict(updated=datetime.now(timezone.utc).isoformat(),
                 workers=[(p.pid,item) for p,(item,_) in active.items()],queued_ranges=len(pending)))
            last_collect=time.monotonic()
        # Existing foreground work can continue. Leave headroom for native graph compilation.
        limit = args.workers
        tuning = HERE/"worker-limit.json"
        if tuning.is_file():
            limit = min(args.workers,max(1,int(read(tuning)["workers"])))
        if pending and len(active)<limit and psutil.virtual_memory().available>2.6*1024**3:
            item = pending.pop(0)
            key = "-".join(map(str,item))
            log = (logs/f"{key}.log").open("ab")
            proc = subprocess.Popen([sys.executable,__file__,"--worker",*map(str,item)],
                                    stdout=log,stderr=subprocess.STDOUT)
            active[proc] = (item,log)
        time.sleep(2)
    for job in jobs:
        completed = collect(job)
        if completed != {i for i,_ in jobs_for(job,0)}:
            raise RuntimeError(f"Incomplete scaling job {job}")
    print("All scaling rows reached 1,000,000 scored shots.",flush=True)


if __name__=="__main__":
    main()
