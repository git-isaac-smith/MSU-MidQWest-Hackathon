# Batched erasure matching for the million-shot studies

The adapter calls Sakidja's pinned `ErasureDecoder` constructor for the ordered graph, edge weights, observable labels and herald-to-edge lists. A compiled wrapper uses PyMatching 2.4.0's sparse blossom implementation to score dense Stim detector records. Each shot still receives its own zero-weight heralded edges.

PyMatching rebuilds its native graph after a public edge replacement. The wrapper builds the native adjacency lists once per batch, retains the same neighbor order, and writes each shot's weights into those lists. It computes PyMatching's scale from the greatest surviving edge weight and uses the integral-weight branch whenever every surviving weight is integral. Syndrome insertion, flood notifications and blossom extraction follow the single-observable positive-weight driver. The adapter checks the assumptions on construction. It accepts dense shots for the graphlike, one-logical studies here. [PyMatching release source](https://pypi.org/project/PyMatching/2.4.0/) · [Author decoder](https://github.com/sakidja/RQL_Quantum_bath/blob/94572516042297d6b6018fe76eb2f2b547e3c46f/part1/dual_rail_structured_bath.py#L428-L484)

## Build and verify

From the repository root, use the original stress experiment's Python environment. The recorded Windows build used the Visual Studio x64 C++ toolset version 14.44.35207. The build script downloads and verifies the released PyMatching source tarball. Generated binaries and compiler files stay in its ignored `build/` directory. The checked Windows binary used in the studies had SHA-256 `6c8c6f0ab9b7176b55ab3410da4f34928fad61cadf27cdf8c611a6cc2a7d9d32`. Linker timestamps can change this digest between builds.

```powershell
python experiments/million-engine-03/build_engine.py
python experiments/million-engine-03/verify.py $upstream
python experiments/million-engine-03/verify_edge_cases.py $upstream
```

Six fresh batches contained 6,144 physical records and 768 artificial records across the two arms, five layouts and three backgrounds (0.1%, 1% and 4%). The artificial records used random syndromes and all, no, or random heralds. All predictions matched the original Python driver, including when the adapter processed the batches in reverse. A four-node fixture exhausted all 256 combinations of four heralds and four syndrome bits. This exercised changed weight scales, the transition to integral weights, overlapping heralds, missing boundary costs and ties. All predictions matched in both orders. The mixed-workload timings include detector-column selection and native matching under concurrent machine load. [Per-batch results](validation.json)
