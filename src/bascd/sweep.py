from dataclasses import asdict

from .decoders import CHOICES
from .measure import measure, sample
from .records import manifest, save_csv, save_json
from .stats import paired_counts


def sweep(out, cases, seed, shots, repeats, latency_shots, command):
    out.mkdir(parents=True, exist_ok=False)
    protocol = manifest(command, seed, shots, repeats)
    protocol.update(scenarios=[asdict(c) for c in cases], split="sweep", latency_shots=latency_shots,
                    decoders=[s.to_dict() for s in CHOICES[:2]], decoder_order="alternates by scenario")
    save_json(out / "manifest.json", protocol)
    rows, comparisons = [], []
    for index, case in enumerate(cases):
        data = sample(case, seed, "sweep", shots)
        masks = {}
        order = CHOICES[:2] if index % 2 == 0 else CHOICES[:2][::-1]
        for position, spec in enumerate(order):
            row, masks[spec.decoder_type] = measure(data, spec, repeats, latency_shots)
            rows.append(dict(**row, timing_position=position + 1))
        counts = paired_counts(masks["mwpm"], masks["correlated_mwpm"])
        comparisons.append(dict(**data.metadata, **counts,
                                disagreements=counts["mwpm_only_failed"] + counts["correlated_only_failed"]))
    save_json(out / "benchmark.json", {"measurements": rows, "paired": comparisons})
    save_csv(out / "benchmark.csv", rows)
    save_csv(out / "paired.csv", comparisons)
    plot_sweep(out, rows)
    return rows, comparisons


def plot_sweep(out, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {3: "#185f92", 5: "#bd5728", 7: "#368060"}
    distances = sorted({r["distance"] for r in rows})
    bases = sorted({r["basis"] for r in rows})
    views = [
        ("logical-error-rate.png", [("ler", "Logical error rate (%)")]),
        ("inference-runtime.png", [("batch_us_per_shot", "Batch inference (µs/shot)"),
                                   ("single_p95_us", "Single-shot p95 (µs)")]),
    ]
    for filename, metrics in views:
        fig, axes = plt.subplots(len(metrics), len(bases), squeeze=False, figsize=(5.3 * len(bases), 3.6 * len(metrics) + 1))
        for col, basis in enumerate(bases):
            for idx, (metric, label) in enumerate(metrics):
                ax = axes[idx, col]
                for d in distances:
                    for kind, style, marker in (("mwpm", "-", "o"), ("correlated_mwpm", "--", "s")):
                        group = sorted((r for r in rows if (r["basis"], r["distance"], r["decoder"]) ==
                                        (basis, d, kind)), key=lambda r: r["p"])
                        prob = [r["p"] for r in group]
                        if metric == "ler":
                            # Plot zero observations and the full upper Wilson bound on the same axis.
                            y = [100 * r["ler"] for r in group]
                            errors = [[100 * (r["ler"] - r["wilson95_low"]) for r in group],
                                      [100 * (r["wilson95_high"] - r["ler"]) for r in group]]
                            ax.errorbar(prob, y, yerr=errors, capsize=3, color=colors[d],
                                        linestyle=style, marker=marker, markersize=4)
                        else:
                            ax.plot(prob, [r[metric] for r in group], color=colors[d], linestyle=style,
                                    marker=marker, markersize=4)
                ax.set(title=f"{basis.upper()} memory", xlabel="Physical error probability p", ylabel=label)
                ticks = sorted({r["p"] for r in rows})
                ax.set_xticks(ticks, [f"{p:g}" for p in ticks])
                if metric == "ler":
                    ax.set_yscale("symlog", linthresh=0.05)
                    ax.set_ylim(bottom=0)
                else:
                    ax.set_yscale("log")
                ax.grid(alpha=0.22)
        handles = [Line2D([], [], color=colors[d], label=f"d = {d}") for d in distances]
        handles += [Line2D([], [], color="#333333", ls=ls, marker=m, label=s) for ls, m, s in
                    (("-", "o", "MWPM"), ("--", "s", "Correlated MWPM"))]
        fig.legend(handles=handles, loc="upper center", ncol=min(5, len(handles)), frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        fig.savefig(out / filename, dpi=160, bbox_inches="tight")
        plt.close(fig)
