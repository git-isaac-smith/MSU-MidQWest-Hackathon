"""Render the completed fixed-noise headline comparisons on one page."""
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

from build_million_error_sheet import FIXED, PDF, TARGET, Sheet, fixed_rows


def main():
    rows = fixed_rows()
    grid = [r for r in rows if r["group"] == "grid" and r["basis"] == "Z" and r["probability"] == .005]
    order = {"mwpm": 0, "fast_hybrid": 1, "css_routed": 1, "correlated": 2}
    grid.sort(key=lambda r: (r["distance"], order[r["method"]]))
    policies = [r for r in rows if r["group"] == "pooled_total"]
    faults = {r["method"]: r for r in rows if r["group"] == "fault" and r["probability"] == .01}
    if (len(grid), len(policies), len(faults)) != (9, 6, 5):
        raise ValueError("Missing a planned fixed-noise comparison")
    for row in [*grid, *policies, *faults.values()]:
        if row["shots"] != TARGET or row["target_shots"] != TARGET or row["us_per_shot"] is None:
            raise ValueError(f"Incomplete fixed-noise row for {row['method']}")

    last_score = max(path.stat().st_mtime for group in ("grid", "pooled", "fault")
                     for path in (FIXED / "checkpoints" / group).glob("*.json"))
    stamp = datetime.fromtimestamp(last_score, timezone.utc).strftime("Fixed-noise records complete %Y-%m-%d %H:%M UTC")
    output = PDF.with_name("Decoder error rates - one million fixed-noise shots.pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, suffix=".tmp", delete=False) as temp:
        temporary = Path(temp.name)
    try:
        page = Sheet(temporary, stamp, "COMPLETE  /  FIXED-NOISE COMPARISONS",
                     footer="MidQWest  /  Surface-code memory")
        page.page("Fixed-circuit surface-code decoding", "Logical error and decoder runtime",
                  "One million scored shots per row. Each comparison uses the same Stim records for its decoders.")
        page.table("Z memory at 0.5% circuit noise", grid,
                   lambda r: f"d{r['distance']}    {r['method_label']}", "Distance / decoder")
        page.paragraph("Each circuit runs as many syndrome rounds as its code distance. A logical failure is a "
                       "wrong prediction of the stored logical observable. The circuit uses provisional simple_uniform noise.")
        page.table("Decoder policies across 24 circuit settings", policies, lambda r: r["method_label"], "Policy")
        page.paragraph("The pooled sample covers distances 3, 5 and 7, both X and Z memories, and 0.1%, 0.3%, "
                       "0.5% and 1% circuit noise. Each setting supplies 41,666 or 41,667 shared shots.")
        by_method = {r["method"]: r for r in policies}
        native, hybrid = by_method["native"], by_method["hybrid"]
        reduction = 100 * (native["wrong"] - hybrid["wrong"]) / native["wrong"]
        slowdown = hybrid["us_per_shot"] / native["us_per_shot"]
        page.paragraph(f"The distance-based hybrid made {native['wrong'] - hybrid['wrong']:,} fewer logical errors "
                       f"than native correlated matching ({reduction:.1f}% fewer) at {slowdown:.1f} times its batch decoding time.")
        page.paragraph(f"At distance 7 and 1% Z-memory noise, a separate paired fault-removal simulation gave "
                       f"{faults['full']['wrong']:,} logical failures with all faults and {faults['gate']['wrong']:,} "
                       "when gate faults were removed, out of one million shots each.")
        page.paragraph("LER is logical error rate. Intervals are Wilson 95% limits, approximate for the pooled mix. "
                       "Times measure batch inference on a shared CPU. Sampling and decoder setup are excluded.")
        page.c.save()
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(output)


if __name__ == "__main__":
    main()
