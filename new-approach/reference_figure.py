"""Plot logical and physical error rates for the decoding comparison."""

from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter


def plot_percent_comparison(path, logical_rows, site_rows, five_rounds_only):
    """Save matching and FFT + CNN error rates on the same percentage scale."""
    panels = (
        ([*logical_rows, five_rounds_only], ("PyMatching (MWPM)", "FFT + CNN\n+ final matching",
                                            "FFT + CNN alone\n(five check rounds)"),
         "Logical error", "Error rate (% of shots)"),
        (site_rows, ("Site-tracking MWPM", "FFT + CNN"),
         "Qubit localization error", "Error rate (% of physical data sites across all shots)"),
    )
    fig = Figure(figsize=(12, 3.8), layout="constrained")
    axes = fig.subplots(1, 2, sharex=True)
    for ax, (rows, labels, title, xlabel) in zip(axes, panels):
        positions = list(reversed(range(len(rows))))
        rates = [row["error_percent"] for row in rows]
        ax.barh(positions, rates, height=0.55, color=["#3c5685", "#197365", "#447f75"][:len(rows)])
        ax.barh(positions, [100 - rate for rate in rates], left=rates,
                height=0.55, color="#e7edf3")
        for position, rate, row in zip(positions, rates, rows):
            rate_label = f"{rate:.3f}%" if rate == 0 or rate >= 0.001 else f"{rate:.3g}%"
            # Place the label in the correct-outcome segment when it has room.
            label_x = rate if rate <= 67 else 0
            light_text = rate > 67
            ax.annotate(
                rate_label, (label_x, position), xytext=(6, 2),
                textcoords="offset points", color="white" if light_text else "#142033", fontsize=11,
                weight="bold", va="bottom",
            )
            ax.annotate(
                f'{row["errors"]:,} / {row["total"]:,}', (label_x, position),
                xytext=(6, -2), textcoords="offset points",
                color="white" if light_text else "#435268",
                fontsize=9, va="top",
            )

        ax.set(yticks=positions, yticklabels=labels, xlim=(0, 100), ylim=(-0.5, len(rows) - 0.5),
               xticks=[0, 25, 50, 75, 100],
               xlabel=xlabel)
        ax.set_title(title, loc="left", fontsize=14, weight="bold", pad=12)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#e0e6ee", linewidth=0.8)
        ax.tick_params(axis="both", length=0, pad=6, colors="#435268")
        for name in ("top", "right", "left"):
            ax.spines[name].set_visible(False)
        ax.spines["bottom"].set_color("#c3ccd9")

    fig.savefig(path, dpi=170, facecolor="white")
