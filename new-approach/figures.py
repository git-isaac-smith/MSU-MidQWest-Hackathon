"""Diagnostic plots for the time-weighted Fourier syndrome experiment."""

from pathlib import Path

import numpy as np
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def _log_amplitude(spectrum):
    return np.log1p(np.abs(np.fft.fftshift(spectrum, axes=(-2, -1))))


def _frequency_extent(height, width):
    x = np.fft.fftshift(np.fft.fftfreq(width))
    y = np.fft.fftshift(np.fft.fftfreq(height))
    return (x[0] - 0.5 / width, x[-1] + 0.5 / width,
            y[0] - 0.5 / height, y[-1] + 0.5 / height)


def _frequency_panel(ax, amplitude, extent, norm):
    return ax.imshow(amplitude, extent=extent, origin="lower",
                     interpolation="nearest", cmap="magma", norm=norm)


def _save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white")


def _plot_frame_spectra(outdir, observed, frame_spectra, extent, shot_index, check_mask):
    amplitudes = _log_amplitude(frame_spectra)
    norm = Normalize(0, amplitudes.max() or 1.0)
    labels = ["Quiet check (0)", "Flipped check (1)"]
    colors = ["#b9d3e5", "#21446e"]
    checks = np.asarray(observed, dtype=bool).astype(np.uint8)
    if check_mask is not None:
        if np.shape(check_mask) != checks.shape[-2:]:
            raise ValueError("Check mask must match the measured check grid")
        labels.append("Inactive site")
        colors.append("#eef0f2")
        checks = np.where(check_mask, checks, 2)
    cmap = ListedColormap(colors)

    fig = Figure(figsize=(12.2, 5.8), layout="constrained")
    grid = fig.add_gridspec(3, len(checks) + 1, height_ratios=[1, 0.14, 1],
                            width_ratios=[1] * len(checks) + [0.09])
    for step, (frame, amplitude) in enumerate(zip(checks, amplitudes)):
        field_ax = fig.add_subplot(grid[0, step])
        field_ax.imshow(frame, interpolation="nearest", cmap=cmap,
                        vmin=-0.5, vmax=len(colors) - 0.5, origin="upper")
        field_ax.set_title(f"Round {step + 1} · {np.count_nonzero(frame == 1)} flipped", fontsize=10)
        field_ax.set_xticks(np.arange(frame.shape[1] + 1) - 0.5, minor=True)
        field_ax.set_yticks(np.arange(frame.shape[0] + 1) - 0.5, minor=True)
        field_ax.grid(which="minor", color="white", linewidth=0.7)
        field_ax.set(xticks=[], yticks=[])
        field_ax.tick_params(which="minor", length=0)
        for spine in field_ax.spines.values():
            spine.set_color("#98a4af")

        spectrum_ax = fig.add_subplot(grid[2, step])
        image = _frequency_panel(spectrum_ax, amplitude, extent, norm)
        if step == 0:
            field_ax.set_ylabel("Measured Z checks")
            spectrum_ax.set_ylabel("Vertical frequency\n(cycles per grid cell)")
        else:
            spectrum_ax.tick_params(labelleft=False)

    legend_ax = fig.add_subplot(grid[1, :-1])
    legend_ax.set_axis_off()
    legend_ax.legend(handles=[Patch(facecolor=color, label=label)
                              for color, label in zip(colors, labels)],
                     loc="center", frameon=False, ncol=len(labels), fontsize=9)
    color_ax = fig.add_subplot(grid[2, -1])
    fig.colorbar(image, cax=color_ax, label="ln(1 + amplitude)")
    fig.supxlabel("Horizontal frequency (cycles per grid cell)", fontsize=11)
    fig.suptitle(f"Measured Z checks and spatial spectra · shot {shot_index}", fontsize=15)
    _save(fig, outdir / "frequency_by_time_step.png")


def _plot_stack(outdir, stacked, filtered_stacked, weights, extent, shot_index):
    amplitudes = _log_amplitude(np.stack([stacked, filtered_stacked]))
    norm = Normalize(0, amplitudes.max() or 1.0)
    fig = Figure(figsize=(8.0, 4.2), layout="constrained")
    axes = fig.subplots(1, 2, sharex=True, sharey=True)
    for ax, amplitude, title in zip(axes, amplitudes, ("Before filtering", "After filtering")):
        image = _frequency_panel(ax, amplitude, extent, norm)
        ax.set_title(title, fontsize=11)
    axes[0].set_ylabel("Vertical frequency (cycles per grid cell)")
    fig.supxlabel("Horizontal frequency (cycles per grid cell)", fontsize=11)
    fig.colorbar(image, ax=axes, shrink=0.86, label="ln(1 + amplitude)")
    weight_text = ", ".join(f"{weight:.3g}" for weight in weights)
    fig.suptitle(f"Time-weighted spatial spectrum · shot {shot_index}\n"
                 f"Step weights [{weight_text}]", fontsize=13)
    _save(fig, outdir / "frequency_stack_filter_comparison.png")


_QUIET = ("No error", "o", "#dce4ee", "#688098", "#182e43")
_ACTUAL = ("Actual error", "o", "#345785", "#345785", "white")
_MATCH = ("Correct selection", "o", "#16675e", "#16675e", "white")
_FALSE = ("False positive", "s", "#a43d27", "#a43d27", "white")
_MISSED = ("Missed error", "D", "white", "#763696", "#763696")


def _plot_lattice_panel(ax, data_ids, categories, title):
    height, width = data_ids.shape
    ax.set(xlim=(-0.6, width - 0.4), ylim=(height - 0.4, -0.6),
           xticks=np.arange(width), yticks=np.arange(height), title=title)
    ax.set_aspect("equal")
    ax.set_axisbelow(True)
    ax.grid(color="#dfe6ee", linewidth=0.65)
    ax.tick_params(length=0, labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    label_ids = max(height, width) <= 9
    marker_size = min(290, (188 / max(height, width)) ** 2)
    for mask, (_, marker, fill, edge, ink) in categories:
        y, x = np.nonzero(mask)
        ax.scatter(x, y, s=marker_size, marker=marker, c=fill, edgecolors=edge,
                   linewidths=1.15, zorder=2)
        if label_ids:
            for row, col in zip(y, x):
                ax.text(col, row, str(data_ids[row, col]), color=ink, ha="center",
                        va="center", fontsize=7, zorder=3)


def _legend_handle(style):
    label, marker, fill, edge, _ = style
    return Line2D([], [], linestyle="none", marker=marker, markersize=7,
                  markerfacecolor=fill, markeredgecolor=edge, label=label)


def _plot_qubit_selection(outdir, data_ids, target, probabilities, threshold, shot_index):
    target = np.asarray(target, dtype=bool)
    chosen = probabilities >= threshold
    correct = target & chosen
    false = ~target & chosen
    missed = target & ~chosen
    fig = Figure(figsize=(8.7, 5.0), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 0.12])
    actual_ax = fig.add_subplot(grid[0, 0])
    chosen_ax = fig.add_subplot(grid[0, 1])
    legend_ax = fig.add_subplot(grid[1, :])
    _plot_lattice_panel(actual_ax, data_ids, [(~target, _QUIET), (target, _ACTUAL)],
                        f"Actual errors · {target.sum()} sites")
    _plot_lattice_panel(chosen_ax, data_ids,
                        [(~target & ~chosen, _QUIET), (correct, _MATCH),
                         (false, _FALSE), (missed, _MISSED)],
                        f"CNN selections · {chosen.sum()} sites")
    legend_ax.set_axis_off()
    styles = (_QUIET, _ACTUAL, _MATCH, _FALSE, _MISSED)
    legend_ax.legend(handles=[_legend_handle(style) for style in styles],
                     loc="center", frameon=False, ncol=len(styles), fontsize=9)
    fig.suptitle(f"Data-qubit errors and CNN selections · shot {shot_index}\n"
                 f"Selection threshold p ≥ {threshold:.3g}", fontsize=14)
    _save(fig, outdir / "qubit_selection_example.png")


def plot_example(outdir, observed, frame_spectra, stacked, filtered_stacked, weights,
                 data_ids, target, probabilities, threshold, shot_index, check_mask=None):
    """Save frequency and qubit-selection figures for one held-out shot."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    frame_extent = _frequency_extent(*frame_spectra.shape[-2:])
    stack_extent = _frequency_extent(*stacked.shape)
    _plot_frame_spectra(outdir, observed, frame_spectra, frame_extent, shot_index, check_mask)
    _plot_stack(outdir, stacked, filtered_stacked, weights, stack_extent, shot_index)
    _plot_qubit_selection(outdir, data_ids, target, probabilities, threshold, shot_index)


def plot_joint_spectrum(outdir, spectrum, passband):
    """Save radially binned temporal spectra before and after masking Fourier modes."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    steps, height, width = spectrum.shape
    ky = np.fft.fftfreq(height)[:, None]
    kx = np.fft.fftfreq(width)[None, :]
    radial_step = 1 / max(height, width)
    radial_bins = np.floor(np.hypot(kx, ky) / radial_step + 0.5).astype(int).ravel()
    counts = np.bincount(radial_bins)
    radial_edges = (np.arange(len(counts) + 1) - 0.5) * radial_step

    power = np.abs(spectrum) ** 2
    panels = []
    for values in (power, np.where(passband, power, 0)):
        sums = np.array([np.bincount(radial_bins, weights=plane.ravel(), minlength=len(counts))
                         for plane in values])
        # Removed modes contribute zero to the same denominator used before filtering.
        means = np.fft.fftshift(sums, axes=0) / np.ma.masked_equal(counts, 0)
        panels.append(np.log1p(means))

    freqs = np.fft.fftshift(np.fft.fftfreq(steps))
    extent = (radial_edges[0], radial_edges[-1], freqs[0] - 0.5 / steps,
              freqs[-1] + 0.5 / steps)
    norm = Normalize(0, panels[0].max() or 1.0)
    fig = Figure(figsize=(9.1, 4.4), layout="constrained")
    axes = fig.subplots(1, 2, sharex=True, sharey=True)
    for ax, panel, title in zip(axes, panels, ("Before filtering", "After filtering")):
        image = ax.imshow(panel, extent=extent, origin="lower", interpolation="nearest",
                          cmap="magma", norm=norm, aspect="auto")
        image.set_cmap(image.get_cmap().with_extremes(bad="#dfe6ee"))
        ax.set(xlim=(0, radial_edges[-1]), title=title)
    axes[0].set_ylabel("Temporal frequency (cycles per step)")
    fig.supxlabel("Radial spatial frequency |k| (cycles per grid cell)", fontsize=11)
    fig.colorbar(image, ax=axes, shrink=0.86, label="ln(1 + mean power per mode)")
    fig.suptitle("Joint wavenumber–frequency spectrum", fontsize=15)
    _save(fig, outdir / "wavenumber_frequency.png")


def plot_errors(outdir, rows, paired=None):
    """Save per-shot site-error counts and the optional paired rate difference."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    metrics = (("fp", "False positives"), ("fn", "Missed sites (false negatives)"))
    rates = np.array([[100 * row[metric] / row["shots"] for row in rows]
                      for metric, _ in metrics])
    limit = max(rates.max(), 1.0)
    method_names = [row["method"].replace("_", " ") for row in rows]
    fig = Figure(figsize=(10.5, 3.8 if paired is not None else 3.5), layout="constrained")
    axes = fig.subplots(1, 2, sharex=True, sharey=True)
    y = np.arange(len(rows))
    colors = ["#345785", "#16675e"]
    for ax, (_, title), values in zip(axes, metrics, rates):
        if len(values) == 2:
            title += f"\nChange from first to second  {values[1] - values[0]:+.2f}"
        ax.barh(y, values, color=colors, height=0.55, zorder=2)
        ax.set(xlim=(0, 1.2 * limit), yticks=y, yticklabels=method_names, title=title)
        ax.grid(axis="x", color="#dfe6ee", linewidth=0.7)
        ax.tick_params(axis="y", length=0)
        ax.spines[["top", "right", "left"]].set_visible(False)
        for value, center in zip(values, y):
            ax.text(value + 0.018 * limit, center, f"{value:.2f}", va="center", fontsize=9)
    axes[0].invert_yaxis()
    fig.supxlabel("Sites per 100 held-out shots", fontsize=11)
    title = "CNN site errors on the same held-out shots"
    if paired is not None:
        summary = (f"Paired total site-error change (bottom − top)  "
                   f"{100 * paired['difference']:+.3f} percentage points")
        if paired["ci95"] is not None:
            low, high = paired["ci95"]
            summary += f" · shot-based 95% CI [{100 * low:+.3f}, {100 * high:+.3f}]"
        title = f"{title}\n{summary}"
    fig.suptitle(title, fontsize=12 if paired is not None else 15)
    _save(fig, outdir / "cnn_site_error_comparison.png")
