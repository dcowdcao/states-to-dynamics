"""Panels shared by the move-legality figures. Drawing only; the statistics are in
`spade.move_legality.localize`.

Every panel here is event-locked to the spliced move at t0 and obeys FIGURE_GUIDELINES 4.2:
a thin dashed gray rule at r = 0 behind the data, the pre-event window shaded and named,
and -- the one this study adds -- the post-event region shaded as *no ground truth*, since
after an illegal move no board exists for the replayed suffix to be legal on.
"""

import numpy as np
from matplotlib.patches import Patch

from spade.figures.paper_style import GRID, MUTED, NULL

PRE_EVENT_FACE = "#f2f2f2"
UNDEFINED_HATCH = "///"


def event_axes(ax, window, ylabel: str, undefined: bool = True) -> None:
    """Shade the two annotated regions and put the dashed rule on the event."""
    offsets = list(window)
    ax.axvspan(offsets[0] - 0.5, -0.5, color=PRE_EVENT_FACE, zorder=0, lw=0)
    if undefined:
        ax.axvspan(0.5, offsets[-1] + 0.5, facecolor="none", edgecolor=GRID,
                   hatch=UNDEFINED_HATCH, zorder=0, lw=0)
    ax.axvline(0, color=MUTED, ls=(0, (4, 3)), lw=0.9, zorder=1)
    ax.set_xlim(offsets[0] - 0.5, offsets[-1] + 0.5)
    ax.set_xlabel("move index relative to the spliced move  ($r = t - t_0$)")
    ax.set_ylabel(ylabel)


def region_key(undefined: bool = True) -> list[Patch]:
    """Legend handles for the two shaded regions, so neither reads as an artifact."""
    handles = [Patch(facecolor=PRE_EVENT_FACE, edgecolor="none",
                     label="$r<0$: shared prefix, identical in every branch")]
    if undefined:
        handles.append(Patch(facecolor="none", edgecolor=GRID, hatch=UNDEFINED_HATCH,
                             label="$r>0$: replayed suffix, no board to judge it on"))
    return handles


def curve(ax, window, mean, lo, hi, style: dict, label: str | None = None,
          markevery: int = 4) -> None:
    """Mean with a shaded 95% band. Markers on the body of the curve, not every point."""
    offsets = np.array(list(window))
    ax.fill_between(offsets, lo, hi, color=style["color"], alpha=0.18, lw=0, zorder=2)
    ax.plot(offsets, mean, color=style["color"], marker=style.get("marker", "o"),
            ls=style.get("ls", "-"), lw=style.get("lw", 1.8), markersize=4.2,
            markevery=markevery, zorder=style.get("zorder", 4),
            label=label if label is not None else style.get("label"))


def item_heatmap(ax, matrix: np.ndarray, window, cbar_label: str, fig=None) -> None:
    """One row per branch, sorted by t0; color is the probe's output at that position."""
    offsets = list(window)
    image = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis",
                      vmin=0, vmax=1, interpolation="nearest",
                      extent=(offsets[0] - 0.5, offsets[-1] + 0.5, 0, len(matrix)))
    ax.axvline(0, color="white", ls=(0, (4, 3)), lw=0.9)
    ax.set_ylabel(f"branch (n = {len(matrix)}, sorted by $t_0$)")
    ax.set_xlabel("$r = t - t_0$")
    ax.grid(False)
    if fig is not None:
        bar = fig.colorbar(image, ax=ax, pad=0.02, fraction=0.045)
        bar.set_label(cbar_label, fontsize=8.5)
        bar.ax.tick_params(labelsize=7.5)


def rate_by_t0(ax, t0_values, series: list[dict], chance: np.ndarray | None = None) -> None:
    """Localization rate against t0, with the 1/(t0+1) pick-a-position chance line."""
    if chance is not None:
        ax.plot(t0_values, chance, color=NULL["color"], ls=(0, (1, 1.6)), lw=1.2,
                zorder=2, label="chance, $1/(t_0+1)$")
    for s in series:
        ax.errorbar(t0_values, s["mean"], yerr=[s["mean"] - s["lo"], s["hi"] - s["mean"]],
                    color=s["style"]["color"], marker=s["style"].get("marker", "o"),
                    ls=s["style"].get("ls", "-"), lw=s["style"].get("lw", 1.6),
                    markersize=4, elinewidth=0.8, capsize=1.5,
                    zorder=s["style"].get("zorder", 4), label=s["label"])
    ax.set_xlabel("position of the spliced move, $t_0$")
    ax.set_ylabel("argmax over $t \\leq t_0$ lands on $t_0$")
    ax.set_ylim(0, 1.02)


def auroc_bars(ax, rows: list[dict], title: str = "") -> None:
    """Horizontal AUROC intervals on the full 0-1 axis; sub-0.5 is never flipped."""
    y = np.arange(len(rows))[::-1]
    for pos, row in zip(y, rows):
        ax.errorbar(row["auroc"], pos,
                    xerr=[[row["auroc"] - row["lo"]], [row["hi"] - row["auroc"]]],
                    color=row["style"]["color"], marker=row["style"].get("marker", "o"),
                    markersize=5.5, elinewidth=1.2, capsize=2.5, ls="none",
                    zorder=row["style"].get("zorder", 4))
    ax.axvline(0.5, color=NULL["color"], ls=NULL["ls"], lw=1.2, zorder=2)
    ax.text(0.5, 0.015, " chance", color=NULL["color"], fontsize=7.5, ha="left",
            va="bottom", transform=ax.get_xaxis_transform())
    ax.set_yticks(y, [row["label"] for row in rows], fontsize=8)
    ax.set_xlim(-0.015, 1.015)   # the full 0-1 axis, plus enough not to clip a marker at 1.0
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("AUROC at $t_0$, illegal branch vs. its own legal control\n"
                  "($>0.5$: the illegal move scores higher)")
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(axis="y", visible=False)
    if title:
        ax.set_title(title)


def confusion(ax, matrix: np.ndarray, row_labels, col_labels, fig=None) -> None:
    """4-class posterior at t0, averaged over the branches of each row. Rows sum to 1.

    Not a confusion matrix of argmax counts: the mean posterior keeps the leak that an
    argmax hides, and the leak is where the two legal rows and the R1 row meet.
    """
    image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if matrix[i, j] < 0.55 else "black")
    ax.set_xticks(range(len(col_labels)), col_labels, fontsize=8)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=8)
    ax.set_xlabel("mean posterior at $t_0$")
    ax.set_ylabel("the branch played at $t_0$")
    ax.grid(False)
    if fig is not None:
        fig.colorbar(image, ax=ax, pad=0.02, fraction=0.045).ax.tick_params(labelsize=7.5)
