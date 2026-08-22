"""Ground truth for the un-normalised divergence-by-space curves, straight from the npz.

`make_figures.py` draws every curve as a multiple of that space-and-game's legal
control. A `--raw` flag draws the numerator instead, and this is the independent
check on it: the same reduction (`series_rows`' rule grouping, mean over member
categories, mean over games, 1.96 sd / sqrt(n)) with the control divisor removed,
written out per drawn point so a candidate implementation can be diffed against
numbers that did not come from it.

Three things it reports:

  1. Four CSVs, one per figure, of every drawn point in RAW units.
  2. A consistency proof. Dividing PER GAME by that game's control mean over r >= 0
     and reducing reproduces the committed ratio CSVs. Doing the division on the
     game-averaged curves instead does NOT -- E[x/c] != E[x]/E[c] -- and the size
     of that gap is reported too, because it is the reason a raw figure and a ratio
     figure cannot be read off one another.
  3. The magnitudes a README has to quote: the control's raw level per space, each
     rule's raw peak and where it falls, and the dynamic range across the nine
     panels, which is what decides the y-axis treatment.

R^21's two figures are NOT in physical units: `curvature_plane.fit_plane` divides
each descriptor column by its per-game SD and `Plane.project` re-applies that
scaling, so both distances there are in per-game descriptor-SD units.

Usage: PYTHONPATH=src python analysis/violation_divergence/divergence_by_space/verification/raw_ground_truth.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from spade.common.paths import results_dir
from spade.common.study import variant_dir
from spade.othello.taxonomy import RULE_MEMBERS

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from make_figures import (  # noqa: E402  (sibling module)
    CONTROL, TABLE_MEASURES, VARIANTS, mean_ci, series_rows)

SUFFIX = "_n100"
FIGURES = [(plane, variant) for plane in ("displacement", "r21")
           for variant in VARIANTS]


def raw(z, key: str) -> np.ndarray:
    """[S, G, series, T]: `ratios` without the control divisor -- each rule collapsed
    onto the mean of its member categories, in the space's own units."""
    return np.stack([z[key][:, :, members].mean(2) for members in series_rows(z).values()],
                    axis=2)


def committed(plane: str, variant: str) -> pd.DataFrame:
    tag = "" if plane == "displacement" else f"_{plane}"
    name = f"divergence_by_space{tag}{'_' + variant if variant != 'spliced' else ''}{SUFFIX}.csv"
    return pd.read_csv(results_dir("violation_divergence") / name)


def load(plane: str, variant: str) -> dict:
    tag = "" if plane == "displacement" else f"_{plane}"
    base = variant_dir("divergence", variant=VARIANTS[variant])
    z = dict(np.load(base / "divergence_by_space" / f"curves{tag}{SUFFIX}.npz"))
    z["space"], z["role"] = list(z["space"]), list(z["role"])
    return z


def table(z, plane: str, variant: str) -> pd.DataFrame:
    rows = []
    for key in TABLE_MEASURES:
        x = raw(z, key)
        for si, space in enumerate(z["space"]):
            for ci, name in enumerate(series_rows(z)):
                mu, half = mean_ci(x[si, :, ci])
                rows.append(pd.DataFrame({
                    "variant": variant, "plane": plane, "space": space, "series": name,
                    "measured_in": key, "r": z["window"], "n_games": x.shape[1],
                    "mean": mu, "ci_lo": mu - half, "ci_hi": mu + half}))
    return pd.concat(rows, ignore_index=True)


def reproduce(z, plane: str, variant: str) -> pd.DataFrame:
    """The committed ratio CSV rebuilt from the raw arrays, two ways.

    `per_game` divides each game by its own control before averaging, which is what
    `normalised` does. `aggregate` divides the game-averaged raw curve by the
    game-averaged control instead -- the reduction a reader would use to convert a
    raw figure back into the ratio figure by eye.
    """
    control = list(series_rows(z)).index(CONTROL)
    post = z["window"] >= 0
    rows = []
    for key in TABLE_MEASURES:
        x = raw(z, key)
        divisor = x[:, :, control][..., post].mean(-1)
        per_game = mean_ci(np.moveaxis(x / divisor[:, :, None, None], 1, 0))[0]
        agg = x.mean(1) / x.mean(1)[:, control][..., post].mean(-1)[:, None, None]
        for si, space in enumerate(z["space"]):
            for ci, name in enumerate(series_rows(z)):
                rows.append(pd.DataFrame({
                    "space": space, "series": name, "measured_in": key, "r": z["window"],
                    "per_game": per_game[si, ci], "agg_ratio": agg[si, ci]}))
    got = pd.concat(rows, ignore_index=True)
    keys = ["space", "series", "measured_in", "r"]
    return got.merge(committed(plane, variant)[keys + ["ratio"]], on=keys, validate="1:1")


def magnitudes(z, df: pd.DataFrame) -> str:
    """The control level per space, each rule's raw peak, and the panel-to-panel range."""
    post = df[df.r >= 0]
    out = [f"{'space':<34}{'control plane':>14}{'control full':>14}"
           + "".join(f"{rule.split()[0] + ' peak (r)':>20}" for rule in RULE_MEMBERS)]
    for space in z["space"]:
        cell = post[post.space == space]
        ctl = cell[cell.series == CONTROL]
        line = (f"{space:<34}"
                f"{ctl[ctl.measured_in == 'plane']['mean'].mean():>14.4g}"
                f"{ctl[ctl.measured_in == 'full']['mean'].mean():>14.4g}")
        for rule in RULE_MEMBERS:
            c = cell[(cell.series == rule) & (cell.measured_in == "plane")]
            top = c.loc[c["mean"].idxmax()]
            line += f"{top['mean']:>13.4g} ({int(top.r):>+3d})"
        out.append(line)

    ctl = post[(post.series == CONTROL) & (post.measured_in == "plane")]
    level = ctl.groupby("space")["mean"].mean()
    peak = post[post.measured_in == "plane"].groupby("space")["mean"].max()
    floor = df[(df.r >= 0) & (df.measured_in == "plane")].groupby("space")["mean"].min()
    out.append(f"panel range, in-plane control level: {level.min():.4g} ({level.idxmin()}) "
               f"to {level.max():.4g} ({level.idxmax()}), {level.max() / level.min():.4g}x")
    out.append(f"panel range, largest drawn point:   {peak.min():.4g} ({peak.idxmin()}) "
               f"to {peak.max():.4g} ({peak.idxmax()}), {peak.max() / peak.min():.4g}x")
    out.append(f"smallest drawn point at r >= 0:     {floor.min():.4g} ({floor.idxmin()}); "
               f"the whole r < 0 prefix is |x| <= {df[df.r < 0]['mean'].abs().max():.3g}")
    return "\n".join(out)


def flip_note(chk: pd.DataFrame) -> str:
    """How often the ratio figure and a raw panel put a rule on OPPOSITE sides of its
    control, over r >= 0. The divisor is per game, so the ratio figure estimates
    E[rule/control] while a raw panel invites the eye to compare E[rule] against
    E[control]; Jensen puts the first above the second, by enough to change the sign
    of the claim. This is why a raw figure cannot be read as the ratio figure rescaled."""
    post = chk[(chk.r >= 0) & (chk.series != CONTROL)]
    cell = post.groupby(["measured_in", "space", "series"], sort=False)[
        ["per_game", "agg_ratio"]].mean()
    crossed = np.sign(cell.per_game - 1) != np.sign(cell.agg_ratio - 1)
    gap = (cell.per_game - cell.agg_ratio).abs()
    worst = gap.idxmax()
    return ("ratio figure E[rule/control] vs raw-panel E[rule]/E[control], mean over r >= 0:\n"
            + "\n".join(
                f"  {m:<6} {int(crossed[m].sum())}/{len(cell.loc[m])} cells land on opposite "
                f"sides of the control" for m in TABLE_MEASURES)
            + f"\n  widest cell: {worst[1]} {worst[2]} ({worst[0]}) "
              f"{cell.per_game[worst]:.3f} as a ratio, {cell.agg_ratio[worst]:.3f} by eye")


def axis_note(df: pd.DataFrame) -> str:
    """What one shared linear y-axis, autoscaled over the nine panels, does to each
    panel. The figure draws the in-plane curve only, so that is what is measured;
    the axis runs from 0 (the shared prefix is exactly zero) to the largest upper
    CI edge anywhere in the grid."""
    drawn = df[df.measured_in == "plane"]
    top = drawn.ci_hi.max()
    out = [f"one shared linear y over the nine panels: 0 to {top:.4g}",
           f"{'space':<34}{'panel extent':>14}{'% of axis':>11}"
           f"{'max rule-control gap':>22}{'% of axis':>11}"]
    for space, cell in drawn.groupby("space", sort=False):
        post = cell[cell.r >= 0]
        extent = post.ci_hi.max() - post["mean"].min()
        wide = post.pivot_table(index="r", columns="series", values="mean")
        gap = wide[list(RULE_MEMBERS)].max(1).sub(wide[CONTROL]).max()
        out.append(f"{space:<34}{extent:>14.4g}{100 * extent / top:>10.2f}%"
                   f"{gap:>22.4g}{100 * gap / top:>10.2f}%")
    return "\n".join(out)


def main():
    report = []
    for plane, variant in FIGURES:
        z = load(plane, variant)
        df = table(z, plane, variant)
        csv = HERE / f"raw_divergence_by_space{'' if plane == 'displacement' else '_' + plane}" \
                     f"{'_' + variant if variant != 'spliced' else ''}{SUFFIX}.csv"
        df.to_csv(csv, index=False, float_format="%.6g")

        chk = reproduce(z, plane, variant)
        dev = (chk.per_game - chk.ratio).abs()
        rel = dev / chk.ratio.abs()
        gap = (chk.agg_ratio - chk.ratio).abs()
        units = ("per-game descriptor-SD units (fit_plane column-scales by SD, "
                 "Plane.project re-applies it)" if plane == "r21" else "raw activation units")
        report += [
            f"\n=== {plane} / {variant}  n={df.n_games.iloc[0]} games, "
            f"{len(z['space'])} spaces, r = {z['window'][0]:+d}..{z['window'][-1]:+d} ===",
            f"units: {units}",
            f"wrote {csv.relative_to(HERE.parents[3])}  ({len(df)} rows)",
            f"per-game normalisation vs committed ratio CSV: max abs dev {dev.max():.3g}, "
            f"max rel dev {rel[chk.ratio.abs() > 1e-6].max():.3g} over "
            f"{len(chk)} points",
            f"aggregate normalisation vs committed ratio CSV: max abs dev {gap.max():.3g} "
            f"(E[x/c] != E[x]/E[c]; a raw figure is not the ratio figure rescaled)",
            flip_note(chk), magnitudes(z, df), axis_note(df)]
    text = "\n".join(report).lstrip("\n")
    print(text)
    (HERE / "raw_magnitudes.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
