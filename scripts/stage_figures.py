"""Copy the paper's 16 figures into figures/ under the names the paper numbers them by.

Each figure is drawn by the study that owns it, into that study's own gitignored output
directory, under a name that records what it is rather than where the paper puts it -- and
several mislead out of context: `fig1_flip_and_basis` is Fig 7, `figure4_othello_with_spade`
is Fig 6, and Figs 9-16 are the `_l0_matched` twins, which their captions do not say. SOURCES
is that mapping, and running this file is the only way anything gets into figures/: edit a
figure in the study that owns it, re-render there, then stage.

Copies, not symlinks: figures/ is what gets uploaded, and a symlink does not survive the trip.
Each figure is a `.png` (what the paper embeds) and the `.pdf` its owner renders under its
camera-ready flag; a missing file is reported and skipped.

Usage: python scripts/stage_figures.py
"""

import shutil

from spade.common.paths import figures_dir, repo_root

DETECTOR = "analysis/violation_divergence/illegal_move_detector/figures"
FORK = "analysis/violation_divergence/fork_and_board/figures/game_1235_fork_and_board"
DISTANCE = "analysis/violation_divergence/divergence_by_space/figures/divergence_by_space"

# figures/<name> -> the stem its owner writes, under the owner named above each entry.
SOURCES = {
    # python -m spade.figures.figure_rules_and_generation
    "fig01_rules_and_generation": "build/rules_and_generation/rules_and_generation",
    # analysis/probe_directions_and_detection/make_figure.py
    "fig02_probe_geometry_and_detection":
        "analysis/probe_directions_and_detection/figures/probe_directions_and_detection",
    # illegal_move_detector/figure_overview.py
    "fig03_detector_example": f"{DETECTOR}/detector_overview",
    # illegal_move_detector/figure_sae_overview.py
    "fig04_detector_sae": f"{DETECTOR}/sae_detector_overview",
    # analysis/board_probe_sweep/layer_sweep.py
    "fig05_board_layer_sweep": "analysis/board_probe_sweep/figures/board_layer_sweep",
    # python -m spade.l0_operating_point.plot_pareto, then .combine_figure
    "fig06_karvonen_sae_sweep": "build/l0_operating_point/figure4_othello_with_spade",
    # analysis/bracket_flip/make_figures.py --only fig1
    "fig07_flanking": "analysis/bracket_flip/figures/fig1_flip_and_basis",
    # changepoint_localization/quantity_ablation.py --l0-matched
    "fig08_detector_metric_comparison": ("analysis/violation_divergence/changepoint_localization/"
                                         "figures/quantity_ablation_l0_matched"),
    # fork_and_board/make_figures.py --l0-matched [--variant rollout] [--plane r21]
    "fig09_fork_displacement_spliced": f"{FORK}_displacement_pca_l0_matched",
    "fig10_fork_displacement_rollout": f"{FORK}_displacement_pca_rollout_l0_matched",
    "fig11_fork_r21_spliced": f"{FORK}_r21_pca_l0_matched",
    "fig12_fork_r21_rollout": f"{FORK}_r21_pca_rollout_l0_matched",
    # divergence_by_space/make_figures.py --n_games 100 --l0-matched [--raw] [--plane r21]
    "fig13_distance_displacement": f"{DISTANCE}_n100_l0_matched",
    "fig14_distance_displacement_raw": f"{DISTANCE}_n100_raw_l0_matched",
    "fig15_distance_r21": f"{DISTANCE}_r21_n100_l0_matched",
    "fig16_distance_r21_raw": f"{DISTANCE}_r21_n100_raw_l0_matched",
}


def main() -> None:
    for name, stem in SOURCES.items():
        for suffix in (".png", ".pdf"):
            src = repo_root() / (stem + suffix)
            if not src.exists():
                print(f"missing  {stem}{suffix}")
                continue
            shutil.copy2(src, figures_dir() / (name + suffix))
            print(f"staged   {name}{suffix}  <- {stem}{suffix}")


if __name__ == "__main__":
    main()
