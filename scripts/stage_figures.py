"""Copy the paper's main-text figures here under the names paper.tex asks for.

The figures are written by seven different studies into seven different directories, and
paper.tex renames every one on its way into Overleaf. Nothing recorded that mapping, which is
how the appendix once shipped the L0-matched twins under the canonical names. SOURCES is the
mapping, and running this file is the only way anything gets into this directory.

Copies, not symlinks: this directory is what gets uploaded, and a symlink does not survive
the trip.

Usage: python figures/paper/stage.py
"""

import shutil
from pathlib import Path

from spade.common.paths import repo_root

HERE = Path(__file__).resolve().parent
DETECTOR = "analysis/violation_divergence/illegal_move_detector/figures"

# paper.tex's name -> the study file that owns it.
SOURCES = {
    "rules_and_generation": "figures/rules_and_generation/rules_and_generation",
    "flanking": "analysis/bracket_flip/figures/fig1_flip_and_basis",
    "probe_geometry_and_detection":
        "figures/probe_directions_and_detection/probe_directions_and_detection",
    "detector_example": f"{DETECTOR}/detector_overview",
    "detector_SAE": f"{DETECTOR}/sae_detector_overview",
    "karvonen_sae_sweep": "figures/l0_operating_point/figure4_othello_with_spade",
}


def main() -> None:
    for name, stem in SOURCES.items():
        for suffix in (".png", ".pdf"):
            src = repo_root() / (stem + suffix)
            if not src.exists():
                print(f"missing  {src.relative_to(repo_root())}")
                continue
            shutil.copy2(src, HERE / (name + suffix))
            print(f"staged   {name}{suffix}  <- {stem}{suffix}")


if __name__ == "__main__":
    main()
