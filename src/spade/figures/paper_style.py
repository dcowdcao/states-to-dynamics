"""Shared figure style. See FIGURE_GUIDELINES.md for the rules this encodes.

Import the palette and call apply_style() instead of setting rcParams or naming
colors in a plotting script, so a class/method keeps one identity everywhere.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Okabe-Ito, the CVD-safe set ICML's accessibility guide links to.
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
             "#E69F00", "#56B4E9", "#F0E442", "#000000"]

INK, MUTED, GRID = "#0b0b0b", "#6e6e6e", "#dcdcdc"

# Reference/baseline and null styling: never in the categorical cycle.
REFERENCE = {"color": "#000000", "marker": "o", "lw": 2.4, "zorder": 5}
NULL = {"color": "#8c8c8c", "marker": "v", "ls": (0, (5, 2)), "lw": 1.5, "zorder": 2}

# Straightness-study game classes, in legend order: the figures draw exactly
# these. random_tokens is a block of games.pt but has no entry here, so it is
# neither drawn nor scored (docs/claims_status.md).
# color = class identity, marker = redundant cue.
CLASS_STYLE = {
    "legal":         {"color": "#000000", "marker": "o", "label": "legal (baseline)"},
    "rollout":       {"color": "#0072B2", "marker": "D", "label": "OthelloGPT rollout"},
    "violation":     {"color": "#009E73", "marker": "^", "label": "violation (1 illegal move)"},
    "championship":  {"color": "#CC79A7", "marker": "P", "label": "championship (human)"},
}


# The three trajectory quantities, for figures that overlay them. theta keeps the
# violation color it carries in the per-quantity panels; the other two take hues
# no game class uses, so a curve never means two things across figures.
QUANTITY_STYLE = {
    "theta": {"color": "#009E73", "marker": "^", "label": "turning angle θ"},
    "d":     {"color": "#E69F00", "marker": "o", "label": "step length ‖Δz‖"},
    "norm":  {"color": "#56B4E9", "marker": "s", "label": "activation norm ‖z‖"},
}


# Rules study (docs/othello_rules.md). Constraint rules R1-R3 and the six
# illegal categories: hue = rule, lightness = detection depth within the rule,
# marker = rule. A grayscale reader still recovers both groupings. legal is the
# reference class in every rules figure.
RULE_STYLE = {
    "legal":         {"color": "#000000", "marker": "o", "label": "legal (baseline)"},
    "R1 placement":  {"color": "#0072B2", "marker": "s", "label": "R1 placement"},
    "R2 bracketing": {"color": "#D55E00", "marker": "^", "label": "R2 bracketing"},
    "R3 turn":       {"color": "#009E73", "marker": "D", "label": "R3 turn"},
}

# Keys and order follow othello.taxonomy.CATEGORIES, which is sorted by how much
# board computation the violation takes to detect; the R2 triple is an ordered
# variable, so it takes a light-to-dark ramp of one hue rather than three
# categorical ones.
CATEGORY_STYLE = {
    "legal":            {"color": "#000000", "marker": "o", "label": "legal (baseline)"},
    "occupied_own":     {"color": "#56B4E9", "marker": "s", "label": "R1 occupied (own)"},
    "occupied_opp":     {"color": "#0072B2", "marker": "s", "label": "R1 occupied (opp)"},
    "empty_detached":   {"color": "#E69F00", "marker": "^", "label": "R2 detached"},
    "empty_own_only":   {"color": "#D55E00", "marker": "^", "label": "R2 own-only"},
    "empty_no_bracket": {"color": "#8C3200", "marker": "^", "label": "R2 no bracket"},
    "opponent_only":    {"color": "#009E73", "marker": "D", "label": "R3 opponent-only"},
}

# One line per rule, for the key printed under a rules figure. Condensed from
# docs/othello_rules.md; a reader should not need that file to read the legend.
RULE_TEXT = {
    "R1": "R1 placement — a move puts one disc of the mover's color on an empty square.",
    "R2": "R2 bracketing — the disc must close a gapless line of opponent discs with one of the mover's own.",
    "R3": "R3 turn — colors alternate; legality is judged for the mover, not the opponent.",
    "R4": "R4 forfeit — a player with no legal move anywhere is skipped, and the opponent moves again.",
    "R5": "R5 flipping — placing a disc flips every bracketed opponent disc, in all directions at once.",
}


def rule_key(owner, rules: list[str], y: float = -0.005, x: float = 0.005) -> None:
    """Print the rule definitions under the panels, left-aligned.

    `owner` is the figure, or an axes when the key belongs beside a panel rather
    than under the whole figure; coordinates are that owner's. Under a figure,
    savefig's tight bbox grows to include the text, so it never collides with
    the bottom row.
    """
    owner.text(x, y, "\n".join(RULE_TEXT[r] for r in rules), ha="left",
               va="top", fontsize=7.5, color=MUTED, linespacing=1.5)


# Un-flip edit populations. `random` is the direction-matched null, so it takes
# NULL styling and goes last.
POPULATION_STYLE = {
    "flipped": {"color": "#0072B2", "marker": "o", "short": "just-flipped",
                "label": "just-flipped square"},
    "settled": {"color": "#009E73", "marker": "^", "short": "long-settled",
                "label": "long-settled square (age $\\geq$ 10)"},
    "placed":  {"color": "#CC79A7", "marker": "s", "short": "freshly-placed",
                "label": "freshly-placed disc"},
    "random":  {"color": NULL["color"], "marker": NULL["marker"],
                "short": "random dir.\n(null)", "label": "random direction (null)"},
}

# Bracket-flip study (docs/bracket_flip_design.md). Absolute disc colors draw
# themselves: a black disc is a filled black circle, a white disc an open one,
# so the panel needs almost no legend and reads identically in grayscale.
DISC_STYLE = {
    "empty": {"color": GRID, "marker": ".", "label": "empty"},
    "black": {"color": "#000000", "marker": "o", "label": "black disc"},
    "white": {"color": "#ffffff", "marker": "o", "label": "white disc"},
}
DISC_EDGE = "#000000"  # white discs need an edge to exist on a white page

# Mover-relative fillers, the probe's native basis. Up/down triangles carry the
# perspective flip when color is gone.
FILLER_STYLE = {
    "empty":  {"color": GRID, "marker": ".", "label": "empty"},
    "theirs": {"color": "#0072B2", "marker": "v", "label": "theirs (opponent)"},
    "mine":   {"color": "#E69F00", "marker": "^", "label": "mine (to move)"},
}

# A square's role in one flip event. `flipped` is the focal class and takes the
# hero treatment; unflipped discs are the reference behavior (they all swap).
SQUARE_ROLE_STYLE = {
    "flipped":        {"color": "#D55E00", "marker": "D", "label": "flipped square"},
    # The two unflipped classes travel in opposite directions and overlap in the
    # displacement field, so they must differ by shape, not only by lightness.
    "unflipped_own":  {"color": "#0072B2", "marker": "o", "label": "unflipped, mover's"},
    "unflipped_opp":  {"color": "#56B4E9", "marker": "P", "label": "unflipped, opponent's"},
    "placed":         {"color": "#CC79A7", "marker": "s", "label": "placed disc"},
    "empty":          {"color": MUTED, "marker": ".", "label": "empty square"},
}

# A square's role, spelled out under a bracket-flip figure. Same contract as
# RULE_TEXT: the legend should read without the design doc open.
ROLE_TEXT = {
    "flipped": "flipped — the disc s that m flanks: opponent's colour before, mover's after",
    "unflipped_own": "unflipped, mover's — a disc of the mover's colour, not flipped by m",
    "unflipped_opp": "unflipped, opponent's — an opponent disc other than s, not flipped by m",
    "placed": "placed — the square m itself: empty before, the mover's colour after",
    "empty": "empty — empty before m and still empty after",
}

# Violation-divergence read-outs plot two branches at the SAME square, so hue is
# already spent on the rule category. Line style carries which branch is being
# read, and the reference-at-the-same-square anchor is the thing the illegal
# curve is claimed to depart from — dotted so it reads as a reference, not a
# second condition.
BRANCH_ROLE_STYLE = {
    "illegal": {"ls": "-", "label": "illegal branch"},
    "reference_at_square": {"ls": (0, (1, 1.4)), "label": "reference branch, same square"},
}


def role_key(fig, roles: list[str], y: float = -0.005) -> None:
    """One line per square-role code, under the panels. See rule_key()."""
    fig.text(0.005, y, "\n".join(ROLE_TEXT[r] for r in roles), ha="left",
             va="top", fontsize=7.5, color=MUTED, linespacing=1.5)


# Probe family and label basis: two dimensions that cross in the depth sweep.
# The linear probe is the capacity reference, so it takes REFERENCE's black; the
# basis rides on line style and marker rather than spending a second hue.
PROBE_STYLE = {
    "linear": {"color": REFERENCE["color"], "marker": "o", "label": "linear probe"},
    "tpr":    {"color": "#E69F00", "marker": "D", "label": "TPR ($d_r$=52, $d_f$=2)"},
}
BASIS_STYLE = {
    "relative": {"ls": "-", "marker": "^", "label": "mover-relative"},
    "absolute": {"ls": (0, (4, 2)), "marker": "v", "label": "absolute colour"},
}

# Conditions in a flip event. `before` is the reference state; `control` is the
# matched non-flipping move at the same prefix, a control and not a null.
EVENT_STYLE = {
    "before":  {"color": "#000000", "marker": "o", "label": "before (baseline)"},
    "after":   {"color": "#D55E00", "marker": "D", "label": "after the flanking move"},
    "control": {"color": "#009E73", "marker": "s", "label": "matched non-flipping move"},
}

# Patched component in a layer sweep.
COMPONENT_STYLE = {
    "attn": {"color": "#0072B2", "marker": "o", "label": "attn_out"},
    "mlp":  {"color": "#D55E00", "marker": "s", "label": "mlp_out"},
}


# Sparsity hyperparameter per seed-42 SAE family, for figure labels. Values are
# the canonical trainer configs, not the checkpoints' config.json: every
# config.json under checkpoints/main is a Mac smoke config written over real GPU
# weights, so none of them labels its own weights (ARCHITECTURE.md, PROCESS.md).
# `SpaDE` is the default SpaDE configuration, λ=1e-3 γ=1e-4.
SAE_HPARAMS = {
    "PAnneal": "p-anneal, pen 0.042",
    "Standard": "L1 0.059",
    "GatedSAE": "L1 1.37",
    "TopK": "k=80",
    "GatedAnneal": "p-anneal, pen 0.955",
    "SpaDE": "λ=1e-3, γ=1e-4",
    "BatchTopK": "k=80",
    "CosineSpaDE": "γ=1.0",
    "SpaDE-lam1e-03-gam1e-04": "λ=1e-3, γ=1e-4",
}


def sae_label(family: str) -> str:
    """Family name with its sparsity setting, e.g. 'SpaDE (λ=1e-3, γ=1e-4)'."""
    hp = SAE_HPARAMS.get(family)
    base = "SpaDE" if family.startswith("SpaDE-") else family
    return f"{base} ({hp})" if hp else family


def apply_style(camera_ready: bool = False) -> None:
    """Working PNG by default; camera_ready switches to vector PDF + serif."""
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.linewidth": 0.7, "lines.linewidth": 1.8, "grid.alpha": 0.7,
        "legend.frameon": False, "legend.handlelength": 2.0,
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,  # TrueType; Type 3 fails NeurIPS
        "savefig.bbox": "tight", "savefig.dpi": 300,
        "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),
    })
    if camera_ready:
        mpl.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman No9 L", "DejaVu Serif"],
            "mathtext.fontset": "stix", "savefig.format": "pdf",
        })


def save_figure(fig, stem: Path, camera_ready: bool = False,
                dpi: int | None = None) -> Path:
    """Working PNG at `stem.png`, or a vector PDF when camera_ready.

    Dense scatter artists should be created with rasterized=True: the PDF then
    keeps vector text and axes over a raster cloud, which is what ICML's vector
    mandate asks for without a 40 MB file.
    """
    out = stem.with_suffix(".pdf" if camera_ready else ".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi or mpl.rcParams["savefig.dpi"], bbox_inches="tight")
    plt.close(fig)
    return out
