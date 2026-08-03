"""The house figure style. The only module in the tree where a color literal may appear.

FIGURE_GUIDELINES.md is the normative document and plotting code cites it by section
number; this is its executable half. Okabe-Ito palette, viridis for sequential, never
color alone as an encoding (color AND marker, plus line style beyond three series) so
every figure survives grayscale at final printed size.

Style dicts are keyed by the domain quantity, not by the plot: class, rule, category,
population, disc, square role, branch role, probe, event, component. Two figures showing
the same class therefore agree without either knowing about the other. `apply_style` takes
camera_ready as a flag, not a rewrite — the same script emits a working PNG and a vector
PDF, and camera-ready drops the in-figure title.
"""

# def apply_style(camera_ready: bool = False) -> None: ...   # rcParams; pdf.fonttype=42
# def save_figure(fig, path): ...                            # bbox_inches="tight", dpi>=300
# CLASS_STYLE = RULE_STYLE = CATEGORY_STYLE = ... = {}       # color + marker + linestyle
# REFERENCE = ...   # black/dark gray, first in legend, labeled "(baseline)"
# NULL = ...        # gray, dashed, low zorder, last — never in the categorical cycle
# def rule_key(...), role_key(...): ...                      # in-figure keys, same constants as labels
# SAE_HPARAMS = {}  # family -> sparsity setting, so every label carries its hyperparameter
