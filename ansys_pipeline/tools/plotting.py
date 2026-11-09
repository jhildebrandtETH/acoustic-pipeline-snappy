"""Publication defaults and paired vector/raster exports for pipeline figures."""
from pathlib import Path

import matplotlib
from cycler import cycler

matplotlib.use("Agg")
from matplotlib import pyplot  # noqa: E402

COLORS = ("#176287", "#BB572B", "#007D68", "#83559B", "#996600", "#415669")
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "mathtext.fontset": "dejavusans",
    "text.color": "#182735",
    "axes.labelcolor": "#182735",
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 8,
    "axes.edgecolor": "#647583",
    "axes.linewidth": .75,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.prop_cycle": cycler(color=COLORS),
    "axes.formatter.use_mathtext": True,
    "xtick.color": "#435364",
    "ytick.color": "#435364",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "grid.color": "#9DAEBA",
    "grid.alpha": .35,
    "grid.linewidth": .65,
    "lines.linewidth": 1.45,
    "legend.frameon": False,
    "legend.fontsize": "small",
    "image.cmap": "viridis",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
})


def save_figure(figure, path, **options):
    """Keep the report PNG and write an editable vector PDF beside it."""
    path = Path(path)
    options.pop("dpi", None)
    figure.savefig(path.with_suffix(".pdf"), dpi=300, **options)
    figure.savefig(path, dpi=300, **options)
