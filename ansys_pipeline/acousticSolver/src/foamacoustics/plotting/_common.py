"""Internal validation and style helpers shared by plotting modules."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import torch

from ..utils.validation import positive_float

ArrayLike = (
    torch.Tensor | Sequence[float] | Sequence[Sequence[float]] | npt.NDArray[np.generic]
)
FigureSize = tuple[float, float]
FrequencyLimits = tuple[float | None, float | None]

CLASSIC_COLORS = ("b", "g", "r", "c", "m", "y", "k")
LINE_STYLES = ("-", "--", "-.", ":")


def finite_1d(values: ArrayLike, name: str) -> torch.Tensor:
    """Return a finite one-dimensional CPU tensor suitable for plotting."""
    data = torch.as_tensor(values, dtype=torch.float64).detach().cpu()
    if data.ndim != 1 or data.numel() == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array.")
    if not torch.isfinite(data).all().item():
        raise ValueError(f"{name} must contain only finite values.")
    return data


def finite_numpy_1d(values: ArrayLike, name: str) -> npt.NDArray[np.float64]:
    """Return a finite one-dimensional NumPy array suitable for Matplotlib."""
    data = finite_1d(values, name).numpy()
    return np.asarray(data, dtype=np.float64)


def validated_figsize(figsize: FigureSize) -> FigureSize:
    """Validate and normalize a Matplotlib figure size."""
    if len(figsize) != 2:
        raise ValueError("figsize must contain width and height.")
    return (
        positive_float(figsize[0], "figsize[0]"),
        positive_float(figsize[1], "figsize[1]"),
    )


def line_properties(index: int) -> tuple[str, str]:
    """Return a deterministic classic color and line-style pair."""
    return (
        CLASSIC_COLORS[index % len(CLASSIC_COLORS)],
        LINE_STYLES[index % len(LINE_STYLES)],
    )
