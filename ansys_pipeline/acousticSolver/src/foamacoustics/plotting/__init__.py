"""Matplotlib helpers for acoustic histories, spectra, and diagnostics.

The public functions are grouped into submodules for discoverability, while
this package namespace preserves the original ``foamacoustics.plotting``
imports.
"""

from ._common import ArrayLike, FigureSize, FrequencyLimits
from .diagnostics import (
    plot_blade_metric,
    plot_blade_pressure_snapshot,
    plot_error_concentration,
    plot_face_pressure_histories,
    plot_harmonic_levels,
    plot_signed_contributions,
    plot_surface_pressure_preview,
    plot_trace_spectrum_comparison,
)
from .directivity import (
    DirectivityPlot,
    configure_polar_axis,
    plot_directivity_metrics,
    plot_oaspl_directivity,
)
from .pressure import (
    MultiAxesPlot,
    plot_pressure_history,
    plot_pressure_traces,
    plot_result_pressure_comparison,
)
from .spectra import (
    ObserverSpectrumPlots,
    SPLPlotResult,
    SpectrumLike,
    plot_observer_spectra,
    plot_spectrum_comparison,
    plot_spl_spectrum,
)

# Compatibility alias from the original single-module plotting API.
LevelLike = ArrayLike

__all__ = [
    "ArrayLike",
    "DirectivityPlot",
    "FigureSize",
    "FrequencyLimits",
    "LevelLike",
    "MultiAxesPlot",
    "ObserverSpectrumPlots",
    "SPLPlotResult",
    "SpectrumLike",
    "configure_polar_axis",
    "plot_blade_metric",
    "plot_blade_pressure_snapshot",
    "plot_directivity_metrics",
    "plot_error_concentration",
    "plot_face_pressure_histories",
    "plot_harmonic_levels",
    "plot_oaspl_directivity",
    "plot_observer_spectra",
    "plot_pressure_history",
    "plot_pressure_traces",
    "plot_result_pressure_comparison",
    "plot_signed_contributions",
    "plot_spectrum_comparison",
    "plot_spl_spectrum",
    "plot_surface_pressure_preview",
    "plot_trace_spectrum_comparison",
]
