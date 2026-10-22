"""Farassat 1A acoustics tools for face-sampled surface data."""

from .core import F1AResult, F1ASolver
from .io import (
    NPZSurfaceSeries,
    OpenFOAMSurfaceFrame,
    OpenFOAMSurfaceSeries,
    SurfaceChunk,
    SurfaceFrame,
    VTKSectionError,
)
from .sampling import CylinderSurface, SphereSurface, interpolate_openfoam_fields
from .utils import (
    PressureSpectrum,
    pressure_to_oaspl,
    pressure_to_spl_spectrum,
    spl_spectrum_to_a_weighted_oaspl,
    spl_spectrum_to_oaspl,
)

__all__ = [
    "F1AResult",
    "F1ASolver",
    "NPZSurfaceSeries",
    "OpenFOAMSurfaceFrame",
    "OpenFOAMSurfaceSeries",
    "PressureSpectrum",
    "SurfaceChunk",
    "SurfaceFrame",
    "CylinderSurface",
    "SphereSurface",
    "VTKSectionError",
    "interpolate_openfoam_fields",
    "pressure_to_oaspl",
    "pressure_to_spl_spectrum",
    "spl_spectrum_to_a_weighted_oaspl",
    "spl_spectrum_to_oaspl",
]
