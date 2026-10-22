"""Signal-processing and validation helpers."""

from .spectrum import (
    REFERENCE_PRESSURE_PA,
    PressureSpectrum,
    pressure_to_oaspl,
    pressure_to_spl_spectrum,
    spl_spectrum_to_a_weighted_oaspl,
    spl_spectrum_to_oaspl,
)
from .validation import finite_float, nonnegative_float, positive_float, positive_int

__all__ = [
    "REFERENCE_PRESSURE_PA",
    "PressureSpectrum",
    "finite_float",
    "nonnegative_float",
    "positive_float",
    "positive_int",
    "pressure_to_oaspl",
    "pressure_to_spl_spectrum",
    "spl_spectrum_to_a_weighted_oaspl",
    "spl_spectrum_to_oaspl",
]
