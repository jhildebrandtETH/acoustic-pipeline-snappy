"""Small reusable input validators."""

import math


def finite_float(value: float, name: str) -> float:
    """Coerce a scalar to ``float`` and reject non-finite values.

    Args:
        value: Scalar value to validate.
        name: Parameter name used in the validation error.

    Raises:
        ValueError: If ``value`` is not finite.
    """
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def positive_float(value: float, name: str) -> float:
    """Return ``value`` as a finite positive float.

    Args:
        value: Scalar value to validate.
        name: Parameter name used in the validation error.

    Raises:
        ValueError: If ``value`` is not finite and greater than zero.
    """
    result = finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def nonnegative_float(value: float, name: str) -> float:
    """Return ``value`` as a finite non-negative float.

    Args:
        value: Scalar value to validate.
        name: Parameter name used in the validation error.

    Raises:
        ValueError: If ``value`` is not finite or is less than zero.
    """
    result = finite_float(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return result


def positive_int(value: int, name: str) -> int:
    """Return ``value`` as a positive integer without silent truncation.

    Args:
        value: Integer value to validate.
        name: Parameter name used in the validation error.

    Raises:
        TypeError: If ``value`` is not an integer.
        ValueError: If ``value`` is less than one.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be positive.")
    return value
