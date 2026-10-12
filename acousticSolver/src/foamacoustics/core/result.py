"""Output model for an F1A observer prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ..utils.spectrum import pressure_to_oaspl


@dataclass(frozen=True, slots=True)
class F1AResult:
    """Observer traces on the original surface-source time grid.

    A single-observer result stores pressure arrays with shape ``(times,)`` and
    ``observer_m`` with shape ``(3,)``. A multi-observer result stores pressure
    arrays with shape ``(observers, times)`` and observer coordinates with
    shape ``(observers, 3)``.

    Attributes:
        observer_times_s: Observer-time samples relative to the first frame.
        thickness_pressure_pa: Farassat thickness contribution in pascals.
        loading_pressure_pa: Farassat loading contribution in pascals.
        total_pressure_pa: Sum of thickness and loading pressure in pascals.
        observer_m: One observer coordinate or an array of coordinates in metres.
        rpm: Rotor speed supplied to the solver.
        maximum_delay_s: Maximum face-arrival delay for each observer.
        response_delay_s: Configured response delay, if any.
        source_time_count: Number of surface-source frames processed.
        propagation_algorithm: Arrival interpolation algorithm used.
        device: Torch device used for the prediction.
        chunk_size: Number of source frames processed per chunk.
    """

    observer_times_s: torch.Tensor
    thickness_pressure_pa: torch.Tensor
    loading_pressure_pa: torch.Tensor
    total_pressure_pa: torch.Tensor
    observer_m: torch.Tensor
    rpm: float
    maximum_delay_s: float | torch.Tensor
    response_delay_s: float | None
    source_time_count: int
    propagation_algorithm: str
    device: str = "cpu"
    chunk_size: int = 1

    @property
    def oaspl_db(self) -> torch.Tensor:
        """OASPL of the stored trace, including any retained mean pressure."""
        if self.total_pressure_pa.ndim == 1:
            return pressure_to_oaspl(self.total_pressure_pa, remove_mean=False)
        return torch.stack(
            [
                pressure_to_oaspl(trace, remove_mean=False)
                for trace in self.total_pressure_pa
            ]
        )

    def to_csv(self, path: str | Path) -> Path:
        """Write a single-observer result to CSV.

        Args:
            path: Destination path. Parent directories must already exist.

        Returns:
            The normalized destination path.

        Raises:
            ValueError: If this result contains multiple observers.
        """
        path = Path(path)
        if self.total_pressure_pa.ndim != 1:
            raise ValueError(
                "to_csv() supports single-observer results only; select one "
                "observer with result.for_observer(index)."
            )
        rows = (
            torch.stack(
                (
                    self.observer_times_s,
                    self.thickness_pressure_pa,
                    self.loading_pressure_pa,
                    self.total_pressure_pa,
                ),
                dim=1,
            )
            .detach()
            .cpu()
            .tolist()
        )
        with path.open("w", encoding="utf-8", newline="") as stream:
            stream.write("time_s,p_thickness_pa,p_loading_pa,p_total_pa\n")
            for row in rows:
                stream.write(",".join(f"{value:.12e}" for value in row) + "\n")
        return path

    def for_observer(self, index: int) -> F1AResult:
        """Select one observer while preserving shared result metadata.

        Args:
            index: Zero-based observer index. Negative indices follow standard
                Python indexing for multi-observer results.

        Returns:
            A single-observer view of the selected pressure traces.

        Raises:
            IndexError: If ``index`` does not identify an available observer.
        """
        if self.total_pressure_pa.ndim == 1:
            if index not in {0, -1}:
                raise IndexError("A single-observer result contains only observer 0.")
            return self
        return F1AResult(
            observer_times_s=self.observer_times_s,
            thickness_pressure_pa=self.thickness_pressure_pa[index],
            loading_pressure_pa=self.loading_pressure_pa[index],
            total_pressure_pa=self.total_pressure_pa[index],
            observer_m=self.observer_m[index],
            rpm=self.rpm,
            maximum_delay_s=float(torch.as_tensor(self.maximum_delay_s)[index]),
            response_delay_s=self.response_delay_s,
            source_time_count=self.source_time_count,
            propagation_algorithm=self.propagation_algorithm,
            device=self.device,
            chunk_size=self.chunk_size,
        )
