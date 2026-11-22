"""Sample GPU board power with NVML and integrate it over an operation.

This reports device-level gross energy, not process-attributed energy. For
interpretable measurements, run workloads with exclusive access to the GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Lock, Thread
from time import perf_counter
from typing import Callable, Protocol, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class GpuEnergyMeasurement:
    label: str
    energy_mwh: float
    duration_seconds: float
    mean_power_w: float
    peak_power_w: float
    sample_count: int
    device_index: int
    device_name: str
    sampling_interval_seconds: float
    method: str = "nvml_trapezoidal_power_integration"


class EnergyMeter(Protocol):
    def measure(
        self,
        operation: Callable[[], T],
        *,
        label: str,
    ) -> tuple[T, GpuEnergyMeasurement]:
        """Run an operation and return its value and measured GPU energy."""


class NvmlEnergyMeter:
    """Measure GPU board energy by sampling NVML power draw.

    NVML power is reported in milliwatts. Adjacent samples are integrated with
    the trapezoidal rule and converted to mWh. The measurement includes gross
    board energy during the operation; it does not subtract idle power and
    cannot attribute energy to one process when the GPU is shared.
    """

    def __init__(
        self,
        device_index: int = 0,
        sampling_interval_seconds: float = 0.05,
        nvml_module=None,
    ) -> None:
        if device_index < 0:
            raise ValueError("device_index cannot be negative.")
        if sampling_interval_seconds <= 0:
            raise ValueError("sampling_interval_seconds must be positive.")

        if nvml_module is None:
            try:
                import pynvml as nvml_module
            except ImportError as exc:
                raise RuntimeError(
                    "NVML measurement requires the nvidia-ml-py package "
                    "(imported as pynvml)."
                ) from exc

        self._nvml = nvml_module
        self.device_index = device_index
        self.sampling_interval_seconds = sampling_interval_seconds
        self._lock = Lock()
        self._closed = False

        try:
            self._nvml.nvmlInit()
            self._handle = self._nvml.nvmlDeviceGetHandleByIndex(device_index)
            raw_name = self._nvml.nvmlDeviceGetName(self._handle)
            self.device_name = (
                raw_name.decode("utf-8", errors="replace")
                if isinstance(raw_name, bytes)
                else str(raw_name)
            )
        except Exception:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            raise

    def measure(
        self,
        operation: Callable[[], T],
        *,
        label: str,
    ) -> tuple[T, GpuEnergyMeasurement]:
        if not label:
            raise ValueError("label is required.")
        if not callable(operation):
            raise TypeError("operation must be callable.")

        with self._lock:
            if self._closed:
                raise RuntimeError("NVML energy meter is closed.")

            samples: list[tuple[float, float]] = []
            sample_errors: list[Exception] = []
            stop_event = Event()

            def sample_power() -> None:
                try:
                    power_mw = float(
                        self._nvml.nvmlDeviceGetPowerUsage(self._handle)
                    )
                    if power_mw < 0:
                        raise RuntimeError("NVML returned negative GPU power.")
                    samples.append((perf_counter(), power_mw))
                except Exception as exc:
                    sample_errors.append(exc)

            def sampling_loop() -> None:
                while not stop_event.wait(self.sampling_interval_seconds):
                    sample_power()

            sample_power()
            if sample_errors:
                raise RuntimeError("Unable to sample GPU power through NVML.") from sample_errors[0]

            thread = Thread(
                target=sampling_loop,
                name="nvml-power-sampler",
                daemon=True,
            )
            thread.start()
            try:
                value = operation()
            except BaseException:
                stop_event.set()
                thread.join()
                sample_power()
                raise
            else:
                stop_event.set()
                thread.join()
                sample_power()

            if sample_errors:
                raise RuntimeError("NVML GPU power sampling failed.") from sample_errors[0]
            if len(samples) < 2:
                raise RuntimeError("NVML measurement needs at least two power samples.")

            samples.sort(key=lambda item: item[0])
            duration_seconds = samples[-1][0] - samples[0][0]
            if duration_seconds <= 0:
                raise RuntimeError("NVML measurement duration was not positive.")

            energy_mwh = 0.0
            for (time_a, power_a_mw), (time_b, power_b_mw) in zip(
                samples,
                samples[1:],
            ):
                delta_seconds = time_b - time_a
                energy_mwh += (
                    0.5 * (power_a_mw + power_b_mw) * delta_seconds / 3600.0
                )

            mean_power_w = energy_mwh * 3.6 / duration_seconds
            peak_power_w = max(power for _, power in samples) / 1000.0

            measurement = GpuEnergyMeasurement(
                label=label,
                energy_mwh=energy_mwh,
                duration_seconds=duration_seconds,
                mean_power_w=mean_power_w,
                peak_power_w=peak_power_w,
                sample_count=len(samples),
                device_index=self.device_index,
                device_name=self.device_name,
                sampling_interval_seconds=self.sampling_interval_seconds,
            )
            return value, measurement

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._nvml.nvmlShutdown()
                self._closed = True

    def __enter__(self) -> "NvmlEnergyMeter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
