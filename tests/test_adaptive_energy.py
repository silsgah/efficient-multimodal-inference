from unittest.mock import patch

import pytest

from src.adaptive.energy import NvmlEnergyMeter


class FakeNvml:
    def __init__(self):
        self.shutdown_calls = 0
        self.init_calls = 0
        self.power_calls = 0

    def nvmlInit(self):
        self.init_calls += 1

    def nvmlShutdown(self):
        self.shutdown_calls += 1

    def nvmlDeviceGetHandleByIndex(self, index):
        assert index == 0
        return "fake-handle"

    def nvmlDeviceGetName(self, handle):
        assert handle == "fake-handle"
        return b"Test GPU"

    def nvmlDeviceGetPowerUsage(self, handle):
        assert handle == "fake-handle"
        self.power_calls += 1
        return 100_000


def test_nvml_energy_meter_integrates_power_in_mwh():
    fake_nvml = FakeNvml()
    with patch(
        "src.adaptive.energy.perf_counter",
        side_effect=[10.0, 12.0],
    ):
        with NvmlEnergyMeter(
            device_index=0,
            sampling_interval_seconds=3600,
            nvml_module=fake_nvml,
        ) as meter:
            value, measurement = meter.measure(
                lambda: "completed",
                label="unit-test",
            )

    assert value == "completed"
    assert fake_nvml.init_calls == 1
    assert fake_nvml.shutdown_calls == 1
    assert fake_nvml.power_calls == 2
    assert measurement.energy_mwh == pytest.approx(55.5555556)
    assert measurement.duration_seconds == pytest.approx(2.0)
    assert measurement.mean_power_w == pytest.approx(100.0)
    assert measurement.peak_power_w == pytest.approx(100.0)
    assert measurement.sample_count == 2
    assert measurement.device_name == "Test GPU"


def test_nvml_energy_meter_requires_positive_interval():
    with pytest.raises(ValueError, match="sampling_interval_seconds"):
        NvmlEnergyMeter(
            sampling_interval_seconds=0,
            nvml_module=FakeNvml(),
        )


def test_nvml_energy_meter_refuses_use_after_close():
    fake_nvml = FakeNvml()
    meter = NvmlEnergyMeter(nvml_module=fake_nvml)
    meter.close()

    with pytest.raises(RuntimeError, match="closed"):
        meter.measure(lambda: None, label="closed-test")
