"""Run monotonic duration-bucket state-machine experiments in Brian2."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from stnbeta.phase4.gpu import ensure_cuda_runtime_libraries

from .neuron_model import (
    bucket_equations,
    encoder_equations,
    phase5_cluster_exc_equations,
    phase5_cluster_inh_equations,
    phase5_readout_equations,
    quiet_equations,
    readout_equations,
)
from .synapse_model import (
    excitatory_on_pre,
    phase5_exc_to_inh_on_pre,
    phase5_forward_on_pre,
    phase5_inhibitory_on_pre,
    phase5_input_on_pre,
    phase5_readout_on_pre,
    phase5_recurrent_on_pre,
    phase5_reset_on_pre,
    reset_on_pre,
)
from .topologies.nsm_monotonic_duration import (
    DurationBucketClusterConfig,
    MonotonicStateMachineConfig,
)


def _lazy_brian(import_cuda: bool = False):
    import brian2 as b2

    if import_cuda:
        import brian2cuda  # noqa: F401

    return b2


@dataclass(frozen=True)
class StateMachineResult:
    """Spike outputs from the duration-bucket state machine."""

    duration_s: float
    encoder_spike_times_s: np.ndarray
    encoder_spike_indices: np.ndarray
    quiet_spike_times_s: np.ndarray
    bucket_spike_times_s: np.ndarray
    bucket_spike_indices: np.ndarray
    readout_spike_times_s: np.ndarray
    bucket_voltages: np.ndarray | None
    bucket_thresholds_ms: tuple[float, ...]
    band_roles: tuple[str, ...]
    readout_bucket_index: int


def derive_quiet_drive(
    encoder_currents: np.ndarray,
    band_roles: Sequence[str],
) -> np.ndarray:
    """Derive a quiet/reset drive from rectified band currents."""
    beta_mask = np.array([role == "beta" for role in band_roles], dtype=bool)
    if not beta_mask.any():
        beta_energy = np.mean(encoder_currents, axis=0)
    else:
        beta_energy = np.mean(encoder_currents[beta_mask], axis=0)
    beta_energy = np.clip(beta_energy, 0.0, None)
    scale = np.percentile(beta_energy, 95.0) + 1e-9
    return np.clip(1.0 - beta_energy / scale, 0.0, 1.5).astype(np.float32)


def _bucket_threshold_array(config: MonotonicStateMachineConfig) -> np.ndarray:
    return np.asarray(config.bucket_thresholds, dtype=np.float32)


def _configure_backend(
    b2,
    *,
    backend: str,
    build_dir: Path | None = None,
    compute_capability: float | None = None,
    cuda_runtime_version: float | None = None,
) -> None:
    if backend == "runtime":
        b2.start_scope()
        return

    if backend == "cuda_standalone":
        ensure_cuda_runtime_libraries()
    b2.set_device(backend, build_on_run=False, directory=None if build_dir is None else str(build_dir))
    if backend == "cuda_standalone":
        if compute_capability is not None:
            b2.prefs.devices.cuda_standalone.cuda_backend.detect_gpus = False
            b2.prefs.devices.cuda_standalone.cuda_backend.compute_capability = float(compute_capability)
        if cuda_runtime_version is not None:
            b2.prefs.devices.cuda_standalone.cuda_backend.cuda_runtime_version = float(cuda_runtime_version)


def _build_network(
    b2,
    *,
    n_steps: int,
    n_inputs: int,
    band_roles: Sequence[str],
    config: MonotonicStateMachineConfig,
    encoder_values: np.ndarray,
    quiet_values: np.ndarray,
    record_voltage: bool,
):
    b2.defaultclock.dt = config.dt_ms * b2.ms
    encoder_drive = b2.TimedArray(
        np.asarray(encoder_values, dtype=np.float32).T,
        dt=config.dt_ms * b2.ms,
        name="encoder_drive",
    )
    quiet_drive = b2.TimedArray(
        np.asarray(quiet_values, dtype=np.float32),
        dt=config.dt_ms * b2.ms,
        name="quiet_drive",
    )

    encoder = b2.NeuronGroup(
        n_inputs,
        encoder_equations(),
        threshold="v > threshold_param",
        reset="v = reset_level",
        refractory="refractory_period",
        method="euler",
        name="encoder",
        namespace={"encoder_drive": encoder_drive},
    )
    encoder.v = 0.0
    encoder.gain = config.encoder_gain
    encoder.bias = config.encoder_bias
    encoder.tau = config.encoder_tau_ms * b2.ms
    encoder.threshold_param = config.encoder_threshold
    encoder.reset_level = config.encoder_reset
    encoder.refractory_period = config.encoder_refractory_ms * b2.ms

    quiet = b2.NeuronGroup(
        1,
        quiet_equations(),
        threshold="v > threshold_param",
        reset="v = reset_level",
        refractory="refractory_period",
        method="euler",
        name="quiet",
        namespace={"quiet_drive": quiet_drive},
    )
    quiet.v = 0.0
    quiet.gain = config.quiet_gain
    quiet.bias = 0.0
    quiet.tau = config.quiet_tau_ms * b2.ms
    quiet.threshold_param = config.quiet_threshold
    quiet.reset_level = 0.0
    quiet.refractory_period = config.quiet_refractory_ms * b2.ms

    buckets = b2.NeuronGroup(
        len(config.bucket_thresholds_ms),
        bucket_equations(),
        threshold="v > theta",
        reset="v = 0",
        refractory="refractory_period",
        method="euler",
        name="buckets",
    )
    buckets.v = 0.0
    buckets.tau = config.bucket_tau_ms * b2.ms
    buckets.theta = _bucket_threshold_array(config)
    buckets.refractory_period = config.bucket_refractory_ms * b2.ms

    readout = b2.NeuronGroup(
        1,
        readout_equations(),
        threshold="v > theta",
        reset="v = 0",
        refractory="refractory_period",
        method="euler",
        name="readout",
    )
    readout.v = 0.0
    readout.tau = config.readout_tau_ms * b2.ms
    readout.theta = config.readout_threshold
    readout.refractory_period = config.readout_refractory_ms * b2.ms

    delay = config.synaptic_delay_ms * b2.ms
    beta_indices = [index for index, role in enumerate(band_roles) if role == "beta"]
    boundary_indices = [index for index, role in enumerate(band_roles) if role != "beta"]
    bucket_targets = np.arange(len(config.bucket_thresholds_ms))

    beta_syn = b2.Synapses(
        encoder,
        buckets,
        "w : 1 (constant)",
        on_pre=excitatory_on_pre(),
        delay=delay,
        name="beta_syn",
    )
    if beta_indices:
        # Only the entry bucket receives direct beta evidence; later buckets require
        # sustained upstream progression instead of a single brief jump.
        beta_syn.connect(i=beta_indices, j=0)
        beta_syn.w = config.input_weight

    boundary_syn = b2.Synapses(
        encoder,
        buckets,
        "w : 1 (constant)",
        on_pre=excitatory_on_pre(),
        delay=delay,
        name="boundary_syn",
    )
    if boundary_indices:
        boundary_syn.connect(
            i=np.repeat(boundary_indices, len(bucket_targets)),
            j=np.tile(bucket_targets, len(boundary_indices)),
        )
        boundary_syn.w = -config.boundary_inhibition_weight

    sustain_syn = b2.Synapses(
        buckets,
        buckets,
        "w : 1 (constant)",
        on_pre=excitatory_on_pre(),
        delay=0 * b2.ms,
        name="sustain_syn",
    )
    sustain_syn.connect(j="i")
    sustain_syn.w = config.sustain_weight

    forward_syn = b2.Synapses(
        buckets,
        buckets,
        "w : 1 (constant)",
        on_pre=excitatory_on_pre(),
        delay=delay,
        name="forward_syn",
    )
    if len(config.bucket_thresholds_ms) > 1:
        forward_syn.connect(i=np.arange(len(config.bucket_thresholds_ms) - 1), j=np.arange(1, len(config.bucket_thresholds_ms)))
        forward_syn.w = config.forward_weight

    reset_syn = b2.Synapses(
        quiet,
        buckets,
        on_pre=reset_on_pre(),
        delay=delay,
        name="reset_syn",
    )
    reset_syn.connect()

    readout_syn = b2.Synapses(
        buckets,
        readout,
        "w : 1 (constant)",
        on_pre=excitatory_on_pre(),
        delay=delay,
        name="readout_syn",
    )
    readout_syn.connect(i=np.arange(config.readout_bucket_index, len(config.bucket_thresholds_ms)), j=0)
    readout_syn.w = config.readout_weight

    reset_readout = b2.Synapses(
        quiet,
        readout,
        on_pre=reset_on_pre(),
        delay=delay,
        name="reset_readout_syn",
    )
    reset_readout.connect()

    encoder_mon = b2.SpikeMonitor(encoder, name="encoder_spikes")
    quiet_mon = b2.SpikeMonitor(quiet, name="quiet_spikes")
    bucket_mon = b2.SpikeMonitor(buckets, name="bucket_spikes")
    readout_mon = b2.SpikeMonitor(readout, name="readout_spikes")
    state_mon = b2.StateMonitor(buckets, "v", record=True, name="bucket_state") if record_voltage else None

    net = b2.Network(
        encoder,
        quiet,
        buckets,
        readout,
        beta_syn,
        boundary_syn,
        sustain_syn,
        forward_syn,
        reset_syn,
        readout_syn,
        reset_readout,
        encoder_mon,
        quiet_mon,
        bucket_mon,
        readout_mon,
    )
    if state_mon is not None:
        net.add(state_mon)

    duration = n_steps * config.dt_ms * b2.ms
    return {
        "network": net,
        "duration": duration,
        "encoder_drive": encoder_drive,
        "quiet_drive": quiet_drive,
        "encoder": encoder,
        "quiet": quiet,
        "buckets": buckets,
        "readout": readout,
        "beta_syn": beta_syn,
        "boundary_syn": boundary_syn,
        "sustain_syn": sustain_syn,
        "forward_syn": forward_syn,
        "readout_syn": readout_syn,
        "encoder_mon": encoder_mon,
        "quiet_mon": quiet_mon,
        "bucket_mon": bucket_mon,
        "readout_mon": readout_mon,
        "state_mon": state_mon,
    }


def _collect_result(b2, handles: dict, config: MonotonicStateMachineConfig, band_roles: Sequence[str]) -> StateMachineResult:
    state_mon = handles["state_mon"]
    bucket_voltages = None if state_mon is None else np.asarray(state_mon.v, dtype=np.float32)
    return StateMachineResult(
        duration_s=float(handles["duration"] / b2.second),
        encoder_spike_times_s=np.asarray(handles["encoder_mon"].t / b2.second, dtype=float),
        encoder_spike_indices=np.asarray(handles["encoder_mon"].i, dtype=int),
        quiet_spike_times_s=np.asarray(handles["quiet_mon"].t / b2.second, dtype=float),
        bucket_spike_times_s=np.asarray(handles["bucket_mon"].t / b2.second, dtype=float),
        bucket_spike_indices=np.asarray(handles["bucket_mon"].i, dtype=int),
        readout_spike_times_s=np.asarray(handles["readout_mon"].t / b2.second, dtype=float),
        bucket_voltages=bucket_voltages,
        bucket_thresholds_ms=config.bucket_thresholds_ms,
        band_roles=tuple(band_roles),
        readout_bucket_index=config.readout_bucket_index,
    )


def run_state_machine(
    encoder_currents: np.ndarray,
    band_roles: Sequence[str],
    config: MonotonicStateMachineConfig,
    *,
    backend: str = "runtime",
    quiet_drive: np.ndarray | None = None,
    record_voltage: bool = False,
    seed: int = 0,
    compute_capability: float | None = None,
    cuda_runtime_version: float | None = None,
) -> StateMachineResult:
    """Run one state-machine simulation, using runtime or standalone backends."""
    if backend != "runtime":
        project = StandaloneStateMachineProject(
            n_steps=encoder_currents.shape[-1],
            n_inputs=encoder_currents.shape[0],
            band_roles=band_roles,
            config=config,
            backend=backend,
            build_dir=Path("results/phase4_synthetic/standalone_build"),
            record_voltage=record_voltage,
            compute_capability=compute_capability,
            cuda_runtime_version=cuda_runtime_version,
        )
        return project.run(encoder_currents, quiet_drive=quiet_drive)

    b2 = _lazy_brian()
    b2.start_scope()
    b2.seed(seed)
    encoder_currents = np.asarray(encoder_currents, dtype=np.float32)
    if quiet_drive is None:
        quiet_drive = derive_quiet_drive(encoder_currents, band_roles)
    handles = _build_network(
        b2,
        n_steps=encoder_currents.shape[-1],
        n_inputs=encoder_currents.shape[0],
        band_roles=band_roles,
        config=config,
        encoder_values=encoder_currents,
        quiet_values=quiet_drive,
        record_voltage=record_voltage,
    )
    handles["network"].run(handles["duration"], namespace={})
    return _collect_result(b2, handles, config, band_roles)


class StandaloneStateMachineProject:
    """Compile once and reuse a standalone Brian2/Brian2CUDA state-machine project."""

    def __init__(
        self,
        *,
        n_steps: int,
        n_inputs: int,
        band_roles: Sequence[str],
        config: MonotonicStateMachineConfig,
        backend: str,
        build_dir: Path,
        record_voltage: bool = False,
        compute_capability: float | None = None,
        cuda_runtime_version: float | None = None,
    ) -> None:
        if backend == "runtime":
            raise ValueError("StandaloneStateMachineProject requires a standalone backend")
        self.band_roles = tuple(band_roles)
        self.config = config
        self.backend = backend
        self.build_dir = Path(build_dir)
        self.record_voltage = record_voltage
        if self.build_dir.exists():
            shutil.rmtree(self.build_dir)
        self.b2 = _lazy_brian(import_cuda=backend == "cuda_standalone")
        self.b2.device.reinit()
        self._configure = _configure_backend(
            self.b2,
            backend=backend,
            build_dir=self.build_dir,
            compute_capability=compute_capability,
            cuda_runtime_version=cuda_runtime_version,
        )
        placeholder_currents = np.zeros((n_inputs, n_steps), dtype=np.float32)
        placeholder_quiet = np.zeros(n_steps, dtype=np.float32)
        self.handles = _build_network(
            self.b2,
            n_steps=n_steps,
            n_inputs=n_inputs,
            band_roles=self.band_roles,
            config=config,
            encoder_values=placeholder_currents,
            quiet_values=placeholder_quiet,
            record_voltage=record_voltage,
        )
        self.handles["network"].run(self.handles["duration"], namespace={})
        self.b2.device.build(directory=str(self.build_dir), compile=True, run=False, with_output=False)

    def _run_args(
        self,
        encoder_currents: np.ndarray,
        quiet_drive: np.ndarray,
        overrides: dict | None = None,
    ) -> dict:
        b2 = self.b2
        config = self.config if not overrides else replace(self.config, **overrides)
        return {
            self.handles["encoder_drive"]: np.ascontiguousarray(
                np.asarray(encoder_currents, dtype=np.float32).T
            ),
            self.handles["quiet_drive"]: np.ascontiguousarray(
                np.asarray(quiet_drive, dtype=np.float32)
            ),
            self.handles["encoder"].gain: config.encoder_gain,
            self.handles["encoder"].bias: config.encoder_bias,
            self.handles["encoder"].tau: config.encoder_tau_ms * b2.ms,
            self.handles["encoder"].threshold_param: config.encoder_threshold,
            self.handles["encoder"].reset_level: config.encoder_reset,
            self.handles["encoder"].refractory_period: config.encoder_refractory_ms * b2.ms,
            self.handles["quiet"].gain: config.quiet_gain,
            self.handles["quiet"].tau: config.quiet_tau_ms * b2.ms,
            self.handles["quiet"].threshold_param: config.quiet_threshold,
            self.handles["quiet"].refractory_period: config.quiet_refractory_ms * b2.ms,
            self.handles["buckets"].tau: config.bucket_tau_ms * b2.ms,
            self.handles["buckets"].theta: _bucket_threshold_array(config),
            self.handles["beta_syn"].w: config.input_weight,
            self.handles["boundary_syn"].w: -config.boundary_inhibition_weight,
            self.handles["forward_syn"].w: config.forward_weight,
            self.handles["sustain_syn"].w: config.sustain_weight,
            self.handles["readout_syn"].w: config.readout_weight,
            self.handles["readout"].tau: config.readout_tau_ms * b2.ms,
            self.handles["readout"].theta: config.readout_threshold,
            self.handles["readout"].refractory_period: config.readout_refractory_ms * b2.ms,
        }

    def run(
        self,
        encoder_currents: np.ndarray,
        *,
        quiet_drive: np.ndarray | None = None,
        results_directory: Path | None = None,
        overrides: dict | None = None,
    ) -> StateMachineResult:
        if quiet_drive is None:
            quiet_drive = derive_quiet_drive(np.asarray(encoder_currents, dtype=np.float32), self.band_roles)
        self.b2.device.run(
            results_directory=None if results_directory is None else str(results_directory),
            with_output=False,
            run_args=self._run_args(encoder_currents, quiet_drive, overrides=overrides),
        )
        result_config = self.config if not overrides else replace(self.config, **overrides)
        return _collect_result(self.b2, self.handles, result_config, self.band_roles)


@dataclass(frozen=True)
class DurationBucketRunResult:
    """Outputs from the Phase 5 clustered duration-bucket state machine."""

    duration_s: float
    encoder_spike_times_s: np.ndarray
    encoder_spike_indices: np.ndarray
    quiet_spike_times_s: np.ndarray
    state_spike_times_s: np.ndarray
    state_spike_indices: np.ndarray
    readout_spike_times_s: np.ndarray
    occupancy: np.ndarray
    readout_trace: np.ndarray
    quiet_drive: np.ndarray
    encoder_currents: np.ndarray
    state_names: tuple[str, ...]
    bucket_thresholds_ms: tuple[float, ...]
    band_roles: tuple[str, ...]
    readout_bucket_index: int
    cluster_exc_size: int
    cluster_inh_size: int

    @property
    def bucket_spike_times_s(self) -> np.ndarray:
        mask = self.state_spike_indices > 0
        return self.state_spike_times_s[mask]

    @property
    def bucket_spike_indices(self) -> np.ndarray:
        mask = self.state_spike_indices > 0
        return self.state_spike_indices[mask] - 1


def aggregate_beta_evidence(
    encoder_currents: np.ndarray,
    band_roles: Sequence[str],
    *,
    mode: str = "mean",
) -> np.ndarray:
    """Aggregate beta-band evidence into one 1-D trace."""
    currents = np.asarray(encoder_currents, dtype=np.float32)
    beta_mask = np.array([role == "beta" for role in band_roles], dtype=bool)
    if currents.ndim != 2:
        raise ValueError(f"Expected encoder currents with shape (n_inputs, n_steps), got {currents.shape}")
    if not beta_mask.any():
        beta_values = currents
    else:
        beta_values = currents[beta_mask]

    normalized = np.clip(beta_values, 0.0, None)
    scale = np.percentile(normalized, 95.0) + 1e-9
    normalized = normalized / scale

    mode_normalized = mode.lower()
    if mode_normalized == "sum":
        aggregated = normalized.sum(axis=0) / max(1, normalized.shape[0])
    elif mode_normalized == "max":
        aggregated = normalized.max(axis=0)
    elif mode_normalized in {"top2_mean", "topk2_mean", "top_k_2_mean"}:
        k = max(1, min(2, normalized.shape[0]))
        if k == normalized.shape[0]:
            aggregated = normalized.mean(axis=0)
        else:
            aggregated = np.partition(normalized, normalized.shape[0] - k, axis=0)[-k:].mean(axis=0)
    else:
        aggregated = normalized.mean(axis=0)
    return np.asarray(aggregated, dtype=np.float32)


def prepare_phase5_entry_currents(
    encoder_currents: np.ndarray,
    band_roles: Sequence[str],
    *,
    mode: str = "raw",
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return model inputs after explicit beta aggregation before D0.

    ``raw`` preserves one encoder channel per input band. Aggregated modes collapse
    beta evidence to one pooled beta current while retaining non-beta channels for
    diagnostics and future boundary/reset aids. The pooled beta current is still
    routed only to D0 by the Phase 5 topology.
    """
    currents = np.asarray(encoder_currents, dtype=np.float32)
    if currents.ndim != 2:
        raise ValueError(f"Expected encoder currents with shape (n_inputs, n_steps), got {currents.shape}")
    if len(band_roles) != currents.shape[0]:
        raise ValueError(
            f"band_roles length {len(band_roles)} does not match current rows {currents.shape[0]}"
        )

    mode_normalized = mode.lower().strip()
    if mode_normalized in {"raw", "none"}:
        return currents, tuple(band_roles)

    beta_indices = [index for index, role in enumerate(band_roles) if role == "beta"]
    if not beta_indices:
        return currents, tuple(band_roles)
    beta_values = currents[beta_indices]

    if mode_normalized in {"sum"}:
        pooled = beta_values.sum(axis=0)
    elif mode_normalized in {"mean", "beta_mean", "pooled_mean", "channel_mean"}:
        pooled = beta_values.mean(axis=0)
    elif mode_normalized in {"max"}:
        pooled = beta_values.max(axis=0)
    elif mode_normalized in {"top2_mean", "topk2_mean", "top_k_2_mean"}:
        k = max(1, min(2, beta_values.shape[0]))
        if k == beta_values.shape[0]:
            pooled = beta_values.mean(axis=0)
        else:
            pooled = np.partition(beta_values, beta_values.shape[0] - k, axis=0)[-k:].mean(axis=0)
    else:
        raise ValueError(
            f"Unsupported evidence_aggregation_mode={mode!r}; "
            "expected one of raw, sum, mean, max, top2_mean"
        )

    non_beta_indices = [index for index, role in enumerate(band_roles) if role != "beta"]
    if non_beta_indices:
        prepared = np.vstack([pooled[np.newaxis, :], currents[non_beta_indices]])
        prepared_roles = ("beta",) + tuple(band_roles[index] for index in non_beta_indices)
    else:
        prepared = pooled[np.newaxis, :]
        prepared_roles = ("beta",)
    return np.asarray(prepared, dtype=np.float32), prepared_roles


def _entry_aggregation_divisor(
    n_beta_inputs: int,
    mode: str,
) -> float:
    mode_normalized = mode.lower().strip()
    if mode_normalized in {"raw", "none", "sum", "max"}:
        return 1.0
    if mode_normalized in {"mean", "beta_mean", "pooled_mean", "channel_mean"}:
        return float(max(n_beta_inputs, 1))
    if mode_normalized in {"top2_mean", "topk2_mean", "top_k_2_mean"}:
        return float(max(min(n_beta_inputs, 2), 1))
    raise ValueError(
        f"Unsupported evidence_aggregation_mode={mode!r}; "
        "expected one of raw, sum, mean, max, top2_mean"
    )


def phase5_entry_weight(
    config: DurationBucketClusterConfig,
    band_roles: Sequence[str],
) -> float:
    """Return the effective D0 input weight after entry aggregation scaling."""
    n_beta_inputs = int(sum(role == "beta" for role in band_roles))
    divisor = _entry_aggregation_divisor(n_beta_inputs, config.evidence_aggregation_mode)
    return float(config.input_weight / divisor)


def _exp_filter(samples: np.ndarray, tau_ms: float, dt_ms: float) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    if tau_ms <= 0.0:
        return samples
    alpha = float(np.exp(-dt_ms / tau_ms))
    out = np.zeros_like(samples, dtype=np.float32)
    if samples.ndim == 1:
        prev = 0.0
        for index, value in enumerate(samples):
            prev = alpha * prev + (1.0 - alpha) * float(value)
            out[index] = prev
        return out

    prev = np.zeros(samples.shape[0], dtype=np.float32)
    for index in range(samples.shape[1]):
        prev = alpha * prev + (1.0 - alpha) * samples[:, index]
        out[:, index] = prev
    return out


def _phase5_state_blocks(config: DurationBucketClusterConfig) -> dict[str, dict[str, np.ndarray]]:
    blocks: dict[str, dict[str, np.ndarray]] = {}
    for state_index, state_name in enumerate(config.state_names):
        exc_start = state_index * config.cluster_exc_size
        inh_start = state_index * config.cluster_inh_size
        blocks[state_name] = {
            "state_index": np.asarray([state_index], dtype=int),
            "exc": np.arange(exc_start, exc_start + config.cluster_exc_size, dtype=int),
            "inh": np.arange(inh_start, inh_start + config.cluster_inh_size, dtype=int),
        }
    return blocks


def _phase5_heterogeneity(n_units: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.standard_normal(n_units).astype(np.float32)
    values -= values.mean()
    std = float(values.std())
    if std > 0.0:
        values /= std
    values = np.clip(values, -1.75, 1.75)
    return values


def _phase5_threshold_bases(config: DurationBucketClusterConfig) -> np.ndarray:
    values = []
    for scale in config.state_threshold_scales:
        values.append(np.full(config.cluster_exc_size, config.threshold_base * scale, dtype=np.float32))
    return np.concatenate(values)


def _phase5_bias_values(config: DurationBucketClusterConfig) -> np.ndarray:
    values = []
    for state_name in config.state_names:
        bias = config.idle_bias if state_name == "D_idle" else 0.0
        values.append(np.full(config.cluster_exc_size, bias, dtype=np.float32))
    return np.concatenate(values)


def _phase5_spike_histogram(
    spike_times_s: np.ndarray,
    spike_indices: np.ndarray,
    *,
    n_states: int,
    n_steps: int,
    dt_ms: float,
    cluster_size: int,
) -> np.ndarray:
    counts = np.zeros((n_states, n_steps), dtype=np.float32)
    if len(spike_times_s) == 0:
        return counts
    sample_index = np.clip(
        np.round(np.asarray(spike_times_s, dtype=float) * 1000.0 / dt_ms).astype(int),
        0,
        n_steps - 1,
    )
    for state_index, step_index in zip(np.asarray(spike_indices, dtype=int), sample_index, strict=False):
        counts[state_index, step_index] += 1.0 / max(cluster_size, 1)
    return counts


def _phase5_collect_result(
    b2,
    handles: dict,
    *,
    config: DurationBucketClusterConfig,
    band_roles: Sequence[str],
    quiet_drive: np.ndarray,
    encoder_currents: np.ndarray,
    n_steps: int,
) -> DurationBucketRunResult:
    exc_indices = np.asarray(handles["exc_mon"].i, dtype=int)
    state_spike_indices = exc_indices // max(config.cluster_exc_size, 1)
    state_spike_times_s = np.asarray(handles["exc_mon"].t / b2.second, dtype=float)
    occupancy_counts = _phase5_spike_histogram(
        state_spike_times_s,
        state_spike_indices,
        n_states=len(config.state_names),
        n_steps=n_steps,
        dt_ms=config.dt_ms,
        cluster_size=config.cluster_exc_size,
    )
    occupancy = _exp_filter(occupancy_counts, config.occupancy_tau_ms, config.dt_ms)
    readout_counts = _phase5_spike_histogram(
        np.asarray(handles["readout_mon"].t / b2.second, dtype=float),
        np.zeros(len(handles["readout_mon"].t), dtype=int),
        n_states=1,
        n_steps=n_steps,
        dt_ms=config.dt_ms,
        cluster_size=1,
    )[0]
    readout_trace = _exp_filter(readout_counts, config.readout_tau_syn_ms, config.dt_ms)
    return DurationBucketRunResult(
        duration_s=float(handles["duration"] / b2.second),
        encoder_spike_times_s=np.asarray(handles["encoder_mon"].t / b2.second, dtype=float),
        encoder_spike_indices=np.asarray(handles["encoder_mon"].i, dtype=int),
        quiet_spike_times_s=np.asarray(handles["quiet_mon"].t / b2.second, dtype=float),
        state_spike_times_s=state_spike_times_s,
        state_spike_indices=state_spike_indices,
        readout_spike_times_s=np.asarray(handles["readout_mon"].t / b2.second, dtype=float),
        occupancy=occupancy,
        readout_trace=np.asarray(readout_trace, dtype=np.float32),
        quiet_drive=np.asarray(quiet_drive, dtype=np.float32),
        encoder_currents=np.asarray(encoder_currents, dtype=np.float32),
        state_names=tuple(config.state_names),
        bucket_thresholds_ms=tuple(config.bucket_thresholds_ms),
        band_roles=tuple(band_roles),
        readout_bucket_index=config.readout_bucket_index,
        cluster_exc_size=config.cluster_exc_size,
        cluster_inh_size=config.cluster_inh_size,
    )


def _build_duration_bucket_network(
    b2,
    *,
    n_steps: int,
    n_inputs: int,
    band_roles: Sequence[str],
    config: DurationBucketClusterConfig,
    encoder_values: np.ndarray,
    quiet_values: np.ndarray,
    seed: int,
):
    b2.defaultclock.dt = config.dt_ms * b2.ms
    encoder_drive = b2.TimedArray(
        np.asarray(encoder_values, dtype=np.float32).T,
        dt=config.dt_ms * b2.ms,
        name="phase5_encoder_drive",
    )
    quiet_drive = b2.TimedArray(
        np.asarray(quiet_values, dtype=np.float32),
        dt=config.dt_ms * b2.ms,
        name="phase5_quiet_drive",
    )

    encoder = b2.NeuronGroup(
        n_inputs,
        encoder_equations(),
        threshold="v > threshold_param",
        reset="v = reset_level",
        refractory="refractory_period",
        method="euler",
        name="phase5_encoder",
        namespace={"encoder_drive": encoder_drive},
    )
    encoder.v = 0.0
    encoder.gain = config.encoder_gain
    encoder.bias = config.encoder_bias
    encoder.tau = config.encoder_tau_ms * b2.ms
    encoder.threshold_param = config.encoder_threshold
    encoder.reset_level = config.encoder_reset
    encoder.refractory_period = config.encoder_refractory_ms * b2.ms

    quiet = b2.NeuronGroup(
        1,
        quiet_equations(),
        threshold="v > threshold_param",
        reset="v = reset_level",
        refractory="refractory_period",
        method="euler",
        name="phase5_quiet",
        namespace={"quiet_drive": quiet_drive},
    )
    quiet.v = 0.0
    quiet.gain = config.quiet_gain
    quiet.bias = 0.0
    quiet.tau = config.quiet_tau_ms * b2.ms
    quiet.threshold_param = config.quiet_threshold
    quiet.reset_level = 0.0
    quiet.refractory_period = max(config.quiet_refractory_ms, config.quiet_holdoff_ms) * b2.ms

    n_states = len(config.state_names)
    total_exc = n_states * config.cluster_exc_size
    total_inh = n_states * config.cluster_inh_size
    blocks = _phase5_state_blocks(config)
    exc = b2.NeuronGroup(
        total_exc,
        phase5_cluster_exc_equations(),
        threshold="v > theta",
        reset="v = reset_level",
        refractory="refractory_period",
        method="euler",
        name="phase5_exc",
    )
    exc.v = 0.0
    exc.bias = 0.0
    exc.tau_input = config.input_tau_syn_ms * b2.ms
    exc.tau_forward = config.feedforward_tau_syn_ms * b2.ms
    exc.tau_recurrent = config.recurrent_tau_syn_ms * b2.ms
    exc.tau_inh = config.inhibitory_tau_syn_ms * b2.ms
    exc.tau_reset = config.reset_tau_syn_ms * b2.ms
    exc.tau_m_base = config.neuron_tau_ms * b2.ms
    exc.threshold_base = _phase5_threshold_bases(config)
    exc.bias = _phase5_bias_values(config)
    exc.mismatch_scale = config.mismatch_cov_pct / 100.0
    exc.reset_level = config.reset_level
    exc.refractory_period = config.refractory_ms * b2.ms
    exc.hetero = _phase5_heterogeneity(total_exc, seed + 101)

    inh = b2.NeuronGroup(
        total_inh,
        phase5_cluster_inh_equations(),
        threshold="v > theta",
        reset="v = reset_level",
        refractory="refractory_period",
        method="euler",
        name="phase5_inh",
    )
    inh.v = 0.0
    inh.tau_exc = config.recurrent_tau_syn_ms * b2.ms
    inh.tau_reset = config.reset_tau_syn_ms * b2.ms
    inh.tau_m_base = config.inhibitory_tau_ms * b2.ms
    inh.threshold_base = config.threshold_base
    inh.mismatch_scale = config.mismatch_cov_pct / 100.0
    inh.reset_level = config.reset_level
    inh.refractory_period = config.refractory_ms * b2.ms
    inh.hetero = _phase5_heterogeneity(total_inh, seed + 202)

    readout = b2.NeuronGroup(
        1,
        phase5_readout_equations(),
        threshold="v > theta",
        reset="v = 0",
        refractory="refractory_period",
        method="euler",
        name="phase5_readout",
    )
    readout.v = 0.0
    readout.tau_state = config.readout_tau_syn_ms * b2.ms
    readout.tau_reset = config.reset_tau_syn_ms * b2.ms
    readout.tau_m = 10.0 * b2.ms
    readout.theta = config.readout_threshold
    readout.refractory_period = config.readout_refractory_ms * b2.ms

    beta_indices = [index for index, role in enumerate(band_roles) if role == "beta"]
    beta_syn = b2.Synapses(
        encoder,
        exc,
        "w : 1 (constant)",
        on_pre=phase5_input_on_pre(),
        delay=0 * b2.ms,
        name="phase5_beta_syn",
    )
    if beta_indices:
        entry_exc = blocks["D0"]["exc"]
        beta_syn.connect(i=np.repeat(beta_indices, len(entry_exc)), j=np.tile(entry_exc, len(beta_indices)))
        beta_syn.w = phase5_entry_weight(config, band_roles)

    idle_recurrent_syn = b2.Synapses(
        exc,
        exc,
        "w : 1 (constant)",
        on_pre=phase5_recurrent_on_pre(),
        delay=0 * b2.ms,
        name="phase5_idle_recurrent_syn",
    )
    idle_exc = blocks["D_idle"]["exc"]
    idle_recurrent_syn.connect(i=np.repeat(idle_exc, len(idle_exc)), j=np.tile(idle_exc, len(idle_exc)))
    idle_recurrent_syn.w = config.idle_recurrent_weight

    recurrent_syn = b2.Synapses(
        exc,
        exc,
        "w : 1 (constant)",
        on_pre=phase5_recurrent_on_pre(),
        delay=0 * b2.ms,
        name="phase5_recurrent_syn",
    )
    recurrent_src = []
    recurrent_dst = []
    for state_name in config.bucket_state_names:
        exc_idx = blocks[state_name]["exc"]
        recurrent_src.append(np.repeat(exc_idx, len(exc_idx)))
        recurrent_dst.append(np.tile(exc_idx, len(exc_idx)))
    if recurrent_src:
        recurrent_syn.connect(i=np.concatenate(recurrent_src), j=np.concatenate(recurrent_dst))
        recurrent_syn.w = config.recurrent_weight

    e_to_i_syn = b2.Synapses(
        exc,
        inh,
        "w : 1 (constant)",
        on_pre=phase5_exc_to_inh_on_pre(),
        delay=0 * b2.ms,
        name="phase5_e_to_i_syn",
    )
    i_to_e_syn = b2.Synapses(
        inh,
        exc,
        "w : 1 (constant)",
        on_pre=phase5_inhibitory_on_pre(),
        delay=0 * b2.ms,
        name="phase5_i_to_e_syn",
    )
    exc_conn_src = []
    exc_conn_dst = []
    inh_conn_src = []
    inh_conn_dst = []
    for state_name, state_block in blocks.items():
        exc_idx = state_block["exc"]
        inh_idx = state_block["inh"]
        exc_conn_src.append(np.repeat(exc_idx, len(inh_idx)))
        exc_conn_dst.append(np.tile(inh_idx, len(exc_idx)))
        inh_conn_src.append(np.repeat(inh_idx, len(exc_idx)))
        inh_conn_dst.append(np.tile(exc_idx, len(inh_idx)))
    e_to_i_syn.connect(i=np.concatenate(exc_conn_src), j=np.concatenate(exc_conn_dst))
    i_to_e_syn.connect(i=np.concatenate(inh_conn_src), j=np.concatenate(inh_conn_dst))
    e_to_i_syn.w = config.local_ei_weight
    i_to_e_syn.w = config.local_ie_weight

    forward_syn = b2.Synapses(
        exc,
        exc,
        "w : 1 (constant)",
        on_pre=phase5_forward_on_pre(),
        delay=0 * b2.ms,
        name="phase5_forward_syn",
    )
    forward_src = []
    forward_dst = []
    for current_state, next_state in zip(config.bucket_state_names, config.bucket_state_names[1:]):
        src_idx = blocks[current_state]["exc"]
        dst_idx = blocks[next_state]["exc"]
        forward_src.append(np.repeat(src_idx, len(dst_idx)))
        forward_dst.append(np.tile(dst_idx, len(src_idx)))
    if forward_src:
        forward_syn.connect(i=np.concatenate(forward_src), j=np.concatenate(forward_dst))
        forward_syn.w = config.feedforward_weight

    lateral_syn = b2.Synapses(
        exc,
        exc,
        "w : 1 (constant)",
        on_pre=phase5_inhibitory_on_pre(),
        delay=0 * b2.ms,
        name="phase5_lateral_syn",
    )
    lateral_src = []
    lateral_dst = []
    for later_offset, later_state in enumerate(config.state_names[1:], start=1):
        src_idx = blocks[later_state]["exc"]
        previous_states = config.state_names[:later_offset]
        dst_idx = np.concatenate([blocks[name]["exc"] for name in previous_states])
        lateral_src.append(np.repeat(src_idx, len(dst_idx)))
        lateral_dst.append(np.tile(dst_idx, len(src_idx)))
    if lateral_src:
        lateral_syn.connect(i=np.concatenate(lateral_src), j=np.concatenate(lateral_dst))
        lateral_syn.w = config.lateral_inhibition_weight

    reset_exc_syn = b2.Synapses(
        quiet,
        exc,
        "w : 1 (constant)",
        on_pre=phase5_reset_on_pre(),
        delay=0 * b2.ms,
        name="phase5_reset_exc_syn",
    )
    non_idle_exc = np.concatenate([blocks[name]["exc"] for name in config.bucket_state_names])
    reset_exc_syn.connect(i=np.zeros(len(non_idle_exc), dtype=int), j=non_idle_exc)
    reset_exc_syn.w = config.reset_weight

    reset_inh_syn = b2.Synapses(
        quiet,
        inh,
        "w : 1 (constant)",
        on_pre=phase5_reset_on_pre(),
        delay=0 * b2.ms,
        name="phase5_reset_inh_syn",
    )
    all_inh = np.arange(total_inh, dtype=int)
    reset_inh_syn.connect(i=np.zeros(len(all_inh), dtype=int), j=all_inh)
    reset_inh_syn.w = config.reset_weight

    readout_syn = b2.Synapses(
        exc,
        readout,
        "w : 1 (constant)",
        on_pre=phase5_readout_on_pre(),
        delay=0 * b2.ms,
        name="phase5_readout_syn",
    )
    readout_src = np.concatenate(
        [blocks[name]["exc"] for name in config.state_names[config.readout_bucket_index + 1 :]]
    )
    if len(readout_src):
        readout_syn.connect(i=readout_src, j=np.zeros(len(readout_src), dtype=int))
        readout_syn.w = config.readout_weight

    reset_readout_syn = b2.Synapses(
        quiet,
        readout,
        "w : 1 (constant)",
        on_pre=phase5_reset_on_pre(),
        delay=0 * b2.ms,
        name="phase5_reset_readout_syn",
    )
    reset_readout_syn.connect()
    reset_readout_syn.w = config.reset_weight

    encoder_mon = b2.SpikeMonitor(encoder, name="phase5_encoder_spikes")
    quiet_mon = b2.SpikeMonitor(quiet, name="phase5_quiet_spikes")
    exc_mon = b2.SpikeMonitor(exc, name="phase5_exc_spikes")
    readout_mon = b2.SpikeMonitor(readout, name="phase5_readout_spikes")

    net = b2.Network(
        encoder,
        quiet,
        exc,
        inh,
        readout,
        beta_syn,
        idle_recurrent_syn,
        recurrent_syn,
        e_to_i_syn,
        i_to_e_syn,
        forward_syn,
        lateral_syn,
        reset_exc_syn,
        reset_inh_syn,
        readout_syn,
        reset_readout_syn,
        encoder_mon,
        quiet_mon,
        exc_mon,
        readout_mon,
    )
    duration = n_steps * config.dt_ms * b2.ms
    return {
        "network": net,
        "duration": duration,
        "encoder_drive": encoder_drive,
        "quiet_drive": quiet_drive,
        "encoder": encoder,
        "quiet": quiet,
        "exc": exc,
        "inh": inh,
        "readout": readout,
        "beta_syn": beta_syn,
        "idle_recurrent_syn": idle_recurrent_syn,
        "recurrent_syn": recurrent_syn,
        "e_to_i_syn": e_to_i_syn,
        "i_to_e_syn": i_to_e_syn,
        "forward_syn": forward_syn,
        "lateral_syn": lateral_syn,
        "reset_exc_syn": reset_exc_syn,
        "reset_inh_syn": reset_inh_syn,
        "readout_syn": readout_syn,
        "reset_readout_syn": reset_readout_syn,
        "encoder_mon": encoder_mon,
        "quiet_mon": quiet_mon,
        "exc_mon": exc_mon,
        "readout_mon": readout_mon,
        "state_blocks": blocks,
    }


def run_duration_bucket_state_machine(
    encoder_currents: np.ndarray,
    band_roles: Sequence[str],
    config: DurationBucketClusterConfig,
    *,
    backend: str = "runtime",
    quiet_drive: np.ndarray | None = None,
    seed: int = 0,
    compute_capability: float | None = None,
    cuda_runtime_version: float | None = None,
) -> DurationBucketRunResult:
    """Run the clustered Phase 5 duration-bucket state machine."""
    encoder_currents = np.asarray(encoder_currents, dtype=np.float32)
    if encoder_currents.ndim != 2:
        raise ValueError(f"Expected encoder currents with shape (n_inputs, n_steps), got {encoder_currents.shape}")
    if quiet_drive is None:
        quiet_drive = derive_quiet_drive(encoder_currents, band_roles)
    quiet_drive = np.asarray(quiet_drive, dtype=np.float32)

    if backend != "runtime":
        project = StandaloneDurationBucketProject(
            n_steps=encoder_currents.shape[-1],
            n_inputs=encoder_currents.shape[0],
            band_roles=band_roles,
            config=config,
            backend=backend,
            build_dir=Path("results/phase5_synthetic/standalone_build"),
            seed=seed,
            compute_capability=compute_capability,
            cuda_runtime_version=cuda_runtime_version,
        )
        return project.run(encoder_currents, quiet_drive=quiet_drive)

    b2 = _lazy_brian()
    b2.start_scope()
    b2.prefs.codegen.target = "numpy"
    b2.seed(seed)
    handles = _build_duration_bucket_network(
        b2,
        n_steps=encoder_currents.shape[-1],
        n_inputs=encoder_currents.shape[0],
        band_roles=band_roles,
        config=config,
        encoder_values=encoder_currents,
        quiet_values=quiet_drive,
        seed=seed,
    )
    handles["network"].run(handles["duration"], namespace={})
    return _phase5_collect_result(
        b2,
        handles,
        config=config,
        band_roles=band_roles,
        quiet_drive=quiet_drive,
        encoder_currents=encoder_currents,
        n_steps=encoder_currents.shape[-1],
    )


class StandaloneDurationBucketProject:
    """Compile once and reuse a standalone Phase 5 duration-bucket project."""

    def __init__(
        self,
        *,
        n_steps: int,
        n_inputs: int,
        band_roles: Sequence[str],
        config: DurationBucketClusterConfig,
        backend: str,
        build_dir: Path,
        seed: int = 0,
        compute_capability: float | None = None,
        cuda_runtime_version: float | None = None,
    ) -> None:
        if backend == "runtime":
            raise ValueError("StandaloneDurationBucketProject requires a standalone backend")
        self.band_roles = tuple(band_roles)
        self.config = config
        self.backend = backend
        self.build_dir = Path(build_dir)
        self.seed = seed
        if self.build_dir.exists():
            shutil.rmtree(self.build_dir)
        self.b2 = _lazy_brian(import_cuda=backend == "cuda_standalone")
        self.b2.device.reinit()
        _configure_backend(
            self.b2,
            backend=backend,
            build_dir=self.build_dir,
            compute_capability=compute_capability,
            cuda_runtime_version=cuda_runtime_version,
        )
        placeholder_currents = np.zeros((n_inputs, n_steps), dtype=np.float32)
        placeholder_quiet = np.zeros(n_steps, dtype=np.float32)
        self.handles = _build_duration_bucket_network(
            self.b2,
            n_steps=n_steps,
            n_inputs=n_inputs,
            band_roles=self.band_roles,
            config=config,
            encoder_values=placeholder_currents,
            quiet_values=placeholder_quiet,
            seed=seed,
        )
        self.handles["network"].run(self.handles["duration"], namespace={})
        self.b2.device.build(directory=str(self.build_dir), compile=True, run=False, with_output=False)

    def _run_args(
        self,
        encoder_currents: np.ndarray,
        quiet_drive: np.ndarray,
        overrides: dict | None = None,
    ) -> dict:
        b2 = self.b2
        config = self.config if not overrides else replace(self.config, **overrides)
        return {
            self.handles["encoder_drive"]: np.ascontiguousarray(np.asarray(encoder_currents, dtype=np.float32).T),
            self.handles["quiet_drive"]: np.ascontiguousarray(np.asarray(quiet_drive, dtype=np.float32)),
            self.handles["encoder"].gain: config.encoder_gain,
            self.handles["encoder"].bias: config.encoder_bias,
            self.handles["encoder"].tau: config.encoder_tau_ms * b2.ms,
            self.handles["encoder"].threshold_param: config.encoder_threshold,
            self.handles["encoder"].reset_level: config.encoder_reset,
            self.handles["encoder"].refractory_period: config.encoder_refractory_ms * b2.ms,
            self.handles["quiet"].gain: config.quiet_gain,
            self.handles["quiet"].tau: config.quiet_tau_ms * b2.ms,
            self.handles["quiet"].threshold_param: config.quiet_threshold,
            self.handles["quiet"].refractory_period: max(config.quiet_refractory_ms, config.quiet_holdoff_ms) * b2.ms,
            self.handles["exc"].tau_input: config.input_tau_syn_ms * b2.ms,
            self.handles["exc"].tau_forward: config.feedforward_tau_syn_ms * b2.ms,
            self.handles["exc"].tau_recurrent: config.recurrent_tau_syn_ms * b2.ms,
            self.handles["exc"].tau_inh: config.inhibitory_tau_syn_ms * b2.ms,
            self.handles["exc"].tau_reset: config.reset_tau_syn_ms * b2.ms,
            self.handles["exc"].tau_m_base: config.neuron_tau_ms * b2.ms,
            self.handles["exc"].mismatch_scale: config.mismatch_cov_pct / 100.0,
            self.handles["exc"].reset_level: config.reset_level,
            self.handles["exc"].refractory_period: config.refractory_ms * b2.ms,
            self.handles["inh"].tau_exc: config.recurrent_tau_syn_ms * b2.ms,
            self.handles["inh"].tau_reset: config.reset_tau_syn_ms * b2.ms,
            self.handles["inh"].tau_m_base: config.inhibitory_tau_ms * b2.ms,
            self.handles["inh"].threshold_base: config.threshold_base,
            self.handles["inh"].mismatch_scale: config.mismatch_cov_pct / 100.0,
            self.handles["inh"].reset_level: config.reset_level,
            self.handles["inh"].refractory_period: config.refractory_ms * b2.ms,
            self.handles["beta_syn"].w: phase5_entry_weight(config, self.band_roles),
            self.handles["idle_recurrent_syn"].w: config.idle_recurrent_weight,
            self.handles["recurrent_syn"].w: config.recurrent_weight,
            self.handles["e_to_i_syn"].w: config.local_ei_weight,
            self.handles["i_to_e_syn"].w: config.local_ie_weight,
            self.handles["forward_syn"].w: config.feedforward_weight,
            self.handles["lateral_syn"].w: config.lateral_inhibition_weight,
            self.handles["reset_exc_syn"].w: config.reset_weight,
            self.handles["reset_inh_syn"].w: config.reset_weight,
            self.handles["readout_syn"].w: config.readout_weight,
            self.handles["reset_readout_syn"].w: config.reset_weight,
            self.handles["readout"].tau_state: config.readout_tau_syn_ms * b2.ms,
            self.handles["readout"].tau_reset: config.reset_tau_syn_ms * b2.ms,
            self.handles["readout"].theta: config.readout_threshold,
            self.handles["readout"].refractory_period: config.readout_refractory_ms * b2.ms,
        }

    def run(
        self,
        encoder_currents: np.ndarray,
        *,
        quiet_drive: np.ndarray | None = None,
        results_directory: Path | None = None,
        overrides: dict | None = None,
    ) -> DurationBucketRunResult:
        encoder_currents = np.asarray(encoder_currents, dtype=np.float32)
        if quiet_drive is None:
            quiet_drive = derive_quiet_drive(encoder_currents, self.band_roles)
        quiet_drive = np.asarray(quiet_drive, dtype=np.float32)
        self.b2.device.run(
            results_directory=None if results_directory is None else str(results_directory),
            with_output=False,
            run_args=self._run_args(encoder_currents, quiet_drive, overrides=overrides),
        )
        result_config = self.config if not overrides else replace(self.config, **overrides)
        return _phase5_collect_result(
            self.b2,
            self.handles,
            config=result_config,
            band_roles=self.band_roles,
            quiet_drive=quiet_drive,
            encoder_currents=encoder_currents,
            n_steps=encoder_currents.shape[-1],
        )
