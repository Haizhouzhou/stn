"""Brian2 equation strings for the Phase 4 and Phase 5 state-machine benchmarks."""

from __future__ import annotations


def encoder_equations(input_name: str = "encoder_drive") -> str:
    return f"""
    dv/dt = (-v + gain * {input_name}(t, i) + bias) / tau : 1 (unless refractory)
    gain : 1 (shared, constant)
    bias : 1 (shared, constant)
    tau : second (shared, constant)
    threshold_param : 1 (shared, constant)
    reset_level : 1 (shared, constant)
    refractory_period : second (shared, constant)
    """


def quiet_equations(input_name: str = "quiet_drive") -> str:
    return f"""
    dv/dt = (-v + gain * {input_name}(t) + bias) / tau : 1 (unless refractory)
    gain : 1 (shared, constant)
    bias : 1 (shared, constant)
    tau : second (shared, constant)
    threshold_param : 1 (shared, constant)
    reset_level : 1 (shared, constant)
    refractory_period : second (shared, constant)
    """


def bucket_equations() -> str:
    return """
    dv/dt = -v / tau : 1 (unless refractory)
    tau : second (shared, constant)
    theta : 1 (constant)
    refractory_period : second (shared, constant)
    """


def readout_equations() -> str:
    return """
    dv/dt = -v / tau : 1 (unless refractory)
    tau : second (shared, constant)
    theta : 1 (shared, constant)
    refractory_period : second (shared, constant)
    """


def phase5_cluster_exc_equations() -> str:
    return """
    dv/dt = (
        -v
        + bias
        + g_input
        + g_forward
        + g_recurrent
        - g_inh
        - g_reset
    ) / tau_m : 1 (unless refractory)
    dg_input/dt = -g_input / tau_input : 1
    dg_forward/dt = -g_forward / tau_forward : 1
    dg_recurrent/dt = -g_recurrent / tau_recurrent : 1
    dg_inh/dt = -g_inh / tau_inh : 1
    dg_reset/dt = -g_reset / tau_reset : 1
    bias : 1
    tau_input : second (shared, constant)
    tau_forward : second (shared, constant)
    tau_recurrent : second (shared, constant)
    tau_inh : second (shared, constant)
    tau_reset : second (shared, constant)
    tau_m_base : second (shared, constant)
    threshold_base : 1 (constant)
    mismatch_scale : 1 (shared, constant)
    hetero : 1 (constant)
    tau_m = tau_m_base * clip(1.0 + 0.5 * mismatch_scale * hetero, 0.4, 2.5) : second
    theta = threshold_base * clip(1.0 + mismatch_scale * hetero, 0.4, 2.5) : 1
    reset_level : 1 (shared, constant)
    refractory_period : second (shared, constant)
    """


def phase5_cluster_inh_equations() -> str:
    return """
    dv/dt = (-v + g_exc - g_reset) / tau_m : 1 (unless refractory)
    dg_exc/dt = -g_exc / tau_exc : 1
    dg_reset/dt = -g_reset / tau_reset : 1
    tau_exc : second (shared, constant)
    tau_reset : second (shared, constant)
    tau_m_base : second (shared, constant)
    threshold_base : 1 (shared, constant)
    mismatch_scale : 1 (shared, constant)
    hetero : 1 (constant)
    tau_m = tau_m_base * clip(1.0 + 0.5 * mismatch_scale * hetero, 0.4, 2.5) : second
    theta = threshold_base * clip(1.0 + mismatch_scale * hetero, 0.4, 2.5) : 1
    reset_level : 1 (shared, constant)
    refractory_period : second (shared, constant)
    """


def phase5_readout_equations() -> str:
    return """
    dv/dt = (-v + g_state - g_reset) / tau_m : 1 (unless refractory)
    dg_state/dt = -g_state / tau_state : 1
    dg_reset/dt = -g_reset / tau_reset : 1
    tau_state : second (shared, constant)
    tau_reset : second (shared, constant)
    tau_m : second (shared, constant)
    theta : 1 (shared, constant)
    refractory_period : second (shared, constant)
    """
