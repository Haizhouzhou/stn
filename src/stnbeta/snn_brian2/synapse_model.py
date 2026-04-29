"""Brian2 synapse snippets for the Phase 4 and Phase 5 state-machine benchmarks."""

from __future__ import annotations


def excitatory_on_pre() -> str:
    return "v_post += w"


def reset_on_pre() -> str:
    return "v_post = 0"


def phase5_input_on_pre() -> str:
    return "g_input_post += w"


def phase5_forward_on_pre() -> str:
    return "g_forward_post += w"


def phase5_recurrent_on_pre() -> str:
    return "g_recurrent_post += w"


def phase5_inhibitory_on_pre(target_var: str = "g_inh_post") -> str:
    return f"{target_var} += w"


def phase5_exc_to_inh_on_pre() -> str:
    return "g_exc_post += w"


def phase5_reset_on_pre(target_var: str = "g_reset_post") -> str:
    return f"{target_var} += w"


def phase5_readout_on_pre() -> str:
    return "g_state_post += w"
