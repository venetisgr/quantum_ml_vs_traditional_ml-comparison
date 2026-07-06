"""Model registry: maps config `model:` names to build functions.

E0 registers the minimal classical set only; E1 completes the classical zoo,
and E2/E3/E5 add quantum-kernel, VQC, and hybrid entries here.
"""
from __future__ import annotations

from models.classical import SPECS as _CLASSICAL_SPECS
from models.classical import ModelSpec

SPECS: dict[str, ModelSpec] = {**_CLASSICAL_SPECS}


def available_models() -> list[str]:
    return sorted(SPECS)


def get_spec(name: str) -> ModelSpec:
    try:
        return SPECS[name]
    except KeyError:
        raise KeyError(
            f"Unknown model '{name}'. Available: {available_models()}"
        ) from None
