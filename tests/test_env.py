"""Environment sanity: the pinned stack (environment.yml) imports and is wired up.

Import-only for the quantum libraries — E0 ships no quantum code. The single
device-construction check validates the compiled lightning.qubit simulator
that every later experiment depends on; it builds no circuit.
"""
import importlib
import sys

import pytest

PACKAGES = [
    "numpy",
    "scipy",
    "sklearn",
    "pandas",
    "matplotlib",
    "yaml",
    "torch",
    "xgboost",
    "optuna",
    "pennylane",
    "pennylane_qiskit",  # needed for IBM hardware in E8; verified against qiskit 2.x
    "qiskit",
    "qiskit_machine_learning",
    "qiskit_aer",
    "qiskit_ibm_runtime",
]


@pytest.mark.parametrize("pkg", PACKAGES)
def test_import(pkg):
    importlib.import_module(pkg)


def test_python_version():
    assert sys.version_info >= (3, 11)


def test_qiskit_major_version_is_2():
    import qiskit

    assert qiskit.__version__.split(".")[0] == "2"


def test_lightning_qubit_device_constructs():
    import pennylane as qml

    dev = qml.device("lightning.qubit", wires=2)
    assert dev.name == "lightning.qubit"
