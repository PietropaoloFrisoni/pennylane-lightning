# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for the ``quimb`` interface.
"""

import itertools
import math

import numpy as np
import pennylane as qml
import pytest
import quimb.tensor as qtn
from conftest import LightningDevice  # tested device
from pennylane.devices import DefaultQubit
from pennylane.wires import Wires
from scipy.sparse import csr_matrix

from pennylane_lightning.lightning_tensor import LightningTensor

# if LightningDevice._CPP_BINARY_AVAILABLE:
#    pytest.skip("Device doesn't have C++ support yet.", allow_module_level=True)


# gates for which device support is tested
ops = {
    "Identity": qml.Identity(wires=[0]),
    "BlockEncode": qml.BlockEncode([[0.1, 0.2], [0.3, 0.4]], wires=[0, 1]),
    "CNOT": qml.CNOT(wires=[0, 1]),
    "CRX": qml.CRX(0, wires=[0, 1]),
    "CRY": qml.CRY(0, wires=[0, 1]),
    "CRZ": qml.CRZ(0, wires=[0, 1]),
    "CRot": qml.CRot(0, 0, 0, wires=[0, 1]),
    "CSWAP": qml.CSWAP(wires=[0, 1, 2]),
    "CZ": qml.CZ(wires=[0, 1]),
    "CCZ": qml.CCZ(wires=[0, 1, 2]),
    "CY": qml.CY(wires=[0, 1]),
    "CH": qml.CH(wires=[0, 1]),
    "DiagonalQubitUnitary": qml.DiagonalQubitUnitary(np.array([1, 1]), wires=[0]),
    "Hadamard": qml.Hadamard(wires=[0]),
    "MultiRZ": qml.MultiRZ(0, wires=[0]),
    "PauliX": qml.X(0),
    "PauliY": qml.Y(0),
    "PauliZ": qml.Z(0),
    "X": qml.X([0]),
    "Y": qml.Y([0]),
    "Z": qml.Z([0]),
    "PhaseShift": qml.PhaseShift(0, wires=[0]),
    "PCPhase": qml.PCPhase(0, 1, wires=[0, 1]),
    "ControlledPhaseShift": qml.ControlledPhaseShift(0, wires=[0, 1]),
    "CPhaseShift00": qml.CPhaseShift00(0, wires=[0, 1]),
    "CPhaseShift01": qml.CPhaseShift01(0, wires=[0, 1]),
    "CPhaseShift10": qml.CPhaseShift10(0, wires=[0, 1]),
    "QubitUnitary": qml.QubitUnitary(np.eye(2), wires=[0]),
    "SpecialUnitary": qml.SpecialUnitary(np.array([0.2, -0.1, 2.3]), wires=1),
    "ControlledQubitUnitary": qml.ControlledQubitUnitary(np.eye(2), control_wires=[1], wires=[0]),
    "MultiControlledX": qml.MultiControlledX(wires=[1, 2, 0]),
    "IntegerComparator": qml.IntegerComparator(1, geq=True, wires=[0, 1, 2]),
    "RX": qml.RX(0, wires=[0]),
    "RY": qml.RY(0, wires=[0]),
    "RZ": qml.RZ(0, wires=[0]),
    "Rot": qml.Rot(0, 0, 0, wires=[0]),
    "S": qml.S(wires=[0]),
    "Adjoint(S)": qml.adjoint(qml.S(wires=[0])),
    "SWAP": qml.SWAP(wires=[0, 1]),
    "ISWAP": qml.ISWAP(wires=[0, 1]),
    "PSWAP": qml.PSWAP(0, wires=[0, 1]),
    "ECR": qml.ECR(wires=[0, 1]),
    "Adjoint(ISWAP)": qml.adjoint(qml.ISWAP(wires=[0, 1])),
    "T": qml.T(wires=[0]),
    "Adjoint(T)": qml.adjoint(qml.T(wires=[0])),
    "SX": qml.SX(wires=[0]),
    "Adjoint(SX)": qml.adjoint(qml.SX(wires=[0])),
    "Toffoli": qml.Toffoli(wires=[0, 1, 2]),
    "QFT": qml.templates.QFT(wires=[0, 1, 2]),
    "IsingXX": qml.IsingXX(0, wires=[0, 1]),
    "IsingYY": qml.IsingYY(0, wires=[0, 1]),
    "IsingZZ": qml.IsingZZ(0, wires=[0, 1]),
    "IsingXY": qml.IsingXY(0, wires=[0, 1]),
    "SingleExcitation": qml.SingleExcitation(0, wires=[0, 1]),
    "SingleExcitationPlus": qml.SingleExcitationPlus(0, wires=[0, 1]),
    "SingleExcitationMinus": qml.SingleExcitationMinus(0, wires=[0, 1]),
    "DoubleExcitation": qml.DoubleExcitation(0, wires=[0, 1, 2, 3]),
    "DoubleExcitationPlus": qml.DoubleExcitationPlus(0, wires=[0, 1, 2, 3]),
    "DoubleExcitationMinus": qml.DoubleExcitationMinus(0, wires=[0, 1, 2, 3]),
    "QubitCarry": qml.QubitCarry(wires=[0, 1, 2, 3]),
    "QubitSum": qml.QubitSum(wires=[0, 1, 2]),
    "PauliRot": qml.PauliRot(0, "XXYY", wires=[0, 1, 2, 3]),
    "U1": qml.U1(0, wires=0),
    "U2": qml.U2(0, 0, wires=0),
    "U3": qml.U3(0, 0, 0, wires=0),
    "SISWAP": qml.SISWAP(wires=[0, 1]),
    "Adjoint(SISWAP)": qml.adjoint(qml.SISWAP(wires=[0, 1])),
    "OrbitalRotation": qml.OrbitalRotation(0, wires=[0, 1, 2, 3]),
    "FermionicSWAP": qml.FermionicSWAP(0, wires=[0, 1]),
    "GlobalPhase": qml.GlobalPhase(0.123, wires=[0, 1]),
}

all_ops = ops.keys()

# observables for which device support is tested
obs = {
    "Identity": qml.Identity(wires=[0]),
    "Hadamard": qml.Hadamard(wires=[0]),
    "Hermitian": qml.Hermitian(np.eye(2), wires=[0]),
    "PauliX": qml.PauliX(0),
    "PauliY": qml.PauliY(0),
    "PauliZ": qml.PauliZ(0),
    "X": qml.X(0),
    "Y": qml.Y(0),
    "Z": qml.Z(0),
    "Projector": [
        qml.Projector(np.array([1]), wires=[0]),
        qml.Projector(np.array([0, 1]), wires=[0]),
    ],
    "SparseHamiltonian": qml.SparseHamiltonian(csr_matrix(np.eye(8)), wires=[0, 1, 2]),
    "Hamiltonian": qml.Hamiltonian([1, 1], [qml.Z(0), qml.X(0)]),
    "LinearCombination": qml.ops.LinearCombination([1, 1], [qml.Z(0), qml.X(0)]),
}

all_obs = obs.keys()

# All qubit observables should be available to test in the device test suite
all_available_obs = qml.ops._qubit__obs__.copy()  # pylint: disable=protected-access
# Note that the identity is not technically a qubit observable
all_available_obs |= {"Identity"}

if not set(all_obs) == all_available_obs | {"LinearCombination"}:
    raise ValueError(
        "A qubit observable has been added that is not being tested in the "
        "device test suite. Please add to the obs dictionary in "
        "pennylane/devices/tests/test_measurements.py"
    )

# single qubit Hermitian observable
A = np.array([[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]])

obs_lst = [
    qml.X(0) @ qml.Y(1),
    qml.X(1) @ qml.Y(0),
    qml.X(1) @ qml.Z(2),
    qml.X(2) @ qml.Z(1),
    qml.Identity(wires=0) @ qml.Identity(wires=1) @ qml.Z(2),
    qml.Z(0) @ qml.X(1) @ qml.Y(2),
]

obs_permuted_lst = [
    qml.Y(1) @ qml.X(0),
    qml.Y(0) @ qml.X(1),
    qml.Z(2) @ qml.X(1),
    qml.Z(1) @ qml.X(2),
    qml.Z(2) @ qml.Identity(wires=0) @ qml.Identity(wires=1),
    qml.X(1) @ qml.Y(2) @ qml.Z(0),
]

label_maps = [[0, 1, 2], ["a", "b", "c"], ["beta", "alpha", "gamma"], [3, "beta", "a"]]


def sub_routine(label_map):
    """Quantum function to initalize state in tests"""
    qml.Hadamard(wires=label_map[0])
    qml.RX(0.12, wires=label_map[1])
    qml.RY(3.45, wires=label_map[2])


class TestSupportedObservables:
    """Test that the device can implement all observables that it supports."""

    @pytest.mark.parametrize("observable", all_obs)
    def test_supported_observables_can_be_implemented(self, observable):
        """Test that the device can implement all its supported observables."""
        dev = LightningTensor(
            wires=Wires(range(3)), backend="quimb", method="mps", c_dtype=np.complex64
        )

        if dev.shots and observable == "SparseHamiltonian":
            pytest.skip("SparseHamiltonian only supported in analytic mode")

        tape = qml.tape.QuantumScript(
            [qml.PauliX(0)],
            [qml.expval(obs[observable])],
        )
        result = dev.execute(circuits=tape)

        print(result)
        print(type(result))

        if observable == "Projector":
            for o in obs[observable]:
                assert isinstance(result, (float, np.ndarray))
        else:
            assert isinstance(result, (float, np.ndarray))

    # def test_tensor_observables_can_be_implemented(self, device_kwargs):
    #    """Test that the device can implement a simple tensor observable.
    #    This test is skipped for devices that do not support tensor observables."""
    #    device_kwargs["wires"] = 2
    #    dev = qml.device(**device_kwargs)
    #    supports_tensor = isinstance(dev, qml.devices.Device) or (
    #        "supports_tensor_observables" in dev.capabilities()
    #        and dev.capabilities()["supports_tensor_observables"]
    #    )
    #    if not supports_tensor:
    #        pytest.skip("Device does not support tensor observables.")

    #    @qml.qnode(dev)
    #    def circuit():
    #        qml.PauliX(0)
    #        return qml.expval(qml.Identity(wires=0) @ qml.Identity(wires=1))

    #    assert isinstance(circuit(), (float, np.ndarray))


THETA = np.linspace(0.11, 1, 3)
PHI = np.linspace(0.32, 1, 3)


@pytest.mark.parametrize("backend", ["quimb"])
@pytest.mark.parametrize("method", ["mps"])
class TestQuimbMPS:
    """Tests for the MPS method."""

    @pytest.mark.parametrize("num_wires", [None, 4])
    @pytest.mark.parametrize("c_dtype", [np.complex64, np.complex128])
    def test_device_init(self, num_wires, c_dtype, backend, method):
        """Test the class initialization and returned properties."""

        wires = Wires(range(num_wires)) if num_wires else None
        dev = LightningTensor(wires=wires, backend=backend, method=method, c_dtype=c_dtype)
        assert isinstance(dev._interface.state, qtn.MatrixProductState)
        assert isinstance(dev._interface.state_to_array(), np.ndarray)

        program, config = dev.preprocess()
        assert config.device_options["backend"] == backend
        assert config.device_options["method"] == method


class TestSupportedGates:
    """Test that the device can implement all gates that it claims to support."""

    @pytest.mark.parametrize("operation", all_ops)
    def test_supported_gates_can_be_implemented(self, operation):
        """Test that the device can implement all its supported gates."""

        dev = LightningTensor(
            wires=Wires(range(4)), backend="quimb", method="mps", c_dtype=np.complex64
        )

        # if isinstance(dev, qml.Device):
        #    if operation not in dev.operations:
        #        pytest.skip("operation not supported.")
        # else:
        #    if ops[operation].name == "QubitDensityMatrix":
        #        prog = dev.preprocess()[0]
        #        tape = qml.tape.QuantumScript([ops[operation]])
        #        try:
        #            prog((tape,))
        #        except qml.DeviceError:
        #            pytest.skip("operation not supported on the device")

        if not ops[operation].has_matrix:
            print(ops[operation])
            assert False

        tape = qml.tape.QuantumScript(
            [ops[operation]],
            [qml.expval(qml.Identity(wires=0))],
        )

        result = dev.execute(circuits=tape)

        assert np.allclose(result, 1.0)
