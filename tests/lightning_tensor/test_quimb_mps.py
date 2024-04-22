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


@pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
class TestExpval:
    """Test expectation value calculations."""

    def test_identity(self, theta, phi):
        """Tests applying identities."""

        ops = [
            qml.Identity(0),
            qml.Identity((0, 1)),
            qml.Identity((1, 2)),
            qml.RX(theta, 0),
            qml.RX(phi, 1),
        ]
        measurements = [qml.expval(qml.PauliZ(0))]
        tape = qml.tape.QuantumScript(ops, measurements)

        dev = LightningTensor(wires=tape.wires, backend="quimb", method="mps", c_dtype=np.complex64)
        result = dev.execute(circuits=tape)
        expected = np.cos(theta)
        assert np.allclose(result, expected)

    def test_identity_expectation(self, theta, phi):
        """Tests identity expectations."""

        ops = [qml.RX(theta, wires=[0]), qml.RX(phi, wires=[1]), qml.CNOT(wires=[0, 1])]
        measurements = [
            qml.expval(qml.Identity(wires=[0])),
            qml.expval(qml.Identity(wires=[1])),
        ]
        tape = qml.tape.QuantumScript(ops, measurements)

        dev = LightningTensor(wires=tape.wires, backend="quimb", method="mps", c_dtype=np.complex64)
        result = dev.execute(circuits=tape)
        expected = 1.0
        assert np.allclose(result, expected)

    def test_multi_wire_identity_expectation(self, theta, phi):
        """Tests multi-wire identity."""

        ops = [qml.RX(theta, wires=[0]), qml.RX(phi, wires=[1]), qml.CNOT(wires=[0, 1])]
        measurements = [qml.expval(qml.Identity(wires=[0, 1]))]
        tape = qml.tape.QuantumScript(ops, measurements)

        dev = LightningTensor(wires=tape.wires, backend="quimb", method="mps", c_dtype=np.complex64)
        result = dev.execute(circuits=tape)
        expected = 1.0
        assert np.allclose(result, expected)

    @pytest.mark.parametrize(
        "Obs, Op, expected_fn",
        [
            (
                [qml.PauliX(wires=[0]), qml.PauliX(wires=[1])],
                qml.RY,
                lambda theta, phi: np.array([np.sin(theta) * np.sin(phi), np.sin(phi)]),
            ),
            (
                [qml.PauliY(wires=[0]), qml.PauliY(wires=[1])],
                qml.RX,
                lambda theta, phi: np.array([0, -np.cos(theta) * np.sin(phi)]),
            ),
            (
                [qml.PauliZ(wires=[0]), qml.PauliZ(wires=[1])],
                qml.RX,
                lambda theta, phi: np.array([np.cos(theta), np.cos(theta) * np.cos(phi)]),
            ),
            (
                [qml.Hadamard(wires=[0]), qml.Hadamard(wires=[1])],
                qml.RY,
                lambda theta, phi: np.array(
                    [
                        np.sin(theta) * np.sin(phi) + np.cos(theta),
                        np.cos(theta) * np.cos(phi) + np.sin(phi),
                    ]
                )
                / np.sqrt(2),
            ),
        ],
    )
    def test_single_wire_observables_expectation(self, Obs, Op, expected_fn, theta, phi):
        """Test that expectation values for single wire observables are correct."""

        tape = qml.tape.QuantumScript(
            [Op(theta, wires=[0]), Op(phi, wires=[1]), qml.CNOT(wires=[0, 1])],
            [qml.expval(Obs[0]), qml.expval(Obs[1])],
        )

        dev = LightningTensor(wires=tape.wires, backend="quimb", method="mps", c_dtype=np.complex64)
        result = dev.execute(circuits=tape)
        expected = expected_fn(theta, phi)

        assert np.allclose(result, expected)


class TestExpvalHamiltonian:
    """Tests expval for Hamiltonians"""

    @pytest.mark.parametrize(
        "obs, coeffs, expected",
        [
            ([qml.PauliX(0) @ qml.PauliZ(1)], [1.0], 0.0),
            ([qml.PauliZ(0) @ qml.PauliZ(1)], [1.0], math.cos(0.4) * math.cos(-0.2)),
            (
                [
                    qml.PauliX(0) @ qml.PauliZ(1),
                    qml.Hermitian(
                        [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 3.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0, 1.0],
                            [0.0, 0.0, 1.0, -2.0],
                        ],
                        wires=[0, 1],
                    ),
                ],
                [0.3, 1.0],
                0.9319728930156066,
            ),
        ],
    )
    def test_expval_hamiltonian(self, obs, coeffs, expected):
        """Test expval with Hamiltonian."""

        tape = qml.tape.QuantumScript(
            [qml.RX(0.4, wires=[0]), qml.RY(-0.2, wires=[1])],
            [qml.expval(qml.Hamiltonian(coeffs, obs))],
        )

        dev = LightningTensor(wires=tape.wires, backend="quimb", method="mps", c_dtype=np.complex64)
        result = dev.execute(circuits=tape)

        assert np.allclose(result, expected, atol=1.0e-8)


@pytest.mark.parametrize("phi", PHI)
class TestExpOperatorArithmetic:
    """Test integration with SProd, Prod, and Sum."""

    def test_sprod(self, phi, tol):
        """Test the `SProd` class."""
        tape = qml.tape.QuantumScript(
            [qml.RX(phi, wires=0)],
            [qml.expval(qml.s_prod(0.5, qml.PauliZ(0)))],
        )
        dev = LightningTensor(wires=tape.wires, backend="quimb", method="mps", c_dtype=np.complex64)
        result = dev.execute(circuits=tape)
        expected = 0.5 * np.cos(phi)

        assert np.allclose(result, expected, tol)

    def test_prod(self, phi, tol):
        """Test the `Prod` class."""
        tape = qml.tape.QuantumScript(
            [qml.RX(phi, wires=0), qml.Hadamard(1), qml.PauliZ(1)],
            [qml.expval(qml.prod(qml.PauliZ(0), qml.PauliX(1)))],
        )
        dev = LightningTensor(wires=tape.wires, backend="quimb", method="mps", c_dtype=np.complex64)
        result = dev.execute(circuits=tape)
        expected = -np.cos(phi)

        assert np.allclose(result, expected, tol)

    @pytest.mark.parametrize("theta", THETA)
    def test_sum(self, phi, theta, tol):
        """Test the `Sum` class."""
        tape = qml.tape.QuantumScript(
            [qml.RX(phi, wires=0), qml.RY(theta, wires=1)],
            [qml.expval(qml.sum(qml.PauliZ(0), qml.PauliX(1)))],
        )
        dev = LightningTensor(wires=tape.wires, backend="quimb", method="mps", c_dtype=np.complex64)
        result = dev.execute(circuits=tape)
        expected = np.cos(phi) + np.sin(theta)

        assert np.allclose(result, expected, tol)


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
