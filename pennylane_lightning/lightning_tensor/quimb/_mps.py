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
Class implementation for MPS manipulation based on the `quimb` Python package.
"""


import quimb.tensor as qtn
from pennylane.wires import Wires

from typing import Callable, Sequence, Union

import pennylane as qml
from pennylane import numpy as np
from pennylane.devices import DefaultExecutionConfig, ExecutionConfig
from pennylane.tape import QuantumTape, QuantumScript
from pennylane.typing import Result, ResultBatch

from pennylane.measurements import (
    MeasurementProcess,
)

Result_or_ResultBatch = Union[Result, ResultBatch]
QuantumTapeBatch = Sequence[QuantumTape]
QuantumTape_or_Batch = Union[QuantumTape, QuantumTapeBatch]
PostprocessingFn = Callable[[ResultBatch], Result_or_ResultBatch]

# TODO: understand if supporting all operations and observables is feasible for the first release
# I comment the following lines since otherwise Codecov complaints

# _operations = frozenset({})
# The set of supported operations.

# _observables = frozenset({})
# The set of supported observables.


class QuimbMPS:
    """Quimb MPS class.

    Interfaces with `quimb` for MPS manipulation.
    """

    def __init__(self, num_wires, dtype=np.complex128):

        if dtype not in [np.complex64, np.complex128]:  # pragma: no cover
            raise TypeError(f"Unsupported complex type: {dtype}")

        self._num_wires = num_wires
        self._wires = Wires(range(num_wires))
        self._dtype = dtype
        self._circuitMPS = qtn.CircuitMPS(psi0=self._initial_mps())
        self._verbosity = True

    @property
    def state(self):
        """Current MPS handled by the interface."""
        return self._circuitMPS.psi

    def _reset_state(self):
        """Reset the MPS."""
        if self._verbosity:
            print("LOG: resetting the MPS")
        self._circuitMPS = qtn.CircuitMPS(psi0=self._initial_mps())

    def state_to_array(self, digits: int = 5):
        """Contract the MPS into a dense array."""
        return self._circuitMPS.to_dense().round(digits)

    def _initial_mps(self):
        r"""
        Returns an initial state to :math:`\ket{0}`.

        Returns:
            array: The initial MPS of a circuit.
        """

        return qtn.MPS_computational_state(
            "0" * max(1, self._num_wires),
            dtype=self._dtype.__name__,
            tags=[str(l) for l in self._wires.labels],
        )

    # pylint: disable=unused-argument
    def execute(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ) -> Result_or_ResultBatch:
        """Execute a circuit or a batch of circuits and turn it into results.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the quantum circuits to be executed
            execution_config (ExecutionConfig): a datastructure with additional information required for execution

        Returns:
            TensorLike, tuple[TensorLike], tuple[tuple[TensorLike]]: A numeric result of the computation.
        """

        if self._verbosity:
            print(f"LOG: execute called with:\ncircuits={circuits}\n")

        results = []
        for circuit in circuits:
            circuit = circuit.map_to_standard_wires()
            results.append(self._simulate(circuit))

        results = tuple(results)

        if self._verbosity:
            print(f"\nLOG: execute results={results}\n")
            print(f"LOG: MPS after execution:\n{self._circuitMPS.psi}")

        return results

    def _simulate(self, circuit: QuantumScript) -> Result:
        """Simulate a single quantum script. This function assumes that all operations provide matrices.

        Args:
            circuit (QuantumScript): The single circuit to simulate.

        Returns:
            Tuple[TensorLike]: The results of the simulation.

        """

        self._reset_state()

        ##############################################################
        ### PART 1: Applying operations
        ##############################################################

        for op in circuit.operations:
            self._apply_operation(op)

        ##############################################################
        ### PART 2: Measurements
        ##############################################################

        if len(circuit.measurements) == 1:
            return self._measure(circuit.measurements[0])

        return tuple(self._measure(mp) for mp in circuit.measurements)

    def _measure(self, measurementprocess: MeasurementProcess):
        """Measure the expectation value over the MPS.

        Args:
            measurementprocess (MeasurementProcess): measurement to apply to the state.

        Returns:
            TensorLike: the result of the measurement.
        """

        obs = measurementprocess.obs

        if self._verbosity:
            print(f"\nLOG: measuring the expval of obs {obs}...")

        return np.real(
            self._circuitMPS.local_expectation(
                G=obs.matrix(),
                where=tuple(obs.wires),
                dtype=self._dtype.__name__,
                simplify_sequence="ADCRS",
                simplify_atol=0.0,
            )
        )

    def _apply_operation(self, op: qml.operation.Operator):
        """Apply a single operator to the circuit, keeping the state always in a MPS form.

        Args:
            op (Operator): The operation to apply.
        """

        if self._verbosity:
            print(f"\nLOG: applying {op} to the circuit...")

        # TODO: investigate in `quimb` how to pass parameters required by PRD (cutoff, max_bond, etc.)
        self._circuitMPS.apply_gate(
            op.matrix(), *op.wires, contract="swap+split", parametrize=None
        )

        if self._verbosity:
            print(f"LOG: MPS after operation:\n{self._circuitMPS.psi}")
