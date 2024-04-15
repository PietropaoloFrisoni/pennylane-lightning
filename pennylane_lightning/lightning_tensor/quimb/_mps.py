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
from pennylane.tape import QuantumTape
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

        # TODO: allows users to specify initial state
        self._circuit = qtn.CircuitMPS(psi0=self._set_initial_mps())

    @property
    def state(self):
        """Current MPS handled by the device."""
        return self._circuit.psi

    def state_to_array(self, digits: int = 5):
        """Contract the MPS into a dense array."""
        return self._circuit.to_dense().round(digits)

    def _set_initial_mps(self):
        r"""
        Returns an initial state to :math:`\ket{0}`.

        Returns:
            array: The initial state of a circuit.
        """

        return qtn.MPS_computational_state(
            "0" * max(1, self._num_wires),
            dtype=self._dtype.__name__,
            tags=[str(l) for l in self._wires.labels],
        )

    def execute(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ) -> Result_or_ResultBatch:
        """
        ...
        """

        print(
            f"LIGHTNING TENSOR execute called with:\nexecution_config={execution_config}\ncircuits={circuits}\n"
        )

        results = []
        for circuit in circuits:
            circuit = circuit.map_to_standard_wires()
            results.append(self.simulate(circuit))

        print(results)

        return tuple(results)

    def simulate(self, tape: qml.tape.QuantumScript) -> Result:
        """Simulate a single quantum script.

        Args:
            tape (QuantumScript): The single circuit to simulate

        Returns:
            tuple(TensorLike): The results of the simulation

        # TODO: understand
        Note that this function can return measurements for non-commuting observables simultaneously.

        # TODO: understand
        It does currently not support sampling or observables without diagonalizing gates.

        This function assumes that all operations provide matrices.
        """

        if set(tape.wires) != set(range(tape.num_wires)):
            print("La condizione si e' verificata")
            wire_map = {w: i for i, w in enumerate(tape.wires)}
            tape = qml.map_wires(tape, wire_map)

        for op in tape.operations:

            print(self._circuit.to_dense())

            print(f"Applying {op}")

            self._circuit.apply_gate(
                op.matrix(), *op.wires, contract=False, parametrize=None
            )

            print(self._circuit.to_dense())

        def measure(measurementprocess: MeasurementProcess):
            """Apply a measurement to state when the measurement process has an observable with diagonalizing gates.

            Args:
                measurementprocess (StateMeasurement): measurement to apply to the state
                state (TensorLike): state to apply the measurement to

            Returns:
                TensorLike: the result of the measurement
            """
            fs_opts = {
                "simplify_sequence": "ADCRS",
                "simplify_atol": 0.0,
            }

            obs = measurementprocess.obs

            return np.real(
                self._circuit.local_expectation(
                    obs.matrix(), tuple(obs.wires), **fs_opts
                )
            )

        return tuple(measure(mp) for mp in tape.measurements)
