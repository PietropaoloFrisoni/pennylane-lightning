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
Class implementation for state-tensor manipulation.
"""

import numpy as np
import pennylane as qml
from pennylane import DeviceError

from pennylane.wires import Wires


class LightningStateTensor:
    """Lightning state-tensor class.

    Interfaces with C++ python binding methods for state-tensor manipulation.
    """

    def __init__(self, num_wires, dtype=np.complex128, device_name="lightning.tensor"):
        self._num_wires = num_wires
        self._wires = Wires(range(num_wires))
        self._dtype = dtype

        if dtype not in [np.complex64, np.complex128]:  # pragma: no cover
            raise TypeError(f"Unsupported complex type: {dtype}")

        if device_name != "lightning.tensor":
            raise DeviceError(f'The device name "{device_name}" is not a valid option.')

        self._device_name = device_name
        # TODO: add binding to Lightning Managed state tensor C++ class.

    @property
    def dtype(self):
        """Returns the state tensor data type."""
        return self._dtype

    @property
    def device_name(self):
        """Returns the state tensor device name."""
        return self._device_name

    @property
    def wires(self):
        """All wires that can be addressed on this device"""
        return self._wires

    @property
    def num_wires(self):
        """Number of wires addressed on this device"""
        return self._num_wires

    @property
    def state_tensor(self):
        """Returns a handle to the state vector."""
        return self._qubit_state
