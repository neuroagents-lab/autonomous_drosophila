"""Detect worlds whose physics went non-finite, without downloading the batched state.

Invalid worlds are ended (``failure_reason="nonfinite_physics"``) and reset by
``FlyGymWarpVecEnv``; a world that is invalid again right after its reset raises.
"""

import numpy as np
import warp as wp


@wp.kernel
def _nonfinite_rows(
    first: wp.array2d(dtype=wp.float32),
    second: wp.array2d(dtype=wp.float32),
    invalid: wp.array(dtype=wp.bool),
):
    row = wp.tid()
    bad = bool(False)  # noqa: UP018 (Warp needs a typed, mutable local)
    for column in range(first.shape[1]):
        bad = bad or not wp.isfinite(first[row, column])
    for column in range(second.shape[1]):
        bad = bad or not wp.isfinite(second[row, column])
    invalid[row] = bad


@wp.kernel
def _clear_warmstart_rows(
    warmstart: wp.array2d(dtype=wp.float32),
    rows: wp.array(dtype=wp.int32),
):
    row, column = wp.tid()
    warmstart[rows[row], column] = 0.0


class PhysicsRecovery:
    def __init__(self, data):
        self.data = data
        self.device = data.qpos.device
        self.num_envs = data.qpos.shape[0]
        self._flags = wp.empty(self.num_envs, dtype=wp.bool, device=self.device)
        self._recovered = np.zeros(self.num_envs, dtype=bool)
        self.invalid = np.zeros(self.num_envs, dtype=bool)

    def _invalid_rows(self, first, second):
        wp.launch(
            _nonfinite_rows,
            dim=self.num_envs,
            inputs=[first, second, self._flags],
            device=self.device,
        )
        return self._flags.numpy().copy()

    def check_decision(self):
        self.invalid = self._invalid_rows(self.data.qpos, self.data.qvel)
        self._raise_if_invalid(self.invalid & self._recovered, "next decision")
        self._recovered[:] = False
        return self.invalid

    def _raise_if_invalid(self, invalid, phase):
        indices = np.flatnonzero(invalid)
        if indices.size:
            raise RuntimeError(f"Physics recovery failed ({phase}) for worlds {indices.tolist()}")

    def clear_invalid_state(self, indices):
        rows = wp.array(indices, dtype=wp.int32, device=self.device)
        # Forward seeds qacc from qacc_warmstart, even after qpos/qvel are restored.
        wp.launch(
            _clear_warmstart_rows,
            dim=(len(indices), self.data.qacc_warmstart.shape[1]),
            inputs=[self.data.qacc_warmstart, rows],
            device=self.device,
        )

    def validate_reset(self, indices):
        invalid = self._invalid_rows(self.data.qpos, self.data.qvel)
        invalid |= self._invalid_rows(self.data.qacc, self.data.qacc_warmstart)
        selected = np.zeros(self.num_envs, dtype=bool)
        selected[indices] = True
        self._raise_if_invalid(invalid & selected, "after reset")
        self._recovered[indices] = True
