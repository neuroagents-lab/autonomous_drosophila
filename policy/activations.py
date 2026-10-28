from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import gymnasium as gym
import h5py
import numpy as np
import torch as th
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from torch import nn
from torch.utils.hooks import RemovableHandle

ActivationDType = Literal["float16", "float32"]
MetadataValue = str | int | float | bool


class RecurrentActivationRecorder:
    """Run recurrent policy inference while recording actor and critic activations."""

    def __init__(self, model) -> None:
        if not isinstance(model.action_space, gym.spaces.Discrete):
            raise TypeError("Recurrent activation recording requires a Discrete policy")

        self.policy = model.policy
        self.actor_lstm = getattr(self.policy, "lstm_actor", None)
        if not isinstance(self.actor_lstm, nn.LSTM):
            raise TypeError("Recurrent activation recording requires an actor LSTM")
        self.critic_lstm = getattr(self.policy, "lstm_critic", None)
        if self.critic_lstm is not None and not isinstance(self.critic_lstm, nn.LSTM):
            raise TypeError("policy.lstm_critic must be a torch LSTM or None")

        self._unit_counts = {
            "input_embedding": int(self.actor_lstm.input_size),
            "actor_lstm_hidden": int(self.actor_lstm.num_layers) * int(self.actor_lstm.hidden_size),
            "policy_hidden": int(self.policy.mlp_extractor.latent_dim_pi),
            "value_hidden": int(self.policy.mlp_extractor.latent_dim_vf),
            "value": 1,
        }
        if self.critic_lstm is not None:
            self._unit_counts["critic_lstm_hidden"] = int(self.critic_lstm.num_layers) * int(
                self.critic_lstm.hidden_size
            )

        self._history: dict[str, list[np.ndarray]] = {}
        self._captured: dict[str, np.ndarray] = {}
        self._recording = False
        self._num_envs: int | None = None
        self._state: RNNStates | None = None
        self._handles: list[RemovableHandle] = []
        self._register_hooks()
        self.reset()

    def reset(self) -> None:
        self._history = {name: [] for name in self._unit_counts}
        self._captured = {}
        self._num_envs = None
        self._state = None

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: RNNStates | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, RNNStates]:
        self.policy.set_training_mode(False)
        observation_tensor, _ = self.policy.obs_to_tensor(observation)
        num_envs = self._observation_batch_size(observation_tensor)
        if self._num_envs is None:
            self._num_envs = num_envs
        elif num_envs != self._num_envs:
            raise ValueError(f"recorder batch size changed from {self._num_envs} to {num_envs}")

        recurrent_state = state or self._state or self._initial_state(num_envs)
        starts = (
            np.zeros(num_envs, dtype=np.float32)
            if episode_start is None
            else np.asarray(episode_start, dtype=np.float32)
        )
        if starts.shape != (num_envs,):
            raise ValueError(f"episode_start must have shape {(num_envs,)}, got {starts.shape}")

        self._captured = {}
        self._recording = True
        with th.no_grad():
            actions, values, _, next_state = self.policy(
                observation_tensor,
                recurrent_state,
                th.as_tensor(starts, device=self.policy.device),
                deterministic=deterministic,
            )
        self._recording = False
        self._captured["value"] = self._tensor_units(values)
        self._append_step(num_envs)
        self._state = next_state
        action_array = actions.detach().cpu().numpy().reshape(num_envs)
        return action_array, next_state

    def activations(self) -> dict[str, np.ndarray]:
        num_envs = 0 if self._num_envs is None else self._num_envs
        result = {}
        for name, records in self._history.items():
            if records:
                result[name] = np.stack(records, axis=1)
            else:
                result[name] = np.empty((num_envs, 0, self._unit_counts[name]), dtype=np.float32)
        return result

    def take_activations(self) -> dict[str, np.ndarray]:
        result = self.activations()
        self._history = {name: [] for name in self._unit_counts}
        return result

    def _register_hooks(self) -> None:
        self._handles.append(
            self.policy.pi_features_extractor.register_forward_hook(
                self._tensor_hook("input_embedding")
            )
        )
        self._handles.append(
            self.actor_lstm.register_forward_hook(self._lstm_hidden_hook("actor_lstm_hidden"))
        )
        if self.critic_lstm is not None:
            self._handles.append(
                self.critic_lstm.register_forward_hook(self._lstm_hidden_hook("critic_lstm_hidden"))
            )
        self._handles.append(
            self.policy.mlp_extractor.policy_net.register_forward_hook(
                self._tensor_hook("policy_hidden")
            )
        )
        self._handles.append(
            self.policy.mlp_extractor.value_net.register_forward_hook(
                self._tensor_hook("value_hidden")
            )
        )

    def _tensor_hook(self, name: str):
        def capture(_module, _inputs, output) -> None:
            if self._recording:
                self._capture(name, self._tensor_units(output))

        return capture

    def _lstm_hidden_hook(self, name: str):
        def capture(_module, _inputs, output) -> None:
            if not self._recording:
                return
            hidden = output[1][0]
            units = hidden.detach().transpose(0, 1).flatten(start_dim=1)
            self._capture(
                name,
                units.cpu().numpy().astype(np.float32, copy=False),
            )

        return capture

    def _capture(self, name: str, values: np.ndarray) -> None:
        if name in self._captured:
            raise RuntimeError(f"activation hook fired more than once for {name}")
        self._captured[name] = values

    def _append_step(self, num_envs: int) -> None:
        missing = [name for name in self._unit_counts if name not in self._captured]
        if missing:
            raise RuntimeError(f"activation hooks did not capture {missing}")
        for name, unit_count in self._unit_counts.items():
            values = self._captured[name]
            expected = (num_envs, unit_count)
            if values.shape != expected:
                raise RuntimeError(
                    f"{name} activation shape is {values.shape}, expected {expected}"
                )
            self._history[name].append(values)

    def _initial_state(self, num_envs: int) -> RNNStates:
        actor_state = self._zero_lstm_state(self.actor_lstm, num_envs)
        value_state = (
            self._zero_lstm_state(self.critic_lstm, num_envs)
            if self.critic_lstm is not None
            else self._zero_lstm_state(self.actor_lstm, num_envs)
        )
        return RNNStates(pi=actor_state, vf=value_state)

    @staticmethod
    def _zero_lstm_state(lstm: nn.LSTM, num_envs: int) -> tuple[th.Tensor, th.Tensor]:
        shape = (lstm.num_layers, num_envs, lstm.hidden_size)
        hidden = th.zeros(shape, dtype=lstm.weight_ih_l0.dtype, device=lstm.weight_ih_l0.device)
        return hidden, hidden.clone()

    @staticmethod
    def _observation_batch_size(observation) -> int:
        if isinstance(observation, dict):
            return int(next(iter(observation.values())).shape[0])
        return int(observation.shape[0])

    @staticmethod
    def _tensor_units(values: th.Tensor) -> np.ndarray:
        return (
            values.detach()
            .reshape(values.shape[0], -1)
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )


def write_activation_hdf5(
    path: str | Path,
    episodes: Sequence[Mapping[str, np.ndarray]],
    timestamps_seconds: np.ndarray,
    episode_seeds: np.ndarray,
    *,
    pool_interval_seconds: float,
    dtype: ActivationDType = "float32",
    metadata: Mapping[str, MetadataValue] | None = None,
) -> Path:
    output_path = Path(path)
    names, time_count, unit_counts = _episode_activation_shape(episodes)
    episode_count = len(episodes)
    timestamps = _validated_timestamps(timestamps_seconds, time_count)
    seeds = np.asarray(episode_seeds, dtype=np.int64)
    if seeds.shape != (episode_count,):
        raise ValueError(f"episode_seeds must have shape {(episode_count,)}, got {seeds.shape}")
    storage_dtype = _storage_dtype(dtype)
    starts, counts, pooled_timestamps = _pooling_plan(
        timestamps,
        pool_interval_seconds,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as output:
        output.attrs["format_version"] = 1
        output.attrs["axes"] = "episode,time,unit"
        output.attrs["activation_dtype"] = storage_dtype.name
        output.attrs["pool_interval_seconds"] = float(pool_interval_seconds)
        if metadata is not None:
            for key, value in metadata.items():
                output.attrs[key] = value
        output.create_dataset("episode_seeds", data=seeds)
        raw_group = output.create_group("raw")
        raw_group.attrs["sampling"] = "one_per_policy_decision"
        raw_group.attrs["timestamp_reference"] = "policy_decision"
        _write_timestamps(
            raw_group,
            timestamps,
        )
        pooled_group = output.create_group("pooled")
        pooled_group.attrs["derived_from"] = "/raw"
        pooled_group.attrs["pooling"] = "mean"
        pooled_group.attrs["timestamp_reference"] = "left_bin_edge"
        _write_timestamps(
            pooled_group,
            pooled_timestamps,
        )
        for name in names:
            unit_count = unit_counts[name]
            raw_dataset = _create_activation_dataset(
                raw_group,
                name,
                (episode_count, time_count, unit_count),
                storage_dtype,
            )
            pooled_dataset = _create_activation_dataset(
                pooled_group,
                name,
                (episode_count, pooled_timestamps.size, unit_count),
                storage_dtype,
            )
            for episode, activations in enumerate(episodes):
                values = np.asarray(activations[name], dtype=np.float32)
                raw_dataset[episode] = values.astype(storage_dtype, copy=False)
                pooled = np.add.reduceat(values, starts, axis=0)
                pooled /= counts[:, None]
                pooled_dataset[episode] = pooled.astype(storage_dtype, copy=False)
    return output_path


def _episode_activation_shape(
    episodes: Sequence[Mapping[str, np.ndarray]],
) -> tuple[tuple[str, ...], int, dict[str, int]]:
    if not episodes:
        raise ValueError("episodes must not be empty")
    names = tuple(episodes[0])
    if not names:
        raise ValueError("activation episodes must not be empty")

    time_count = None
    unit_counts = {}
    for episode, activations in enumerate(episodes):
        if tuple(activations) != names:
            raise ValueError(f"activation sources changed in episode {episode}")
        for name, values in activations.items():
            array = np.asarray(values)
            if array.ndim != 2:
                raise ValueError(
                    f"episode {episode} {name} must have axes time,unit; got shape {array.shape}"
                )
            if array.shape[0] < 1 or array.shape[1] < 1:
                raise ValueError(f"episode {episode} {name} must contain time and units")
            if time_count is None:
                time_count = array.shape[0]
            elif array.shape[0] != time_count:
                raise ValueError(
                    f"episode {episode} {name} has {array.shape[0]} time points, "
                    f"expected {time_count}"
                )
            if episode == 0:
                unit_counts[name] = array.shape[1]
            elif array.shape[1] != unit_counts[name]:
                raise ValueError(
                    f"episode {episode} {name} has {array.shape[1]} units, "
                    f"expected {unit_counts[name]}"
                )
    return names, time_count, unit_counts


def _pooling_plan(
    timestamps: np.ndarray,
    interval_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("interval_seconds must be finite and positive")
    scaled = (timestamps - timestamps[0]) / float(interval_seconds)
    nearest = np.rint(scaled)
    scaled = np.where(np.isclose(scaled, nearest, rtol=0.0, atol=1e-10), nearest, scaled)
    bin_indices = np.floor(scaled).astype(np.int64)
    starts = np.concatenate(([0], np.flatnonzero(np.diff(bin_indices)) + 1)).astype(np.int64)
    counts = np.diff(np.append(starts, timestamps.size)).astype(np.float32)
    pooled_timestamps = timestamps[0] + bin_indices[starts].astype(np.float64) * interval_seconds
    return starts, counts, pooled_timestamps


def _validated_timestamps(timestamps_seconds: np.ndarray, time_count: int) -> np.ndarray:
    timestamps = np.asarray(timestamps_seconds, dtype=np.float64)
    if timestamps.shape != (time_count,):
        raise ValueError(
            f"timestamps_seconds must have shape {(time_count,)}, got {timestamps.shape}"
        )
    if time_count == 0:
        raise ValueError("timestamps_seconds must not be empty")
    if not np.isfinite(timestamps).all():
        raise ValueError("timestamps_seconds must be finite")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps_seconds must be strictly increasing")
    return timestamps


def _storage_dtype(dtype: ActivationDType) -> np.dtype:
    if dtype not in {"float16", "float32"}:
        raise ValueError("dtype must be 'float16' or 'float32'")
    return np.dtype(dtype)


def _write_timestamps(
    group: h5py.Group,
    timestamps: np.ndarray,
) -> None:
    group.attrs["axes"] = "episode,time,unit"
    group.create_dataset(
        "timestamps_seconds",
        data=timestamps,
        compression="gzip",
        shuffle=True,
    )


def _create_activation_dataset(
    group: h5py.Group,
    name: str,
    shape: tuple[int, int, int],
    dtype: np.dtype,
) -> h5py.Dataset:
    chunks = (1, min(shape[1], 256), min(shape[2], 256))
    return group.create_dataset(
        name,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        compression="gzip",
        shuffle=True,
    )
