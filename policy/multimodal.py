from __future__ import annotations

import gymnasium as gym
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from envs.observations import (
    FLY_VISION_OBSERVATION,
    ODOR_OBSERVATION,
    PROPRIOCEPTION_OBSERVATION,
    normalize_observation_components,
)


class SharedEyeCNN(nn.Module):
    """One shared CNN applied independently to both eyes."""

    def __init__(
        self,
        eye_shape: tuple[int, int, int],
        latent_dim: int,
    ) -> None:
        super().__init__()

        self.eye_shape = eye_shape
        self.cnn = nn.Sequential(
            nn.Conv2d(eye_shape[0], 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            n_flatten = int(self.cnn(th.zeros(1, *eye_shape)).shape[1])
        self.projection = nn.Sequential(nn.Linear(n_flatten, int(latent_dim)), nn.ReLU())

    def forward(self, vision: th.Tensor) -> th.Tensor:
        if vision.ndim != 5 or vision.shape[1:] != (2, *self.eye_shape):
            raise ValueError(
                f"Vision must have shape (batch, 2, {self.eye_shape}); got {tuple(vision.shape)}"
            )
        batch = vision.shape[0]
        eyes = vision.float().reshape(batch * 2, *vision.shape[2:])
        encoded = self.projection(self.cnn(eyes))
        return encoded.reshape(batch, 2 * encoded.shape[-1])


class FlyGymMultimodalExtractor(BaseFeaturesExtractor):
    """Encode each observation component and concatenate the results.

    ``base_features`` excludes the ``skill`` observation; ``forward`` appends
    it unchanged. Odor and proprioception also pass through raw.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        *,
        fly_vision_latent_dim: int = 128,
        odor_latent_dim: int = 16,
        proprioception_latent_dim: int = 64,
        odor_hidden_dim: int = 32,
        proprioception_hidden_dim: int = 128,
    ) -> None:
        spaces = observation_space.spaces
        components = normalize_observation_components([key for key in spaces if key != "skill"])
        skill_dim = int(spaces["skill"].shape[0]) if "skill" in spaces else 0
        base_dim = 0
        if FLY_VISION_OBSERVATION in components:
            base_dim += int(fly_vision_latent_dim)
        for key, latent_dim in (
            (ODOR_OBSERVATION, odor_latent_dim),
            (PROPRIOCEPTION_OBSERVATION, proprioception_latent_dim),
        ):
            if key in components:
                base_dim += int(latent_dim) + int(spaces[key].shape[0])
        super().__init__(observation_space, base_dim + skill_dim)
        self.base_features_dim = base_dim
        self.has_skill = skill_dim > 0

        self.fly_vision_encoder = None
        if FLY_VISION_OBSERVATION in components:
            fly_shape = spaces[FLY_VISION_OBSERVATION].shape
            if len(fly_shape) != 4 or fly_shape[:2] != (2, 2):
                raise ValueError(f"fly_vision space must have shape (2, 2, H, W); got {fly_shape}")
            if int(fly_vision_latent_dim) % 2:
                raise ValueError("fly_vision_latent_dim must be even")
            self.fly_vision_encoder = SharedEyeCNN(fly_shape[1:], int(fly_vision_latent_dim) // 2)

        self.odor_encoder = None
        if ODOR_OBSERVATION in components:
            self.odor_encoder = _mlp(
                spaces[ODOR_OBSERVATION].shape[0], odor_hidden_dim, odor_latent_dim
            )

        self.proprioception_encoder = None
        if PROPRIOCEPTION_OBSERVATION in components:
            self.proprioception_encoder = _mlp(
                spaces[PROPRIOCEPTION_OBSERVATION].shape[0],
                proprioception_hidden_dim,
                proprioception_latent_dim,
            )

    def base_features(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        features: list[th.Tensor] = []
        if self.fly_vision_encoder is not None:
            features.append(self.fly_vision_encoder(observations[FLY_VISION_OBSERVATION]))
        if self.odor_encoder is not None:
            odor = observations[ODOR_OBSERVATION].float()
            features.extend((self.odor_encoder(odor), odor))
        if self.proprioception_encoder is not None:
            proprio = observations[PROPRIOCEPTION_OBSERVATION].float()
            features.extend((self.proprioception_encoder(proprio), proprio))
        return th.cat(features, dim=1)

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        base = self.base_features(observations)
        if not self.has_skill:
            return base
        return th.cat((base, observations["skill"].float()), dim=1)


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(int(in_dim), int(hidden_dim)),
        nn.ReLU(),
        nn.Linear(int(hidden_dim), int(out_dim)),
        nn.ReLU(),
    )
