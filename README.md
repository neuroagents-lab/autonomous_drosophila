# Autonomous Drosophila

This is the code implementation of the paper "Intrinsic Goals for Autonomous Agents: Hierarchical Exploration Captures Future Volitional Action in Drosophila"

Project website: [https://neuroagents-lab.github.io/autonomous_drosophila](https://neuroagents-lab.github.io/autonomous_drosophila)

arXiv: [https://arxiv.org/abs/2610.34638](https://arxiv.org/abs/2610.34638)

bioRxiv: [https://www.biorxiv.org/content/10.64898/2026.10.26.761878v1](https://www.biorxiv.org/content/10.64898/2026.10.26.761878v1)

## Setup

This project uses [uv](https://docs.astral.sh/uv/) and requires Python 3.12.

Clone the repository, run `uv sync`, and run experiment commands from the repository root.

```bash
uv sync
```

Setup installs SSRL for the IDM implementations.

Activate `.venv` before using the `python` commands below, or use `uv run python`.

### GPU training

The default setup runs the simulation and the PPO policy on an NVIDIA GPU and the IDM on CPU with JAX. You can move it to the GPU as well with JAX CUDA.

On a machine with no display, MuJoCo needs an EGL context 

```bash
export MUJOCO_GL=egl
```

To run the IDMs on an NVIDIA GPU on Linux, enable CUDA 12 support with the project's pinned JAX version:

```bash
uv add "jax[cuda12]==0.4.33"
JAX_PLATFORMS=cuda uv run python -c "import jax; print(jax.devices())"
```

This updates your local JAX dependency so subsequent `uv sync` commands retain CUDA support. A compatible NVIDIA driver is required; see the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu).

The verification command must list a CUDA device and fails if CUDA is unavailable.

For GPU IDMs and a GPU policy, launch training with:

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python scripts/train.py \
  envs=flygym_intrinsic_exploration idm=diayn device=cuda
```

`device=cuda` selects the PyTorch policy device; JAX selects its device separately.  

Disabling [JAX memory preallocation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)
leaves GPU memory available for PyTorch and the Warp environments.  

Through our experimentation we found differing benefits to putting the policy/IDM on the CPU or GPU depending on the algorithm.

## Environments

There are three environments implemented. The first is an odor navigation environment where the fly can be trained to navigate to an odor source. The second is the intrinsic reward environment which was used for the main paper results. It has three odor sources and 12-50 obstacles. The last is the ball treadmill evaluation environment.

## Train

`scripts/train.py` loads `configs/configs.yaml` with Hydra.

```bash
python scripts/train.py envs=flygym_intrinsic_exploration idm=diayn seed=1 envs.env_num=256
python scripts/train.py envs=flygym_intrinsic_exploration idm=rnd algo.learning_rate=1e-4 wandb.enabled=false
```

For the extrinsic odor navigation baseline:

```bash
python scripts/train.py envs=flygym_odor_navigation
```

`idm=<name>` loads `configs/idm/<name>.yaml` and sets `idm.type` and its hyperparameters. Omit `idm=` for an extrinsic run (`idm.type=none`).

Useful overrides: `envs.env_num`, `envs.steps`, `seed`, `device`, `wandb.enabled`.

Runs are written to `logdir` (default `runs/<date>/<time>`), including `.hydra/config.yaml` and checkpoints.

## Evaluate

```bash
python scripts/eval_ball.py --run-dir runs/... --episodes 9 --duration-seconds 1800 \
  --out-dir local_data/ball_eval --save-activations
python scripts/eval_odor.py --run-dir runs/... --episodes 10 \
  --output-json local_data/odor_eval/metrics.json
```

`ball` rolls it out on the tethered-ball treadmill as done in the paper. `odor` rolls the policy out in the navigation arena used for training, this collects metrics such as "success rate", ie how often the policy navigates to the odor source. Both default to the GPU (Warp simulation, CUDA policy) and read the run's `.hydra/config.yaml`, `latest_model.zip`, and matching `vecnormalize.pkl` from `--run-dir`. Add `--backend cpu --device cpu` for CPU evaluation, or `--model-path runs/.../best_model/best_model.zip` for a selected checkpoint. Normalization statistics are always loaded beside the selected model.

Ball evaluation writes `rollouts.npz`, `summary.json`, and one episode yaw PNG per episode to `--out-dir` (default `local_data/ball_eval/ball_policy_eval`). `--save-activations` adds `activations.h5` (raw and mean-pooled actor/critic activations) and `observations.h5` (aligned raw proprioception). Skill policies use their saved skill schedule. Odor evaluation prints episode metrics; `--output-json` saves them, and `--record-dir local_data/odor_eval/videos` records videos and a trajectory overview.

Training defaults to the Warp backend and a CUDA policy (`device=cuda`). For CPU training use `envs.backend=cpu envs.env_num=1 device=cpu`. Checkpoints and `.hydra/config.yaml` are written under `logdir` (default `runs/<date>/<time>`); intrinsic runs also write `idm_state.pkl`.

The odor baseline preset trains for 10M decisions; intrinsic exploration trains for 50M.
Intrinsic runs save their final checkpoint without periodic navigation evaluation.

## Grooming recording

Training and ball evaluation replay a recorded grooming bout from `data/grooming.npz`,  
which contains 6,000 frames of anatomical joint angles in radians. The default replay  
uses frames 4766:4801 (35 frames).  

The grooming replay was obtained from the [NeuroMechFly project](https://github.com/NeLy-EPFL/NeuroMechFly).

The kinematic replay is demonstrated in **Aymanns et al. (2022), Figure 5C and Video 5**.

**Reference:** [Aymanns et al., *eLife* (2022)](https://doi.org/10.7554/eLife.81527)

## IDM implementations

The twenty intrinsic objectives use the implementations from:
[SSRL Agent](https://github.com/neuroagents-lab/ssrl-agent)'s `ssrl.idms` package.
`policy/idm.py` maps configurations to constructors and samples skills;
`policy/idm_manager.py` handles replay, Optax updates, and IDM checkpointing.
`policy/idm_rollout.py` connects intrinsic rewards to recurrent PPO rollouts.

For editable development against a local clone, replace the pinned dependency locally:

```bash
uv add --editable /path/to/ssrl-agent
uv sync
```



## Citation

If you use this code in your research, please cite our paper:

```bibtex
@misc{kutscher2026intrinsic,
  title         = {Intrinsic Goals for Autonomous Agents: Hierarchical Exploration Captures Future Volitional Action in Drosophila},
  author        = {Declan Kutscher and Reece Keller and Alyn Kirsch and Felix Pei and Aran Nayebi},
  year          = {2026},
  eprint        = {2610.34638},
  archivePrefix = {arXiv},
  primaryClass  = {q-bio.NC},
  url           = {https://arxiv.org/abs/2610.34638}
}
```



## License

Licensed under the [MIT License](LICENSE).
