# CSCI 645 Homework: Learning In-Hand Cube Rotation

## Overview

In this assignment, you will train and analyze a reinforcement-learning policy for in-hand cube rotation using a simulated LEAP Hand. You will first install and validate the codebase, reproduce a baseline policy, and then evaluate principled extensions.

**See the end of this document for the precise deliverables.**

**DEADLINE: October 01 at 11:59pm in Pacific Time.**

Changelog:

- 08/27/2026: initial release.

---

## Part 0: Installation and System Validation

### Hardware

A CUDA-capable NVIDIA GPU is strongly recommended.

Reference development machine:

- Ubuntu 22.04
- Python 3.10
- NVIDIA GeForce RTX 4080 SUPER
- 16 GB GPU memory
- NVIDIA driver 560.35.03
- PyTorch 2.9.0 with CUDA 12.6

A CPU-only run may be enough to open a single-environment visualization, but full policy training with thousands of environments requires a GPU.

### Install `uv`

Install `uv` using the official instructions, then verify:

```bash
uv --version
```

If the command is not found immediately after installation:

```bash
source ~/.bashrc
```

### Install the project

```bash
git clone <COURSE_REPOSITORY_URL>
cd <COURSE_REPOSITORY_NAME>
uv sync
```

This creates a local virtual environment in:

```text
.venv/
```

Do not change the pinned package versions in `pyproject.toml` or `uv.lock`.

### Verify PyTorch and CUDA

```bash
uv run python -c "import torch; \
print('Torch:', torch.__version__); \
print('Torch CUDA:', torch.version.cuda); \
print('CUDA available:', torch.cuda.is_available()); \
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

Expected output should resemble:

```text
Torch: 2.9.0+cu126
Torch CUDA: 12.6
CUDA available: True
GPU: NVIDIA GeForce RTX 4080 SUPER
```

The key line is:

```text
CUDA available: True
```

### Test the pretrained policy

```bash
uv run python scripts/play.py \
  Mjlab-Leap-Left-Custom-HandCube-Rotate \
  --checkpoint-file ckpts/leap_left_custom_model_4900.pt
```

A MuJoCo viewer should open and display the LEAP Hand manipulating a cube.

The terminal should report something similar to:

```text
Devices:
  "cpu"    : "x86_64"
  "cuda:0" : "NVIDIA GeForce RTX 4080 SUPER"

Base Environment
Number of environments | 1
Environment device     | cuda:0
```

The first run may be slower because Warp compiles and caches GPU kernels.

Stop the viewer with `Ctrl+C`. A final `KeyboardInterrupt` traceback after pressing `Ctrl+C` is expected.

### Test random and zero actions

```bash
uv run python scripts/play.py \
  Mjlab-Leap-Left-HandCube-Rotate \
  --agent random
```

```bash
uv run python scripts/play.py \
  Mjlab-Leap-Left-HandCube-Rotate \
  --agent zero
```

These commands are only environment checks. Random and zero actions are not expected to solve the task.

### Configure Weights & Biases

```bash
uv run wandb login --verify
```

Then verify:

```bash
uv run wandb status
```

---

## Part 1: Baseline Training

### Initial test

Start with 256 parallel environments:

```bash
uv run python scripts/train.py \
  Mjlab-Leap-Left-HandCube-Rotate \
  --env.scene.num-envs 256
```

Let this run for at least 50 to 100 iterations.

A successful run should:

- report `Environment device | cuda:0`;
- advance through learning iterations;
- print finite rewards and losses;
- create a W&B run;
- create a directory under `logs/rsl_rl/`;
- avoid NaNs and CUDA errors.

Reference usage on an RTX 4080 SUPER:

```text
Training-process GPU memory: approximately 0.6 to 0.7 GB
GPU utilization: approximately 60%
```

These values are only examples.

Monitor the GPU in another terminal:

```bash
watch -n 0.5 nvidia-smi
```

### Full baseline run

After the initial test succeeds:

```bash
uv run python scripts/train.py \
  Mjlab-Leap-Left-HandCube-Rotate \
  --env.scene.num-envs 4096
```

Reference measurements on an RTX 4080 SUPER:

```text
Number of environments:       4096
Training-process GPU memory:  approximately 4.5 GB
Total GPU memory in use:      approximately 5.4 GB
GPU utilization:              approximately 80 to 85%
Training throughput:          approximately 29,700 steps/s
Iteration time:               approximately 4.4 s
Estimated 5,000-iteration run: approximately 6 hours
```

If 4,096 environments cause an out-of-memory error, reduce to 2,048 or 1,024:

```bash
uv run python scripts/train.py \
  Mjlab-Leap-Left-HandCube-Rotate \
  --env.scene.num-envs 2048
```

### Typical training output

```text
Learning iteration 348/5000

Total steps:              45744128
Steps per second:         29724
Collection time:          4.229s
Learning time:            0.180s
Mean reward:              1.87
Mean episode length:      365.63
Iteration time:           4.41s
```

### What to monitor

`Train/mean_reward` is the main high-level learning signal, but it should not be used alone.

Also monitor:

- `Episode_Metrics/rotation_progress`
- `Episode_Metrics/linear_speed`
- `Episode_Metrics/position_error`
- `Episode_Metrics/tilt_error`
- `Episode_Metrics/fingertip_contact_fraction`
- `Episode_Termination/cube_fell`
- `Episode_Termination/cube_pose_deviation`
- `Episode_Termination/nan`

A better policy should improve rotation while retaining the cube and avoiding unstable behavior.

### Checkpoints

Training outputs are saved under:

```text
logs/rsl_rl/<experiment_name>/<timestamp>/
```

Find recent checkpoints:

```bash
find logs/rsl_rl -name 'model_*.pt' \
  -printf '%T@ %p\n' | sort -n | tail
```

Play a trained checkpoint:

```bash
uv run python scripts/play.py \
  Mjlab-Leap-Left-HandCube-Rotate \
  --checkpoint-file logs/rsl_rl/<experiment>/<timestamp>/model_<iteration>.pt
```

---

## Part 2: Understand the Baseline

Before modifying the code, we suggest you look at and understand:

1. the actor observation space;
2. the privileged critic observation space;
3. the 16-dimensional action space;
4. the reward terms;
5. the termination conditions;
6. the domain randomization parameters;
7. the purpose of the grasp cache.

This way it makes it easier to test the open-ended extensions, described next.

---

## Part 3: Open-Ended Extension

Choose TWO of the four tracks below. For each selected track, implement and evaluate one principled modification to the baseline (it does not have to be explicitly listed under the track, as long as it is a reasonable fit). 

### Track A: Reward Design

Examples:

- improve rotation while limiting tilt;
- improve object retention;
- modify a reward curriculum;
- encourage different types of rotation behavior.

### Track B: Observation Design

Examples:

- change observation history length;
- remove or add selected proprioceptive signals;
- introduce substantial observation noise;
- study partial observability.

### Track C: Domain Randomization and Robustness

Examples:

- change cube mass, size, or friction ranges;
- change actuator delay randomization;
- change PD gain randomization;
- modify grasp cache initialization;
- evaluate untested / new physical parameters.

### Track D: Policy or Training Configuration

Examples:

- change network width or depth;
- change PPO hyperparameters;
- change rollout length;
- modify entropy or action-noise schedules;
- study the number of parallel environments.

Simple parameter sweeps without a clear hypothesis are not sufficient.

---

## Part 4: BONUS update code to most recent mjlab version

The code right now uses an older version of `mjlab`, so for extra credit, you are welcome to update the code and API to deal with the most recent version as of the release date of this assignment (and then answer the above questions). The bonus will be no more than 20% of the overall grade.

---

## Part 5: Experimental Requirements

Your submission must include:

1. a concrete hypothesis;
2. a clearly described implementation;
3. a baseline comparison using an equal training budget;
4. fixed evaluation conditions;
5. analysis of both improvements and failure cases.

Changing only the numerical scale of the reward does not count as an improvement.

Suggested evaluation metrics:

- rotation progress;
- cube-retention or failure rate;
- mean episode length;
- position error;
- tilt error;
- fingertip contact;
- torque and work;
- performance under held-out domain randomization.

---

## Deliverables

Submit **one PDF report** on Brightspace by the deadline (see above and course website). You do **not** need to submit source code. Answer the following questions **in order**, using the numbered headings below. Unless otherwise noted, aim for about **1-2 paragraphs per question**. If you would like to provide additional supporting material, place it in an appendix at the end of the PDF. The appendix is optional and may not be reviewed in detail.

### 1. Baseline

What command did you use to train the baseline, and what training budget did you use? Provide the W&B link for the baseline run (be sure the link is accessible!). Briefly summarize the baseline performance and indicate where we can see the baseline policy video.

### 2. Baseline evaluation

Report the task level evaluation metrics for the baseline. Include the results in a clear table and briefly explain what they indicate.

### 3. Modification 1

What was your first principled modification to the baseline, and why did you expect it to help? Clearly describe what you changed.

### 4. Modification 1 results

How did Modification 1 perform compared with the baseline? Provide the W&B link, report the same evaluation metrics, and briefly interpret the result.

### 5. Modification 2

What was your second principled modification to the baseline, and why did you expect it to help? Clearly describe what you changed.

### 6. Modification 2 results

How did Modification 2 perform compared with the baseline? Provide the W&B link, report the same evaluation metrics, and briefly interpret the result.

### 7. Overall discussion

Which approach worked best, and why? Discuss the main conclusions, limitations, and what you would try next with more time or compute. Aim for **1-2 short paragraphs**.

Include **one final table or figure** comparing the baseline, Modification 1, and Modification 2 on the same evaluation metrics, along with links to videos of the resulting policies.

Your grade is based on the quality of experimental methodology and analysis. Negative results are acceptable when the experiments are well designed and clearly analyzed.
