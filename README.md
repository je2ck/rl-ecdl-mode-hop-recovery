# Image-Based Reinforcement Learning for Robust Mode Hop Recovery in External-Cavity Diode Lasers

[![DOI](https://zenodo.org/badge/1202613242.svg)](https://doi.org/10.5281/zenodo.22857783)

Reference implementation for the paper **“Image-Based Reinforcement Learning
for Robust Mode Hop Recovery in External-Cavity Diode Lasers”** by Jaeick Bae,
Kyoungsik Yu, Yunheung Song, Jeong Ho Han, and Jongchul Mun.

The system trains a Rainbow DQN policy on an empirical current-frequency model
of an external-cavity diode laser (ECDL). The trained image-based policy can then
recover a target single-mode state by applying discrete laser-current actions.
The paper reports 100% recovery across 163 forced temperature-drift events and
999 successful recoveries in 1,000 abrupt-current trials. Those values are
experimental results from the paper; reproducing them requires the corresponding
empirical data, checkpoint, and hardware.

## Architecture

```text
empirical sweep data
        |
        v
Rust simulator (PyO3) -> current-frequency image -> Rainbow DQN -> action
        ^                                                        |
        |________________________________________________________|

hardware mode: DLC Pro + wavemeter + FP oscilloscope -> same RL interface
```

- `simulator/`: empirical laser transition model and image renderer in Rust.
- `rainbow/`: environment, replay memory, CNN/Rainbow agent, training, and
  evaluation.
- `hardware/`: TOPTICA DLC Pro, wavemeter, Rigol oscilloscope, and FP-mode
  classification adapters.
- `tests/`: dependency-light regression tests for configuration and control
  safety, plus optional PyTorch replay-memory tests.
- `scripts/audit_public_tree.py`: checks tracked files for machine addresses,
  local home paths, loopback-host fallbacks, and embedded secrets.

## Requirements

- Python 3.10 or newer
- PyTorch 2.0 or newer
- A Rust toolchain with edition 2024 support
- `maturin` for building the PyO3 extension

## Installation

Create and activate a virtual environment, then install the simulation and RL
dependencies:

```bash
python -m pip install -r requirements.txt
cd simulator
maturin develop --release
cd ..
```

For real-hardware evaluation, install the additional drivers:

```bash
python -m pip install -r requirements-hardware.txt
```

## Empirical simulator data

The simulator requires the measured files described in
`simulator/data/README.md`. Place them in `simulator/data/`, or provide their
directory at runtime:

```bash
export LASER_SIM_DATA_DIR="<path-to-empirical-data>"
```

The data used in the paper are available from the authors upon request. They are
intentionally absent from this repository, so simulator construction and the
data-dependent Rust tests cannot run until the files are supplied.

## Training in simulation

Simulation is the default runtime. This command makes the mode explicit and
uses the image-only enhanced CNN reported in the paper:

```bash
python -m rainbow.train \
    --simulation \
    --image-only \
    --use-deep-conv \
    --T-max 5000000 \
    --target-frequency 751.52630
```

The CLI defaults for history length, frame skip, discount, multi-step return,
prioritized replay, target-network update, optimizer, batch size, hidden size,
and NoisyLinear standard deviation match Table 4 of the paper. Run
`python -m rainbow.train --help` for all overrides.

## Evaluating on hardware

Hardware access is opt-in through `--hardware`. Connection values must be
provided at runtime; the program exits before opening a device when any required
value is missing or malformed.

```bash
export DLC_HOST="<controller-hostname-or-ip>"
export WAVEMETER_URL="http://<wavemeter-hostname-or-ip>"
export WAVEMETER_PORT="<wavemeter-port>"
export OSCILLOSCOPE_RESOURCE="USB"  # or an explicit VISA resource

python -m rainbow.train \
    --hardware \
    --evaluate \
    --model "<path-to-model.pth>" \
    --image-only \
    --use-deep-conv \
    --control-mode current
```

The same settings can be supplied with `--dlc-host`, `--wavemeter-url`,
`--wavemeter-port`, and `--oscilloscope-resource`. Do not commit real laboratory
addresses or VISA resource identifiers. Hardware mode sends current commands to
the configured laser controller and should first be exercised with a validated
checkpoint and the laser's operating limits independently confirmed.

## Validation

The checks that do not require the private empirical data or a laboratory setup
can be run with:

```bash
python scripts/audit_public_tree.py
python -m unittest discover -s tests -v
cd simulator
cargo fmt --check
cargo test --locked
```

Data-dependent Rust tests are marked ignored and can be run after installing the
empirical files with `cargo test --locked -- --ignored`.

## Citation

To cite the archived `v1.0.0` software release, use:

```bibtex
@software{bae2026modehop,
  author    = {Bae, Jaeick and Yu, Kyoungsik and Song, Yunheung and Han, Jeong Ho and Mun, Jongchul},
  title     = {Image-Based Reinforcement Learning for Robust Mode Hop Recovery in External-Cavity Diode Lasers},
  version   = {1.0.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22857784},
  url       = {https://doi.org/10.5281/zenodo.22857784}
}
```

The DOI badge links to the all-versions record, which resolves to the latest
archived release.

## Acknowledgments and license

The Rainbow DQN implementation is based on
[Kaixhin/Rainbow](https://github.com/Kaixhin/Rainbow), licensed under MIT. This
work was supported by the National Research Foundation of Korea under grants
RS-2023-NR119928, RS-2023-00283259, and RS-2025-25464182, and by the Institute
for Information & Communications Technology Planning & Evaluation under grant
RS-2023-00223497.

This repository is released under the [MIT License](LICENSE).
