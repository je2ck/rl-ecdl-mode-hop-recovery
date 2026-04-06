# Image-Based Reinforcement Learning for Robust Mode Hop Recovery in External-Cavity Diode Lasers

This repository contains the code for the paper:

> **Image-Based Reinforcement Learning for Robust Mode Hop Recovery in External-Cavity Diode Lasers**
>
> Jaeick Bae, Kyoungsik Yu, Yunheung Song, Jeong Ho Han, and Jongchul Mun
>
> [Paper Link (DOI TBD)]

A Rainbow DQN agent trained entirely in simulation achieves 100% recovery success across 163 continuous temperature-drift events and 99.9% success under abrupt current jumps (999/1000 trials), then deploys directly to a real ECDL system without retraining.

## Repository Structure

```
├── simulator/          # Rust-based laser simulator with PyO3 Python bindings
│   ├── src/            # Simulator source code
│   └── data/           # Empirical frequency-current mapping data (not included)
├── rainbow/            # Rainbow DQN agent
│   ├── model.py        # CNN architecture (Table 3 in paper)
│   ├── agent.py        # Rainbow DQN training agent
│   ├── memory.py       # Prioritized experience replay
│   ├── interface.py    # Simulator-RL bridge
│   ├── env.py          # Environment wrapper
│   └── train.py        # Training entry point
└── hardware/           # Hardware control interfaces for real experiments
    ├── dlc_controller.py   # TOPTICA DLC Pro controller
    ├── wavemeter.py        # Wavelength meter API
    ├── oscilloscope.py     # Rigol oscilloscope driver
    └── cavity_classifier.py # FP cavity mode classifier
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- Rust toolchain (for building the simulator)
- [maturin](https://github.com/PyO3/maturin) (for building PyO3 bindings)

## Installation

### 1. Build the Rust simulator

```bash
cd simulator
pip install maturin
maturin develop --release
```

This compiles the `laser_sim` Python module from Rust source.

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up data directory

Place the empirical laser data files in `simulator/data/`, or set the environment variable:

```bash
export LASER_SIM_DATA_DIR=/path/to/your/data
```

## Usage

### Training in simulation

```bash
python -m rainbow.train \
    --simulation \
    --image-only \
    --use-deep-conv \
    --history-length 6 \
    --frame-skip-num 6 \
    --T-max 5000000 \
    --target-frequency 751.52630
```

### Deploying to hardware

```bash
# Set hardware connection parameters
export DLC_IP=<your-dlc-ip>
export WAVEMETER_URL=http://<your-wavemeter-ip>

python -m rainbow.train \
    --evaluate \
    --model results/default/model.pth \
    --image-only \
    --use-deep-conv
```

## Data Availability

The experimental laser frequency-current mapping data used for simulator construction is available from the authors upon request. Contact: jcmun@kriss.re.kr

## Citation

```bibtex
@article{bae2025modehop,
  title={Image-Based Reinforcement Learning for Robust Mode Hop Recovery in External-Cavity Diode Lasers},
  author={Bae, Jaeick and Yu, Kyoungsik and Song, Yunheung and Han, Jeong Ho and Mun, Jongchul},
  journal={TBD},
  year={2025}
}
```

## Acknowledgments

- The Rainbow DQN implementation is based on [Kaixhin/Rainbow](https://github.com/Kaixhin/Rainbow), licensed under MIT.
- This work was supported by the National Research Foundation of Korea (NRF) under Grant No. RS-2023-NR119928 and RS-2023-00283259, and by the Institute for Information & Communications Technology Planning & Evaluation under Grant No. RS-2023-00223497.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
