# Exoplanet Oblateness Characterization ($J_2$, $k_2$) via 1D-CNNs

## Research Overview
The current bottleneck in characterizing planetary interiors is the computational inefficiency of traditional Bayesian models when analyzing extremely subtle (parts-per-million) potential field anomalies in noisy photometric data. 

This project investigates whether a 1D Convolutional Neural Network (CNN) can extract gravitational field perturbations—specifically the quadrupole moment ($J_2$) and Love number ($k_2$)—from noisy transit light curves more efficiently and accurately than traditional Markov Chain Monte Carlo (MCMC) forward-modeling.

## Project Architecture
* `data/` 
  * `synthetic/`: Generated `.npy` arrays of phase-folded light curves and $J_2$ target labels (Ignored by Git).
* `notebooks/`: Interactive environments for pipeline execution.
  * `01_data_simulation.ipynb`: Synthetic dataset generation.
  * `02_cnn_training.ipynb`: Model training and evaluation.
  * `03_mcmc_inference.ipynb`: Baseline Bayesian parameter retrieval.
  * `04_results_comparison.ipynb`: Final benchmarking and visualization.
* `src/`: Core Python modules.
  * `components/data_simulator.py`: Physics engine using `batman` for spherical transits and oblateness injection.
  * `components/cnn_model.py`: PyTorch 1D-CNN architecture.
  * `components/mcmc_baseline.py`: Bayesian inference pipeline.
* `models/weights/`: Saved PyTorch model states (`.pt`).
* `results/`: Output figures and benchmarking logs.

## Setup and Installation
This project requires Python 3.10+ and uses a dedicated virtual environment to manage astrophysics dependencies.

    # Clone the repository
    git clone <your-repo-url>
    cd ExoCNN_Project

    # Create and activate the virtual environment
    python3 -m venv exo_cnn
    source exo_cnn/bin/activate

    # Install dependencies (Note: requires setuptools for legacy package support)
    pip install setuptools
    pip install -r requirements.txt

## Current Status
- [x] **Phase 1: Data Simulation** - Physics engine built and synthetic dataset (10,000 curves) generated.
- [ ] **Phase 2: Deep Learning** - 1D-CNN architecture and PyTorch training loop.
- [ ] **Phase 3: MCMC Baseline** - Bayesian inference pipeline for benchmarking.
- [ ] **Phase 4: Analysis** - Inference time and accuracy comparison.

## Author
**Atharva**
*HBSc Physics & Astrophysics, University of Toronto*# Exo_Characterisation
