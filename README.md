# PolicyMOF

Policy-optimized language modelling for end-to-end metal-organic framework
(MOF) discovery.

This repository contains the code and example data accompanying the manuscript
**"Policy-Optimized Language Modelling for End-to-End Metal--Organic Framework
Discovery"**. The project uses MOFid-style sequence representations, LLaMA-3.1
LoRA fine-tuning, property-conditioned supervised fine-tuning, group relative
policy optimization (GRPO), deterministic SMILES-to-CIF reconstruction, and
physics-based validation for MOF candidate discovery.

## Overview

The workflow connects four stages:

1. **Continuous pre-training (CPT)** on MOF-derived sequence corpora.
2. **Supervised fine-tuning (SFT)** for property-conditioned representation
   learning and generation.
3. **GRPO optimization** with a composite reward for chemical validity,
   novelty, reconstructability, and adsorption-related property scores.
4. **Simulation validation** using CIF reconstruction, UFF4MOF/LAMMPS
   relaxation, Zeo++ pore analysis, and GCMC adsorption calculations.

CO2 adsorption is used as the representative target property in the current
release.

## Repository Structure

```text
.
├── CPT/                 # Continuous pre-training scripts and MOF corpus example
├── SFT/                 # Supervised fine-tuning and inference examples
├── GRPO/                # GRPO training, inference, and MOF reconstruction utils
├── reward/              # Composite reward plugin for MOF generation
├── simulation/          # LAMMPS, Zeo++, and GCMC validation workflow
├── visualization/       # Attention-analysis scripts used for interpretation
├── lora_example/        # Example LoRA/GRPO training outputs
├── manuscript.tex       # Manuscript source
├── README.md
├── LICENSE
└── .gitignore
```

## Main Components

- `CPT/`: domain adaptation of the base language model to MOF sequences.
- `SFT/`: property-conditioned supervised fine-tuning for regression,
  classification, and generation.
- `GRPO/`: policy optimization scripts and utilities for MOFid-style sequence
  processing and TOBACCO-based reconstruction.
- `reward/mof_reward.py`: composite reward used during GRPO.
- `simulation/`: downstream validation pipeline for generated CIF structures.
- `visualization/`: attention attribution and feature-level interpretation.

## Environment

The full workflow requires both Python packages and external scientific
software. At minimum, expect to configure:

- Python 3.10+
- PyTorch
- transformers
- peft
- ms-swift/SWIFT
- RDKit
- NumPy, pandas, scikit-learn, SciPy
- pymatgen, ASE
- matplotlib, seaborn
- LAMMPS with UFF4MOF support
- Zeo++
- MOFid and TOBACCO-related reconstruction utilities

Model paths, dataset paths, and external executable paths should be configured
for the local machine before running the training or simulation scripts.

## Example Workflow

Continuous pre-training:

```bash
bash CPT/pretrain.sh
```

Supervised fine-tuning:

```bash
bash SFT/shared_backbone/train.sh
```

GRPO optimization:

```bash
bash GRPO/train_GRPO_func.sh
```

Attention visualization:

```bash
bash visualization/run_attention_visualization_paper.sh
```

Simulation validation:

```bash
cd simulation
python 01_prepare_lammps.py
python 02_run_lammps.py
python 03_analyze_relaxation.py
python 04_run_zeopp.py data/cif_candidates
python 05_prepare_gcmc.py
python 06_run_gcmc.py
python 07_analyze_adsorption.py
```

See `simulation/README.md` for details.

## Data and Checkpoints

Small example datasets are included for reproducibility of the code structure.
Large model checkpoints, full generated structures, and large simulation outputs
should be stored outside normal Git history or released through Git LFS, Zenodo,
Hugging Face, or another data repository.

## Citation

If you use this code, please cite the accompanying manuscript:

```text
Hongye Liu, Bingxu Wang, Guibo Luo, Feng Pan.
Policy-Optimized Language Modelling for End-to-End Metal--Organic Framework Discovery.
```

## License

This project is released under the MIT License. See `LICENSE` for details.