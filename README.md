# PolicyMOF

<div align='center'>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Data: Zenodo](https://img.shields.io/badge/Data-Zenodo-blue.svg)](https://doi.org/10.5281/zenodo.19809194)

</div>

**Title** - Policy-Optimized Language Modelling for End-to-End Metal--Organic Framework Discovery

**Authors** - Hongye Liu, Bingxu Wang, Guibo Luo and Feng Pan

---

## Table of Contents

- [PolicyMOF](#policymof)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Model Architecture](#model-architecture)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Example Workflow](#example-workflow)
  - [Datasets](#datasets)
  - [Model Files](#model-files)
  - [License](#license)
  - [Citation](#citation)
  - [Acknowledgements](#acknowledgements)

---

## Introduction

PolicyMOF is an end-to-end large language model framework for metal-organic framework (MOF) discovery. The framework represents reticular chemistry with MOFid-style sequences and connects property prediction, conditional generation, policy optimization, deterministic CIF reconstruction, and physics-based validation in one workflow. It uses LLaMA-3.1 with LoRA fine-tuning, property-conditioned supervised fine-tuning, and group relative policy optimization (GRPO) with a composite reward for chemical validity, novelty, reconstructability, and adsorption-related property scores.

In the current release, CO2 adsorption is used as the representative target property. Generated candidates can be reconstructed into CIF structures and evaluated through UFF4MOF/LAMMPS relaxation, Zeo++ pore analysis, and GCMC adsorption simulation.

> **Keywords**: Metal-organic frameworks, large language models, policy optimization, reinforcement learning, inverse design, adsorption.

## Model Architecture

The PolicyMOF workflow contains four main stages:

1. **Continuous pre-training (CPT)** on MOF-derived sequence corpora to adapt the base language model to reticular chemistry.
2. **Supervised fine-tuning (SFT)** for property-conditioned representation learning, regression, classification, and generation.
3. **GRPO optimization** with a composite reward that evaluates chemical validity, novelty, reconstructability, structural quality, and adsorption performance.
4. **Simulation validation** through SMILES-to-CIF reconstruction, UFF4MOF/LAMMPS relaxation, Zeo++ pore analysis, and GCMC CO2/N2 adsorption calculations.

The implementation also includes attention-analysis scripts for interpreting chemically meaningful tokens such as metal centers, functional groups, and topology identifiers.

## Getting Started

### Prerequisites

The full workflow requires Python packages and external scientific software:

- Python>=3.10
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

Model paths, dataset paths, and external executable paths should be configured for the local machine before running the training or simulation scripts. Large model checkpoints are not included in this repository.

### Example Workflow

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

See `simulation/README.md` for more details.

## Datasets

The released dataset is available on Zenodo:

```text
https://doi.org/10.5281/zenodo.19809194
```

It includes processed MOF data used by this project, including training/test data and generated output data. Raw structures derived from public databases should be accessed through the original database sources cited in the manuscript.

Small example files are also included in this repository to document the expected input formats:

```text
CPT/mof_pretrain_data.jsonl
SFT/best_regression/train/reg_reg_train.jsonl
SFT/best_regression/test/reg_reg_test.jsonl
SFT/shared_backbone/train/data_train.jsonl
SFT/shared_backbone/test/data_test.jsonl
GRPO/train/gene_mix_train.jsonl
GRPO/test/gene_mix_test.jsonl
```

## Model Files

This repo contains the following main components:

- `CPT/` - Continuous pre-training scripts and MOF sequence corpus examples.
- `SFT/` - Supervised fine-tuning scripts for regression, classification, and shared-backbone training.
- `GRPO/` - GRPO training, inference, and MOF sequence-processing utilities.
- `reward/` - Composite reward plugin and node/linker reference files for MOF generation.
- `simulation/` - LAMMPS, Zeo++, and GCMC validation workflow for generated CIF structures.
- `visualization/` - Attention-analysis scripts for model interpretation.
- `lora_example/` - Example training curves from LoRA/GRPO runs.
- `ms-swift-MOF_master/` - Modified ms-swift/SWIFT training framework used for MOF sequence modelling.

Large checkpoints, generated CIF collections, and full simulation outputs should be stored outside normal Git history or released through Zenodo, Hugging Face, Git LFS, or another data repository.

If you find any bugs or have questions, please open an issue in this repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this code or dataset in your work, please cite:

- Hongye Liu, Bingxu Wang, Guibo Luo and Feng Pan. "Policy-Optimized Language Modelling for End-to-End Metal--Organic Framework Discovery."

## Acknowledgements

This repository builds on open-source scientific and machine-learning tools, including ms-swift/SWIFT, MOFid, TOBACCO, LAMMPS, Zeo++, RDKit, pymatgen, ASE, PyTorch, transformers, and peft.
