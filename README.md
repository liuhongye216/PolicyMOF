# PolicyMOF

<div align='center'>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Data: Zenodo](https://img.shields.io/badge/Data-Zenodo-blue.svg)](https://doi.org/10.5281/zenodo.19809194)
[![Checkpoints: Zenodo](https://img.shields.io/badge/Checkpoints-Zenodo-blue.svg)](https://doi.org/10.5281/zenodo.20742424)

</div>

**Manuscript** - Structure-Aware Language Models for Closed-Loop Materials Discovery

**Authors** - Hongye Liu, Bingxu Wang, Feng Pan and Guibo Luo

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

PolicyMOF is a closed-loop, end-to-end language-model framework for metal-organic framework (MOF) discovery. The framework represents reticular chemistry with MOFid-derived sequences and connects property prediction, conditional generation, policy optimization, deterministic CIF reconstruction, and physics-based validation in one workflow. It uses Llama 3.1 with LoRA fine-tuning, property-conditioned supervised fine-tuning, and structure-aware group relative policy optimization (SR-GRPO) with a composite reward for chemical validity, novelty, reconstructability, and adsorption-related property scores.

In the current release, CO2 adsorption is used as the representative target property. Generated candidates can be reconstructed into CIF structures and evaluated through UFF4MOF/LAMMPS relaxation, Zeo++ pore analysis, and GCMC adsorption simulation.

> **Keywords**: Metal-organic frameworks, large language models, policy optimization, reinforcement learning, inverse design, adsorption.

## Model Architecture

The PolicyMOF workflow contains four main stages:

1. **Continuous pre-training (CPT)** on MOF-derived sequence corpora to adapt the base language model to reticular chemistry.
2. **Supervised fine-tuning (SFT)** for property-conditioned representation learning, regression, classification, and generation.
3. **SR-GRPO optimization** with hierarchical validity, novelty, reconstruction, and target-property signals derived from a ten-component composite reward.
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

Access to the gated `meta-llama/Llama-3.1-8B` checkpoint requires accepting Meta's license terms and authenticating with Hugging Face. Install the repository's modified SWIFT implementation before running the training scripts:

```bash
pip install -e ms-swift-MOF_master
```

Model paths, dataset paths, and external executable paths should be configured for the local machine before running the training or simulation scripts. Large model checkpoints are not included in this repository and are released separately on Zenodo at `https://doi.org/10.5281/zenodo.20742424`.

The reward plugin reads its reference CSV files from `reward/` by default. For
real-time CIF reconstruction, prepare a TOBACCO work directory containing the
required `templates/` and `nodes/` assets, then configure the external paths:

```bash
export MOF_TOBACCO_WORKDIR=/path/to/tobacco_workdir
export MOF_ADSORPTION_MODEL=/path/to/co2_regression_checkpoint
```

Generated `edges/` and `output_cifs/` directories are created inside this work
directory. Each variable used by the training scripts, including `MODEL_PATH`,
`DATASET_PATH`, `OUTPUT_DIR`, and `CUDA_VISIBLE_DEVICES`, can also be overridden
without editing the repository.

### Example Workflow

Continuous pre-training with the domain-specific chemical vocabulary:

```bash
bash CPT/train_chemical.sh
```

Supervised fine-tuning:

```bash
bash SFT/shared_backbone/train.sh
```

Structure-aware GRPO optimization:

```bash
bash GRPO/train_SR_GRPO_func.sh
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

It includes processed MOF data used by this project, including training/test data and generated output data. Approximately 180,000 hypothetical compositions were enumerated for synthetic augmentation; after SMILES-level validity and deduplication filtering, approximately 80,000 synthetic sequences were retained, giving a combined fine-tuning corpus of approximately 180,000 sequences. Raw structures derived from public databases should be accessed through the original database sources cited in the manuscript.

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
- `GRPO/` - Baseline GRPO and manuscript-aligned SR-GRPO training, inference, and MOF sequence-processing utilities.
- `reward/` - Composite reward plugin and node/linker reference files for MOF generation.
- `simulation/` - LAMMPS, Zeo++, and GCMC validation workflow for generated CIF structures.
- `visualization/` - Attention-analysis scripts for model interpretation.
- `lora_example/` - Example training curves from LoRA/GRPO runs. The corresponding LoRA/GRPO checkpoint files are not stored on GitHub; they are available from Zenodo at `https://doi.org/10.5281/zenodo.20742424`.
- `ms-swift-MOF_master/` - Modified ms-swift/SWIFT training framework used for MOF sequence modelling; this bundled third-party component retains its Apache-2.0 license.

Large checkpoints, generated CIF collections, and full simulation outputs should be stored outside normal Git history. The released PolicyMOF model checkpoint files are available on Zenodo:

```text
https://doi.org/10.5281/zenodo.20742424
```

If you find any bugs or have questions, please open an issue in this repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this code or dataset in your work, please cite:

- Hongye Liu, Bingxu Wang, Feng Pan and Guibo Luo. "Structure-Aware Language Models for Closed-Loop Materials Discovery."

## Acknowledgements

This repository builds on open-source scientific and machine-learning tools, including ms-swift/SWIFT, MOFid, TOBACCO, LAMMPS, Zeo++, RDKit, pymatgen, ASE, PyTorch, transformers, and peft. PolicyMOF-specific code is released under the repository's MIT license; bundled ms-swift/SWIFT code remains under Apache-2.0.
