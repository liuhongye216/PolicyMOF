---
base_model: outputs/shared_backbone
library_name: peft
license: other
tags:
  - materials-science
  - metal-organic-frameworks
  - lora
  - sr-grpo
---

# PolicyMOF SR-GRPO Example Metadata

This directory contains lightweight example metadata from SR-GRPO training.
Model weights are intentionally excluded from Git history.

The released LoRA checkpoint is available on Zenodo:

https://doi.org/10.5281/zenodo.20742424

## Model

- Base family: Llama 3.1 8B
- Initialization: PolicyMOF shared-backbone SFT checkpoint
- Adaptation: LoRA, rank 8, alpha 32, applied to `q_proj` and `v_proj`
- Optimization: SR-GRPO with four completions per prompt
- GRPO clipping coefficient: 0.2
- KL coefficient: 0.04

Model use is subject to the Meta Llama 3.1 license. The PolicyMOF repository
code is released under the MIT License.

## Intended Use

The checkpoint supports research on property-conditioned generation of
MOFid-derived reticular sequences. Generated strings require deterministic CIF
reconstruction and physics-based validation before they can be interpreted as
candidate materials.

## Limitations

Language-model validity does not establish synthetic feasibility, structural
stability, or experimental performance. The model is intended for research use
and candidate prioritization, not autonomous experimental deployment.

## Related Resources

- Repository: https://github.com/liuhongye216/PolicyMOF
- Dataset: https://doi.org/10.5281/zenodo.19809194
- Manuscript: "Structure-Aware Language Models for Closed-Loop Materials Discovery"
