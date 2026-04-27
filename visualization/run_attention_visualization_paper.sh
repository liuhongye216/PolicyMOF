#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Nature Publication-Quality Attention Visualization Script
# =============================================================================

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ======================== Configuration ========================
MODEL_PATH="${MODEL_PATH:-outputs/cpt_chemical_tokens}"
ADAPTER_PATH="${ADAPTER_PATH:-outputs/sft_regression/checkpoint-last}"

# SMILES strings (MOFid already included in the SMILES input)
CANONICAL_SMILES="COC(=C(C(=O)[O-])[O])C(=O)[O-].[Zn][Zn] MOFid-v1.pcu.cat0"
ISOMERIC_SMILES="O=C([O-])C(=C(OC)C(=O)[O-])[O].[Zn][Zn] MOFid-v1.pcu.cat0"

# Control SMILES for cross-molecule invariance validation (separated by ;;)
# Choose 3+ structurally distinct MOFs from your dataset to serve as controls.
# The script will compare canonical MOF vs each control to establish a
# cross-molecule baseline; if same-molecule DTW r >> cross-molecule DTW r,
# representation invariance is confirmed beyond universal MOF SMILES properties.
#
# >>> IMPORTANT: Replace these examples with real MOFs from your training set <<<
CONTROL_SMILES="OC(=O)c1cc(C(=O)O)cc(C(=O)O)c1.[Cu][Cu] MOFid-v1.tbo.cat0;;\
OC(=O)c1ccc(C(=O)O)cc1.[Zr] MOFid-v1.fcu.cat0;;\
OC(=O)c1ccc(C(=O)O)cc1.[Zn][Zn] MOFid-v1.pcu.cat0"

# Output directory
OUTPUT_DIR="./attention_plots_paper"

# Sample name prefix
SAMPLE_NAME="MOF_example"

# =============================================================================
# Streamlined Execution: Generate Flagship Combined Figures
# =============================================================================

echo "=========================================="
echo "  Nature Publication-Quality Figure Generation"
echo "  (with Cross-Molecule Invariance Control)"
echo "=========================================="
echo ""
echo "Model:   ${MODEL_PATH}"
echo "Adapter: ${ADAPTER_PATH}"
echo "Output:  ${OUTPUT_DIR}"
echo ""

python run_attention_visual.py \
    --model_path "${MODEL_PATH}" \
    --adapter_path "${ADAPTER_PATH}" \
    --smiles "${CANONICAL_SMILES}" \
    --isomeric_smiles "${ISOMERIC_SMILES}" \
    --control_smiles "${CONTROL_SMILES}" \
    --output_dir "${OUTPUT_DIR}" \
    --sample_name "${SAMPLE_NAME}" \
    --device cuda \
    --deep_layers_only

# =============================================================================
# Done
# =============================================================================

echo ""
echo "=========================================="
echo "  Flagship Figures Generated!"
echo "  Output formats: PNG, TIFF, PDF"
echo "=========================================="
echo ""
echo "Please check ${OUTPUT_DIR} for the results."
