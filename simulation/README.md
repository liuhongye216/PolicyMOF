# MOF Simulation Validation Workflow

This directory contains the downstream validation workflow used for generated
MOF candidates in the manuscript. The pipeline converts generated CIF files into
LAMMPS inputs, relaxes structures with UFF4MOF, computes pore descriptors with
Zeo++, and prepares/runs GCMC adsorption simulations for CO2 and N2.

## Directory Layout

```text
simulation/
├── data/
│   ├── cif_candidates/      # Input CIF files generated from MOF sequences
│   ├── lammps_inputs/       # Generated LAMMPS inputs and relaxed outputs
│   └── gcmc_inputs/         # Generated GCMC simulation inputs
├── results/                 # Aggregated relaxation, pore, and adsorption results
├── 01_prepare_lammps.py     # Convert CIF files to LAMMPS/UFF4MOF inputs
├── 02_run_lammps.py         # Run geometry optimization jobs
├── 03_analyze_relaxation.py # Analyze relaxation success and structural changes
├── 04_run_zeopp.py          # Run Zeo++ pore analysis on CIF files
├── 05_prepare_gcmc.py       # Prepare GCMC inputs for CO2/N2 adsorption
├── 06_run_gcmc.py           # Run GCMC simulation jobs
├── 07_analyze_adsorption.py # Summarize adsorption and selectivity results
└── lammps_template.in       # LAMMPS input template
```

## Requirements

- Python 3.10+
- `pymatgen`, `numpy`, `scipy`, `matplotlib`
- `lammps-interface`
- LAMMPS executable available as `lmp`, or configured in `06_run_gcmc.py`
- Zeo++ executable available as `network`, or passed to `04_run_zeopp.py`
- Generated candidate CIF files in `simulation/data/cif_candidates/`

## Running the Pipeline

From the repository root:

```bash
cd simulation

# Step 1-3: UFF4MOF geometry relaxation
python 01_prepare_lammps.py
python 02_run_lammps.py
python 03_analyze_relaxation.py

# Step 4: Zeo++ pore analysis
python 04_run_zeopp.py data/cif_candidates

# Step 5-7: GCMC adsorption simulations and analysis
python 05_prepare_gcmc.py
python 06_run_gcmc.py
python 07_analyze_adsorption.py
```

The default GCMC setup follows the manuscript protocol: rigid framework
approximation, TraPPE CO2/N2 adsorbates, 298 K, and pressure points spanning
0.01-1.0 bar.

## Notes

The scripts are intended as reproducible workflow templates. External
executables, force-field installations, and cluster-specific launch commands may
need to be adjusted for a new machine.
