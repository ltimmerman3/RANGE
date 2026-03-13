# RANGE User Guide

RANGE (Robust Adaptive Nature-inspired Global Explorer) is a Python tool for exploring potential energy surfaces and finding low-energy molecular/material structures using a hybrid ABC+GA algorithm.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Setting Up a Model](#setting-up-a-model)
4. [Constraint Types](#constraint-types)
5. [Calculators](#calculators)
6. [Hybrid ML/DFT Pipeline](#hybrid-mldft-pipeline)
7. [Algorithm Parameters](#algorithm-parameters)
8. [Output and Analysis](#output-and-analysis)
9. [Restarting a Search](#restarting-a-search)

---

## Installation

Requires the Atomic Simulation Environment (ASE). Install from [ase-lib.org](https://ase-lib.org/install.html).

```bash
# From the RANGE root directory:
pip install .
```

---

## Quick Start

A RANGE calculation has four steps:

```python
from RANGE_go.ga_abc import GA_ABC
from RANGE_go.cluster_model import cluster_model
from RANGE_go.energy_calculation import energy_computation

# 1. Define molecules and placement rules
cluster = cluster_model(
    molecules=['substrate.xyz', 'adsorbate.xyz'],
    num_of_molecules=[1, 5],
    constraint_type=['at_position', 'in_box'],
    constraint_value=[(0,0,0,0,0,0), (-5,-5,5, 5,5,10)],
)
templates, bounds, rules = cluster.generate_bounds()

# 2. Set up energy calculator
computation = energy_computation(
    templates=templates,
    go_conversion_rule=rules,
    calculator=your_ase_calculator,
    calculator_type='ase',
    geo_opt_para=dict(fmax=0.05, steps=100),
)

# 3. Run the search
opt = GA_ABC(computation.obj_func_compute_energy, bounds,
             colony_size=20, max_iteration=100)
opt.run(print_interval=1)

# 4. Analyze results (see Analysis section)
```

---

## Setting Up a Model

Use `cluster_model` to define what molecules to include and where they can be placed.

```python
cluster = cluster_model(
    molecules,           # List of ASE Atoms objects or XYZ file paths
    num_of_molecules,    # List of int: how many of each molecule
    constraint_type,     # List of str: placement rule per molecule type
    constraint_value,    # List of tuples: parameters for each rule
    pbc_box=None,        # Tuple (Lx, Ly, Lz) for periodic boundary conditions
    pbc_applied=(True, True, True),  # Which directions are periodic
)
templates, bounds, rules = cluster.generate_bounds()
```

The `templates` are ASE Atoms for each molecule instance, `bounds` define the search space, and `rules` encode how to convert algorithm vectors to 3D coordinates.

---

## Constraint Types

Each molecule type gets a placement rule:

| Type | Parameters | Description |
|------|-----------|-------------|
| `at_position` | `(x, y, z, euler_x, euler_y, euler_z)` | Fixed position, orientation optimized if euler = 0 |
| `in_box` | `(xlo, ylo, zlo, xhi, yhi, zhi)` | Molecule placed inside a box region |
| `in_box_out` | `(xlo, ylo, zlo, xhi, yhi, zhi, ixlo, iylo, izlo, ixhi, iyhi, izhi)` | Inside outer box but outside inner box |
| `in_sphere_shell` | `(cx, cy, cz, Rx, Ry, Rz[, dR_ratio])` | Inside ellipsoid; optional shell with `dR_ratio` |
| `on_surface` | `(substrate_id, (lo, hi), atom_id, direction_id)` | Adsorbed on a substrate surface |
| `in_pore` | `(substrate_id, atom_indices, grid_spacing)` | Inside pore space defined by grid points |
| `layer` | `(points..., spacing, n1, n2)` | On a planar grid |
| `micelle` | `(A, B, C, center, spacing, n1, n2)` | On an ellipsoidal grid |
| `replace` | `[atom_indices]` | Replace atoms in the substrate |

---

## Calculators

### ASE Calculator (recommended)

Any ASE-compatible calculator works: MACE, CHGNet, xTB, EMT, etc.

```python
from mace.calculators import mace_mp
calc = mace_mp(model='medium', dispersion=True, default_dtype="float64", device='cpu')

computation = energy_computation(
    templates=templates,
    go_conversion_rule=rules,
    calculator=calc,
    calculator_type='ase',
    geo_opt_para=dict(fmax=0.05, steps=100),
)
```

#### Singlepoint only (no geometry optimization)

Set `geo_opt_para=None`:

```python
computation = energy_computation(
    ...,
    geo_opt_para=None,  # singlepoint energy only
)
```

#### ASE constraints

```python
from ase.constraints import FixAtoms
constraint = FixAtoms(indices=[0, 1, 2, 3])
geo_opt_para = dict(fmax=0.05, steps=100, ase_constraint=constraint)
```

#### Dual-stage optimization

First optimize with constraints, then remove them:

```python
geo_opt_para = dict(
    fmax=0.2, steps=20,
    ase_constraint=constraint,
    Dual_stage_optimization=dict(fmax=0.05, steps=100),
)
```

### External Calculator

For software without ASE interfaces (xTB CLI, CP2K, ORCA, Gaussian, LAMMPS):

```python
computation = energy_computation(
    ...,
    calculator="xtb --gfn2 {input_xyz} --opt normal",
    calculator_type='external',
    geo_opt_para=dict(method='xTB'),
)
```

See `examples/` for CP2K, DFTB+, Gaussian, ORCA, SPARC, and LAMMPS templates.

### Coarse Pre-optimization

UFF-based rigid-body pre-relaxation avoids bad initial geometries and speeds up the fine optimizer:

```python
coarse_para = dict(
    coarse_calc_eps='UFF',       # LJ epsilon ('UFF' or dict of element values)
    coarse_calc_sig='UFF',       # LJ sigma
    coarse_calc_chg=0,           # Atomic charges (0, float, list, or dict)
    coarse_calc_step=20,         # Max BFGS steps
    coarse_calc_fmax=10,         # Force convergence (eV/A)
    coarse_calc_constraint=None, # Optional ASE constraint
)

computation = energy_computation(
    ...,
    if_coarse_calc=True,
    coarse_calc_para=coarse_para,
)
```

### Structure-only mode

Generate structures without computing energies:

```python
computation = energy_computation(
    ...,
    calculator_type='structural',
)
```

---

## Hybrid ML/DFT Pipeline

For workflows where ML potentials relax geometry but DFT provides final energies:

```
Coarse UFF (optional) --> ML Relaxation --> DFT Singlepoint
```

```python
from mace.calculators import mace_mp
from ase.calculators.emt import EMT  # placeholder for real DFT

ml_calc = mace_mp(model='medium', dispersion=True, default_dtype="float64", device='cpu')
dft_calc = EMT()  # replace with SPARC, GPAW, etc.

hybrid_para = dict(
    ml_calculator=ml_calc,       # ASE calculator for ML relaxation
    ml_fmax=0.05,                # Force convergence for ML stage
    ml_steps=200,                # Max steps for ML stage
    ml_constraint=constraint,    # Optional ASE constraint for ML stage
    dft_calculator=dft_calc,     # ASE calculator for DFT singlepoint
    dft_constraint=None,         # Optional constraint for DFT stage
)

computation = energy_computation(
    templates=templates,
    go_conversion_rule=rules,
    calculator_type='ase',
    if_coarse_calc=True,
    coarse_calc_para=coarse_para,
    if_hybrid_calc=True,
    hybrid_calc_para=hybrid_para,
    save_output_level='Simple',
)
```

When `if_hybrid_calc=True`:
- The `calculator` and `geo_opt_para` parameters are not needed
- **`pbc_box` is required** — hybrid mode will raise `ValueError` if the cell is all zeros. Set `pbc_box` in `cluster_model` to define the simulation cell.
- ML relaxation failure returns sentinel energy `5555555`
- DFT singlepoint failure returns sentinel energy `6666666`
- In `Full` output mode, saves `ml_final.xyz` and `dft_singlepoint.xyz`

See `examples/hybrid_ML_DFT/input_example.py` for a complete working example (requires MACE), or `examples/hybrid_ML_DFT/input_example_emt.py` for a self-contained example using only ASE's built-in EMT calculator.

---

## Algorithm Parameters

```python
opt = GA_ABC(
    obj_func,                       # Energy function (from energy_computation)
    bounds,                         # Search space bounds (from cluster_model)
    colony_size=20,                 # Number of bees (food sources)
    limit=40,                       # Abandon threshold for scout conversion
    max_iteration=100,              # Maximum iterations
    initial_population_scaler=5,    # Multiplier for initial random guesses
    ga_interval=1,                  # Iterations between GA operations
    ga_parents=10,                  # Number of GA mutations per activation
    mutate_rate=0.5,                # GA mutation probability
    mutat_sigma=0.05,               # GA mutation spread
    output_directory='results',     # Output folder
    output_database='structure_pool.db',  # Database file name
    restart_from_pool=None,         # Path to .db or directory for restart
    restart_strategy='lowest',      # 'lowest' or 'random' restart selection
    apply_algorithm='ABC_GA',       # 'ABC_GA', 'ABC_random', or 'GA_native'
    if_clip_candidate=True,         # Clip candidates to bounds
    early_stop_parameter=None,      # dict with 'Max_candidate', 'Max_ratio', or 'Max_lifetime'
)
opt.run(print_interval=1)
```

---

## Output and Analysis

### Output levels

- `save_output_level='Simple'`: Only the database file is written (fastest)
- `save_output_level='Full'`: Per-structure folders with `start.xyz`, `final.xyz`, `vec.txt`, `energy.txt`, optimization logs

### Output file format

All `.xyz` output files (`start.xyz`, `final.xyz`, `coarse_final.xyz`, `ml_final.xyz`, `dft_singlepoint.xyz`) are written in **extended XYZ (extxyz)** format. This preserves the unit cell (`Lattice=`) and periodic boundary conditions (`pbc=`) in the comment line, ensuring that downstream tools (e.g., `convert_xyz_to_lmps`) correctly recover cell information. The `.xyz` extension is retained since extxyz is a superset of standard XYZ.

### Database

All structures are saved to `structure_pool.db` (ASE database format). This is the primary output.

### Analysis tools

Located in `examples/analysis_tools/`:

| Script | Purpose |
|--------|---------|
| `energy_plots.py` | Energy profile, summary log, optional sorted XYZ trajectory |
| `clean_structure_pool.py` | Filter by connectivity, similarity, custom conditions |
| `frame_capture.py` | Extract specific frames from trajectory |
| `optimize_MACE.py` | Re-optimize selected structures with MACE |
| `optimize_coraseFF.py` | Re-optimize with coarse force field |

```bash
# Generate energy summary
python examples/analysis_tools/energy_plots.py

# Filter structures by group
python examples/analysis_tools/clean_structure_pool.py --input pool.xyz --group 0 1

# Extract specific frames
python examples/analysis_tools/frame_capture.py --frame 0 1 7 21
```

---

## Restarting a Search

To continue a previous search, point `restart_from_pool` to the database or results directory:

```python
opt = GA_ABC(
    ...,
    restart_from_pool='structure_pool.db',  # or 'results/' directory
    restart_strategy='lowest',              # pick lowest-energy candidates
)
```

The search resumes with the best candidates from the previous run as the starting colony.

---

## Sentinel Energies

When calculations fail, RANGE assigns large sentinel energies so the search continues:

| Value | Meaning |
|-------|---------|
| `5555555` | Geometry optimization failed (ASE or ML relaxation) |
| `6666666` | Singlepoint energy calculation failed (ASE or DFT) |
| `7777777` | External calculator failed |

These structures are effectively discarded by the algorithm since lower-energy structures are always preferred.
