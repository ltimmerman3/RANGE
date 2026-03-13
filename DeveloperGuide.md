# RANGE Developer Guide

This guide covers the internal architecture of RANGE for contributors and developers extending the codebase.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Module Reference](#module-reference)
3. [Data Flow](#data-flow)
4. [Adding a New Calculator](#adding-a-new-calculator)
5. [Adding a New Constraint Type](#adding-a-new-constraint-type)
6. [The Hybrid ML/DFT Pipeline](#the-hybrid-mldft-pipeline)
7. [Key Design Patterns](#key-design-patterns)
8. [Testing](#testing)

---

## Architecture Overview

RANGE has four core modules in `RANGE_go/`:

```
RANGE_go/
  cluster_model.py        # Molecular system definition and search space generation
  energy_calculation.py   # Energy evaluation pipeline (calculators, optimization)
  ga_abc.py               # ABC+GA optimization algorithm
  utility.py              # Coordinate transforms, UFF parameters, structure checks
  input_output.py         # File I/O, database operations, external code helpers
```

The user-facing workflow is:

```
cluster_model  -->  energy_computation  -->  GA_ABC
  (what)              (how)                  (search)
```

---

## Module Reference

### `cluster_model.py`

**Class `cluster_model`**: Defines the molecular system.

- `__init__(molecules, num_of_molecules, constraint_type, constraint_value, pbc_box, pbc_applied)`: Validates inputs, reads XYZ files, assigns residue names, sets PBC.
- `generate_bounds()`: Returns `(templates, bounds, go_conversion_rule)`.
  - `templates`: List of ASE `Atoms`, one per molecule instance (e.g., 12 Cu atoms = 12 single-atom `Atoms`).
  - `bounds`: `(N*6, 2)` array of `(lo, hi)` for each degree of freedom.
  - `go_conversion_rule`: List of tuples encoding how to interpret the 6-DOF vector for each molecule (coordinate system, faces, grid points, etc.).
  - Emits a `warnings.warn()` if any `in_box` constraint extends beyond `pbc_box` (i.e., hi > cell or lo < 0). This catches misconfigurations where atoms could be placed outside the periodic cell.

### `energy_calculation.py`

**Class `RigidLJQ_calculator`**: ASE calculator for coarse UFF rigid-body optimization. Computes LJ + Coulomb interactions between rigid molecules using vectorized neighbor lists.

**Class `energy_computation`**: The energy evaluation pipeline.

- `__init__(...)`: Stores calculator configuration. Initializes coarse (UFF) and hybrid (ML/DFT) calculators if enabled.
- `vector_to_cluster(vec)`: Converts a 1D optimization vector to a 3D ASE `Atoms` structure using `go_conversion_rule`.
- `cluster_to_vector(cluster, vec)`: Inverse of `vector_to_cluster` — extracts translation + rotation from optimized positions.
- `obj_func_compute_energy(vec, computing_id, save_output_directory)`: The objective function called by `GA_ABC`. Returns `(vec, energy, atoms)`.

### `ga_abc.py`

**Class `GA_ABC`**: The hybrid ABC+GA search algorithm.

- `__init__(obj_func, bounds, ...)`: Configures the algorithm.
- `run(print_interval, if_return_results)`: Main loop. Handles initialization, employed/onlooker/scout bee phases, GA crossover/mutation, database saving, and early stopping.
- `_init_colony()`: Initializes or restarts the colony from random generation or a database.

### `utility.py`

Stateless helper functions:
- Coordinate transforms: `cartesian_to_ellipsoidal_deg`, `ellipsoidal_to_cartesian_deg`, `rotate_atoms_by_euler`, `get_translation_and_euler_from_positions`
- UFF parameters: `get_UFF_para(element)` returns `(epsilon, sigma)` for LJ
- Structure validation: `check_structure(atoms, energy, sanity_params)` — checks for unreasonable geometries (too-close atoms, broken connectivity)
- Surface normals: `correct_surface_normal`
- Diversity selection: `select_max_diversity`, `compute_differences`

### `input_output.py`

- `save_structure_to_db(db_path, atoms, name, vec, energy)`: Appends to ASE database
- `read_structure_from_db(db_path, strategy, n)`: Reads top-n structures for restart
- `read_structure_from_directory(dir_path, strategy, n)`: Reads from result directories
- `get_CP2K_run_info(...)`: Parses CP2K output
- `convert_xyz_to_lmps(...)`: Reads extxyz, writes LAMMPS data format (preserves cell)
- `print_code_info(mode)`: Prints the RANGE header/footer

---

## Data Flow

### Per-structure evaluation

```
GA_ABC calls obj_func_compute_energy(vec, id, dir)
  |
  +--> vector_to_cluster(vec) --> atoms (ASE Atoms)
  |
  +--> [if coarse] UFF rigid-body BFGS --> update vec
  |
  +--> [if hybrid]
  |      ML calculator + BFGS --> update vec
  |      DFT calculator singlepoint --> energy
  |    [else if ase]
  |      ASE calculator + BFGS (or singlepoint) --> energy, update vec
  |    [else if external]
  |      Write input, run subprocess, parse output --> energy, atoms
  |
  +--> check_structure(atoms, energy, sanity) --> final energy
  +--> SinglePointCalculator(atoms, energy) --> attach energy to atoms
  |
  +--> return (vec, energy, atoms)
```

### Vector encoding

Each molecule instance has 6 degrees of freedom in the optimization vector:
- Indices `[6*i : 6*i+3]`: Translation (Cartesian, spherical, grid index, or surface parameters depending on `go_conversion_rule`)
- Indices `[6*i+3 : 6*i+6]`: Rotation (Euler angles ZXZ, or surface-specific rotation)

The `go_conversion_rule[i][0]` string determines how these 6 values are interpreted:
- `'at_position'`, `'in_box'`, `'in_box_out'`: Cartesian + Euler
- `'in_sphere_shell'`: Ellipsoidal `(rho, theta, phi)` + Euler
- `'on_surface'`: `(face_idx, s1, s2, distance, angle, 0)` — surface parametric coordinates
- `'in_pore'`, `'layer'`, `'micelle'`: `(grid_idx, 0, 0)` + Euler
- `'replace'`: `(atom_idx, 0, 0)` + Euler

---

## Adding a New Calculator

### ASE-based calculator

No code changes needed. Pass any ASE-compatible calculator:

```python
computation = energy_computation(
    ...,
    calculator=my_new_ase_calculator,
    calculator_type='ase',
    geo_opt_para=dict(fmax=0.05, steps=100),
)
```

### External calculator

To support a new external code, add a branch in `call_external_calculation()` (around line 587 in `energy_calculation.py`):

```python
elif geo_opt_para_line['method'] == 'MyCode':
    # 1. Prepare input files from start_xyz
    # 2. Run the external command via subprocess
    # 3. Parse energy from output
    # 4. Read final structure
    # 5. Return (atoms, energy)
```

Follow the existing xTB or DFTB+ patterns. Key requirements:
- Use `start_xyz` as input (which is `'start.xyz'` or `'coarse_final.xyz'` if coarse is enabled)
- Handle both `Full` and `Simple` output levels
- Return sentinel energy `7777777` on failure
- Always `os.chdir(job_directory)` before running, and results stay in that directory

---

## Adding a New Constraint Type

Constraint types are placement rules that define where molecules can be placed.

### Step 1: `cluster_model.generate_bounds()`

Add a new branch in `generate_bounds()` to:
1. Parse `constraint_value` for the new type
2. Set appropriate `bounds` (lo/hi for each of 6 DOFs)
3. Set the `go_conversion_rule` entry (a tuple starting with the type name string)

### Step 2: `energy_computation.vector_to_cluster()`

Add an `elif self.go_conversion_rule[i][0] == 'my_new_type':` branch to convert the 6-value vector slice into a positioned/rotated ASE `Atoms`.

### Step 3: `energy_computation.cluster_to_vector()`

Add the corresponding inverse branch to extract the 6 parameters from relaxed atomic positions.

### Checklist

- The 6-DOF encoding must be invertible (`vector_to_cluster` and `cluster_to_vector` are inverses)
- Bounds must be tight enough for efficient search but cover the relevant space
- The `go_conversion_rule` tuple carries any metadata needed at runtime (grid points, face vertices, etc.)

---

## The Hybrid ML/DFT Pipeline

Added in the latest version. Located in `energy_calculation.py`.

### Design

The hybrid pipeline is a new branch in `obj_func_compute_energy()` that activates when `calculator_type == 'ase'` and `if_hybrid_calc == True`. It runs before the standard `ase` branch (which becomes an `elif`).

### Flow

```
1. atoms.calc = ml_calculator
2. Apply ml_constraint (if any)
3. BFGS(fmax=ml_fmax, steps=ml_steps)
4. cluster_to_vector(atoms, vec) --> update vec
5. Save ml_final.xyz (Full mode)
6. atoms.set_constraint() --> remove ML constraints
7. atoms.calc = dft_calculator
8. Apply dft_constraint (if any)
9. energy = atoms.get_potential_energy()  # singlepoint
10. Save dft_singlepoint.xyz (Full mode)
```

### Error handling

- ML relaxation wrapped in try/except: failure sets `energy = 5555555` and skips DFT
- DFT singlepoint wrapped in try/except: failure sets `energy = 6666666`
- The `ml_success` flag controls whether to proceed to DFT

### Parameters stored in `__init__`

```python
self.hybrid_ml_calculator   # ASE calculator for ML relaxation
self.hybrid_ml_fmax         # Force convergence criterion
self.hybrid_ml_steps        # Max BFGS steps
self.hybrid_ml_constraint   # Optional ASE constraint (or None)
self.hybrid_dft_calculator  # ASE calculator for DFT singlepoint
self.hybrid_dft_constraint  # Optional ASE constraint (or None)
```

### Cell validation

Hybrid mode requires a valid periodic cell. The `__init__` checks `templates[0].get_cell()` — if it is all zeros (i.e., `pbc_box` was not set in `cluster_model`), a `ValueError` is raised immediately. This prevents silent failures in ML/DFT calculators that rely on cell information.

### Backwards compatibility

When `if_hybrid_calc=False` (the default), the code falls through to the existing `elif self.calculator_type == 'ase':` branch. No existing behavior is changed.

---

## Key Design Patterns

### Sentinel energies

Failed calculations return large sentinel values instead of raising exceptions, so the search continues:
- `5555555`: Geometry optimization failure
- `6666666`: Singlepoint energy failure
- `7777777`: External calculator failure

The algorithm naturally discards these since it minimizes energy.

### Vector-structure duality

The algorithm operates on 1D vectors (`vec`), but energy evaluation requires 3D structures. The `vector_to_cluster` / `cluster_to_vector` pair handles this conversion. After geometry optimization, the relaxed structure is converted back to a vector so the algorithm can track the improved position.

### Output levels

- `'Simple'`: No per-job directories for ASE calculations. Only the database is written. Best for production runs.
- `'Full'`: Creates a directory per structure with input/output files, logs, and vectors. Best for debugging.

### Boolean flags for optional stages

Optional pipeline stages use `if_XXX` boolean flags with corresponding `XXX_para` dictionaries:
- `if_coarse_calc` / `coarse_calc_para`
- `if_hybrid_calc` / `hybrid_calc_para`

This pattern keeps the `__init__` signature clean and the default behavior unchanged.

---

## Testing

### Backwards compatibility

Run any existing example (e.g., `examples/MACE_calc/input_batio3_Cu13.py`) to verify existing behavior is unchanged.

### Hybrid pipeline

Use EMT (built into ASE, no external dependencies) as a stand-in for both ML and DFT calculators:

```python
from ase.calculators.emt import EMT

hybrid_para = dict(
    ml_calculator=EMT(),
    ml_fmax=0.1,
    ml_steps=50,
    dft_calculator=EMT(),
)

computation = energy_computation(
    ...,
    calculator_type='ase',
    if_hybrid_calc=True,
    hybrid_calc_para=hybrid_para,
)
```

Verify:
1. Coarse UFF runs (if enabled)
2. ML relaxation runs and geometry updates
3. DFT singlepoint produces the final energy
4. `ml_final.xyz` and `dft_singlepoint.xyz` appear in Full mode
5. Sentinel energies returned for deliberately bad structures

### Structure sanity

The `check_structure_sanity` parameter applies after all optimization stages, including the hybrid pipeline. Test with `check_structure_sanity=None` (distance check only) and with connectivity checking.
