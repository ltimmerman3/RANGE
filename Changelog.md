# Changelog

All notable changes to the RANGE project are documented in this file.

---

## [Unreleased]

### Added
- **Hybrid ML/DFT pipeline** (`energy_calculation.py`): New three-stage optimization workflow that combines ML-potential geometry relaxation with DFT singlepoint energy evaluation.
  - New parameters `if_hybrid_calc` and `hybrid_calc_para` in `energy_computation.__init__`
  - `hybrid_calc_para` accepts: `ml_calculator`, `ml_fmax`, `ml_steps`, `ml_constraint` (optional), `dft_calculator`, `dft_constraint` (optional)
  - ML relaxation failure returns sentinel energy `5555555`; DFT singlepoint failure returns `6666666`
  - Output files `ml_final.xyz` and `dft_singlepoint.xyz` saved when `save_output_level='Full'`
- **Cell validation for hybrid mode** (`energy_calculation.py`): `energy_computation.__init__` now raises `ValueError` if `if_hybrid_calc=True` and the unit cell is all zeros (i.e., `pbc_box` was not set in `cluster_model`)
- **Constraint-vs-cell warning** (`cluster_model.py`): `generate_bounds()` now emits a `warnings.warn()` if any `in_box` constraint extends beyond the `pbc_box` dimensions
- **Example scripts** (`examples/hybrid_ML_DFT/`):
  - `input_example.py`: Hybrid pipeline using MACE (ML) + EMT (DFT stand-in)
  - `input_example_emt.py`: Self-contained example using only ASE built-in EMT calculator for both stages
- `.gitignore` for Python artifacts, IDE files, and RANGE output files

### Changed
- **extxyz output** (`energy_calculation.py`): All `write()` calls now use `format='extxyz'` instead of `format='xyz'`, preserving cell and PBC information in `Lattice=` and `pbc=` comment-line headers. Affects `start.xyz`, `coarse_final.xyz`, `ml_final.xyz`, `dft_singlepoint.xyz`, and `final.xyz`.
- **extxyz input** (`input_output.py`): `convert_xyz_to_lmps()` and `convert_xyz_to_gro()` now read with `format='extxyz'`, correctly recovering cell dimensions from the extxyz header
- `energy_computation.__init__` signature extended with two new optional keyword arguments (fully backwards-compatible)
- `obj_func_compute_energy` gains a new branch for `calculator_type='ase'` + `if_hybrid_calc=True`, inserted before the existing `ase` branch
- `numpy>=1.20` and `scipy>=1.5` added as install dependencies in `setup.py`

### Notes
- extxyz is a superset of standard XYZ — all downstream tools that read `.xyz` files continue to work
- Existing workflows are unaffected when `if_hybrid_calc=False` (the default)

---

## Previous Changes (from git history)

### 2025-06 through 2026-03

- Concise log output in summary
- Single atom adsorbate support for `on_surface` constraint
- Analysis tools: energy grouping, structure similarity, structure cleaning
- Constraint and settings updates
- Bug fixes for constraints, on-surface placement
- README cleanup
- ASE constraint detail updates
- Analysis script updates for atom composition
- Structure cleaning for surface modeling
- Connectivity check with TS prediction integration
- DFTB+ calculator support and examples
- Structure sanity check with `if_return_results`
- CP2K, ORCA, Gaussian, SPARC, LAMMPS external calculator support
- `on_surface`, `in_pore`, `in_box_out`, `in_sphere_shell`, `replace`, `layer`, `micelle` constraint types
- ABC+GA hybrid global optimization algorithm
- Coarse UFF rigid-body pre-optimization
- Dual-stage optimization with constraint removal
- Restart from database or directory
- Early stopping criteria
