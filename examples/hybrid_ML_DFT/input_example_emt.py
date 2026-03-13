# -*- coding: utf-8 -*-
"""
Example: Hybrid ML/DFT pipeline using only ASE built-in calculators

This example uses EMT for both the ML relaxation and DFT singlepoint stages,
so it runs without any external dependencies beyond ASE. It demonstrates the
full hybrid pipeline: coarse UFF -> ML relaxation -> DFT singlepoint.

For a real application, replace the ML calculator with e.g. mace_mp() and
the DFT calculator with e.g. SPARC, GPAW, or CP2K.
"""

from RANGE_go.ga_abc import GA_ABC
from RANGE_go.cluster_model import cluster_model
from RANGE_go.energy_calculation import energy_computation

import os
from ase import Atoms
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT

print("Step 0: Preparation")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Build a small Cu slab as substrate (FCC 111-like, 2x2x2)
from ase.build import fcc111
substrate = fcc111('Cu', size=(3, 3, 3), vacuum=10.0, periodic=True)

# Adsorbate: single Cu atom
adsorbate = Atoms('Cu', positions=[(0, 0, 0)])

input_molecules = [substrate, adsorbate]
input_num_of_molecules = [1, 4]

# Fix substrate at its position, place adsorbates in a box above it
slab_zmax = substrate.positions[:, 2].max()
cell = substrate.get_cell().diagonal()

input_constraint_type = ['at_position', 'in_box']
input_constraint_value = [
    (0, 0, 0, 0, 0, 0),  # substrate: fixed position and orientation
    (0, 0, slab_zmax + 1.5, cell[0], cell[1], slab_zmax + 6.0),  # adsorbates: box above slab
]


print("Step 1: Setting cluster")
cluster = cluster_model(
    input_molecules, input_num_of_molecules,
    input_constraint_type, input_constraint_value,
    pbc_box=tuple(cell),  # required for hybrid mode
)
cluster_template, cluster_boundary, cluster_conversion_rule = cluster.generate_bounds()


print("Step 2: Setting calculators")

# Both stages use EMT here (replace with real ML/DFT calculators)
ml_calculator = EMT()
dft_calculator = EMT()

# Fix substrate atoms during optimization
substrate_indices = list(range(len(substrate)))
ase_constraint = FixAtoms(indices=substrate_indices)

# Coarse UFF pre-relaxation parameters
coarse_opt_parameter = dict(
    coarse_calc_eps='UFF', coarse_calc_sig='UFF', coarse_calc_chg=0,
    coarse_calc_step=20, coarse_calc_fmax=10,
    coarse_calc_constraint=ase_constraint,
)

# Hybrid ML/DFT parameters
hybrid_calc_para = dict(
    ml_calculator=ml_calculator,
    ml_fmax=0.05,
    ml_steps=100,
    ml_constraint=ase_constraint,
    dft_calculator=dft_calculator,
)

computation = energy_computation(
    templates=cluster_template,
    go_conversion_rule=cluster_conversion_rule,
    calculator_type='ase',
    if_coarse_calc=True,
    coarse_calc_para=coarse_opt_parameter,
    if_hybrid_calc=True,
    hybrid_calc_para=hybrid_calc_para,
    save_output_level='Full',  # 'Full' to inspect intermediate files
)


print("Step 3: Run")
optimization = GA_ABC(
    computation.obj_func_compute_energy, cluster_boundary,
    colony_size=10, limit=20, max_iteration=5,
    ga_interval=1, ga_parents=5, mutate_rate=0.5, mutat_sigma=0.05,
    output_directory='results_emt_hybrid',
    apply_algorithm='ABC_GA',
    if_clip_candidate=True,
)
optimization.run(print_interval=1)

print("Step 4: Done. Check results_emt_hybrid/ for output files.")
print("  In Full mode, each job folder contains:")
print("    start.xyz          - initial structure (extxyz with cell info)")
print("    coarse_final.xyz   - after UFF pre-relaxation")
print("    ml_final.xyz       - after ML geometry optimization")
print("    dft_singlepoint.xyz - structure with DFT energy")
print("    final.xyz          - final structure saved to database")
