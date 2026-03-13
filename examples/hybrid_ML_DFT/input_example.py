# -*- coding: utf-8 -*-
"""
Example: Hybrid ML/DFT pipeline
  Stage 1 (coarse): UFF rigid-body pre-relaxation
  Stage 2 (ML):     MACE geometry optimization
  Stage 3 (DFT):    Singlepoint energy from a second calculator

This example uses EMT as a stand-in for DFT (since EMT is built into ASE).
Replace it with your actual DFT calculator (e.g., SPARC, GPAW, CP2K).
"""

from RANGE_go.ga_abc import GA_ABC
from RANGE_go.cluster_model import cluster_model
from RANGE_go.energy_calculation import energy_computation

import os
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from mace.calculators import mace_mp

print("Step 0: Preparation and user input")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

xyz_path = '../xyz_structures/'
substrate = os.path.join(xyz_path, 'BaTiO3-cub-7layer-TiO.xyz')
adsorb = os.path.join(xyz_path, 'Cu.xyz')

input_molecules = [substrate, adsorb]
input_num_of_molecules = [1, 12]

input_constraint_type = ['at_position', 'in_box']
input_constraint_value = [(0,0,0,0,0,0), (-2.5,-2.5,7.5, 2.5,2.5,12.5)]


print("Step 1: Setting cluster")
cluster = cluster_model(input_molecules, input_num_of_molecules,
                        input_constraint_type, input_constraint_value,
                        pbc_box=(20.26028, 20.26028, 32.13928),
                        )
cluster_template, cluster_boundary, cluster_conversion_rule = cluster.generate_bounds()


print("Step 2: Setting calculators")

# ML calculator for geometry relaxation
ml_calculator = mace_mp(model='medium', dispersion=True, default_dtype="float64", device='cpu')

# DFT calculator for singlepoint energy (EMT used here as a placeholder)
# Replace with your actual DFT calculator, e.g.:
#   from sparc.calculator import SPARC
#   dft_calculator = SPARC(...)
dft_calculator = EMT()

# Constraint: fix substrate atoms during optimization
ase_constraint = FixAtoms(indices=[at.index for at in cluster.system_atoms if at.symbol != 'Cu'])

# Coarse (UFF) parameters
coarse_opt_parameter = dict(coarse_calc_eps='UFF', coarse_calc_sig='UFF', coarse_calc_chg=0,
                            coarse_calc_step=20, coarse_calc_fmax=10,
                            coarse_calc_constraint=ase_constraint)

# Hybrid ML/DFT parameters
hybrid_calc_para = dict(
    ml_calculator=ml_calculator,
    ml_fmax=0.05,
    ml_steps=200,
    ml_constraint=ase_constraint,
    dft_calculator=dft_calculator,
    # dft_constraint=None,  # optional: constraint for DFT singlepoint
)

computation = energy_computation(
    templates=cluster_template,
    go_conversion_rule=cluster_conversion_rule,
    calculator_type='ase',
    if_coarse_calc=True,
    coarse_calc_para=coarse_opt_parameter,
    if_hybrid_calc=True,
    hybrid_calc_para=hybrid_calc_para,
    save_output_level='Simple',
)


print("Step 3: Run")
output_folder_name = 'results'
optimization = GA_ABC(computation.obj_func_compute_energy, cluster_boundary,
                      colony_size=40, limit=40, max_iteration=5000,
                      ga_interval=1, ga_parents=20, mutate_rate=0.5, mutat_sigma=0.05,
                      output_directory=output_folder_name,
                      restart_from_pool='structure_pool.db',
                      apply_algorithm='ABC_GA',
                      if_clip_candidate=True,
                      early_stop_parameter={'Max_candidate': 6000},
                      )
optimization.run(print_interval=1)

print("Step 4: See results: use analysis script")
