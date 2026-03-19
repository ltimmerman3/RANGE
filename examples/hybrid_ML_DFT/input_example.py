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

# Provide user input
Pt = 'Pt.xyz'
input_molecules = [Pt]
# input_num_of_molecules = [13]
input_num_of_molecules = [4]
input_constraint_type = ['in_box']
input_constraint_value = [(0,0,0,5,5,5) ]


print("Step 1: Setting cluster")
cluster = cluster_model(input_molecules, input_num_of_molecules,
                        input_constraint_type, input_constraint_value,
                        pbc_box=(15., 15., 15.), pbc_applied=(False, False, False)
                        )
cluster_template, cluster_boundary, cluster_conversion_rule = cluster.generate_bounds()

print("Step 2: Setting calculators")

# ML calculator for geometry relaxation
ml_calculator = mace_mp(model='medium', dispersion=True, default_dtype="float64", device='cpu')

# DFT calculator for singlepoint energy
from sparc.calculator import SPARC

calc_params = {
    "EXCHANGE_CORRELATION": "GGA_PBE",
    "KPOINT_GRID": [1, 1, 1],
    "MESH_SPACING": 0.35,
    "TOL_SCF": 0.0001,
    "MAXIT_SCF": 40,
    "CALC_STRESS": 0,
    "PRINT_RESTART_FQ": 10,
    "PRINT_ATOMS": 1,
    "PRINT_FORCES": 1,
    "SPIN_TYP": 1,
}

dft_calculator = SPARC(use_socket=True, **calc_params)

# Coarse (UFF) parameters
coarse_opt_parameter = dict(coarse_calc_eps='UFF', coarse_calc_sig='UFF', coarse_calc_chg=0,
                            coarse_calc_step=20, coarse_calc_fmax=10, coarse_calc_constraint=None)

# Hybrid ML/DFT parameters
hybrid_calc_para = dict(
    ml_calculator=ml_calculator,
    ml_fmax=0.05,
    ml_steps=200,
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
    save_output_level='Full',
)


print("Step 3: Run")
output_folder_name = 'results'
optimization = GA_ABC(computation.obj_func_compute_energy, cluster_boundary,
                      colony_size=40, limit=40, max_iteration=5,
                      ga_interval=1, ga_parents=20, mutate_rate=0.5, mutat_sigma=0.05,
                      output_directory=output_folder_name,
                      # restart_from_pool='structure_pool.db',
                      apply_algorithm='ABC_GA',
                      if_clip_candidate=True,
                      early_stop_parameter={'Max_candidate': 500},
                      )
optimization.run(print_interval=1)

print("Step 4: Done. Check results_emt_hybrid/ for output files.")
print("  In Full mode, each job folder contains:")
print("    start.xyz          - initial structure (extxyz with cell info)")
print("    coarse_final.xyz   - after UFF pre-relaxation")
print("    ml_final.xyz       - after ML geometry optimization")
print("    dft_singlepoint.xyz - structure with DFT energy")
print("    final.xyz          - final structure saved to database")
