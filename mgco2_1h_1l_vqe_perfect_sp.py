# --------------------------------------------------------------------
# ******************  Importing libraries ****************************
# --------------------------------------------------------------------
import timeit
import numpy as np
import pandas as pd
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.algorithms.optimizers import SLSQP, SPSA
from qiskit.primitives import Estimator
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.mappers import QubitConverter
import qchem

# --------------------------------------------------------------------
# ****************************  Inputs *******************************
# --------------------------------------------------------------------
implementation = 'perfect'
no_homo = 1
no_lumo = 1
assumption = f'{no_homo}H_{no_lumo}L'
no_q = f'{(no_homo + no_lumo) * 2}q'


name_excel_vqe = f'mgco2_vqe_{assumption}_{no_q}_{implementation}_sp.xlsx'
name_excel_parameters = f'mgco2_optimized_parameters_{implementation}_{assumption}_{no_q}_sp.xlsx'


# ---------> Molecules <---------
ion = 'Mg'

target = 'C; ' \
         'O 1 1.19630; ' \
         'O 1 1.19628 2 179.97438; '

ligand = 'C; ' \
         'O 1 1.19630; ' \
         'O 1 1.19628 2 179.97438; ' \
         'Mg 3 1.97923 1 120.07450 2 180.00000'

a_el_ion = 2
a_el_target = 2
a_el_ligand = 2

# Number of active orbitals ---> An integer
a_or_ion = 2
a_or_target = 2
a_or_ligand = 2

# List of all active orbitals ---> A list of integers
a_mos_ion = [4, 5]
a_mos_target = [10, 11]
a_mos_ligand = [15, 16]

charge_ion = 2
charge_target = 0
charge_ligand = 2


bonding_distances = np.arange(1.2, 4, 0.3)
basis = "sto-3g"
charge = 2
spin = 0
geo_format = None

# ---------> Classical Solver <---------
numpy_solver = NumPyMinimumEigensolver()

# ---------> Ansatz Parameterization <---------
num_qubits = a_or_ion * 2
reps = 1
entanglement= "full"
insert_barriers=True
rotation_blocks = "ry"
entanglement_blocks = "cz"

# ---------> VQE Circuit  <---------
mapper = ParityMapper()
two_q_reduc = True
z_sym = None
shots = 1000

# ---> Ansatz Parameterization <---
opt_iter = 1000
optimization_level = 3
optimizer = SLSQP(opt_iter)

# --------------------------------------------------------------------
# ******************** Molecule Construction *************************
# --------------------------------------------------------------------
ion_q_prob = qchem.q_driver(ion, charge_ion, spin, basis, geo_format)
target_q_prob = qchem.q_driver(target, charge_target, spin, basis, geo_format)
ligand_q_prob = qchem.q_driver(ligand, charge_ligand, spin, basis, geo_format)

# --------------------------------------------------------------------
# **************** Active Space Trasnformation ***********************
# --------------------------------------------------------------------
ion_q_prob = qchem.as_transformation(ion_q_prob, a_el_ion, a_or_ion, a_mos_ion)
target_q_prob = qchem.as_transformation(target_q_prob, a_el_target, a_or_target, a_mos_target)
ligand_q_prob = qchem.as_transformation(ligand_q_prob, a_el_ligand, a_or_ligand, a_mos_ligand)

# --------------------------------------------------------------------
# ****************** Hamiltonian & Qubit Encoding ********************
# --------------------------------------------------------------------
fer_op_ion = ion_q_prob.hamiltonian.second_q_op()
fer_op_target = target_q_prob.hamiltonian.second_q_op()
fer_op_ligand = ligand_q_prob.hamiltonian.second_q_op()

converter_ion = QubitConverter(mapper=mapper,
                               two_qubit_reduction=two_q_reduc,
                               z2symmetry_reduction=z_sym)

converter_target = QubitConverter(mapper=mapper,
                                  two_qubit_reduction=two_q_reduc,
                                  z2symmetry_reduction=z_sym)

converter_ligand = QubitConverter(mapper=mapper,
                                  two_qubit_reduction=two_q_reduc,
                                  z2symmetry_reduction=z_sym)

hamiltonian_ion = converter_ion.convert(fer_op_ion, num_particles=ion_q_prob.num_particles)
hamiltonian_target = converter_target.convert(fer_op_target, num_particles=target_q_prob.num_particles)
hamiltonian_ligand = converter_ligand.convert(fer_op_ligand, num_particles=ligand_q_prob.num_particles)

# --------------------------------------------------------------------
# ************************ Analytical Reference **********************
# --------------------------------------------------------------------
numpy_solver_ion = NumPyMinimumEigensolver().compute_minimum_eigenvalue(hamiltonian_ion)
numpy_solver_target = NumPyMinimumEigensolver().compute_minimum_eigenvalue(hamiltonian_target)
numpy_solver_ligand = NumPyMinimumEigensolver().compute_minimum_eigenvalue(hamiltonian_ligand)

# calc_ion_diag = GroundStateEigensolver(ion_q_prob, numpy_solver_ion)
# calc_target_diag = GroundStateEigensolver(target_q_prob, numpy_solver_target)
# calc_ligand_diag = GroundStateEigensolver(ligand_q_prob, numpy_solver_ligand)

start_gse_ion_diag = timeit.default_timer()
gse_ion_diag = ion_q_prob.interpret(numpy_solver_ion)
stop_gse_ion_diag = timeit.default_timer()

start_gse_target_diag = timeit.default_timer()
gse_target_diag = target_q_prob.interpret(numpy_solver_target)
stop_gse_target_diag = timeit.default_timer()

start_gse_ligand_diag = timeit.default_timer()
gse_ligand_diag = ligand_q_prob.interpret(numpy_solver_ligand)
stop_gse_ligand_diag = timeit.default_timer()

# -
# ------> Electronic Energy - Diagonalization <------
elec_en_ion_diag = gse_ion_diag.electronic_energies
elec_en_target_diag = gse_target_diag.electronic_energies
elec_en_ligand_diag = gse_ligand_diag.electronic_energies

# ------> Total Energy - Diagonalization <------
tot_en_ion_diag = gse_ion_diag.total_energies
tot_en_target_diag = gse_target_diag.total_energies
tot_en_ligand_diag = gse_ligand_diag.total_energies

# ------> Computed Energy <------
comp_en_ion_diag = gse_ion_diag.computed_energies
comp_en_target_diag = gse_target_diag.computed_energies
comp_en_ligand_diag = gse_ligand_diag.computed_energies

# ------> Active Space Extracted Energy <------
as_extract_en_ion_diag = elec_en_ion_diag - comp_en_ion_diag
as_extract_en_target_diag = elec_en_target_diag - comp_en_target_diag
as_extract_en_ligand_diag = elec_en_ligand_diag - comp_en_ligand_diag

# --------------------------------------------------------------------
# ************************ VQE Solver **************************
# --------------------------------------------------------------------
# ---> Ansatz <---
# Ion:
ansatz_ion = UCCSD(
    ion_q_prob.num_spatial_orbitals,
    ion_q_prob.num_particles,
    converter_ion,
    reps = reps,
    initial_state=HartreeFock(
        ion_q_prob.num_spatial_orbitals,
        ion_q_prob.num_particles,
        converter_ion,
    ), )

# Target:
ansatz_target = UCCSD(
    target_q_prob.num_spatial_orbitals,
    target_q_prob.num_particles,
    converter_target,
    reps=reps,
    initial_state=HartreeFock(
        target_q_prob.num_spatial_orbitals,
        target_q_prob.num_particles,
        converter_target,
    ), )

# Ligand:
ansatz_ligand = UCCSD(
    ligand_q_prob.num_spatial_orbitals,
    ligand_q_prob.num_particles,
    converter_ligand,
    reps=reps,
    initial_state=HartreeFock(
        ligand_q_prob.num_spatial_orbitals,
        ligand_q_prob.num_particles,
        converter_ligand,
    ), )

# ---> VQE Algorithm <---
vqe_ion = VQE(Estimator(), ansatz_ion, optimizer)
vqe_target = VQE(Estimator(), ansatz_target, optimizer)
vqe_ligand = VQE(Estimator(), ansatz_ligand, optimizer)

vqe_ion.initial_point = [0.0] * ansatz_ion.num_parameters
vqe_target.initial_point = [0.0] * ansatz_target.num_parameters
vqe_ligand.initial_point = [0.0] * ansatz_ligand.num_parameters

calc_ion_vqe = GroundStateEigensolver(converter_ion, vqe_ion)
calc_target_vqe = GroundStateEigensolver(converter_target, vqe_target)
calc_ligand_vqe = GroundStateEigensolver(converter_ligand, vqe_ligand)

start_gse_ion_vqe = timeit.default_timer()
gse_ion_vqe = calc_ion_vqe.solve(ion_q_prob)
stop_gse_ion_vqe = timeit.default_timer()

start_gse_target_vqe = timeit.default_timer()
gse_target_vqe = calc_target_vqe.solve(target_q_prob)
stop_gse_target_vqe = timeit.default_timer()

start_gse_ligand_vqe = timeit.default_timer()
gse_ligand_vqe = calc_ligand_vqe.solve(ligand_q_prob)
stop_gse_ligand_vqe = timeit.default_timer()

# --------------------------------------------------------------------
# ************************ Results **************************
# --------------------------------------------------------------------
# ------> Total energies - Diagonalization <------
tot_en_ion_diag = gse_ion_diag.total_energies
print("Total Energies Ion - Analytical Reference:  \n", gse_ion_diag.total_energies)
tot_en_target_diag = gse_target_diag.total_energies
print("Total Energies Target - Analytical Reference:  \n", gse_target_diag.total_energies)
tot_en_ligand_diag = gse_ligand_diag.total_energies
print("Total Energies Ligand - Analytical Reference:  \n", gse_ligand_diag.total_energies)
print("-" * 60, '\n')

# ------> Total energies - VQE <------
start_tot_en_ion = timeit.default_timer()

tot_en_ion_vqe = gse_ion_vqe.total_energies
print("Total Energies Ion - VQE:  \n", tot_en_ion_vqe)
stop_tot_en_ion = timeit.default_timer()

start_tot_en_target = timeit.default_timer()
tot_en_target_vqe = gse_target_vqe.total_energies
print("Total Energies Target - VQE:  \n", tot_en_target_vqe)
stop_tot_en_target = timeit.default_timer()

start_tot_en_ligand_vqe = timeit.default_timer()
tot_en_ligand_vqe = gse_ligand_vqe.total_energies
print("Total Energies Ligand - VQE:  \n", tot_en_ligand_vqe, "\n")
stop_tot_en_ligand = timeit.default_timer()
print("-" * 60, '\n')

# ------> Electronic Energy <------
elec_en_ion_vqe = gse_ion_vqe.electronic_energies
print("Electronic Energy ion - VQE: \n", elec_en_ion_vqe)
elec_en_target_vqe = gse_target_vqe.electronic_energies
print("Electronic Energy Target - VQE:  \n", elec_en_target_vqe)
elec_en_ligand_vqe = gse_ligand_vqe.electronic_energies
print("Electronic Energy Ligand - VQE:  \n", elec_en_ligand_vqe, "\n")
print("-" * 60, '\n')

# ------> Computed Energy <------
comp_en_ion_vqe = gse_ion_vqe.computed_energies
print("Computed Energy Ion- VQE: \n", comp_en_ion_vqe)
comp_en_target_vqe = gse_target_vqe.computed_energies
print("Computed Energy Target - VQE: \n", comp_en_target_vqe)
comp_en_ligand_vqe = gse_ligand_vqe.computed_energies
print("Computed Energy Ligand - VQE:  \n", comp_en_ligand_vqe, "\n")
print("-" * 60, '\n')

# ------> Active Space Extracted Energy <------
as_extract_en_ion_vqe = elec_en_ion_vqe - comp_en_ion_vqe
print("Active Space Extracted Energy Ion - VQE:  \n", as_extract_en_ion_vqe)
as_extract_en_target_vqe = elec_en_target_vqe - comp_en_target_vqe
print("Active Space Extracted Energy Target - VQE:  \n", as_extract_en_target_vqe)
as_extract_en_ligand_vqe = elec_en_ligand_vqe - comp_en_ligand_vqe
print("Active Space Extracted Energy Ligand - VQE:  \n", as_extract_en_ligand_vqe, "\n")
print("-" * 60, '\n')

# ------> Initial Points <------
opt_pts_ion_vqe = calc_ion_vqe.solve(ion_q_prob).raw_result.optimal_parameters
opt_pts_ion_vqe = list(opt_pts_ion_vqe.values())
print("initial_points_ion:  \n", opt_pts_ion_vqe)

opt_pts_target_vqe = calc_target_vqe.solve(target_q_prob).raw_result.optimal_parameters
opt_pts_target_vqe = list(opt_pts_target_vqe.values())
print("initial_points_target :  \n", opt_pts_target_vqe)

opt_pts_ligand_vqe = calc_ligand_vqe.solve(ligand_q_prob).raw_result.optimal_parameters
opt_pts_ligand_vqe = list(opt_pts_ligand_vqe.values())
print("initial_points_ligand:  \n", opt_pts_ligand_vqe, "\n")

# ------> Run Time <------
runtime_gse_ion_diag = [(stop_gse_ion_diag - start_gse_ion_diag) / 60]
runtime_gse_target_diag = [(stop_gse_target_diag - start_gse_target_diag) / 60]
runtime_gse_ligand_diag = [(stop_gse_ligand_diag - start_gse_ligand_diag) / 60]
runtime_gse_ion_vqe = [(stop_gse_ion_vqe - start_gse_ion_vqe) / 60]
runtime_gse_target_vqe = [(stop_gse_target_vqe - start_gse_target_vqe) / 60]
runtime_gse_ligand_vqe = [(stop_gse_ligand_vqe - start_gse_ligand_vqe) / 60]

print("Runtime - Diagnolacition - ion:  \n", runtime_gse_ion_diag, "\n")
print("Runtime - Diagnolacition - target:  \n", runtime_gse_target_diag, "\n")
print("Runtime - Diagnolacition - Ligand:  \n", runtime_gse_ligand_diag, "\n")
print("Runtime - VQE - ion:  \n", runtime_gse_ion_vqe, "\n")
print("Runtime - VQE - target:  \n", runtime_gse_target_vqe, "\n")
print("Runtime - VQE - Ligand:  \n", runtime_gse_ligand_vqe, "\n")
print("-" * 60, '\n')

# --------------------------------------------------------------------
# ************************ Other Results **************************
# --------------------------------------------------------------------

# nuclear_repulsion_energy = calc.solve(ligand_q_prob).nuclear_repulsion_energy
# print("Nuclear Repulsion Energy ----------------------:  \n", nuclear_repulsion_energy, "\n")

# eigenvalues = calc.solve(ligand_q_prob).eigenvalues[0]
# print("Computed Eigenvalue----------------------:  \n", eigenvalues, "\n")

# aux_operators_evaluated = calc.solve(ligand_q_prob).aux_operators_evaluated
# print("aux_operators_evaluated----------------------:  \n", aux_operators_evaluated, "\n")

# groundenergy = calc.solve(ligand_q_prob).groundenergy
# print("Groundenergy ----------------------: \n", groundenergy, "\n")

# --------------------------------------------------------------------
# ************************ Database **************************
# --------------------------------------------------------------------
# ----> Analytical <----
df_tot_energy_diag = pd.DataFrame(list(zip(tot_en_ion_diag,
                                           tot_en_target_diag,
                                           tot_en_ligand_diag)),
                                  columns=['gse ion - Analytical (ha)',
                                           'gse target - Analytical (ha)',
                                           'gse ligand - Analytical (ha)'])

df_elec_energy_diag = pd.DataFrame(list(zip(elec_en_ion_diag,
                                            elec_en_target_diag,
                                            elec_en_ligand_diag)),
                                   columns=['elec energy ion - Analytical(ha)',
                                            'elec energy target - Analytical(ha)',
                                            'elec energy ligand - Analytical(ha)'])

df_computed_energy_diag = pd.DataFrame(list(zip(comp_en_ion_diag,
                                                comp_en_target_diag,
                                                comp_en_ligand_diag)),
                                       columns=['computed energy ion - Analytical (ha)',
                                                'computed energy target - Analytical (ha)',
                                                'computed energy ligand - Analytical (ha)'])

df_as_extract_energy_diag = pd.DataFrame(list(zip(as_extract_en_ion_diag,
                                                  as_extract_en_target_diag,
                                                  as_extract_en_ligand_diag)),
                                         columns=['as ext energy ion - Analytical (ha)',
                                                  'as ext energy target - Analytical (ha)',
                                                  'as ext energy ligand - Analytical(ha)'])

# ----> VQE <----
df_opt_parameters_vqe = pd.DataFrame(list(zip(opt_pts_ion_vqe,
                                              opt_pts_target_vqe,
                                              opt_pts_ligand_vqe)),
                                     columns=['opt_pts_ion_VQE',
                                              'opt_pts_target_VQE',
                                              'opt_pts_ligand_VQE'])

df_tot_energy_vqe = pd.DataFrame(list(zip(tot_en_ion_vqe,
                                          tot_en_target_vqe,
                                          tot_en_ligand_vqe)),
                                 columns=['gse ion - VQE (ha)',
                                          'gse target - VQE (ha)',
                                          'gse ligand - VQE (ha)'])

df_elec_energy_vqe = pd.DataFrame(list(zip(elec_en_ion_vqe,
                                           elec_en_target_vqe,
                                           elec_en_ligand_vqe)),
                                  columns=['elec energy ion (ha)',
                                           'elec energy target (ha)',
                                           'elec energy ligand (ha)'])

df_computed_energy_vqe = pd.DataFrame(list(zip(comp_en_ion_vqe,
                                               comp_en_target_vqe,
                                               comp_en_ligand_vqe)),
                                      columns=['computed energy ion - VQE (ha)',
                                               'computed energy target - VQE (ha)',
                                               'computed energy ligand - VQE (ha)'])

df_as_extract_energy_vqe = pd.DataFrame(list(zip(as_extract_en_ion_vqe,
                                                 as_extract_en_target_vqe,
                                                 as_extract_en_ligand_vqe)),
                                        columns=['as ext energy ion - VQE (ha)',
                                                 'as ext energy target - VQE (ha)',
                                                 'as ext energy ligand - VQE (ha)'])

df_runtime_tot_energy = pd.DataFrame(list(zip(runtime_gse_ion_diag,
                                              runtime_gse_target_diag,
                                              runtime_gse_ligand_diag,
                                              runtime_gse_ion_vqe,
                                              runtime_gse_target_vqe,
                                              runtime_gse_ligand_vqe
                                              )),
                                     columns=[
                                         'Runtime tot energy ion - Diagonalization (min)',
                                         'Runtime tot energy target - Diagonalization (min)',
                                         'Runtime tot energy ligand - Diagonalization (min)',
                                         'Runtime tot energy ion - VQE (min)',
                                         'Runtime tot energy target - VQE (min)',
                                         'Runtime tot energy ligand - VQE (min)',
                                     ])

with pd.ExcelWriter(name_excel_parameters) as writer:
    df_opt_parameters_vqe.to_excel(writer, sheet_name='Optimized_Parameters_VQE', index=False)
with pd.ExcelWriter(name_excel_vqe) as writer:
    # VQE
    df_computed_energy_vqe.to_excel(writer, sheet_name='Computed - VQE', index=False)
    df_tot_energy_vqe.to_excel(writer, sheet_name='GSE - VQE', index=False)
    df_elec_energy_vqe.to_excel(writer, sheet_name='Electronic - VQE', index=False)
    df_as_extract_energy_vqe.to_excel(writer, sheet_name='AS Extracted - VQE', index=False)
    # Analytical
    df_tot_energy_diag.to_excel(writer, sheet_name='GSE - Analytical', index=False)
    df_computed_energy_diag.to_excel(writer, sheet_name='Computed - Analytical', index=False)
    df_elec_energy_diag.to_excel(writer, sheet_name='Electronic - Analytical', index=False)
    df_as_extract_energy_diag.to_excel(writer, sheet_name='AS Extracted - Analytical', index=False)
    # History
    # df_history_vqe_log.to_excel(writer, sheet_name='Opt Log History - VQE', index=False)
    # Runtime
    df_runtime_tot_energy.to_excel(writer, sheet_name='Runtime', index=False)
