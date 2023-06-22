# --------------------------------------------------------------------
# ******************  Importing libraries ****************************
# --------------------------------------------------------------------
import timeit
from dataclasses import dataclass
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import TwoLocal
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator, Session, Options
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.mappers import QubitConverter
import qchem

# --------------------------------------------------------------------
# ************************* Backends ********************************
# --------------------------------------------------------------------
service = QiskitRuntimeService()
# backend = service.get_backend("ibmq_quito")
# backend = service.get_backend("ibmq_jakarta")
# backend = service.get_backend("ibm_nairobi")
# backend = service.get_backend("ibmq_manila")
# backend = service.get_backend("ibm_lagos")
# backend = service.get_backend("ibmq_belem")
backend = service.get_backend("ibmq_lima")

# backend = service.get_backend("simulator_statevector")
#backend = service.get_backend("ibmq_qasm_simulator")

# --------------------------------------------------------------------
# ****************************  Inputs *******************************
# --------------------------------------------------------------------
# -----> Database <-----
implementation = "physical"
assumption = '1H_1L'
no_q = '4q'

name_excel_vqe = f'mgco2_vqe_{assumption}_{no_q}_{implementation}_sp.xlsx'
name_excel_parameters = f'mgco2_parameters_{assumption}_{no_q}_{implementation}_sp.xlsx'

# -----> Visualization <-----
format = 'png'
dpi = 600
bbox_inches = 'tight'

title_ion = f"VQE Energy - {implementation} Simulation"
title_target = f"VQE Energy - {implementation} Simulation"
title_ligand = f"VQE Energy - {implementation} Simulation"


fig_name_ion = f'mgco2_2hl_vqe_{implementation}_ion_sp.png'
fig_name_target = f'mgco2_2hl_vqe_{implementation}_target_sp.png'
fig_name_lignad = f'mgco2_2hl_vqe_{implementation}_ligand_sp.png'

xlabel_ion = "Iteration"
ylabel_ion = "Energy [Ha]"

xlabel_target = "Iteration"
ylabel_target = "Energy [Ha]"

xlabel_ligand = "Iteration"
ylabel_ligand = "Energy [Ha]"
label_ion_vqe = "VQE Eigenvalue"
label_ion_diag = "Analytical Reference"

label_target_vqe = "VQE Eigenvalue"
label_target_diag = "Analytical Reference"

label_ligand_vqe = "VQE Eigenvalue"
label_lignad_diag = "Analytical Reference"

# -----> Molecules <-----
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
shots = 1000
opt_iter = 0
# opt_iter = input("#Optimization iteration? ")
basis = "sto-3g"
mapper = ParityMapper()
optimizer = SPSA(opt_iter)
two_q_reduc = True
z_sym = None
num_qubits = a_or_ion * 2
reps = 3
rotation_blocks = "ry"
entanglement_blocks = "cz"
charge = 2
spin = 0
geo_format = None
numpy_solver = NumPyMinimumEigensolver()
optimization_level = 3


# ---> Ansatz Parameterization <---
# I. Optimized initial points:
database = 'mgco2_optimized_parameters_perfect_1H_1L_4q_sp.xlsx'
db_opt_pts = pd.read_excel(database, sheet_name='Optimized_Parameters_VQE')

init_pts_ion = db_opt_pts['opt_pts_ion_VQE']
init_pts_target = db_opt_pts['opt_pts_target_VQE']
init_pts_ligand = db_opt_pts['opt_pts_ligand_VQE']

# II. Random initial points:
no_init_points = len(init_pts_ligand)
#init_pts_ion = np.random.uniform(-np.pi, np.pi, no_init_points)
#init_pts_target = np.random.uniform(-np.pi, np.pi, no_init_points)
#init_pts_ligand = np.random.uniform(-np.pi, np.pi, no_init_points)

print(no_init_points)
# --------------------------------------------------------------------
# ******************** Molecule Construction *************************
# --------------------------------------------------------------------
ion_q_prob = qchem.q_driver(ion, charge_ion, spin, basis, geo_format)
target_q_prob = qchem.q_driver(target, charge_target, spin, basis, geo_format)
ligand_q_prob = qchem.q_driver(ligand, charge_ligand, spin, basis, geo_format)

# --------------------------------------------------------------------
# **************** Active Space Trasnformation ***********************
# --------------------------------------------------------------------
ion_q_prob = qchem.as_transformation(ion_q_prob, a_el_ion, a_el_ion, a_mos_ion)
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

# --------------------------------------------------------------------
# ************************ Analytical Reference **********************
# --------------------------------------------------------------------
numpy_solver_ion = NumPyMinimumEigensolver().compute_minimum_eigenvalue(hamiltonian_ion)
numpy_solver_target = NumPyMinimumEigensolver().compute_minimum_eigenvalue(hamiltonian_target)
numpy_solver_ligand = NumPyMinimumEigensolver().compute_minimum_eigenvalue(hamiltonian_ligand)

total_energies_ion_numpy = GroundStateEigensolver(ion_q_prob, numpy_solver_ion)
total_energies_target_numpy = GroundStateEigensolver(target_q_prob, numpy_solver_target)
total_energies_ligand_numpy = GroundStateEigensolver(ligand_q_prob, numpy_solver_ligand)

gse_diag_ion = ion_q_prob.interpret(numpy_solver_ion)
gse_diag_target = target_q_prob.interpret(numpy_solver_target)
gse_diag_ligand = ligand_q_prob.interpret(numpy_solver_ligand)

# ------> Electronic Energy - Diagonalization <------
elec_en_ion_diag = gse_diag_ion.electronic_energies
elec_en_target_diag = gse_diag_target.electronic_energies
elec_en_ligand_diag = gse_diag_ligand.electronic_energies

# ------> Total Energy - Diagonalization <------
tot_en_ion_diag = gse_diag_ion.total_energies
tot_en_target_diag = gse_diag_target.total_energies
tot_en_ligand_diag = gse_diag_ligand.total_energies

# ------> Computed Energy <------
comp_en_ion_diag = gse_ion_diag.computed_energies
comp_en_target_diag = gse_target_diag.computed_energies
comp_en_ligand_diag = gse_ligand_diag.computed_energies

# ------> Active Space Extracted Energy <------
as_extract_en_ion_diag = elec_en_ion_diag - comp_en_ion_diag
as_extract_en_target_diag = elec_en_target_diag - comp_en_target_diag
as_extract_en_ligand_diag = elec_en_ligand_diag - comp_en_ligand_diag

# --------------------------------------------------------------------
# ******************** VQE - Noisy Simulation *************************
# --------------------------------------------------------------------
# ---> I. Real Amplitude Ansatz <---
# ansatz = RealAmplitudes(num_qubits=2, reps=2)


# ---> II. Two Local Ansatz <---
ansatz = TwoLocal(num_qubits=num_qubits,
                  reps=reps,
                  rotation_blocks=rotation_blocks,
                  entanglement_blocks=entanglement_blocks)

# ---> III. Hardware Efficient Ansatz <---



# ---> Optimization History <---
# We create a simple object to log our intermediate results for plotting later:
# Create an object to store intermediate results
@dataclass
class VQELog:
    values: list
    parameters: list

    def update(self, count, parameters, mean, _metadata):
        self.values.append(mean)
        self.parameters.append(parameters)
        print(f"Running circuit {count} of ~350", end="\r", flush=True)


# ---> VQE Implementation for Ion <---
log_ion = VQELog([], [])
with Session(service=service, backend=backend) as session:
    options = Options()
    options.optimization_level = optimization_level

    # ---> Noisy Simulation <---
    vqe_ion = VQE(
        Estimator(session=session, options=options),
        ansatz,
        optimizer,
        callback=log_ion.update,
        initial_point=init_pts_ion,
    )

    start_gse_ion_vqe = timeit.default_timer()
    gse_ion_vqe = vqe_ion.compute_minimum_eigenvalue(hamiltonian_ion)
    stop_gse_ion_vqe = timeit.default_timer()

# ---> VQE Implementation for Target <---
log_target = VQELog([], [])
# ---> VQE Implementation <---
with Session(service=service, backend=backend) as session:
    options = Options()
    options.optimization_level = optimization_level

    # ---> Noisy Simulation <---
    vqe_target = VQE(
        Estimator(session=session, options=options),
        ansatz,
        optimizer,
        callback=log_target.update,
        initial_point=init_pts_target,
    )

    start_gse_target_vqe = timeit.default_timer()
    gse_target_vqe = vqe_target.compute_minimum_eigenvalue(hamiltonian_target)
    stop_gse_target_vqe = timeit.default_timer()

# ---> VQE Implementation for Target <---
log_lignad = VQELog([], [])
# ---> VQE Implementation <---
with Session(service=service, backend=backend) as session:
    options = Options()
    options.optimization_level = optimization_level

    # ---> Noisy Simulation <---
    vqe_ligand = VQE(
        Estimator(session=session, options=options),
        ansatz,
        optimizer,
        callback=log_lignad.update,
        initial_point=init_pts_ligand,
    )

    start_gse_ligand_vqe = timeit.default_timer()
    gse_ligand_vqe = vqe_ligand.compute_minimum_eigenvalue(hamiltonian_ligand)
    stop_gse_ligand_vqe = timeit.default_timer()

# --------------------------------------------------------------------
# ************************** Results *********************************
# --------------------------------------------------------------------

history_log_ion_vqe = log_ion.values
history_log_target_vqe = log_target.values
history_log_ligand_vqe = log_lignad.values
# for a, b, c in range(len(zip(log_ion.values, log_target.values, log_lignad.values))):
#    history_log_ion_vqe += [a
#                   #+ as_extract_en_ligand_diag + nu_ligand_diag
#                   ]
#    history_log_target_vqe += [b
#                       # + as_extract_en_ligand_diag + nu_ligand_diag
#                       ]
#    history_log_ligand_vqe += [c
#                       # + as_extract_en_ligand_diag + nu_ligand_diag
#                       ]


# ------> Computed Energy <------
comp_en_ion_vqe = [gse_ion_vqe.optimal_value]
comp_en_target_vqe = [gse_target_vqe.optimal_value]
comp_en_ligand_vqe = [gse_ligand_vqe.optimal_value]

print("Eigenvalue Ion :  \n", comp_en_ion_vqe)
print("Eigenvalue Target:  \n", comp_en_target_vqe)
print("Eigenvalue Ligand:  \n", comp_en_ligand_vqe, "\n")
print("-" * 60, '\n')

# ------> Total energies - VQE <------
tot_ion_vqe = comp_en_ion_vqe + as_extract_en_ion_diag
tot_target_vqe = comp_en_target_vqe + as_extract_en_ion_diag
tot_ligand_vqe = comp_en_ligand_vqe + as_extract_en_ion_diag
print("Total Energies Ion:  \n", tot_ion_vqe)
print("Total Energies Target:  \n", tot_target_vqe)
print("Total Energies Ligand:  \n", tot_ligand_vqe, "\n")
print("-" * 60, '\n')

nu_ion_diag = [gse_ion_diag.nuclear_repulsion_energy]
nu_target_diag = [gse_target_diag.nuclear_repulsion_energy]
nu_ligand_diag = [gse_ligand_diag.nuclear_repulsion_energy]
print("Nuclear Repulsion energy - Ion:  \n", nu_ion_diag)
print("Nuclear Repulsion energy - Target:  \n", nu_target_diag)
print("Nuclear Repulsion energy - Ligand:  \n", nu_ligand_diag, "\n")
print("-" * 60, '\n')

# ------> Initial Points <------
opt_pts_ion_vqe = [gse_ion_vqe.optimal_parameters]
# init_pts_ion_vqe = list(init_pts_ion_vqe.values())
print("initial_points_ion:  \n", opt_pts_ion_vqe)

opt_pts_target_vqe = [gse_target_vqe.optimal_parameters]
# init_pts_target_vqe = list(init_pts_target_vqe.values())
print("initial_points_target :  \n", opt_pts_target_vqe)

opt_pts_ligand_vqe = [gse_ligand_vqe.optimal_parameters]
# init_pts_ligand_vqe = list(init_pts_ligand_vqe.values())
print("initial_points_ligand:  \n", opt_pts_ligand_vqe, "\n")
print("-" * 60, '\n')

# ------> Run Time <------
runtime_tot_en_ion = [(stop_gse_ion_vqe - start_gse_ion_vqe) / 60]
runtime_tot_en_target = [(stop_gse_target_vqe - start_gse_target_vqe) / 60]
runtime_tot_en_ligand = [(stop_gse_ligand_vqe - start_gse_ligand_vqe) / 60]

print("Runtime Total - VQE - ion:  \n", runtime_tot_en_ion, "\n")
print("Runtime Total - VQE - target:  \n", runtime_tot_en_target, "\n")
print("Runtime Total - VQE - Ligand:  \n", runtime_tot_en_ligand, "\n")
print("-" * 60, '\n')

runtime_opt_en_ion = [gse_ion_vqe.optimizer_time / 60]
runtime_opt_en_target = [gse_target_vqe.optimizer_time / 60]
runtime_opt_en_ligand = [gse_ligand_vqe.optimizer_time / 60]

print("Runtime Optimizer - VQE - ion:  \n", runtime_opt_en_ion, "\n")
print("Runtime Optimizer - VQE - target:  \n", runtime_opt_en_target, "\n")
print("Runtime Optimizer - VQE - Ligand:  \n", runtime_opt_en_ligand, "\n")

runtime_qc_en_ion = [runtime_opt_en_ion[0] - runtime_tot_en_ion[0]]
runtime_qc_en_target = [runtime_opt_en_target[0] - runtime_tot_en_target[0]]
runtime_qc_en_ligand = [runtime_opt_en_target[0] - runtime_tot_en_ligand[0]]

print("Runtime Quantum Computing - VQE - ion:  \n", runtime_qc_en_ion, "\n")
print("Runtime Quantum Computing - VQE - target:  \n", runtime_qc_en_target, "\n")
print("Runtime Quantum Computing - VQE - Ligand:  \n", runtime_qc_en_ligand, "\n")

# --------------------------------------------------------------------
# ************************ Database **************************
# --------------------------------------------------------------------
df_opt_parameters_vqe = pd.DataFrame(list(zip(opt_pts_ion_vqe,
                                              opt_pts_target_vqe,
                                              opt_pts_ligand_vqe)),
                                     columns=['opt pts ion - VQE ',
                                              'opt pts target - VQE ',
                                              'opt pts ligand - VQE '])

df_tot_energy_diag = pd.DataFrame(list(zip(tot_en_ion_diag,
                                           tot_en_target_diag,
                                           tot_en_ligand_diag)),
                                  columns=['gse ion - Analtical Ref (ha)',
                                           'target target - Analtical Ref (ha)',
                                           'ligand ligand - Analtical Ref (ha)'])

df_tot_energy_vqe = pd.DataFrame(list(zip(tot_ion_vqe,
                                          tot_ligand_vqe,
                                          tot_ligand_vqe)),
                                 columns=['gse ion - VQE (ha)',
                                          'target target - VQE (ha)',
                                          'ligand ligand - VQE (ha)'])

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

df_computed_energy_vqe = pd.DataFrame(list(zip(comp_en_ion_vqe,
                                               comp_en_target_vqe,
                                               comp_en_ligand_vqe)),
                                      columns=['computed energy ion - VQE (ha)',
                                               'computed energy target - VQE (ha)',
                                               'computed energy ligand - VQE (ha)'])

df_as_extract_energy_diag = pd.DataFrame(list(zip(as_extract_en_ion_diag,
                                                  as_extract_en_target_diag,
                                                  as_extract_en_ligand_diag)),
                                         columns=['as ext energy ion - Analytical (ha)',
                                                  'as ext energy target - Analytical (ha)',
                                                  'as ext energy ligand - Analytical(ha)'])

df_runtime_tot_energy = pd.DataFrame(list(zip(runtime_tot_en_ion,
                                              runtime_tot_en_target,
                                              runtime_tot_en_ligand,
                                              runtime_opt_en_ion,
                                              runtime_opt_en_target,
                                              runtime_opt_en_ligand,
                                              runtime_qc_en_ion,
                                              runtime_qc_en_target,
                                              runtime_qc_en_ligand
                                              )),
                                     columns=['Runtime tot energy ion - VQE(min)',
                                              'Runtime tot energy target - VQE(min)',
                                              'Runtime tot energy ligand - VQE(min)',
                                              'Runtime opt energy ion - VQE(min)',
                                              'Runtime opt energy target - VQE(min)',
                                              'Runtime opt energy ligand - VQE(min)',
                                              'Runtime qc energy ion - VQE(min)',
                                              'Runtime qc energy target - VQE(min)',
                                              'Runtime qc energy ligand - VQE(min)',
                                              ])

df_history_vqe_log = pd.DataFrame(list(zip(history_log_ion_vqe,
                                           history_log_target_vqe,
                                           history_log_ligand_vqe)),
                                  columns=['Energy Log History ion - VQE ',
                                           'Energy Log History target - VQE ',
                                           'Energy Log History ligand - VQE '])

with pd.ExcelWriter(name_excel_parameters) as writer:
    df_opt_parameters_vqe.to_excel(writer, sheet_name='Optimized Parameters - VQE', index=False)

with pd.ExcelWriter(name_excel_vqe) as writer:
    # VQE
    df_computed_energy_vqe.to_excel(writer, sheet_name='Computed - VQE', index=False)
    df_tot_energy_vqe.to_excel(writer, sheet_name='GSE - VQE', index=False)
    # Analytical
    df_tot_energy_diag.to_excel(writer, sheet_name='GSE - Analytical', index=False)
    df_computed_energy_diag.to_excel(writer, sheet_name='Computed - Analytical', index=False)
    df_elec_energy_diag.to_excel(writer, sheet_name='Electronic - Analytical', index=False)
    df_as_extract_energy_diag.to_excel(writer, sheet_name='AS Extracted - Analytical', index=False)
    # History
    df_history_vqe_log.to_excel(writer, sheet_name='Opt Log History - VQE', index=False)
    # Runtime
    df_runtime_tot_energy.to_excel(writer, sheet_name='Runtime', index=False)

# --------------------------------------------------------------------
# ************************ Visualization **************************
# --------------------------------------------------------------------
plt.rcParams["font.size"] = 14

# Plot energy and reference value
plt.figure(figsize=(12, 6))
plt.plot(history_log_ion_vqe, label=label_ion_vqe)
plt.axhline(y=comp_en_ion_diag, color="tab:red", ls="--", label=label_ion_diag)
plt.legend(loc="best")
plt.grid(True, linestyle='-.', linewidth=0.2, which='major')
plt.xlabel(xlabel_ion)
plt.ylabel(ylabel_ion)
plt.title(title_ion)
plt.savefig(fig_name_ion, format=format, dpi=dpi, bbox_inches=bbox_inches)
plt.show()

# Plot energy and reference value
plt.figure(figsize=(12, 6))
plt.plot(history_log_target_vqe, label=label_target_vqe)
plt.axhline(y=comp_en_target_diag, color="tab:red", ls="--", label=label_target_diag)
plt.legend(loc="best")
plt.grid(True, linestyle='-.', linewidth=0.2, which='major')
plt.xlabel(xlabel_target)
plt.ylabel(ylabel_target)
plt.title(title_target)
plt.savefig(fig_name_target, format=format, dpi=dpi, bbox_inches=bbox_inches)
plt.show()

# Plot energy and reference value
plt.figure(figsize=(12, 6))
plt.plot(history_log_ligand_vqe, label=label_ligand_vqe)
plt.axhline(y=comp_en_ligand_vqe, color="tab:red", ls="--", label=label_lignad_diag)
plt.legend(loc="best")
plt.grid(True, linestyle='-.', linewidth=0.2, which='major')
plt.xlabel(xlabel_ligand)
plt.ylabel(ylabel_ligand)
plt.title(title_ligand)
plt.savefig(fig_name_lignad, format=format, dpi=dpi, bbox_inches=bbox_inches)
plt.show()

