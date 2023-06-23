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
from qiskit.algorithms.optimizers import SPSA, SLSQP
from qiskit.circuit.library import TwoLocal, EfficientSU2, RealAmplitudes
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
# backend = service.get_backend("ibmq_lima")

# backend = service.get_backend("simulator_statevector")
backend = service.get_backend("ibmq_qasm_simulator")

# --------------------------------------------------------------------
# ****************************  Inputs *******************************
# --------------------------------------------------------------------
# ---------> Database <---------
implementation = 'noisy'
title = 'Noisy'

no_homo = 1
no_lumo = 1
assumption = f'{no_homo}H_{no_lumo}L'
no_q = f'{(no_homo + no_lumo) * 2}q'
problem = "full"
name_excel_vqe = f'mgco2_vqe_{assumption}_{no_q}_{implementation}_{problem}.xlsx'
name_excel_parameters = f'mgco2_parameters_{assumption}_{no_q}_{implementation}_{problem}.xlsx'

# ---------> Visualization <---------
format = 'png'
dpi = 600
bbox_inches = 'tight'
font = 10


title_ion = f"VQE Energy - {title} Simulation"
title_target = f"VQE Energy - {title} Simulation"
title_ligand = f"{title} Simulation - {no_homo}H {no_lumo}L - {(no_homo + no_lumo) * 2} qubits: Lignad"

fig_name_ion = f'mgco2_vqe_{assumption}_{no_q}_{implementation}_ion_{problem}.{format}'
fig_name_target = f'mgco2_vqe_{assumption}_{no_q}_{implementation}_target_{problem}.{format}'
fig_name_lignad = f'mgco2_vqe_{assumption}_{no_q}_{implementation}_ligand_{problem}.{format}'


xlabel_ion = "Distance ($\AA$)"
ylabel_ion = "Energy (Ha)"

xlabel_target = "Distance ($\AA$)"
ylabel_target = "Energy (Ha)"

xlabel_ligand = "Distance ($\AA$)"
ylabel_ligand = "Energy (Ha)"

label_ion_vqe = "VQE Eigenvalue"
label_ion_diag = "Analytical Reference"

label_target_vqe = "VQE Eigenvalue"
label_target_diag = "Analytical Reference"

label_ligand_vqe = "VQE Eigenvalue"
label_lignad_diag = "Analytical Reference"

# ---------> Molecules <---------
ion = 'Mg'

target = 'C; ' \
         'O 1 1.19630; ' \
         'O 1 1.19628 2 179.97438;'

ligand = 'C; ' \
         'O 1 1.19630; ' \
         'O 1 1.19628 2 179.97438; ' \
         'Mg 3 1.97923 1 120.07450 2 180.00000'

ligand = 'C; ' \
         'O 1 1.19630; ' \
         'O 1 1.19628 2 179.97438; ' \
         'Mg 3 {} 1 120.07450 2 180.00000'

el = 2*no_homo
a_el_ion = el
a_el_target = el
a_el_ligand = el

# Number of active orbitals ---> An integer
orb = (no_homo + no_lumo)
a_or_ion = orb
a_or_target = orb
a_or_ligand = orb

# List of all active orbitals ---> A list of integers
a_mos_ion = [4,
             5]

a_mos_target = [10,
                11]

a_mos_ligand = [15,
                16]

charge_ion = 2
charge_target = 0
charge_ligand = 2

bonding_distances = np.arange(0.59628, 7, 0.3)
basis = "sto-3g"
charge = 2
spin = 0
geo_format = bonding_distances

# ---------> Classical Solver <---------
numpy_solver = NumPyMinimumEigensolver()

# ---------> Ansatz Parameterization <---------
num_qubits = a_or_ion * 2
reps = 1
entanglement = "full"
insert_barriers = True
rotation_blocks = "ry"
entanglement_blocks = "cz"

# I. Optimized initial points:
database = 'optimized_parameters_database.xlsx'
db_opt_pts = pd.read_excel(database, sheet_name='Optimized_Parameters_VQE')

init_pts_ligand = db_opt_pts['opt_pts_ligand_VQE']

# II. Random initial points:
no_init_points = 4
#init_pts_ligand = np.random.uniform(-np.pi, np.pi, no_init_points)

# print(no_init_points)

# ---------> VQE Circuit  <---------
mapper = ParityMapper()
two_q_reduc = True
z_sym = None
shots = 1000

# ---> Ansatz Parameterization <---
opt_iter = 0
optimization_level = 2
optimizer = SPSA(opt_iter)

# --------------------------------------------------------------------
# ******************** Analytical Reference *************************
# --------------------------------------------------------------------
comp_en_ligand_diag = []
elec_en_ligand_diag = []
tot_en_ligand_diag = []
as_extract_en_ligand_diag = []
nu_ligand_diag = []
for i, d in enumerate(bonding_distances):
    # ------> Electronic Energy - Diagonalization <------
    ligand_q_prob = qchem.q_driver(ligand, charge_ligand, spin, basis, d)
    # ------> Active Space Trasnformation <------
    ligand_q_prob = qchem.as_transformation(ligand_q_prob, a_el_ligand, a_or_ligand, a_mos_ligand)
    # ------> Hamiltonian & Qubit Encoding <------
    converter_ligand = QubitConverter(mapper=mapper,
                                      two_qubit_reduction=two_q_reduc,
                                      z2symmetry_reduction=z_sym)
    fer_op_ligand = ligand_q_prob.hamiltonian.second_q_op()
    hamiltonian_ligand = converter_ligand.convert(fer_op_ligand, num_particles=ligand_q_prob.num_particles)
    # ------> Analytical Solver <------
    numpy_solver_ligand = NumPyMinimumEigensolver().compute_minimum_eigenvalue(hamiltonian_ligand)
    gse_ligand_diag = ligand_q_prob.interpret(numpy_solver_ligand)

    # ------> Electronic Energy - Diagonalization <------
    elec_en_ligand_diag += [gse_ligand_diag.electronic_energies[0]]
    # ------> Total Energy - Diagonalization <------
    tot_en_ligand_diag += [gse_ligand_diag.total_energies[0]]
    # ------> Computed Energy <------
    comp_en_ligand_diag += [gse_ligand_diag.computed_energies[0]]
    # ------> Active Space Extracted Energy <------
    as_extract_en_ligand_diag += [elec_en_ligand_diag[i] - comp_en_ligand_diag[i]]
    # ------> Active Space Extracted Energy <------
    nu_ligand_diag += [gse_ligand_diag.nuclear_repulsion_energy]

print(comp_en_ligand_diag)
print(elec_en_ligand_diag)
print(tot_en_ligand_diag)
print(as_extract_en_ligand_diag)
print(nu_ligand_diag)

# --------------------------------------------------------------------
# ******************** VQE - Noisy Simulation *************************
# --------------------------------------------------------------------
# ---> I. Real Amplitude Ansatz <---
# ansatz = RealAmplitudes(num_qubits=num_qubits,
#                        entanglement=entanglement,
#                        reps=reps)

# ---> II. Hardware Efficient Ansatz <---
# ansatz = EfficientSU2(num_qubits=num_qubits,
#                      reps=reps,
#                      entanglement=entanglement,
#                      insert_barriers = insert_barriers)

# ---> III. Two Local Ansatz <---
ansatz = TwoLocal(num_qubits=num_qubits,
                  reps=reps,
                  rotation_blocks=rotation_blocks,
                  entanglement_blocks=entanglement_blocks,
                  entanglement=entanglement,
                  )



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
comp_en_ligand_vqe = []
tot_en_ligand_vqe = []
opt_pts_ligand_vqe = []
history_log_ligand_vqe = []

for i, d in enumerate(bonding_distances):
    # ------> Electronic Energy - Diagonalization <------
    ligand_q_prob = qchem.q_driver(ligand, charge_ligand, spin, basis, d)
    # ------> Active Space Trasnformation <------
    ligand_q_prob = qchem.as_transformation(ligand_q_prob, a_el_ligand, a_or_ligand, a_mos_ligand)
    # ------> Hamiltonian & Qubit Encoding <------
    converter_ligand = QubitConverter(mapper=mapper,
                                      two_qubit_reduction=two_q_reduc,
                                      z2symmetry_reduction=z_sym)
    fer_op_ligand = ligand_q_prob.hamiltonian.second_q_op()
    hamiltonian_ligand = converter_ligand.convert(fer_op_ligand, num_particles=ligand_q_prob.num_particles)

    log_lignad = VQELog([], [])
    # ---> VQE Implementation <---
    with Session(service=service, backend=backend) as session:
        options = Options()
        options.optimization_level = optimization_level

        # ---> Noisy Simulation <---
        vqe_ligand = VQE(
            Estimator(
            #    session=session, options=options
            ),
            ansatz,
            optimizer,
            callback=log_lignad.update,
            initial_point=init_pts_ligand,
        )
        gse_ligand_vqe = vqe_ligand.compute_minimum_eigenvalue(hamiltonian_ligand)

        # ------> Computed Energy <------
        comp_en_ligand_vqe += [gse_ligand_vqe.optimal_value]
        tot_en_ligand_vqe += [comp_en_ligand_vqe[i] + as_extract_en_ligand_diag[i] + nu_ligand_diag[i]]

        # ------> Initial Points <------
        opt_pts_ligand_vqe += [list(gse_ligand_vqe.optimal_parameters.values())]
        # init_pts_ligand_vqe = opt_pts_ligand_vqe.values()

        history_log_ligand_vqe += [log_lignad.values]

print("Total Energies Ligand:  \n", tot_en_ligand_vqe, "\n")

# --------------------------------------------------------------------
# ************************ Database **************************
# --------------------------------------------------------------------
df_opt_parameters_vqe = pd.DataFrame(list(zip(opt_pts_ligand_vqe[0])),
                                     columns=['opt_pts_ligand_VQE'])

df_tot_energy = pd.DataFrame(list(zip(bonding_distances,
                                      tot_en_ligand_vqe,
                                      comp_en_ligand_vqe,
                                      tot_en_ligand_diag,
                                      comp_en_ligand_diag,
                                      elec_en_ligand_diag,
                                      as_extract_en_ligand_diag,
                                      nu_ligand_diag)),
                             columns=['Distance',
                                      'gse ligand - VQE (ha)',
                                      'computed energy ligand - VQE (ha)',
                                      'gse ligand - Analtical (ha)',
                                      'computed ligand - Analtical (ha)',
                                      'elec ligand - Analtical (ha)',
                                      'as ext energy ligand - Analytical (ha)',
                                      'Nuclear Repulsion - Analytical (ha)'])

df_history_vqe_log = pd.DataFrame(list(zip(history_log_ligand_vqe)),
                                  columns=['Energy Log History ion - VQE '])

with pd.ExcelWriter(name_excel_vqe) as writer:
    # VQE
    df_tot_energy.to_excel(writer, sheet_name='Total Energy', index=False)
    df_history_vqe_log.to_excel(writer, sheet_name='Opt Log History - VQE', index=False)

with pd.ExcelWriter(name_excel_parameters) as writer:
    # VQE
    df_opt_parameters_vqe.to_excel(writer, sheet_name='Opt_parameters', index=False)

# --------------------------------------------------------------------
# ************************ Visualization **************************
# --------------------------------------------------------------------
plt.rcParams["font.size"] = font

# Plot energy and reference value
plt.plot(bonding_distances, tot_en_ligand_vqe, color="tab:blue", ls="-.", label=label_ligand_vqe)
plt.plot(bonding_distances, tot_en_ligand_diag, color="tab:red", ls="--", label=label_lignad_diag)
plt.legend(loc="best")
plt.grid(True, linestyle='-.', linewidth=0.2, which='major')
plt.xlabel(xlabel_ligand)
plt.ylabel(ylabel_ligand)
plt.title(title_ligand)
plt.legend()
plt.savefig(fig_name_lignad, format=format, dpi=dpi, bbox_inches=bbox_inches)
plt.show()
