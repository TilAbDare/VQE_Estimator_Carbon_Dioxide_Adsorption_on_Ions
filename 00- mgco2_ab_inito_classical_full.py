# --------------------------------------------------------------------
# ******************  Importing libraries ****************************
# --------------------------------------------------------------------
import timeit

import pandas as pd
from pyscf import scf, cc
import qchem
import numpy as np
from matplotlib import pyplot as plt

# --------------------------------------------------------------------
# ****************************  Inputs *******************************
# --------------------------------------------------------------------
# ---------> Database <---------
implementation = 'ab-initio classical'
title = 'ab-initio Classical'
no_homo = 1
no_lumo = 1
assumption = 'full'
no_q = f'{(no_homo + no_lumo) * 2}q'
problem = "full"
name_excel = f'mgco2_{assumption}_{implementation}_{problem}.xlsx'

# ---------> Visualization <---------
format = 'png'
dpi = 600
bbox_inches = 'tight'
font = 10
title_ligand = f"{title} Simulations: Lignad"

fig_name_lignad = f'mgco2_{assumption}_{implementation}_ligand_{problem}.{format}'


xlabel_ligand = "Distance ($\AA$)"
ylabel_ligand = "Energy (Ha)"

label_ligand_rhf = "RHF"
label_ligand_ccsd = "CCSD"
label_ligand_ccsdt = "CCSD(T)"
label_ligand_mp2= "MP2"

label_lignad_diag = "Analytical Reference"

# ---------> Molecules <---------
ion = 'Mg'

target = 'C; ' \
         'O 1 1.19630; ' \
         'O 1 1.19628 2 179.97438; '

ligand = 'C; ' \
         'O 1 1.19630; ' \
         'O 1 1.19628 2 179.97438; ' \
         'Mg 3 1.97923 1 120.07450 2 180.00000'
ligand = 'C; ' \
         'O 1 1.19630; ' \
         'O 1 1.19628 2 179.97438; ' \
         'Mg 3 {} 1 120.07450 2 180.00000'

bonding_distances = np.arange(0.59628, 7, 0.3)

el = 2
a_el_ion = el
a_el_target = el
a_el_ligand = el

# Number of active orbitals ---> An integer
orb = 2
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

basis = "sto-3g"
charge = 2
spin = 0
geo_format = None

# --------------------------------------------------------------------
# ******************** Molecule Construction *************************
# --------------------------------------------------------------------

ligand_energy_rhf = []
ligand_energy_ccsd = []
ligand_energy_ccsdt = []
ligand_energy_mp2 = []
runtime_rhf = []
runtime_ccsd = []
runtime_ccsdt = []
runtime_mp2 = []

for i, d in enumerate(bonding_distances):
    # ------> Electronic Energy - Diagonalization <------
    ligand_c_prob = qchem.c_driver(ligand, charge_ion, spin, basis, d)

    #   Classical Method by PySCF library ---> RHF
    ligand_rhf = scf.RHF(ligand_c_prob).run()

    # ----> RHF Energy <----
    start_rhf = timeit.default_timer()
    ligand_energy_rhf += [ligand_rhf.e_tot]
    stop_rhf = timeit.default_timer()

    # ----> CCSD Energy <----
    start_ccsd = timeit.default_timer()

    ligand_ccsd = cc.CCSD(ligand_rhf).run()
    ligand_energy_ccsd += [ligand_ccsd.e_tot]

    stop_ccsd = timeit.default_timer()

    # ----> CCSD(T) Energy <----
    start_ccsdt = timeit.default_timer()

    ligand_energy_ccsdt += [ligand_ccsd.ccsd_t() + ligand_ccsd.e_tot]

    stop_ccsdt = timeit.default_timer()

    # ----> MP2 Energy <----
    start_mp2 = timeit.default_timer()

    ligand_energy_mp2 += [ligand_rhf.MP2().run().e_tot]

    stop_mp2 = timeit.default_timer()

    runtime_rhf += [(stop_rhf - start_rhf) / 60]
    runtime_ccsd += [(stop_ccsd - start_ccsd) / 60]
    runtime_ccsdt += [(stop_ccsdt - start_ccsdt) / 60]
    runtime_mp2 += [(stop_mp2 - start_mp2 / 60)]

# freeze 2 core orbitals
# mymp = mp.MP2(ion_rhf, frozen=[0,1]).run()
# freeze 2 core orbitals and 3 unoccupied orbitals
# mymp = mp.MP2(ion_rhf, frozen=[0,1,16,17,18]).run()



# runtime_dft = stop_dft - start_dft


print('\n', "-" * 30, 'Total Energies', "-" * 30 )
print("Total Energies Ligand - RHF: ", ligand_energy_rhf)
print("-" * 60)

print("Total Energies Ligand - CCSD: ", ligand_energy_ccsd)
print("-" * 60)

print("Total Energies Ligand - CCSD(T): ", ligand_energy_ccsdt)

print("-" * 60)

print("Total Energies Ligand - MP2: ", ligand_energy_mp2)

print("-" * 60)
# print("Total Energies Ion - DFT: ", ion_energy_mp2)
# print("Total Energies Target - DFT: ", target_energy_mp2)
# print("Total Energies Ligand - DFT: ", ligand_energy_mp2)


print("Runtime Total - RHF: ", float("{:.2f}".format(runtime_rhf[0])), 'min')
print("Runtime Total - CCSD: ", float("{:.2f}".format(runtime_ccsd[0])), " min")
print("Runtime Total - CCSD(T): ", float("{:.2f}".format(runtime_ccsdt[0])), 'min')
print("Runtime Total - MP2: ", float("{:.2f}".format(runtime_mp2[0])), "min ")
# print("Runtime Total - DFT: ", float("{:.2f}".format(runtime_dft)) , "min ")

# --------------------------------------------------------------------
# ************************ Database **************************
# --------------------------------------------------------------------

df_ligand_energy = pd.DataFrame(list(zip(bonding_distances,
                                         ligand_energy_rhf,
                                         ligand_energy_ccsd,
                                         ligand_energy_ccsdt,
                                         ligand_energy_mp2,
                                         # ligand_energy_dft
                                         )),
                                columns=['bonding_distances',
                                         'ligand_energy_rhf',
                                         'ligand_energy_ccsd',
                                         'ligand_energy_ccsdt',
                                         'ligand_energy_mp2',
                                         # 'ligand_energy_dft'
                                         ])

df_runtime = pd.DataFrame(list(zip(runtime_rhf,
                                   runtime_ccsd,
                                   runtime_ccsdt,
                                   runtime_mp2,
                                   # runtime_dft
                                   )),
                          columns=['runtime_rhf (min)',
                                   'runtime_ccsd (min)',
                                   'runtime_ccsdt (min)',
                                   'runtime_mp2 (min)',
                                   # 'runtime_dft (min)'
                                   ])

with pd.ExcelWriter(name_excel) as writer:
    df_ligand_energy.to_excel(writer, sheet_name='ligand_energy', index=False)
    df_runtime.to_excel(writer, sheet_name='runtime', index=False)

plt.rcParams["font.size"] = font

# Plot energy and reference value
plt.plot(bonding_distances, ligand_energy_rhf, color="tab:blue", ls="-.", label=label_ligand_rhf)
plt.plot(bonding_distances, ligand_energy_ccsd, color="tab:red", ls="-", label=label_ligand_ccsd)
plt.plot(bonding_distances, ligand_energy_ccsdt, color="tab:green", ls="--", label=label_ligand_ccsdt)
plt.plot(bonding_distances, ligand_energy_mp2, color="tab:brown", ls=":", label=label_ligand_mp2)

plt.legend(loc="best")
plt.grid(True, linestyle='-.', linewidth=0.2, which='major')
plt.xlabel(xlabel_ligand)
plt.ylabel(ylabel_ligand)
plt.title(title_ligand)
plt.legend()
plt.savefig(fig_name_lignad, format=format, dpi=dpi, bbox_inches=bbox_inches)
plt.show()