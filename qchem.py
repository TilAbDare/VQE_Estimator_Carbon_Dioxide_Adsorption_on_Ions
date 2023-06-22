from dataclasses import dataclass
import numpy as np
from matplotlib import pyplot as plt
from pyscf import gto
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import TwoLocal
from qiskit.circuit.library import RealAmplitudes
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator, Session, Options
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.mappers import QubitConverter
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.units import DistanceUnit

@dataclass
class VQELog:
    values: list
    parameters: list

    def update(self, count, parameters, mean, _metadata):
        self.values.append(mean)
        self.parameters.append(parameters)
        print(f"Running circuit {count} of ~350", end="\r", flush=True)



#   Qiskit library - molecule builder:
def q_driver(molecule, charge, spin, basis, geo_format):
    q_Laboratory = PySCFDriver(molecule.format(geo_format),
                               unit=DistanceUnit.ANGSTROM,
                               charge=charge,
                               spin=spin,
                               basis=basis,
                               )

    return q_Laboratory.run()


def ham_construction(problem, mapper, twoq_redu, zsym):
    fermionic_op = problem.hamiltonian.second_q_op()
    converter = QubitConverter(mapper=mapper,
                               two_qubit_reduction=twoq_redu,
                               z2symmetry_reduction = zsym)
    hamiltonian = converter.convert(fermionic_op, num_particles=problem.num_particles)
    return hamiltonian



#   PySCF library - molecule builder:
def c_driver(molecule, charge, spin, basis, geo_format):
    mol = gto.Mole()
    c_laboratory = mol.build(atom=molecule.format(geo_format),
                             charge=charge,
                             spin=spin,
                             basis=basis)
    return c_laboratory



#   Complexity Reduction ---> Active Space Transformer
def as_transformation(problem, ac_elec, ac_orbitals, address):
    transformer = ActiveSpaceTransformer(ac_elec, ac_orbitals, active_orbitals=address)
    return transformer.transform(problem)



#   Solver ---> Output = total energy
def solver(approach, problem, converter):
    gse_solver = GroundStateEigensolver(converter, approach)
    return gse_solver.solve(problem).total_energies[0]



#   Solver ---> Output = Raw data
def solver_raw(approach, problem, converter):
    gse_solver = GroundStateEigensolver(converter, approach)
    return gse_solver.solve(problem)



#   Unit Converter ---> Hartree to kJ/mol
def ha_to_kj(hartree):
    kj = hartree * 2625.5
    return kj
