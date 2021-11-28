from neal import SimulatedAnnealingSampler

from src.code.quantum_annealing.model_builder import ModelBuilder
from src.code.quantum_annealing.results.result import Result


class SA_Model(ModelBuilder):
    def __init__(self, hamiltonian):
        super(SA_Model, self).__init__(hamiltonian)

    def __execute_model(self, num_reads, chain_strength, label):
        sampler = SimulatedAnnealingSampler()

        return sampler.sample(self.BQM, num_reads=num_reads, chain_strength=chain_strength, label=label)

    def solve(self, num_reads, chain_strength, label, do_print):
        sa_response = self.__execute_model(num_reads, chain_strength, label)

        # once we have the responses we can build the result class and visualize the solution
        sa_result = Result(sa_response, do_print=do_print)

        return sa_result
