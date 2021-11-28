import dimod


class ModelBuilder(object):
    model = None

    def __init__(self, hamiltonian):
        self.hamiltonian = hamiltonian
        self.qubo, self.BQM = self.__build_model()

    def __build_model(self):
        self.model = self.hamiltonian.compile()
        qubo, _ = self.model.to_qubo()

        # build the BQM from the qubo
        BQM = dimod.BinaryQuadraticModel.from_qubo(qubo)

        return qubo, BQM

    def __execute_model(self, num_reads, chain_strength, label):
        pass

    def solve(self, num_reads, chain_strength, label, do_print):
        pass
