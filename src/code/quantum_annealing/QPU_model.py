from dwave.system.samplers import DWaveSampler
from dwave.system.samplers import LeapHybridSampler
from dwave.system.composites import AutoEmbeddingComposite, EmbeddingComposite

from src.code.quantum_annealing.model_builder import ModelBuilder
from src.code.quantum_annealing.results.result import Result


class QPU_Model(ModelBuilder):
    anneal_schedule = None

    def __init__(self, hamiltonian, client_conf, emb_name, hybrid):
        super(QPU_Model, self).__init__(hamiltonian)
        self.hybrid = hybrid
        if hybrid:
            self.sampler = LeapHybridSampler(config_file=client_conf['config_path'], profile=client_conf['profile'])
        else:
            self.sampler = DWaveSampler(config_file=client_conf['config_path'], profile=client_conf['profile'])
        self.emb_name = emb_name

    def __build_embedding(self, emb_name):
        if self.hybrid:
            return self.sampler
        else:
            if emb_name == 'auto':
                return AutoEmbeddingComposite(self.sampler)
            elif emb_name == 'composite':
                return EmbeddingComposite(self.sampler)
            else:
                print("Wrong Embedding Name, provide a correct one")

    def __execute_model(self, num_reads, chain_strength, label):
        embedding = self.__build_embedding(self.emb_name)

        if self.hybrid:
            return embedding.sample(self.BQM)
        else:
            if self.anneal_schedule is not None:
                return embedding.sample(self.BQM, num_reads=num_reads, chain_strength=chain_strength, label=label,
                                        anneal_schedule=self.anneal_schedule)
            else:
                return embedding.sample(self.BQM, num_reads=num_reads, chain_strength=chain_strength, label=label)

    def solve(self, num_reads, chain_strength, label, do_print):
        qpu_response = self.__execute_model(num_reads, chain_strength, label)

        # once we have the responses we can build the result class and visualize the solution
        qpu_result = Result(qpu_response, do_print=do_print)

        return qpu_result

    def set_anneal_schedule(self, anneal_schedule):
        self.anneal_schedule = anneal_schedule

    def get_sampler_properties(self, prop):
        return self.sampler.properties[prop]
