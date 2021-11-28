from src.code.model.experiments.single_experiment import SingleExp


class Experiment(object):
    def __init__(self, name, label, num_nodes: int, norm_factors, sa_experiment: SingleExp, qpu_experiment: SingleExp):
        self.name = name
        self.label = label
        self.num_nodes = num_nodes
        self.norm_factors = norm_factors
        self.sa_experiment = sa_experiment
        self.qpu_experiment = qpu_experiment
