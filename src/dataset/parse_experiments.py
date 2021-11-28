import json

from src.code.model.experiments.experiment import Experiment
from src.code.model.experiments.single_experiment import SingleExp

exp_file = '../resources/experiments.json'

with open(exp_file) as jsonFile:
    json_exp = json.load(jsonFile)
    jsonFile.close()

def get_config_path():
    return json_exp['config_path']


# embedding can take one of the following strings, representing each a different embedding:
# "auto" -> AutoEmbeddingComposite
# "composite" -> EmbeddingComposite
def get_embedding():
    return json_exp['embedding']


def parse_experiments():
    experiments = []
    exps = json_exp['experiments']

    for exp in exps:
        exp_name = exp['name']
        exp_label = exp['label']
        sa_qpu_exps = exp['sa_qpu']
        num_nodes = int(exp['num_nodes'])
        abc = exp['norm_factors']
        norm_factors = {'A_Normalization': abc[0], 'B_Normalization': abc[1], 'C_Normalization': abc[2]}

        # parse sa experiment parameters and build sa_exp
        sa_exp = SingleExp(False, (sa_qpu_exps[0])['num_reads_SA'], (sa_qpu_exps[0])['chain_strength_SA'])

        # parse qpu experiment parameters and build qpu_exp
        qpu_exp = SingleExp(True, (sa_qpu_exps[1])['num_reads_QPU'], (sa_qpu_exps[1])['chain_strength_QPU'])

        experiments.append(Experiment(exp_name, exp_label, num_nodes, norm_factors, sa_exp, qpu_exp))

    return experiments


def get_exp_by_name(experiments, name):
    for exp in experiments:
        if exp.name == name:
            return exp
