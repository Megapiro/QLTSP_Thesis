"""
Possible things to be used for tuning and trying to obtain better qpu results:
    (i) Chain Strength
    (ii) Grid search between chain strength and anneal_schedule
    (iii) Variables pruning
    (iv) Pauses in annealing scheduler
"""

import random

from dwave.embedding.chain_strength import uniform_torque_compensation
from src.code.quantum_annealing.QPU_model import QPU_Model
from src.code.quantum_annealing.performance.results_performance import chain_comparer
from dwave.system import EmbeddingComposite, DWaveSampler

cs_num_reads = 5000


def tune_chain_strength(model, exp_chain_strength, num_nodes, client_conf):
    int_list = random.sample(range(100, 250), 10)
    prefactor_list = [x/1000 for x in int_list]
    prefactor_list.insert(0, 1.414)     # insert default prefactor

    chain_strength = __fine_tune(model, num_nodes, client_conf, exp_chain_strength, prefactor_list)

    return chain_strength


def __compare_results_on_chain_strength(results_strengths, prefactor_list, num_nodes):
    best_index = chain_comparer(results_strengths, prefactor_list, num_nodes)

    return (results_strengths[1])[best_index]


def __fine_tune(model: QPU_Model, num_nodes, client_conf, exp_chain_strength, prefactor_list):
    # chain strength tuning is done on chain strength obtained in this order:
    # 1) Chain strength from experiment
    # 2) Chain strength obtained with torque with default prefactor
    # 3) Chain strenghts obtained with torque and random prefactors
    results_strengths = []
    BQM = model.BQM

    # first of all we add the result corresponding to the chain strength present in the experiment
    exp_res = model.solve(cs_num_reads, exp_chain_strength, 'chain_strength fine tuning, num_reads = ' +
                          str(cs_num_reads), do_print=False)
    results_strengths.append([exp_res])
    results_strengths.append([exp_chain_strength])

    # for each prefactor we get a chain strength, execute the model and after comparing all the models we keep the chain
    # that corresponds to the model which returned the best result
    for prefactor in prefactor_list:
        chain_strength = uniform_torque_compensation(BQM,
                                                     embedding=EmbeddingComposite(
                                                         DWaveSampler(config_file=client_conf['config_path'],
                                                                      profile=client_conf['profile'])),
                                                     prefactor=prefactor)
        res_p = model.solve(cs_num_reads, chain_strength, 'chain_strength fine tuning, num_reads = ' +
                            str(cs_num_reads), do_print=False)

        results_strengths[0].append(res_p)
        results_strengths[1].append(chain_strength)

    best_strength = __compare_results_on_chain_strength(results_strengths, prefactor_list, num_nodes)

    return best_strength
