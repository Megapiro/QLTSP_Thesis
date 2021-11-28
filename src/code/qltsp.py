from preprocessing.preprocess import Preprocessing
from quantum_annealing.hamiltonian_builder import HamiltonianBuilder
from quantum_annealing.SA_model import SA_Model
from quantum_annealing.Hybrid_model import Hybrid_Model
from quantum_annealing.results.results_visualizer import *
from quantum_annealing.performance.time_performance import TimingPerformance
from quantum_annealing.tuning.chain_strength_tuning import tune_chain_strength
from quantum_annealing.tuning.anneal_schedule_tuning import *
from model.graph import TSP_Graph
from dataset.parse_experiments import *
from quantum_annealing.results.results_writer import *
from quantum_annealing.results.results_analyzer import *
from model.embedding import Embedding


def init_model(solver, exp_name):
    client_conf = {
        'config_path': get_config_path(),
        'profile': solver
    }
    experiments = parse_experiments()
    exp = get_exp_by_name(experiments, exp_name)
    num_nodes = exp.num_nodes

    preprocessing = Preprocessing.get_instance(num_nodes, time_unit='minutes', testing=True)

    # parse nodes and durations
    preprocessing.parse_nodes()
    preprocessing.parse_durations()

    # parse keys and respective saturations
    preprocessing.parse_charges(',')

    return preprocessing.build_model(), client_conf, exp


def chain_tuning(model, init_chain, num_nodes, client_conf):
    chain_strength = tune_chain_strength(model, init_chain, num_nodes, client_conf)

    return chain_strength


def anneal_tuning(model, num_nodes):
    anneal_mode = "pause_and_quench"
    test_tuning = False

    test_dict = {
        'num_points': 1,
        's_low': 0.5,
        's_high': 0.5
    }

    real_dict = {
        'num_points': 3,
        's_low': 0.3,
        's_high': 0.5
    }

    if test_tuning:
        num_points = test_dict['num_points']
        s_low = test_dict['s_low']
        s_high = test_dict['s_high']
    else:
        num_points = real_dict['num_points']
        s_low = real_dict['s_low']
        s_high = real_dict['s_high']

    if anneal_mode == 'pause':
        schedule = tune_anneal_schedule(model, anneal_mode, num_nodes, num_points, s_low, s_high)
        draw_pause_schedule(schedule)
    elif anneal_mode == 'quench':
        schedule = tune_anneal_schedule(model, anneal_mode, num_nodes, num_points, s_low, s_high)
        draw_quench_schedule(schedule)
    elif anneal_mode == 'pause_and_quench':   # pause and quench
        schedule = tune_anneal_schedule(model, anneal_mode, num_nodes, num_points, s_low, s_high)
        draw_pause_quench_schedule(schedule)
    else:
        raise Exception('Wrong Annealing mode')

    return schedule


def con_tuning(model, exp, client_conf):
    init_chain = exp.qpu_experiment.chain_strength
    num_nodes = exp.num_nodes

    # tuning = 0 -> do both chain and annealing
    # tuning = 1 -> do only chain tuning
    # tuning = 2 -> do only annealing tuning
    # tuning = 3 -> do not do tuning
    tuning = 0
    if tuning == 0:
        chain_strength = chain_tuning(model, init_chain, num_nodes, client_conf)
        schedule = anneal_tuning(model, num_nodes)
    elif tuning == 1:
        chain_strength = chain_tuning(model, init_chain, num_nodes, client_conf)
        schedule = None   # choose a default pause schedule
    elif tuning == 2:
        chain_strength = init_chain
        schedule = anneal_tuning(model, num_nodes)
    elif tuning == 3:
        chain_strength = init_chain
        schedule = None
    else:
        raise Exception('Wrong Tuning Configuration')

    return chain_strength, schedule


def build_hamiltonian(distance_matrix, charge_array, norm_dict):
    hamiltonian_complete = HamiltonianBuilder(distance_matrix, charge_array)
    hamiltonian = hamiltonian_complete.get_hamiltonian(norm_dict)

    return hamiltonian


def simulated_annealing(hamiltonian, exp):
    sa_model = SA_Model(hamiltonian)
    sa_result = sa_model.solve(exp.sa_experiment.num_reads, exp.sa_experiment.chain_strength, exp.label,
                               do_print=True)

    return sa_result


def real_annealing(hamiltonian, exp, client_conf):
    hybrid_sampler = False
    qpu_model = QPU_Model(hamiltonian, client_conf, get_embedding(), hybrid_sampler)

    tuned_chain, tuned_schedule = con_tuning(qpu_model, exp, client_conf)
    if tuned_schedule is not None:
        qpu_model.set_anneal_schedule(tuned_schedule)
        # put tuned chain in .solve of next line when we will have a tuned one

    qpu_result = qpu_model.solve(exp.qpu_experiment.num_reads, tuned_chain, exp.label,
                                 do_print=True)

    # build time performance object for the execution on the qpu and print relevant times
    if not hybrid_sampler:
        time_perf = TimingPerformance(exp.label, qpu_result.response)
        print(f'QPU Annealing Time per Sample: {time_perf.get_time("qpu_anneal_time_per_sample")}μs')
        print(f'QPU Programming Time: {time_perf.get_time("qpu_programming_time")}μs')
        print(f'QPU Sampling Time: {time_perf.get_time("qpu_sampling_time")}μs')
        print(f'QPU Access Time: {time_perf.get_time("qpu_access_time")}μs')

        # checks for timings for annealing scheduler
        anneal_range = qpu_model.get_sampler_properties("annealing_time_range")
        slope = get_annealing_slope(anneal_range)

        # print number of qubits used and min,max chain lengths
        embedding = Embedding(qpu_model.BQM, client_conf)
        emb_graph = embedding.embedding
        qubits_num = qubits_number(emb_graph)
        min_chain_length, max_chain_length = chain_lengths(emb_graph)
        print("Minor Embedding requires {}".format(qubits_num))
        print("Minor Embedding has chains whose length is between {}-{}".format(min_chain_length, max_chain_length))

        # visualize the result on the qpu by using the inspector provided by dwave
        embedding_inspector(qpu_result)

    return qpu_result


def best_qpu(hamiltonian, exp, client_conf):
    hybrid_sampler = False
    qpu_model = QPU_Model(hamiltonian, client_conf, get_embedding(), hybrid_sampler)

    # get best chain and schedule from results csv
    best_chain, best_schedule = get_best_parameters(exp.num_nodes)

    qpu_model.set_anneal_schedule(best_schedule)
    qpu_result = qpu_model.solve(exp.qpu_experiment.num_reads, best_chain, exp.label,
                                 do_print=True)
    embedding_inspector(qpu_result)

    return qpu_result


def best_hybrid(hamiltonian, exp, client_conf):
    # todo: implement if necessary
    return 0


def hybrid_annealing(hamiltonian, exp, client_conf):
    qpu_parameters_dict = {
        'chain_strength': exp.qpu_experiment.chain_strength,
        # 'anneal_schedule': [[0.0, 0.0], [60.0, 0.4], [140.0, 0.4], [230.0, 1.0]],
        'label': exp.label + "-Hybrid"
    }

    hybrid_parameters_dict = {
        'hybrid_num_reads': 1,
        'max_iter': 100,
        'convergence': 3,
        'sa_reads': exp.sa_experiment.num_reads,
        'qpu_reads': exp.qpu_experiment.num_reads,
        'qpu_params': qpu_parameters_dict,
        'max_subproblem_size': 50   # 50 is the default value, 196 is 14*14 the maximum we were able to embed
    }

    # build hybrid model and set its parameters
    hybrid_model = Hybrid_Model(hamiltonian, client_conf)
    hybrid_model.set_parameters(hybrid_parameters_dict)

    # once the hybrid model is built we execute it
    hybrid_result = hybrid_model.solve_hybrid(True)

    return hybrid_result


def qltsp(solver, mode, exp_name):
    # first of all we build the model storing our problem in terms of dataframes
    model, client_conf, exp = init_model(solver, exp_name=exp_name)

    # init model and pre tuning
    normalization = True
    total_inspection_time, distance_matrix = model.get_qubo_D(normalization)
    charge_array = model.get_charges_Q(normalization)

    # pre-tuning
    norm_dict = exp.norm_factors

    # build the hamiltonian
    hamiltonian = build_hamiltonian(distance_matrix, charge_array, norm_dict)

    # Execution Modes:
    # 0 -> Simulated Annealing
    # 1 -> QPU
    # 2 -> Hybrid
    # 3 -> Simulated-QPU
    # 4 -> Simulated-Hybrid
    # 5 -> Best QPU
    # 6 -> Best Hybrid
    # 7 -> Simulated-Best QPU
    # 8 -> Simulated-Best Hybrid
    if mode == 0:
        sa_result = simulated_annealing(hamiltonian, exp)
        sa_solution = sa_result.get_solution()
    elif mode == 1:
        qpu_result = real_annealing(hamiltonian, exp, client_conf)
        qpu_solution = qpu_result.get_solution()
    elif mode == 2:
        hybrid_result = hybrid_annealing(hamiltonian, exp, client_conf)
        hybrid_solution = hybrid_result.get_solution()
    elif mode == 3:
        sa_result = simulated_annealing(hamiltonian, exp)
        qpu_result = real_annealing(hamiltonian, exp, client_conf)

        sa_solution = sa_result.get_solution()
        qpu_solution = qpu_result.get_solution()

        histogram_energies(sa_result.response, qpu_result.response)
    elif mode == 4:
        sa_result = simulated_annealing(hamiltonian, exp)
        hybrid_result = hybrid_annealing(hamiltonian, exp, client_conf)

        sa_solution = sa_result.get_solution()
        hybrid_solution = hybrid_result.get_solution()

        histogram_energies(sa_result.response, hybrid_result.response)
    elif mode == 5:
        best_result = best_qpu(hamiltonian, exp, client_conf)
        best_solution = best_result.get_solution()
    elif mode == 6:
        best_result = best_hybrid(hamiltonian, exp, client_conf)
        best_solution = best_result.get_solution()
    elif mode == 7:
        sa_result = simulated_annealing(hamiltonian, exp)
        best_qpu_result = best_qpu(hamiltonian, exp, client_conf)

        sa_solution = sa_result.get_solution()
        best_solution = best_qpu_result.get_solution()

        histogram_energies(sa_result.response, best_qpu_result.response)
    elif mode == 8:
        sa_result = simulated_annealing(hamiltonian, exp)
        best_hybrid_result = best_hybrid(hamiltonian, exp, client_conf)

        sa_solution = sa_result.get_solution()
        best_solution = best_hybrid_result.get_solution()

        histogram_energies(sa_result.response, best_hybrid_result.response)
    else:
        raise Exception('Wrong Execution Mode')
