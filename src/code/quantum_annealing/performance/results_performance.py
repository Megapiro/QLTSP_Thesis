import numpy as np
import pandas as pd

from src.code.quantum_annealing.performance.time_performance import TimingPerformance


def __compare_results(couple_array, file_path, perf_type):
    """ Compares a list of results executed on the QPU

    Compares a list of results by checking the following properties contained in each response.
    The order in which the properties are listed is the order of importance in the comparison:
        1) Minimal Energy
        2) Mean Energy
        3) Number of correct solutions
        4) Access time to the QPU
        5) Programming Time of the QPU

    """
    results_list = couple_array[0]
    performance_list = couple_array[1]
    if perf_type == 'Norm_A':
        performance_key = "B - C"   # since we have a fixed A, the one used for naming the file and change B and C
    elif perf_type == 'Norm_B':
        performance_key = "A - B - C"
    elif perf_type == 'Chain':
        performance_key = "Chain Strength"
    elif perf_type == 'Annealing':
        performance_key = "Schedule"
    else:   # if other performance types will be needed, add them here
        print("Wrong Performance Type")
        return

    columns = ["Energy", "Mean Energy", "Correct Sol Num", "QPU Access Time", "QPU Programming Time"]
    columns.insert(0, performance_key)

    performance_df = pd.DataFrame(index=range(len(results_list)), columns=columns, data=None)

    comp_dict = {
        "energy": 0.0,
        "mean_energy": 0.0,
        "correct_solutions_number": 0,
        "qpu_access_time": 1000000000,
        "qpu_programming_time": 1000000000,
    }
    best_index = None

    for i in range(len(results_list)):
        result = results_list[i]
        response = result.response
        energy, mean_energy = min(result.pos_energies), np.mean(result.pos_energies)
        correct_solutions, _ = result.split_solution()
        time_perf = TimingPerformance('res_' + str(i), response)

        if energy < comp_dict['energy']:
            comp_dict['energy'] = energy
            best_index = i
        elif energy == comp_dict['energy']:
            if mean_energy < comp_dict['mean_energy']:
                comp_dict['mean_energy'] = mean_energy
                best_index = i
            elif mean_energy == comp_dict['mean_energy']:
                if len(correct_solutions) > comp_dict['correct_solutions_number']:
                    comp_dict["correct_solutions_number"] = len(correct_solutions)
                    best_index = i
                elif len(correct_solutions) == comp_dict['correct_solutions_number']:
                    if time_perf.get_time('qpu_access_time') < comp_dict['qpu_access_time']:
                        comp_dict['qpu_access_time'] = time_perf.get_time('qpu_access_time')
                        best_index = i
                    elif time_perf.get_time('qpu_access_time') == comp_dict['qpu_access_time']:
                        if time_perf.get_time('qpu_programming_time') < comp_dict['qpu_programming_time']:
                            comp_dict['qpu_programming_time'] = time_perf.get_time('qpu_programming_time')
                            best_index = i
        # all the missing elses are: DOES NOTHING -> the best index remains the one of the previous iteration

        performance_df.iloc[i] = {
            performance_key: performance_list[i],
            "Energy": energy,
            "Mean Energy": mean_energy,
            "Correct Sol Num": len(correct_solutions),
            "QPU Access Time": time_perf.get_time('qpu_access_time'),
            "QPU Programming Time": time_perf.get_time('qpu_programming_time')
        }

    performance_df['Best'] = 0
    performance_df.loc[best_index, 'Best'] = 1
    performance_df.to_csv(file_path, index=False)

    return best_index


def dicts_comparer(mode, results_dicts, mode_parameters, num_nodes):
    file_path = '../resources/performance/norm_abc/' + str(num_nodes) + '/abc_tuning_'

    if mode == 'A':
        file_path += str(mode_parameters) + '.csv'
        return __compare_results(results_dicts, file_path, 'Norm_A')
    elif mode == 'B':
        file_path += "-".join(map(str, mode_parameters)) + '.csv'
        return __compare_results(results_dicts, file_path, 'Norm_B')
    else:
        raise Exception('Wrong Mode for tuning on Normalization Parameters')


def chain_comparer(results_strengths, prefactor_list, num_nodes):
    file_path = '../resources/performance/chain_strength/' + str(num_nodes) + '/FINAL_chain_tuning_' + \
                "-".join(map(str, prefactor_list)) + '.csv'

    return __compare_results(results_strengths, file_path, 'Chain')


def anneal_comparer(results_schedules, anneal_mode, num_nodes, anneal_time, pause_or_quench, pause_and_quench=None):
    file_path = '../resources/performance/anneal_schedule/' + str(num_nodes) + '/FINAL_anneal_tuning_' + \
                anneal_mode + "_" + "-".join(map(str, anneal_time)) + "_" + "-".join(map(str, pause_or_quench))

    if pause_and_quench is not None:
        file_path += "_" + "-".join(map(str, pause_and_quench)) + '.csv'
    else:
        file_path += '.csv'

    return __compare_results(results_schedules, file_path, 'Annealing')
