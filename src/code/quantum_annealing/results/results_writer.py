"""
This class takes all the results obtained from the tuning processes and puts together the best ones found. While
executing the software we are able to specify a parameter if we want to write the results at the end of execution. Once
a result csv file is written for an experiment with num_nodes we use the best values of the parameters found and launch
the final model based on them.
"""

import os
import re
import pandas as pd

from src.code.preprocessing.utils import *


def __get_best_lines(dataframe, parameter_type):
    # drop Mean Enery and Best columns
    dataframe = dataframe.drop(columns=['Mean Energy', 'Best'])

    # first get all the lines that have at least a solution
    df_with_sol = dataframe.loc[dataframe['Correct Sol Num'] >= 1]

    # then all the lines that have the minimum energy
    df_min_energy = dataframe.loc[dataframe['Energy'] == dataframe['Energy'].min()]

    df_best_lines = pd.concat([df_with_sol, df_min_energy], axis=0)
    df_best_lines.insert(1, 'Type', parameter_type)

    return df_best_lines


def write_results(num_nodes):
    # first of all we add to the paths the directory of the experiment executed
    results_path = '../resources/results/' + str(num_nodes) + '/'
    csv_path = '../resources/performance/'
    norm_path = csv_path + 'norm_abc/' + str(num_nodes)
    chain_path = csv_path + 'chain_strength/' + str(num_nodes)
    anneal_path = csv_path + 'anneal_schedule/' + str(num_nodes)

    # once we have the paths we can read the datasets
    chain_csv = os.listdir(chain_path)
    anneal_csv = os.listdir(anneal_path)

    chain_df = read_input(os.path.join(chain_path, chain_csv[0]), ',')  # change it if only one chain csv is present
    chain_df = chain_df.rename(columns={'Chain Strength': 'Parameter'})
    pause_df = read_input(os.path.join(anneal_path, anneal_csv[0]), ',')  # anneal csv are always ordered as: p,pq and q
    pause_df = pause_df.rename(columns={'Schedule': 'Parameter'})
    quench_df = read_input(os.path.join(anneal_path, anneal_csv[2]), ',')
    quench_df = quench_df.rename(columns={'Schedule': 'Parameter'})
    pause_quench_df = read_input(os.path.join(anneal_path, anneal_csv[1]), ',')
    pause_quench_df = pause_quench_df.rename(columns={'Schedule': 'Parameter'})

    # create the results dataframe
    results_df = pd.DataFrame(columns=['Parameter', 'Type', 'Energy', 'Correct Sol Num', 'QPU Access Time',
                                       'QPU Programming Time'])

    # for each csv line we keep all lines that have at least one correct solution and those that have the minimum energy
    # 0 -> chain type
    # 1 -> pause, quench, pause or quench
    results_df = pd.concat([results_df, __get_best_lines(chain_df, 0)], axis=0)
    results_df = pd.concat([results_df, __get_best_lines(pause_df, 1)], axis=0)
    results_df = pd.concat([results_df, __get_best_lines(quench_df, 2)], axis=0)
    results_df = pd.concat([results_df, __get_best_lines(pause_quench_df, 3)], axis=0)

    # finally once the results dataframe is built we can write it
    results_path += 'performance_results.csv'
    results_df.to_csv(results_path, index=False)


def __get_best(dataframe):
    temp_df = dataframe.loc[dataframe['Correct Sol Num'] >= 1]

    if len(temp_df.index) == 0:
        temp_df = dataframe.loc[dataframe['Energy'] == dataframe['Energy'].min()]
    else:
        temp_df = temp_df.loc[temp_df['Energy'] == temp_df['Energy'].min()]

    if len(temp_df.index) > 1:
        temp_df = temp_df.loc[temp_df['QPU Access Time'] == temp_df['QPU Access Time'].min()]

    return temp_df['Parameter']


def get_best_parameters(num_nodes):
    csv_results_path = '../resources/results/' + str(num_nodes) + '/performance_results.csv'
    results_df = read_input(csv_results_path, ',')

    best_chain = round(float(__get_best(results_df.loc[results_df['Type'] == 0]).values[0]), 2)
    best_schedule = __get_best(results_df.loc[results_df['Type'] != 0]).values[0]
    bs = best_schedule.replace("[", "").replace("]", "").split(',')

    best_schedule = [[float(bs[i]), float(bs[i + 1])] for i in range(0, len(bs), 2)]

    return best_chain, best_schedule
