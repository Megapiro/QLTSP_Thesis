"""
Consider the following notebook for anneal schedule:
https://cloud.dwavesys.com/learning/user/francesco_2epiro_40mail-bip_2ecom/notebooks/leap/techniques/anneal_schedule/01-anneal-schedule.ipynb

Annealing scheduling is performed by passing two mutually exclusive parameters in the .sample function.
The normalized anneal factor s is an abstract parameter ranging from 0 to 1. s(t) is a continuous function starting
at s=0 for t=0 and ending at s=1 at t=t_f. To specify the anneal schedule we can work on s in 2 exclusive ways:
    (i) annealing_time: sets a number in microseconds to specify the linear growth from s=0 to s=1
    (ii) annealing_schedule: specifies a list of (t,s) pairs specifying points, which are then linearly interpolated.
         It can be done in 2 modes: mid-anneal pause and mid-anneal quench

To build the Anneal Schedule we must follow these rules:
    (i) The first point must be (0,0).
    (ii) Normalized anneal fraction s must increase monotonically.
    (iii) In the final point, s must equal 1 and time t must not exceed the maximum value in the annealing_time_range
          property.
    (iv) The number of points must be ≥2 . The upper bound is system-dependent — check the max_anneal_schedule_points
         property.
    (v) The slopes of each line segment must not violate the maximum slope m_max.

IDEA: since we have long chains it means that the freezeout point will move early in time. Hence probably we will see
improvements by setting the anneal pause early in time when using the anneal schedule.
"""

import numpy as np

from bokeh.io import reset_output
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show
from src.code.quantum_annealing.QPU_model import QPU_Model
from dwave.system.composites import EmbeddingComposite
from src.code.quantum_annealing.performance.results_performance import anneal_comparer
from src.code.quantum_annealing.results.result import Result

as_num_reads = 2000
plot_schedules = False
testing = False

if testing:
    anneal_time = [50, 10]
    pause_duration = [25]
    quench_slopes = [0.25]
    quench_fast_slopes = [1.0]
else:
    anneal_time = [10.0, 50.0, 150]
    pause_duration = [10.0, 80.0, 100.0]
    quench_slopes = [1.0, 0.5, 0.25]
    quench_fast_slopes = [1.0, 0.8, 0.5]


def get_annealing_slope(annealing_time_range):
    max_slope = 1.0 / annealing_time_range[0]
    print("Annealing time range: {}".format(annealing_time_range))
    print("Maximum slope:", max_slope)
    return max_slope


def draw_pause_schedule(schedule):
    print("Schedule: %s" % schedule)
    p = figure(title="Example Anneal Schedule with Pause", x_axis_label='Time [us]',
               y_axis_label='Annealing Parameter s')
    p.line(*np.array(schedule).T)
    show(p)
    reset_output()


def draw_quench_schedule(schedule):
    print("Schedule: %s" % schedule)
    p = figure(title="Example Anneal Schedule with Quench", x_axis_label='Time [us]',
               y_axis_label='Annealing Parameter s')
    p.line(*np.array(schedule).T)
    show(p)
    reset_output()


def draw_pause_quench_schedule(schedule):
    print("Schedule: %s" % schedule)
    p = figure(title="Example Anneal Schedule with Pause and Quench", x_axis_label='Time [us]',
               y_axis_label='Annealing Parameter s')
    p.line(*np.array(schedule).T)
    show(p)
    reset_output()


def __compare_results_on_anneal_schedule(num_nodes, results_schedules, anneal_mode, anneal_array, pause_or_quench,
                                         pause_and_quench=None):
    best_index = anneal_comparer(results_schedules, anneal_mode, num_nodes, anneal_array, pause_or_quench,
                                 pause_and_quench)

    return (results_schedules[1])[best_index]


def __get_pause_schedule(model: QPU_Model, num_points, s_low, s_high):
    results_schedules = []
    sampler_embedded = EmbeddingComposite(model.sampler)
    BQM = model.BQM
    pause_start = np.linspace(s_low, s_high, num=num_points)

    schedules = []
    first = True
    for anneal in anneal_time:
        an_schedules = []
        for pause in pause_duration:
            for start in pause_start:
                schedule = [[0.0, 0.0], [start * anneal, start], [start * anneal + pause, start], [anneal + pause, 1.0]]
                response = sampler_embedded.sample(BQM, anneal_schedule=schedule, num_reads=as_num_reads,
                                                   answer_mode='raw',
                                                   label='Anneal Schedule Pause Tuning - num_reads = ' +
                                                         str(as_num_reads),
                                                   num_spin_reversal_transforms=1)
                result = Result(response, False)
                if first:
                    results_schedules.append([result])
                    results_schedules.append([schedule])

                    first = False
                else:
                    results_schedules[0].append(result)
                    results_schedules[1].append(schedule)

                if plot_schedules:
                    p = figure(title=f"Anneal Schedule of Time={anneal} with Pause={pause}", x_axis_label='Time [us]',
                               y_axis_label='Annealing Parameter s')
                    p.line(*np.array(schedule).T)
                    an_schedules.append(p)

        schedules.append(an_schedules)

    if plot_schedules:
        grid = gridplot(schedules)
        show(grid)
        reset_output()

    return results_schedules


def __get_quench_schedule(model, num_points, s_low, s_high):
    results_schedules = []
    sampler_embedded = EmbeddingComposite(model.sampler)
    BQM = model.BQM
    quench_start = np.linspace(s_low, s_high, num=num_points)

    schedules = []
    first = True
    for anneal in anneal_time:
        an_schedules = []
        for quench in quench_slopes:
            for start in quench_start:
                schedule = [[0.0, 0.0], [start * anneal, start], [(1 - start + quench * start * anneal) / quench, 1.0]]
                response = sampler_embedded.sample(BQM, anneal_schedule=schedule, num_reads=as_num_reads,
                                                   answer_mode='raw',
                                                   label='Anneal Schedule Quench Tuning - num_reads = ' +
                                                         str(as_num_reads),
                                                   num_spin_reversal_transforms=1)
                result = Result(response, False)
                if first:
                    results_schedules.append([result])
                    results_schedules.append([schedule])

                    first = False
                else:
                    results_schedules[0].append(result)
                    results_schedules[1].append(schedule)

                if plot_schedules:
                    p = figure(title=f"Anneal Schedule of Time={anneal} with Quench={quench}", x_axis_label='Time [us]',
                               y_axis_label='Annealing Parameter s')
                    p.line(*np.array(schedule).T)
                    an_schedules.append(p)

        schedules.append(an_schedules)

    if plot_schedules:
        grid = gridplot(schedules)
        show(grid)
        reset_output()

    return results_schedules


def __get_pause_quench_schedule(model, num_points, s_low, s_high):
    results_schedules = []
    sampler_embedded = EmbeddingComposite(model.sampler)
    BQM = model.BQM
    pause_quench_start = np.linspace(s_low, s_high, num=num_points)

    # delta_s specifies the difference between s0 and s1
    delta_s = 0.2

    schedules = []
    first = True
    for anneal in anneal_time:
        an_schedules = []
        for pause, quench in zip(pause_duration, quench_fast_slopes):
            for start in pause_quench_start:
                s0 = start
                s1 = start + delta_s
                schedule = [[0.0, 0.0], [s0 * anneal, s0], [s0 * anneal + pause, s0], [2 * s0 * anneal + pause, s1],
                            [(1 - s1 + quench * (2 * s0 * anneal + pause)) / quench, 1.0]]
                response = sampler_embedded.sample(BQM, anneal_schedule=schedule, num_reads=as_num_reads,
                                                   answer_mode='raw',
                                                   label='Anneal Schedule Pause&Quench Tuning - num_reads = ' +
                                                         str(as_num_reads),
                                                   num_spin_reversal_transforms=1)
                result = Result(response, False)
                if first:
                    results_schedules.append([result])
                    results_schedules.append([schedule])

                    first = False
                else:
                    results_schedules[0].append(result)
                    results_schedules[1].append(schedule)

                if plot_schedules:
                    p = figure(title=f"Anneal Schedule of Time={anneal} with Pause={pause} and Quench={quench}",
                               x_axis_label='Time [us]',
                               y_axis_label='Annealing Parameter s')
                    p.line(*np.array(schedule).T)
                    an_schedules.append(p)

        schedules.append(an_schedules)

    if plot_schedules:
        grid = gridplot(schedules)
        show(grid)
        reset_output()

    return results_schedules


def tune_anneal_schedule(model: QPU_Model, pause_or_quench, num_nodes, num_points, s_low, s_high):
    # if pause_or_quench is True it means that we want to pause, otherwise we quench
    if pause_or_quench == "pause":
        res_scheds = __get_pause_schedule(model, num_points, s_low, s_high)
        return __compare_results_on_anneal_schedule(num_nodes, res_scheds, pause_or_quench, anneal_time, pause_duration)
    elif pause_or_quench == "quench":
        res_scheds = __get_quench_schedule(model, num_points, s_low, s_high)
        return __compare_results_on_anneal_schedule(num_nodes, res_scheds, pause_or_quench, anneal_time, quench_slopes)
    elif pause_or_quench == "pause_and_quench":
        res_scheds = __get_pause_quench_schedule(model, num_points, s_low, s_high)
        return __compare_results_on_anneal_schedule(num_nodes, res_scheds, pause_or_quench, anneal_time, pause_duration,
                                                    quench_slopes)
    else:
        raise Exception('Wrong Annealing Mode')
