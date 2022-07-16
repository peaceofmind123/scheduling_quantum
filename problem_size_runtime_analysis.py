from typing import List
from branch_and_bound_solution import BranchBoundSolver
from TaskSystemGeneration import TaskSystemGenerator
from task_system import TaskSystem
import pickle
import time
import matplotlib.pyplot as plt

""" Initially, independent functional units will be written
    as the full picture of the analysis is not in my head yet"""


def generate_dataset(save_path, num_tasksystems_per_iteration, max_num_tasks, max_num_processors,
                     min_deadline=5, max_deadline=100, exp_scale=2.0):
    """
    Generates the heirarchical dataset for the first analysis
    :return:
    """
    dataset = {}
    for num_tasks in range(2, max_num_tasks + 1):
        dataset[f'{num_tasks}'] = {}
        for num_processors in range(2, max_num_processors + 1):
            # number of tasks should be >= number of processors
            if num_tasks < num_processors:
                continue
            task_system_generator = TaskSystemGenerator(num_tasks, num_processors, min_deadline,
                                                        max_deadline, exp_scale)
            task_systems: List[TaskSystem] = task_system_generator.generate_dataset(num_tasksystems_per_iteration,
                                                                                    '', False)  # do not save
            dataset[f'{num_tasks}'][f'{num_processors}'] = task_systems

    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)

def find_solution_to_dataset(dataset_path, results_save_path, Solver):
    """
        Find the solutions to the entire dataset and save the solutions.
        The solver gives either AnnealingSolver class or a BranchBoundSolution class
        Both have a uniform interface of solve()
        Also, the runtimes will be recorded
        The strategy is to augment the dataset object with solution info and runtime info
        And save it in the same heirarchical (json-like) format
        """

    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    # count the number of infeasible problems in the dataset
    num_infeasible_tasksystems = 0
    total_num_tasksystems = 0
    # iterate through the dataset
    for num_tasks in dataset:  # num_tasks is the first key
        for num_processors in dataset[num_tasks]:
            task_systems_set = dataset[num_tasks][num_processors]

            # this notation is needed for overwriting the values of the list
            for i in range(len(task_systems_set)):
                # record total number of task systems
                total_num_tasksystems += 1
                # initialize the task system
                task_system = task_systems_set[i]
                # initialize the solver
                solver = Solver(task_system)

                try:
                    partitioning, optimal_objective, runtime_ms = solver.solve(False)  # don't display output
                except Exception as e:
                    print(e)
                    # this happens only when the system does not have a feasible solution
                    partitioning = 'infeasible'
                    runtime_ms = 'infeasible'
                    num_infeasible_tasksystems += 1
                # now, replace the task system with the dict containing relevant info
                task_system_info = {
                    'task_system': task_system,
                    'partitioning': partitioning,
                    'runtime_ms': runtime_ms
                }
                task_systems_set[i] = task_system_info
    print(f'Total number of task systems: {total_num_tasksystems}')
    print(f'Number of infeasible task systems: {num_infeasible_tasksystems}')
    with open(results_save_path, 'wb') as f:
        pickle.dump(dataset, f)


def find_branch_bound_solutions_to_dataset(dataset_path, results_save_path):
    """
    Find the branch and bound solutions to the entire dataset and save the solutions.
    Also, the runtimes will be recorded
    The strategy is to augment the dataset object with solution info and runtime info
    And save it in the same heirarchical (json-like) format
    """

    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    # count the number of infeasible problems in the dataset
    num_infeasible_tasksystems = 0
    total_num_tasksystems = 0
    # iterate through the dataset
    for num_tasks in dataset:  # num_tasks is the first key
        for num_processors in dataset[num_tasks]:
            task_systems_set = dataset[num_tasks][num_processors]

            # this notation is needed for overwriting the values of the list
            for i in range(len(task_systems_set)):
                # record total number of task systems
                total_num_tasksystems += 1
                # initialize and solve the task system
                task_system = task_systems_set[i]
                bbs = BranchBoundSolver(task_system)

                try:
                    partitioning, optimal_objective, runtime_ms = bbs.solve(False)  # don't display output
                except TypeError as e:
                    # this happens only when the system does not have a feasible solution
                    partitioning = 'infeasible'
                    runtime_ms = 'infeasible'
                    num_infeasible_tasksystems += 1
                # now, replace the task system with the dict containing relevant info
                task_system_info = {
                    'task_system': task_system,
                    'partitioning': partitioning,
                    'runtime_ms': runtime_ms
                }
                task_systems_set[i] = task_system_info
    print(f'Total number of task systems: {total_num_tasksystems}')
    print(f'Number of infeasible task systems: {num_infeasible_tasksystems}')
    with open(results_save_path, 'wb') as f:
        pickle.dump(dataset, f)


def generate_dataset_script():
    save_path = './datasets/dataset_problem_size_runtime_1.pkl'
    num_task_systems_per_iteration = 10
    max_num_tasks = 10
    max_num_processors = 10
    generate_dataset(save_path, num_task_systems_per_iteration, max_num_tasks, max_num_processors)


def generate_solution_dataset_script():
    dataset_path = './datasets/dataset_problem_size_runtime_1.pkl'
    solution_save_path = './datasets/dataset_solutions_problem_size_runtime_1.pkl'
    find_branch_bound_solutions_to_dataset(dataset_path, solution_save_path)


def get_problem_size_and_runtime_from_solution_dataset(solution_save_path='./datasets/dataset_solutions_problem_size_runtime_1.pkl'):
    with open(solution_save_path, 'rb') as f:
        dataset = pickle.load(f)

    problem_sizes_to_runtime_dict = {}

    for num_tasks in dataset:  # num_tasks is the first key
        for num_processors in dataset[num_tasks]:
            task_systems_set = dataset[num_tasks][num_processors]

            for i in range(len(task_systems_set)):
                task_system_info = task_systems_set[i]
                runtime = task_system_info['runtime_ms']
                # check for feasibility
                if runtime == 'infeasible':
                    continue

                task_system = task_system_info['task_system']
                problem_size = task_system.num_tasks * \
                               task_system.num_processors

                # if the dictionary key already has an associated list, just append
                # otherwise create a list and add
                if problem_size in problem_sizes_to_runtime_dict:
                    problem_sizes_to_runtime_dict[problem_size].append(runtime)
                else:
                    problem_sizes_to_runtime_dict[problem_size] = [runtime]
    return problem_sizes_to_runtime_dict

def generate_problem_size_vs_runtime_curve(problem_size_and_runtime_dict, save_path):
    problem_sizes = []
    runtimes = []

    for problem_size in problem_size_and_runtime_dict:
        problem_sizes.append(problem_size)
        runtimes_individual = problem_size_and_runtime_dict[problem_size]
        runtimes.append(sum(runtimes_individual) / len(runtimes_individual)) # append the average value

    # lambda for generating scatter plot (since just plotting gives horrendous curve due to much noise)
    scatter_lambda = lambda xs, ys, ax: ax.scatter(xs,ys,marker='o')
    generateNCurves('Problem Size','Runtime (ms)',save_path,[problem_sizes],[runtimes],0,plotter_func=scatter_lambda)



# generate N different curves on the same plot
# reused from previous project
def generateNCurves(xlabel, ylabel, save_path, xss,yss, fig_idx,
                    inv=False, legends=None,
                    plotter_func=lambda xs,ys,ax: ax.plot(xs,ys,marker='o')):

    plt.figure(fig_idx)
    _, ax = plt.subplots(1, 1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.tick_params(
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off,
        left=False,
        right=False)

    # plot the curves on the same axes object
    for xs,ys in zip(xss,yss): # get data of each curve
        plotter_func(xs,ys,ax)

    if legends is not None:
        ax.legend(legends)  # provide legends as an array

    if inv:
        ax.invert_xaxis() # invert the x axis so that it is in descending order

    plt.savefig(fname=save_path, dpi=600, bbox_inches='tight')

def end_to_end_analysis():
    dataset_save_path = './datasets/dataset_problem_size_runtime_2.pkl'
    solution_dataset_save_path = './datasets/dataset_solutions_problem_size_runtime_2.pkl'
    max_num_tasks = 50
    max_num_processors = 10
    num_tasksystems_per_iteration = 100
    curve_save_path = './graphs/problem_size_avg_runtime.png'

    # generate the tasksystems dataset
    generate_dataset(dataset_save_path,num_tasksystems_per_iteration,max_num_tasks,max_num_processors)
    # generate dataset with solutions and runtimes
    find_branch_bound_solutions_to_dataset(dataset_save_path,solution_dataset_save_path)

    problem_size_runtime = get_problem_size_and_runtime_from_solution_dataset(solution_dataset_save_path)
    generate_problem_size_vs_runtime_curve(problem_size_runtime,curve_save_path)


if __name__ == '__main__':
    end_to_end_analysis()