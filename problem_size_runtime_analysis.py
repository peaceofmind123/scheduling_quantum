from typing import List

from dwave.system import LeapHybridCQMSampler

from branch_and_bound_solution import BranchBoundSolver
from TaskSystemGeneration import TaskSystemGenerator
from task_system import TaskSystem
import pickle
import time
import matplotlib.pyplot as plt
from annealing_solution import AnnealingSolver, build_cqm, solve_cqm
from tqdm import tqdm

""" Initially, independent functional units will be written
    as the full picture of the analysis is not in my head yet"""


def generate_dataset(save_path, num_tasksystems_per_iteration, max_num_tasks, max_num_processors,
                     min_deadline=5, max_deadline=100, exp_scale=2.0, steps_num_tasks=1,steps_num_processors=1, min_num_tasks=2, min_num_processors=2):
    """
    Generates the heirarchical dataset for the first analysis
    :arg steps_num_tasks: increments in the number of tasks and number of processors
    :return:
    """
    dataset = {}
    for num_tasks in tqdm(range(min_num_tasks, max_num_tasks + 1, steps_num_tasks)):
        dataset[f'{num_tasks}'] = {}
        for num_processors in range(min_num_processors, max_num_processors + 1, steps_num_processors):
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
    for num_tasks in tqdm(dataset):  # num_tasks is the first key
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
    find_solution_to_dataset(dataset_path,results_save_path,BranchBoundSolver)


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

    for num_tasks in tqdm(dataset):  # num_tasks is the first key
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

def find_annealing_lower_bounds(dataset_save_path, annealing_lower_bound_save_path):
    """
    Save the lower bounds given by the annealer on the running time and save it second dataset
    :param dataset_save_path:
    :param annealing_lower_bound_save_path:
    :return:
    """
    sampler = LeapHybridCQMSampler()
    # this will store the lower bounds corresponding to the problem size
    # will be of the form {problem_size1: [lb1, lb2...]}
    # eg: {50: [3,4,3,2],...}
    lower_bounds = {}

    with open(dataset_save_path, 'rb') as f:
        dataset = pickle.load(f)

    for n_tasks in tqdm(dataset):
        num_tasks = int(n_tasks)
        for n_procs in dataset[n_tasks]:
            num_processors = int(n_procs)
            # calculate the problem size
            problem_size = num_tasks * num_processors
            # the set of r task systems with n_tasks tasks and n_procs processors
            task_systems = dataset[n_tasks][n_procs]
            for task_system in task_systems:
                annealing_solver = AnnealingSolver(task_system)
                lower_bound = annealing_solver.get_lower_bound(sampler)
                if problem_size in lower_bounds:
                    lower_bounds[problem_size].append(lower_bound)
                else:
                    lower_bounds[problem_size] = [lower_bound]

    with open(annealing_lower_bound_save_path,'wb') as f:
        pickle.dump(lower_bounds,f)



def end_to_end_analysis():
    dataset_save_path = './datasets/dataset_problem_size_runtime_2.pkl'
    solution_dataset_save_path = './datasets/dataset_solutions_problem_size_runtime_2.pkl'
    annealing_lower_bound_save_path = './datasets/dataset_anneal_lower_bound.pkl'
    max_num_tasks = 503
    max_num_processors = 250
    num_tasksystems_per_iteration = 10
    num_tasks_step_size = 100
    num_processors_step_size = 20
    curve_save_path = './graphs/problem_size_avg_runtime_3.png'

    # generate the tasksystems dataset
    print('starting dataset generation')
    generate_dataset(dataset_save_path, num_tasksystems_per_iteration, max_num_tasks, max_num_processors,
                     steps_num_tasks=num_tasks_step_size,steps_num_processors=num_processors_step_size)
    # generate the lower bounds dataset
    #find_annealing_lower_bounds(dataset_save_path, annealing_lower_bound_save_path)

    # generate dataset with solutions and runtimes
    print('finding solutions')
    find_branch_bound_solutions_to_dataset(dataset_save_path,solution_dataset_save_path)

    print('parsing solutions')
    problem_size_runtime = get_problem_size_and_runtime_from_solution_dataset(solution_dataset_save_path)
    generate_problem_size_vs_runtime_curve(problem_size_runtime,curve_save_path)

def lower_bound_analysis():
    """
    Find out where the lower bound of the annealer starts to grow
    :return:
    """
    min_num_tasks = 100
    min_num_processors = 50
    sampler = LeapHybridCQMSampler()
    lower_bounds = []
    for i in range(10):
        num_tasks = min_num_tasks + i*100
        num_processors = min_num_processors +i*50

        tsg:TaskSystemGenerator = TaskSystemGenerator(num_tasks,num_processors,1000,5000,2.0)
        task_system:TaskSystem = tsg.canonical_generate_tasks()
        annealing_solver = AnnealingSolver(task_system)
        lower_bound = annealing_solver.get_lower_bound(sampler)
        lower_bounds.append(lower_bound)
    print(lower_bounds)

def single_tasksystem_solution_annealing_experiment(
        taskSystem:TaskSystem):
    """
    Experiment of partitioning a single task system of a considerable size with a 5s time limit
    :return:
    """

    # build cqm problem from the tasksystem
    cqm = build_cqm(taskSystem)

    # sample using the leap hybrid cqm sampler
    sampler = LeapHybridCQMSampler()
    solutions = solve_cqm(cqm, sampler, time_limit=5)
    with open('./results/solution_503_250.pkl','wb') as f:
        pickle.dump(solutions,f)

def single_tasksystem_solution_bbs_experiment(
        taskSystem:TaskSystem):
    """
    Branch and bound solution experiment of a single task system
    :return:
    """

    bbs = BranchBoundSolver(taskSystem)

    try:
        partitioning, optimal_objective, runtime_ms = bbs.solve(False)  # don't display output
        print(f'Problem size: {num_tasks * num_processors} Runtime: {runtime_ms} milliseconds')
    except Exception as e:
        print("Infeasible task system")


def estimated_annealer_runtime_analysis():
    """
    Problem size vs estimated annealer runtime analysis
    range: 0-1,000,000
    :return:
    """
    dataset_save_path = './datasets/problem_size_vs_estimated_annealer_runtime/initial/1.pkl'
    solution_dataset_save_path = './datasets/problem_size_vs_estimated_annealer_runtime/solution/1.pkl'
    curve_save_path = './datasets/problem_size_vs_estimated_annealer_runtime/curve/1.png'

    # parameters
    min_num_tasks = 2
    min_num_processors = 2
    max_num_tasks = 10002
    max_num_processors = 122
    steps_num_tasks = 1000
    steps_num_processors = 40
    min_deadline = 100
    max_deadline = 200

    num_tasksystems_per_iteration = 1 # for testing purposes, increase later on

    print('generating task systems')

    generate_dataset(dataset_save_path,num_tasksystems_per_iteration,
                     max_num_tasks,max_num_processors, min_deadline,max_deadline,
                     steps_num_tasks=steps_num_tasks, steps_num_processors=steps_num_processors,
                     min_num_tasks=min_num_tasks,min_num_processors=min_num_processors)

    print('Finding estimated annealer runtimes')
    # generate the lower bounds dataset
    find_annealing_lower_bounds(dataset_save_path, solution_dataset_save_path)

    print('reading runtimes')
    # read the lower bounds and create xss, yss for curve generation
    with open(solution_dataset_save_path,'rb') as f:
        lower_bounds = pickle.load(f)

    print('parsing runtimes to generate graph')
    problem_sizes = []
    worst_case_lower_bounds = []
    for problem_size in lower_bounds:
        problem_sizes.append(problem_size)
        worst_case_lower_bounds.append(max(lower_bounds[problem_size]))

    # generate lower bound curve
    generateNCurves('Problem Size','Estimated Annealer Runtime (s)',
                    curve_save_path,[problem_sizes], [worst_case_lower_bounds],1,plotter_func=lambda xs,ys,ax: ax.scatter(xs,ys))


def single_tasksystem_experiments():
    num_tasks = 7000
    num_processors = 90
    min_deadline = 100
    max_deadline = 200
    exp_scale = 2.0
    # for repeatability of the experiment(s)
    # use different seed for experiments of different sizes
    RANDOM_SEED = 10
    tsg = TaskSystemGenerator(num_tasks, num_processors, min_deadline, max_deadline, exp_scale)
    taskSystem: TaskSystem = tsg.canonical_generate_tasks()

    single_tasksystem_solution_bbs_experiment(taskSystem)
    # single_tasksystem_solution_annealing_experiment(taskSystem)
    pass

def estimated_annealer_runtime_vs_bbs_runtime_analysis():
    """
        Problem size vs estimated annealer runtime (worst case) and
        Problem size vs actual bbs runtime (worst case)
        range: 0-1,000,000
        :return:
        """
    # this dataset needs to be separate because bbs cannot handle 1.2m sized problems

    dataset_for_bbs_save_path = './datasets/problem_size_annealer_bbs_runtimes/initial/1_bbs.pkl'
    dataset_save_path = './datasets/problem_size_annealer_bbs_runtimes/initial/1.pkl'
    bbs_solution_save_path = './datasets/problem_size_annealer_bbs_runtimes/solution/bbs/1.pkl'
    annealer_runtimes_save_path = './datasets/problem_size_annealer_bbs_runtimes/solution/annealer/1.pkl'
    curve_save_path = './datasets/problem_size_annealer_bbs_runtimes/curve/1.png'

    # parameters
    min_num_tasks = 2
    min_num_processors = 2
    max_num_tasks = 5002 # this is 10002 for annealer, implicit in the dataset
    max_num_processors = 122
    steps_num_tasks = 1000
    steps_num_processors = 40
    min_deadline = 100
    max_deadline = 200

    num_tasksystems_per_iteration = 1  # for testing purposes, increase later on

    # generate dataset for bbs
    print('generating task systems for bbs')

    generate_dataset(dataset_for_bbs_save_path, num_tasksystems_per_iteration,
                     max_num_tasks, max_num_processors, min_deadline, max_deadline,
                     steps_num_tasks=steps_num_tasks, steps_num_processors=steps_num_processors,
                     min_num_tasks=min_num_tasks, min_num_processors=min_num_processors)

    # compute solutions
    print("Computing branch and bound solutions")
    find_branch_bound_solutions_to_dataset(dataset_for_bbs_save_path,bbs_solution_save_path)

    print('parsing solutions')
    # get branch and bound solution runtimes
    problem_size_runtime_bbs = get_problem_size_and_runtime_from_solution_dataset(bbs_solution_save_path)
    # get annealer estimated runtimes
    with open(annealer_runtimes_save_path, 'rb') as f:
        lower_bounds = pickle.load(f)

    print('generating graph')

    problem_sizes_annealer = []
    problem_sizes_bbs = []
    annealer_runtimes = []
    bbs_runtimes = []

    for problem_size in lower_bounds:
        problem_sizes_annealer.append(problem_size)
        annealer_runtimes.append(max(lower_bounds[problem_size]))

    for problem_size in problem_size_runtime_bbs:
        problem_sizes_bbs.append(problem_size)
        bbs_runtimes.append(max(problem_size_runtime_bbs[problem_size]) / 1000) # cuz it's in milliseconds

    generateNCurves('Problem Size', 'Runtimes (s)',
                    curve_save_path, [problem_sizes_annealer,problem_sizes_bbs], [annealer_runtimes,bbs_runtimes], 1,
                    plotter_func=lambda xs, ys, ax: ax.scatter(xs, ys), legends=['Quantum Annealing', 'Branch and Bound'])


if __name__ == '__main__':
    #end_to_end_analysis()
    #lower_bound_analysis()
    #single_tasksystem_experiments()
    #estimated_annealer_runtime_analysis()
    estimated_annealer_runtime_vs_bbs_runtime_analysis()