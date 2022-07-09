from typing import List
from branch_and_bound_solution import BranchBoundSolver
from TaskSystemGeneration import TaskSystemGenerator
from task_system import TaskSystem
import pickle
import time
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

    with open(save_path,'wb') as f:
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

    # iterate through the dataset
    for num_tasks in dataset: # num_tasks is the first key
        for num_processors in dataset[num_tasks]:
            task_systems_set = dataset[num_tasks][num_processors]

            # this notation is needed for overwriting the values of the list
            for i in range(len(task_systems_set)):
                # initialize and solve the task system
                task_system = task_systems_set[i]
                bbs = BranchBoundSolver(task_system)

                start = time.time_ns()
                partitioning, optimal_objective = bbs.solve(False) # don't display output
                end = time.time_ns()
                elapsed = end - start

                # now, replace the task system with the dict containing relevant info
                task_system_info = {
                    'task_system': task_system,
                    'partitioning': partitioning,
                    'runtime_ns': elapsed
                }
                task_systems_set[i] = task_system_info

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
    find_branch_bound_solutions_to_dataset(dataset_path,solution_save_path)


if __name__ == '__main__':
    generate_solution_dataset_script()