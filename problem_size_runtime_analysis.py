from typing import List

from TaskSystemGeneration import TaskSystemGenerator
from task_system import TaskSystem
import pickle
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


if __name__ == '__main__':
    save_path = './datasets/dataset_problem_size_runtime_1.pkl'
    num_task_systems_per_iteration = 10
    max_num_tasks = 10
    max_num_processors = 10
    generate_dataset(save_path,num_task_systems_per_iteration, max_num_tasks,max_num_processors)
