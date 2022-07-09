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
    for i in range(2, max_num_tasks + 1):
        dataset[f'{i}'] = {}
        for j in range(2, max_num_processors + 1):
            task_system_generator = TaskSystemGenerator(i, j, min_deadline,
                                                        max_deadline, exp_scale)
            task_systems: List[TaskSystem] = task_system_generator.generate_dataset(num_tasksystems_per_iteration,
                                                                                    '', False)  # do not save
            dataset[f'{i}'][f'{j}'] = task_systems

    with open(save_path,'wb') as f:
        pickle.dump(dataset, f)
