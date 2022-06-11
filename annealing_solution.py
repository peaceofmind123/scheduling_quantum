# This code inspired from https://github.com/dwave-examples/distributed-computing/blob/master/demo.py

from random import random
from collections import defaultdict
import sys

import networkx as nx
import numpy as np
import click
import matplotlib
from dimod import Binary, ConstrainedQuadraticModel, quicksum
from dwave.system import LeapHybridCQMSampler
from task_system import TaskSystem, Task
import itertools
try:
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib.use("agg")
    import matplotlib.pyplot as plt


def build_cqm(taskSystem: TaskSystem):
    """
    Build a cqm model given a task system
    :param taskSystem: the task system
    :return: cqm: A ConstrainedQuadraticModel Object
    """
    num_processors = taskSystem.num_processors
    num_tasks = taskSystem.num_tasks

    # create the task and processor ids
    task_ids = range(num_tasks)
    processor_ids = range(num_processors)

    # Create the decision variables xij
    x = [[Binary(f'x_{i},{j}') for j in processor_ids] for i in task_ids]

    # Initialize the cqm object
    cqm = ConstrainedQuadraticModel()

    # Add the first constraint: each task must have at least one processor
    for i in task_ids:
        cqm.add_constraint(quicksum(x[i][j] for j in processor_ids) >= 1,
                           label=f"at_least_one_processor_{i}")

    # Add the second constraint: EDF schedulability condition
    # i.e. for each processor, the uniprocessor schedule must be feasible
    for j in processor_ids:
        cqm.add_constraint(quicksum(x[i][j]*(taskSystem.tasks[i].wcet_array[j]/taskSystem.tasks[i].deadline) for i in task_ids) <= 1)

    # Define the Objective
    objective = []
    for i in task_ids:
        objective.append([x[i][j]*taskSystem.tasks[i].wcet_array[j]/taskSystem.tasks[i].deadline for j in processor_ids])

    cqm.set_objective(sum(list(itertools.chain(*objective))))

    return cqm

if __name__ == '__main__':
    ts = TaskSystem([Task([1, 3, 6, 2], 7),
                     Task([3, 5, 10, 4], 12),
                     Task([2, 4, 6, 5], 6),
                     Task([1, 1, 2, 1], 3),
                     Task([3, 3, 4, 6], 10)])
    cqm = build_cqm(ts)