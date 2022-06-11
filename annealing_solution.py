# This code inspired from https://github.com/dwave-examples/distributed-computing/blob/master/demo.py

from random import random
from collections import defaultdict
import sys

import networkx as nx
import numpy as np
import click
import matplotlib
from dimod import Binary, ConstrainedQuadraticModel, quicksum, ExactCQMSolver
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

def solve_cqm(cqm, sampler):
    """
    Solve the given cqm using the given sampler
    :param cqm: the constrained quadratic model object
    :param sampler: the sampler used to solve the problem
    :return: sample_set: the set of feasible solutions to the problem
                        or None if no feasible solution is found
    """

    # find the entire solution set
    sample_set = sampler.sample_cqm(cqm, label='Partitioning solution')

    # filter out the infeasible solutions, keeping only the feasible ones
    feasible_sampleset = sample_set.filter(lambda row: row.is_feasible)

    # if no feasible solution found, return None
    if not len(feasible_sampleset):
        print('No feasible solution found!!')
        return None

    return feasible_sampleset

if __name__ == '__main__':
    # a test tasksystem
    ts1 = TaskSystem([Task([6, 3,4], 7),
                      Task([3, 5,6], 12),
                      Task([2, 4,2], 6),
                      Task([4, 10,11], 12),
                      Task([3, 8,2], 10),
                      Task([9, 14,12], 16),
                      ])

    # build cqm problem from the tasksystem
    cqm = build_cqm(ts1)

    # sample using the exact cqm sampler for testing

    sampler = ExactCQMSolver()
    solutions = solve_cqm(cqm,sampler)
    print(solutions)
