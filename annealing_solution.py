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
import re
from RTOS.RTOS_Objects import UniprocessorTask
import time

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

    chained_objective = itertools.chain(*objective)
    chained_objective_list = list(chained_objective)
    sum_objective = sum(chained_objective_list) # the bottleneck
    cqm.set_objective(sum_objective)
    cqm.set_objective(sum(list(itertools.chain(*objective))))

    return cqm

def solve_cqm(cqm, sampler,time_limit=10):
    """
    Solve the given cqm using the given sampler
    :param cqm: the constrained quadratic model object
    :param sampler: the sampler used to solve the problem
    :return: sample_set: the set of feasible solutions to the problem
                        or None if no feasible solution is found
    """
    print(f'min time limit: {sampler.min_time_limit(cqm)}')
    # find the entire solution set (with a time limit of 10 seconds to find the solution)
    sample_set = sampler.sample_cqm(cqm, time_limit=time_limit, label='Partitioning solution')

    # filter out the infeasible solutions, keeping only the feasible ones
    feasible_sampleset = sample_set.filter(lambda row: row.is_feasible)

    # if no feasible solution found, return None
    if not len(feasible_sampleset):
        print('No feasible solution found!!')
        return None
    
    return feasible_sampleset


class AnnealingSolver:
    """
    The annealing solver
    """
    def __init__(self, taskSystem: TaskSystem):
        self.taskSystem = taskSystem
        # the partitions found from the optimal solution
        self.partitions = []
        # the optimal objective
        self.optimal_objective = None

    def get_partitioning_from_solution(self, solution):
        """
        Get the partitioning from the annealing solution
        :param solution: Contains a dict of the form {'x_0,0':0, 'x_0,1':1,...} as .first where 0 indicates
        task non assignment and 1 indicates task assignment
        :return:
        """
        # get the optimal solution
        optimal_solution = solution.first.sample

        # fill this array with arrays of uniprocessor tasks for the corresponding processor
        partitions = [[] for j in range(self.taskSystem.num_processors)]

        for decision_variable in optimal_solution:
            value = optimal_solution[decision_variable]

            # find the i and j values from the decision variables
            # the decision variable is of the form x_i,j
            i, j = re.findall("[0-9]", decision_variable) #TODO: fix this does not work for x_0,100 since it gives [0,1,0,0] as output
            i = int(i) # i is now the task id
            j = int(j) # j is the processor id
            if value == 1:
                # means task i has been assigned to processor j
                # get the task
                task = self.taskSystem.tasks[i]
                # get the wcet of task i on processor j
                wcet = task.wcet_array[j]
                # get the deadline
                deadline = task.deadline

                # i+1 is the task number w.r.t the multiprocessor task system
                uniprocessor_task = UniprocessorTask(wcet,deadline,i+1)
                partitions[j].append(uniprocessor_task)

        self.partitions = partitions
        return partitions

    def solve(self, display_output=True,time_limit=10):
        """
        A wrapper function that matches the external interface of branch and bound solution
        so that downstream tasks can use these functions interchangably
        :arg display_output: dummy arg, to match the interface
        :return:
        """
        # build cqm problem from the tasksystem
        cqm = build_cqm(self.taskSystem)

        # sample using the leap hybrid cqm sampler
        sampler = LeapHybridCQMSampler()
        # the solution given here is actually a set of feasible solutions
        # the optimal solution is found in the subsequent step

        # measure the running time
        before = time.time_ns()
        solution = solve_cqm(cqm, sampler,time_limit)
        after = time.time_ns()
        elapsed = (after - before) / 1e6 # get time elapsed in milliseconds

        # get the partitioning from the optimal solution
        self.get_partitioning_from_solution(solution)

        # set the optimal objective
        self.optimal_objective = solution.first.energy

        return self.partitions, self.optimal_objective, elapsed

    def get_lower_bound(self, sampler:LeapHybridCQMSampler):
        """
        Get the lower bound for solution of the task system
        :return: lower_bound: min time in milliseconds to solve the problem
        """
        cqm = build_cqm(self.taskSystem)
        return sampler.min_time_limit(cqm)

if __name__ == '__main__':
    # a test tasksystem
    ts1 = TaskSystem([Task([5, 10, 1], 17),
                      Task([2, 1, 3], 32),
                      Task([2, 3, 3], 36),
                      Task([2, 8, 3], 10),
                      Task([1, 6, 3], 26),
                      Task([2, 3, 3], 56),
                      Task([2, 8, 3], 100),
                      Task([1, 6, 3], 96),

                      ])

    # build cqm problem from the tasksystem
    cqm = build_cqm(ts1)

    # sample using the leap hybrid cqm sampler

    sampler = LeapHybridCQMSampler()
    solutions = solve_cqm(cqm, sampler)
    print(solutions)
