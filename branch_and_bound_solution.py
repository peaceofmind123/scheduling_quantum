from ortools.linear_solver import pywraplp
from task_system import TaskSystem, Task
import numpy as np

class BranchBoundSolver:
    def __init__(self, task_system:TaskSystem, solver_backend:str='SCIP'):
        self.solver:pywraplp.Solver = pywraplp.Solver.CreateSolver('SCIP')
        self.task_system:TaskSystem = task_system
        self.x = None # the decision variables, initialized below
        self._initialize_solver_params()

    def _initialize_solver_params(self):
        """Internal routine called within init routine to initialize the solver
           parameters given the tasksystem"""
        # initialize the decision variables
        x = [[self.solver.BoolVar(f'x{i}{j}') for j in range(self.task_system.num_processors)] for i in range(self.task_system.num_tasks)]

        # initialize the first set of constraints: each task assigned to at least one processor
        for i in range(self.task_system.num_tasks):
            # initialize the constraint with >=1 as the RHS
            # the first param is the lower bound and second is upper bound for the constraint
            constraint:pywraplp.Constraint = self.solver.RowConstraint(1, self.solver.infinity())
            for j in range(self.task_system.num_processors):
                constraint.SetCoefficient(x[i][j], 1) # each coefficient is 1 in the first set of constraints


        # initialize the second set of constraints: EDF schedulability
        for j in range(self.task_system.num_processors):
            # initialize the constraint with <= 1 on the RHS
            constraint:pywraplp.Constraint = self.solver.RowConstraint(0, 1)
            for i in range(self.task_system.num_tasks):
                task = self.task_system.tasks[i]
                # the worst case execution time for the jth processor and the deadline
                Cij = task.wcet_array[j]
                Ti = task.deadline
                constraint.SetCoefficient(x[i][j], Cij/Ti)

        # define the objective
        objective: pywraplp.Objective = self.solver.Objective()
        for i in range(self.task_system.num_tasks):
            for j in range(self.task_system.num_processors):
                task = self.task_system.tasks[i]
                Cij = task.wcet_array[j]
                Ti = task.deadline

                objective.SetCoefficient(x[i][j], Cij/Ti)
        objective.SetMinimization()

        # save the decision variables
        self.x = x

    def solve(self):
        status = self.solver.Solve()
        if status == pywraplp.Solver.OPTIMAL:
            print('Objective value =', self.solver.Objective().Value())
            for i in range(self.task_system.num_tasks):
                for j in range(self.task_system.num_processors):
                    print(self.x[i][j].name(), ' = ', self.x[i][j].solution_value())
            print()
            print('Problem solved in %f milliseconds' % self.solver.wall_time())
            print('Problem solved in %d iterations' % self.solver.iterations())
            print('Problem solved in %d branch-and-bound nodes' % self.solver.nodes())
        else:
            print('The problem does not have an optimal solution.')


def main():
    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver('SCIP')
    ts1 = TaskSystem([Task([5, 10, 1], 17),
                      Task([2, 1, 3], 32),
                      Task([2, 3, 3], 36),
                      Task([2, 8, 3], 10),
                      Task([1, 6, 3], 26),
                      Task([2, 3, 3], 56),
                      Task([2, 8, 3], 100),
                      Task([1, 6, 3], 96),

                      ])
    branch_and_bound_solver = BranchBoundSolver(ts1)
    branch_and_bound_solver.solve()

if __name__ == '__main__':
    main()