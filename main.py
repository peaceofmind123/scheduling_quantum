"""Overall system integration"""
from TaskSystemGeneration import TaskSystemGenerator
from branch_and_bound_solution import BranchBoundSolver
from RTOS.RTOS_Objects import UniprocessorTask

def get_partition_from_solution(solution, solver):

    x, objective_value = solution

def overall_system():
    num_tasksystems = 10
    num_tasks = 20
    num_processors = 5
    min_deadline = 5
    max_deadline = 100
    exp_scale = 2

    tsg = TaskSystemGenerator(num_tasks,num_processors,min_deadline,max_deadline,exp_scale)
    task_systems = tsg.generate_dataset(num_tasksystems,'', False) # don't save
    bbss = [BranchBoundSolver(task_system) for task_system in task_systems] # the set of branch and bound solvers

    # run the partitioning algorithm and get the results
    solutions = [solver.solve(False) for solver in bbss] # just get solution, do not display
    pass

if __name__ == '__main__':
    overall_system()