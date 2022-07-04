"""Overall system integration"""
from TaskSystemGeneration import TaskSystemGenerator
from branch_and_bound_solution import BranchBoundSolver
from RTOS.RTOS_Objects import UniprocessorTask, MultiprocessorRTOS

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
    solutions:[([[UniprocessorTask]], float)] = [solver.solve(False) for solver in bbss] # just get solution, do not display

    # create a multiprocessor rtos object for each task system
    # solution[0] is the partitioned_tasks
    # solution[1] is the worst-case utilization of the partitioning
    multiprocessors_with_partitioned_tasks = [MultiprocessorRTOS(solution[0],solution[1]) for solution in solutions]

    # generate the chart for the first multiprocessor RTOS for testing
    multiprocessors_with_partitioned_tasks[0].main_loop(1000)
    multiprocessors_with_partitioned_tasks[0].generate_schedule_chart()

if __name__ == '__main__':
    overall_system()