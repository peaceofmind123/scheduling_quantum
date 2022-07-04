import numpy as np
import math
from task_system import Task, TaskSystem
import pickle

class TaskSystemGenerator:
    """A wrapper for multiple task system generation algorithms"""

    def __init__(self, num_tasks:int, num_processors:int,
                 min_deadline:int, max_deadline:int, exp_scale: float):
        self.num_tasks = num_tasks
        self.num_processors = num_processors
        self.min_deadline = min_deadline
        self.max_deadline = max_deadline

        # the parameter of the exponential distribution used to generate the utilization
        self.exp_scale = exp_scale

    def canonical_generate_tasks(self):
        """
        The canonical algorithm, suggested in the paper and mentioned in the proposal
        :return:
        """
        tasks = []
        for i in range(self.num_tasks):
            utilizations = np.ones(self.num_processors) * 2
            wcets = np.zeros(self.num_processors, dtype=np.int64)
            deadline = np.random.randint(self.min_deadline, self.max_deadline+1)


            valid_utilization_counter = 0
            while not valid_utilization_counter == self.num_processors:
                valid_utilization_counter = 0
                for j in range(self.num_processors):
                    # the first condition prevents trivially infeasible task systems
                    # the second condition prevents wcets[j] to be 0
                    if utilizations[j] > 1 or utilizations[j] * deadline < 1:
                        # invalid, so generate new utilization and wcet
                        utilizations[j] = np.random.exponential(self.exp_scale)
                        wcets[j] = int(math.floor(utilizations[j]*deadline))

                        if utilizations[j] <= 1 and utilizations[j] * deadline >= 1:
                            valid_utilization_counter += 1
                    else:
                        # it is already valid, so skip
                        valid_utilization_counter += 1

            # create a task with the valid utilizations and deadline
            task = Task(list(wcets), deadline)
            tasks.append(task)

        ts = TaskSystem(tasks)
        return ts

    def generate_dataset(self, num_tasksystems:int, save_path:str, save:bool = True):
        """Generate a set of task systems with the given parameters"""

        task_systems = [self.canonical_generate_tasks() for i in range(num_tasksystems)]
        if save:
            with open(save_path, 'wb') as f:
                pickle.dump(task_systems, f)

        return task_systems


if __name__ == '__main__':

    num_tasksystems = 10
    num_tasks = 20
    num_processors = 5
    min_deadline = 5
    max_deadline = 100
    exp_scale = 2
    save_path = f'tasksystem-{10}-{20}-{5}.pkl'
    task_system_generator = TaskSystemGenerator(num_tasks, num_processors,
                                                min_deadline, max_deadline, exp_scale)
    task_systems = task_system_generator.generate_dataset(num_tasksystems, save_path)





