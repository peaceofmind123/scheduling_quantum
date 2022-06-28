import numpy as np
import math
from task_system import Task, TaskSystem

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
                    if utilizations[j] > 1:
                        # invalid, so generate new utilization and wcet
                        utilizations[j] = np.random.exponential(self.exp_scale)
                        wcets[j] = int(math.floor(utilizations[j]*deadline))
                        if utilizations[j] <= 1:
                            valid_utilization_counter += 1
                    else:
                        # it is already valid, so skip
                        valid_utilization_counter += 1

            # create a task with the valid utilizations and deadline
            task = Task(list(wcets), deadline)
            tasks.append(task)

        ts = TaskSystem(tasks)
        return ts


if __name__ == '__main__':

    task_system_generator = TaskSystemGenerator(10,5,5,100,2)
    task_system: TaskSystem = task_system_generator.canonical_generate_tasks()
    pass

