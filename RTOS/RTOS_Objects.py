import heapq
import time
import math
from threading import Thread, Lock
from numpy import random

EXP_SCALE = 2.0
class JobQueue:
    def __init__(self):
        self.queue = []
        heapq.heapify(self.queue)

    def enqueue(self, job):
        heapq.heappush(self.queue, job)

    def dequeue(self):
        return heapq.heappop(self.queue)

    def __len__(self):
        return len(self.queue)

class UniprocessorTask:
    def __init__(self, wcet:int, deadline:int):
        self.wcet = wcet
        self.deadline = deadline
        self.next_arrival_time = 0 # arrival time of next job (initially that of the first job)
        self.next_job_number = 0

    def operation_step(self, current_time, job_queue: JobQueue):
        """
        One step of task operation
        :param current_time: the current global logical time
        :return:
        """
        if current_time == self.next_arrival_time:
            exec_time = math.floor(random.uniform(0,self.wcet))
            self.emit_job(current_time, exec_time, self.next_arrival_time, job_queue)
            self.next_arrival_time = self.next_arrival_time + math.floor(random.exponential(EXP_SCALE)) + self.deadline

    def emit_job(self, current_time, exec_time, arrival_time, job_queue:JobQueue):
        job = Job(self,self.next_job_number, arrival_time, exec_time)
        self.next_job_number += 1
        job_queue.enqueue(job)

class Job:
    def __init__(self, task:UniprocessorTask, job_number, arrival_time, execution_time):
        self.task = task
        self.job_number = job_number
        self.arrival_time = arrival_time
        self.execution_time = execution_time
        self.priority = 0

    def set_priority(self, priority: int):
        self.priority = priority

class Running:
    def __init__(self):
        self.running = None # the running job:Job instance

    def has_expired(self, current_time):
        if self.running is None:
            return False # default value (since there is no running job, it cannot expire)

        return self.running.arrival_time + self.running.execution_time >= current_time

    def execute(self, current_time):
        print(f'Current Time: {current_time} Arrival Time: {self.running.arrival_time} '
              f'Execution Time: {self.running.execution_time}')

class TimeCounter:
    def __init__(self):
        """
        Counts logical time steps
        """
        self.start_time = 0
        self.current_time = 0

    def increment_time(self):
        self.current_time += 1

class UniprocessorTaskSystem:
    def __init__(self, tasks):
        self.tasks: [UniprocessorTask] = tasks
        # a temporary job queue to hold jobs that have arrived at the current instant of time only
        self.temp_job_queue = JobQueue()

    def emit_jobs(self, current_time):
        """
        Run one step of the operation of emitting jobs at current time step
        :return:
        """
        for task in self.tasks:
            task.operation_step(current_time, self.temp_job_queue) # fill the temp job queue with new jobs, if any


    def check_new_job_arrival(self):
        return len(self.temp_job_queue) > 0


class RTOS:

    def __init__(self, tasks:[UniprocessorTask]):
        # initialize the task system
        self.task_system = UniprocessorTaskSystem(tasks)

        # initialize the time counter, which acts as a global logical clock
        self.time_counter = TimeCounter()

        # initialize the job queue
        self.job_queue = JobQueue()

        # initialize the running job
        self.running = Running()

    def run_scheduler(self):
        """
        The core scheduler code
        :return:
        """
        print('scheduler to be run')

    def handle_running_job_expiry(self):
        self.run_scheduler()

    def run_task_system_job_generation(self):
        self.task_system.emit_jobs(self.time_counter.current_time)

    def check_new_job_arrival(self):
        return self.task_system.check_new_job_arrival() # delegate the checking to the task system

    def handle_new_job_arrival(self):
        """
        Assumes that new jobs have arrived into the temporary job queue of the UniprocessorTaskSystem
        :return:
        """
        # enqueue the new jobs into the job queue
        # note that this automatically performs the cleanup of the task system's temp queue
        for i in range(len(self.task_system.temp_job_queue)):
            self.job_queue.enqueue(self.task_system.temp_job_queue.dequeue())

        # then, run the scheduler
        self.run_scheduler()

    def run_cleanup(self):
        pass

    def main_loop(self, num_iterations):

        for i in range(num_iterations):
            # check if running job has expired
            if self.check_running_job_expired(): #TODO: only do this on the scheduling instants
                self.handle_running_job_expiry()

            # run the tasks' job arrival mechanism
            self.run_task_system_job_generation()

            # check if new jobs have arrived
            if self.check_new_job_arrival():
                self.handle_new_job_arrival()

            # run the running job for one step
            self.running.execute(self.time_counter.current_time)

            # run cleanup
            self.run_cleanup()

            # progress the clock by 1 logical unit
            self.time_counter.increment_time()


    def check_running_job_expired(self):
        return self.running.has_expired(self.time_counter.current_time)
    def set_running_job_expired(self, value):
        pass

    def set_new_job_arrived(self, value):
        pass

    def set_running(self, new_running):
        pass
