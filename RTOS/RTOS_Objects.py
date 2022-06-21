import heapq
import time
import math
from threading import Thread, Lock
from numpy import random
import pandas as pd
import plotly.express as px

import matplotlib.pyplot as plt

EXP_SCALE = 2.0
class JobQueue:
    def __init__(self):
        self.queue = []
        heapq.heapify(self.queue)

    def enqueue(self, job):
        heapq.heappush(self.queue, job)

    def dequeue(self):
        return heapq.heappop(self.queue)
    def top(self):
        if len(self.queue) == 0:
            return None

        return self.queue[0]

    def __len__(self):
        return len(self.queue)

    def update(self, current_time):
        # need to recreate the priority queue
        jobs = []
        heapq.heapify(jobs)
        for job in self.queue:
            job.update_priority_on_current_time(current_time)
            heapq.heappush(jobs, job)

        self.queue = jobs

class UniprocessorTask:
    def __init__(self, wcet:int, deadline:int):
        self.wcet = wcet
        self.deadline = deadline
        self.next_arrival_time = 0  # arrival time of next job (initially that of the first job)
        self.next_job_number = 1  # job number starting from 1
        self.task_number = None

    def operation_step(self, current_time, job_queue: JobQueue):
        """
        One step of task operation
        :param current_time: the current global logical time
        :param job_queue: the job queue to enqueue potentially arriving job
        :return:
        """
        if current_time == self.next_arrival_time:
            exec_time = math.floor(random.uniform(1,self.wcet))
            self.emit_job(current_time, exec_time, self.next_arrival_time, job_queue)
            self.next_arrival_time = self.next_arrival_time + math.floor(random.exponential(EXP_SCALE)) + self.deadline

    def emit_job(self, current_time, exec_time, arrival_time, job_queue:JobQueue):
        job = Job(self,self.next_job_number, arrival_time, exec_time, current_time)
        self.next_job_number += 1
        job_queue.enqueue(job)

class Job:
    def __init__(self, task:UniprocessorTask, job_number, arrival_time, execution_time, current_time):
        self.task = task
        self.job_number = job_number
        self.arrival_time = arrival_time
        self.execution_time = execution_time
        self.priority = self.arrival_time + self.task.deadline - current_time # set the initial priority when job is created
        self.time_executed = 0 # the amount of time the job has run on the processor

    def set_priority(self, priority: float):
        self.priority = priority

    def update_priority_on_current_time(self, current_time):
        """
        Dynamically update the EDF priority based on the current time
        Should only be called on scheduling instants, not at each time instant
        :param current_time: The current logical time
        :return:
        """
        # CAUTION: the priority is for a min-heap only
        # i.e. smaller value means higher priority
        self.priority = self.arrival_time + self.task.deadline - current_time

    def __lt__(self, other):
        return self.priority < other.priority

    def __gt__(self, other):
        return self.priority > other.priority

    def __le__(self, other):
        return self.priority <= other.priority

    def __ge__(self, other):
        return self.priority >= other.priority

class Running:
    def __init__(self):
        self.running = None # the running job:Job instance
        self.arrival_time_on_processor = None # the time that the current running job arrived on the processor
        self.session_duration = None  # the duration that the current job has been executing, without preemption on the processor

    def has_expired(self, current_time):
        if self.running is None:
            return False # default value (since there is no running job, it cannot expire)

        return self.running.time_executed == self.running.execution_time

    def execute(self, current_time):
        if self.running is None:
            print('IDLE')
            return
        self.running.time_executed += 1
        self.session_duration += 1  # log the current session duration
        print(f'Task Number: {self.running.task.task_number} Job Number: {self.running.job_number} '
              f'Current Time: {current_time+1} Time executed: {self.running.time_executed} Execution Time Left: {self.running.execution_time - self.running.time_executed}'
              )

    def get_event(self):
        """
        Get the event tuple at the current time
        The event tuple records (arrival_time_on_processor, session_duration)

        :return:
        """
        return self.arrival_time_on_processor, self.session_duration

    def set_running(self, job:Job, current_time, logger):
        event = self.get_event()

        # the if clause required to skip the initial null event
        if event[0] is not None and event[1] is not None:
            logger.add_event(self.get_event())
        self.running = job
        self.arrival_time_on_processor = current_time
        self.session_duration = 0


class TimeCounter:
    def __init__(self):
        """
        Counts logical time steps
        """
        self.current_time = 0

    def increment_time(self):
        self.current_time += 1

class UniprocessorTaskSystem:
    def __init__(self, tasks):
        self.tasks: [UniprocessorTask] = tasks
        # assign the tasks their task number
        for i, task in enumerate(self.tasks):
            task.task_number = i+1  # starting from 1

        # test for feasibility of scheduling
        self.uniprocessor_schedulability_test()
        # a temporary job queue to hold jobs that have arrived at the current instant of time only
        self.temp_job_queue = JobQueue()

    def uniprocessor_schedulability_test(self):
        if len(self.tasks) == 0:
            return
        total_utilization = 0.0 # the total system utilization

        for task in self.tasks:
            try:
                total_utilization += task.wcet / task.deadline
            except ZeroDivisionError as e:
                raise ValueError(f'Invalid deadline parameter on task #{task.task_number}') # cast to valueError for uniform interface

        if total_utilization > 1.0:
            raise ValueError('The task system is infeasible upon the platform!')

    def emit_jobs(self, current_time):
        """
        Run one step of the operation of emitting jobs at current time step
        :return:
        """
        for task in self.tasks:
            task.operation_step(current_time, self.temp_job_queue) # fill the temp job queue with new jobs, if any


    def check_new_job_arrival(self):
        return len(self.temp_job_queue) > 0

class Logger:
    def __init__(self):
        self.events = [] # the array of events: it consists of tuples of form (arrival_time, time_executed)

    def add_event(self, event):
        self.events.append(event)

class Grapher:
    def __init__(self):
        pass

class RTOS:

    def __init__(self, tasks:[UniprocessorTask]):
        # initialize the task system
        try:
            self.task_system = UniprocessorTaskSystem(tasks)
        except ValueError as e:
            # this means that the task system was unschedulable
            raise e

        # set the number of iterations (time steps)
        self.max_iterations = 0 # 0 by default, set again by main loop
        # initialize the time counter, which acts as a global logical clock
        self.time_counter = TimeCounter()

        # initialize the job queue
        self.job_queue = JobQueue()

        # initialize the running job
        self.running = Running()

        # initialize the logger
        self.logger = Logger()

        # initialize the grapher
        self.grapher = Grapher()


    def run_scheduler(self):
        """
        The EDF scheduler
        :return:
        """
        print('scheduler running')
        # update the job queue dynamically
        self.job_queue.update(self.time_counter.current_time)

        # also update the running job's priority
        if self.running.running is not None:
            self.running.running.update_priority_on_current_time(self.time_counter.current_time)

        # check if there are no tasks on the job queue at this moment
        if len(self.job_queue) == 0:
            return  # no point in running the scheduler if the job queue is empty

        # check if running job's priority is higher (less, since we are using min-heaps) than the top of the queue
        if self.running.running is not None and self.running.running <= self.job_queue.top():
            # no need to preempt the running task if it has a higher priority
            return

        # now, the running task should be prempted by the top of the queue
        if self.running.running is not None:
            # note that heapreplace first pops the top and then pushes running
            top = heapq.heapreplace(self.job_queue.queue,self.running.running)
            self.set_running(top)

        # if there is no running and job queue is not empty, just pop the queue and place
        # the top job as running
        else:
            top = heapq.heappop(self.job_queue.queue)
            self.set_running(top)


    def handle_running_job_expiry(self):
        print("Running job has expired")
        self.running.running = None # delete the running job cuz it has expired
        # don't run scheduler here, only run at the end of the main loop
        # self.run_scheduler()

    def run_task_system_job_generation(self):
        self.task_system.emit_jobs(self.time_counter.current_time)

    def check_new_job_arrival(self):
        return self.task_system.check_new_job_arrival() # delegate the checking to the task system

    def handle_new_job_arrival(self):
        """
        Assumes that new jobs have arrived into the temporary job queue of the UniprocessorTaskSystem
        :return:
        """
        print('New job(s) have arrived')
        # enqueue the new jobs into the job queue
        # note that this automatically performs the cleanup of the task system's temp queue
        for i in range(len(self.task_system.temp_job_queue)):
            self.job_queue.enqueue(self.task_system.temp_job_queue.dequeue())

        # don't run scheduler here, delegated to the main loop itself
        #self.run_scheduler()

    def run_cleanup(self):
        pass

    def check_missed_deadlines(self):
        for job in self.job_queue.queue:
            if self.time_counter.current_time > job.arrival_time + job.task.deadline:
                if job.time_executed != job.execution_time:
                    raise RuntimeError('Deadline missed!!')

    def main_loop(self, num_iterations):

        # set the max number of iterations
        self.max_iterations = num_iterations

        for i in range(num_iterations):
            # check if any job has missed its deadline # not necessary, shifted to initialization portion
            # self.check_missed_deadlines()
            # check if running job has expired
            has_running_job_expired = False
            have_new_jobs_arrived = False

            if self.check_running_job_expired():
                has_running_job_expired = True
                self.handle_running_job_expiry()

            # run the tasks' job arrival mechanism
            self.run_task_system_job_generation()

            # check if new jobs have arrived
            # TODO: some refactoring to be done, the checking for new job arrival and
            # handling new job arrival could be inbuilt into the task system
            # job generation mechanism
            if self.check_new_job_arrival():
                have_new_jobs_arrived = True
                self.handle_new_job_arrival()

            # run the scheduler if necessary
            if has_running_job_expired or have_new_jobs_arrived:
                self.run_scheduler()

            # run the running job for one step
            self.running.execute(self.time_counter.current_time)

            # run cleanup
            self.run_cleanup()

            # progress the clock by 1 logical unit
            self.time_counter.increment_time()

    def check_running_job_expired(self):
        return self.running.has_expired(self.time_counter.current_time)

    def set_running(self, new_running):
        self.running.set_running(new_running, self.time_counter.current_time, self.logger)

    def generate_schedule_chart(self):
        """
        Generate the schedule chart
        :return:
        """

        # Declaring a figure "gnt"
        fig, gnt = plt.subplots()

        # Setting Y-axis limits
        gnt.set_ylim(0, 10)

        # Setting X-axis limits
        gnt.set_xlim(0, self.max_iterations)

        # Setting labels for x-axis and y-axis
        gnt.set_xlabel('Logical Time')
        gnt.set_ylabel('Processor')

        # Setting ticks on y-axis
        gnt.set_yticks([3])
        # Labelling tickes of y-axis
        gnt.set_yticklabels(['1'])

        # Setting graph attribute
        gnt.grid(True)

        # Declaring a bar in schedule
        #gnt.broken_barh([(40, 50)], (30, 9), facecolors=('tab:orange'))

        # Declaring multiple bars in at same level and same width
        gnt.broken_barh(self.logger.events, (3, 5),
                        facecolors='tab:blue')

        #gnt.broken_barh([(10, 50), (100, 20), (130, 10)], (20, 9),
        #                facecolors=('tab:red'))
        plt.show()
        plt.savefig("gantt1.png")


if __name__ == '__main__':
    tasks = [UniprocessorTask(4,10),
             UniprocessorTask(5,26),
             UniprocessorTask(6,17)]

    try:
        rtos = RTOS(tasks)
        rtos.main_loop(100)
        rtos.generate_schedule_chart()
    except ValueError as e: # means that the task system given is unschedulable
        print(e)

