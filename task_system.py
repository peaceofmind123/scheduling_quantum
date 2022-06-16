class Task:
    def __init__(self, wcet_array:[int], deadline:int):
        self.num_processors = len(wcet_array)
        self.wcet_array = wcet_array
        self.deadline = deadline

class TaskSystem:
    def __init__(self, tasks:[Task]):
        self.tasks = tasks
        self.num_tasks = len(self.tasks)
        self.num_processors = len(self.tasks[0].wcet_array)


if __name__ == '__main__':
    ts = TaskSystem([Task([1,3,6,2],7),
                    Task([3,5,10,4],12),
                    Task([2,4,6,5],6),
                    Task([1,1,2,1],3),
                    Task([3,3,4,6],10)])


