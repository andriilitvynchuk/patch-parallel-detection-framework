import sys
from abc import abstractclassmethod


# this class is created for combining all Runners in one place
# to create full pipeline. Call _recursive_start(runner) to recursive start
# all child runners starting from given as parameter (runner in parameters is meant to be producer runner)
# The same for join (is important to use to make main process wait other processes to end before terminating)


class SimplePipeline:
    def start(self) -> None:
        for runner in vars(self).values():
            runner.start()

    def join(self) -> None:
        producer_runners = [runner for runner in vars(self).values() if len(runner.parents) == 0]
        for runner in producer_runners:
            runner.join()
            # if we joined then producer is closed. TODO: do it better
            sys.exit()

    @abstractclassmethod
    def connect_runners(self) -> None:
        raise NotImplementedError
