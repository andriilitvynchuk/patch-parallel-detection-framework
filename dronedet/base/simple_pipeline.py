from abc import abstractclassmethod

from .simple_runner import SimpleRunner


# this class is created for combining all Runners in one place
# to create full pipeline. Call _recursive_start(runner) to recursive start
# all child runners starting from given as parameter (runner in parameters is meant to be producer runner)
# The same for join (is important to use to make main process wait other processes to end before terminating)


class SimplePipeline:
    def _recursive_start(self, runner: SimpleRunner) -> None:
        runner.start()
        for child in runner.children.values():
            child_instance = child["cls"]
            if not child_instance.is_running:
                self._recursive_start(child_instance)

    def _recursive_join(self, runner: SimpleRunner) -> None:
        try:
            runner.join()
            for child in runner.children:
                self._recursive_join(child)
        except TypeError:
            print("Terminated")

    @abstractclassmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractclassmethod
    def join(self) -> None:
        raise NotImplementedError

    @abstractclassmethod
    def connect_runners(self) -> None:
        raise NotImplementedError
