from abc import ABC, abstractmethod
from typing import Any


class BaseClass(
    ABC
):  # TODO: Try to move more methods and attributes into BaseClass and put it in a separate module. \endtodo
    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, inputs: Any, *args: Any, **kwargs: Any) -> Any:
        ...
