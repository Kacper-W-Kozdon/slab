from abc import ABC, abstractmethod
from typing import Any


class BaseClass(ABC):
    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, inputs: Any, *args: Any, **kwargs: Any) -> Any:
        ...
