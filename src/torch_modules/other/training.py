#
from typing import Any, Callable, Union


def curve_fit(*args: Any, **kwargs: dict[Any, Any]) -> Union[Callable[..., Any], None]:
    """
    Fit using scipy's curve_fit().
    """
    raise NotImplementedError


def nn(*args: Any, **kwargs: dict[Any, Any]) -> Union[Callable[..., Any], None]:
    """
    Fit using torch's nn.Module
    """
    raise NotImplementedError
