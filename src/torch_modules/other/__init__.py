#
from typing import Any, Union

from .training import curve_fit, nn

training_methods = {"curve_fit": curve_fit, "nn": nn}
fitting_method_args: Union[dict[str, tuple[Any, ...]], None] = None
fitting_method_kwargs: Union[dict[str, dict[str, str]], None] = None
