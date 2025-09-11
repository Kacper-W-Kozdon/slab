from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from scipy.signal import convolve
from torch.nn import Module

from ..other.utils import A_parameter, Image_Sources_Positions, Mean_Path_T_R
from ..reflect_transmit.reflect_transmit import (
    Reflectance_Transmittance,
    Reflectance_Transmittance_rho,
    Reflectance_Transmittance_rho_t,
    Reflectance_Transmittance_t,
)
from .base_class import BaseClass

# import datetime
# import pathlib
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.optimize import curve_fit
# from scipy.signal import convolve
# from ..other.utils import A_parameter, Image_Sources_Positions, Mean_Path_T_R
# from ..reflect_transmit.reflect_transmit import (
#     Reflectance_Transmittance,
#     Reflectance_Transmittance_rho,
#     Reflectance_Transmittance_rho_t,
#     Reflectance_Transmittance_t,
# )


class tContini(Module, BaseClass):
    def __init__(
        self,
        s: Union[int, float, None] = None,
        mua: Union[int, float, None] = None,
        musp: Union[int, float, None] = None,
        n1: Union[int, float, None] = None,
        n2: Union[int, float, None] = None,
        anisothropy_coeff: Union[int, float, None] = None,
        phantom: Optional[str] = "",
        DD: Optional[str] = "",
        m: Union[int, None] = None,
        eq: str = "",
        IRF: Union[List[Union[float, int]], None] = None,
        normalize: bool = True,
        log_scale: Union[bool, None] = None,
        values_to_fit: Optional[Union[List[str], Any]] = None,
        free_params: Optional[Union[List[str], Any]] = None,
        offset: Optional[float] = None,
        scaling: Optional[float] = None,
        controls: Union[Dict[Any, Any], None] = None,
    ) -> None:
        """
        The class initiating the slab model with the RTE and DE Green's functions. Source- Contini.

        :param s: The thickness of the diffusing slab in [mm]. Default: 0.
        :type s: Union[int, float]
        :param mua: Absorption coefficient of the slab in [mm^-1]. Default: None.
        :type mua: Union[int, float]
        :param musp: Reduced scattering coefficient of the slab in [mm^-1]. Default: None.
        :type musp: Union[int, float]
        :param n1: Refractive index of the external medium. Default: 0.
        :type n1: Union[int, float]
        :param n2: Refractive index of the diffusing medium (slab). Default: 0.
        :type n2: Union[int, float]
        :param anisothropy_coeff: The anisothropy coefficient g. musp = (1 - g) * mus. Default: 0.85
        :type anisothropy_coeff: Union[int, float, None]
        :param DD: Flag parameter to switch between mu = musp + mua for DD == "Dmuas" and mu = musp for DD == "Dmus" (Default).
        :type DD: str
        :param m: Number of mirror images (delta sources) in the lattice. Default: 100
        :type m: int
        :param eq: Flag parameter to switch between "RTE" and "DE" Green's functions. Default: "RTE"
        :type eq: str
        :param IRF: Instrument Response Function as a list of the function's outputs. Default: None.
        :type IRF: Union[List[Union[float, int]], None]
        :param normalize: Decide whether to normalize the function's output to the data to fit. Default: True.
        :type normalize: bool
        :param values_to_fit: Values passed to scipy.curve_fit(f, ydata, xdata, params) as ydata.
        :type values_to_fit: Union[List[str], Any]
        :param free_params: A list of free parameters passed down to scipy.curve_fit(f, ydata, xdata, params) as params for fitting.
        :type free_params: Union[List[str], Any]
        :param log_scale: bool that controls whether the outputs are rescaled with log. Default: None.
        :type log_scale: Union[bool, None]
        :param controls: A dict collecting all the parameters controlling the forward pass and the fit.
        :type controls: Union[Dict[Any, Any], None]
        :param scaling: Linear scaling in the fit.
        :type scaling: Optional[float]
        """
        super().__init__()

        self._mua = None if not mua else mua * 1e3
        self._musp = None if not musp else musp * 1e3
        self._offset = offset or 0.0
        self._scaling = scaling or 0.0
        if controls:
            self.controls = controls
        else:
            self.controls = {
                "_s": s * 1e-3 if s is not None else 0,  # 40 * 1e-3,
                "n1": n1 if n1 is not None else 0,
                "n2": n2 if n2 is not None else 0,
                "phantom": phantom or "",
                "anisothropy_coeff": anisothropy_coeff or 0.85,
                "DD": DD or "Dmuas",
                "m": m or 100,
                "eq": eq or "RTE",
                "IRF": IRF or None,
                "normalize": normalize or True,
                "values_to_fit": values_to_fit or ["T_rho_t"],
                "free_params": free_params or ["mua", "musp", "offset", "scaling"],
                "log_scale": log_scale or None,
                "err": 1 * 1e-6,
                "ydata_info": {"_max_ydata": 1, "min_ydata": 0},
            }

        # print(f"---INIT---\n{self._mua, self._musp, self._offset}")

    @property
    def scaling(self) -> Union[None, float]:
        return self._scaling

    @scaling.setter
    def scaling(self, value: float) -> None:
        if value is not None:
            self._scaling = value

    @scaling.deleter
    def scaling(self) -> None:
        self._scaling = 0.0

    @property
    def mua(self) -> Union[None, float]:
        return self._mua

    @mua.setter
    def mua(self, value: float) -> None:
        if value is None:
            self._mua = None
        else:
            self._mua = value * 1e3

    @mua.deleter
    def mua(self) -> None:
        self._mua = None

    @property
    def musp(self) -> Union[None, float]:
        return self._musp

    @musp.setter
    def musp(self, value: float) -> None:
        if value is None:
            self._musp = None
        else:
            self._musp = value * 1e3

    @musp.deleter
    def musp(self) -> None:
        self._musp = None

    @property
    def offset(self) -> float:
        return self._offset

    @offset.setter
    def offset(self, value: float) -> None:
        self._offset = value

    @offset.deleter
    def offset(self) -> None:
        self._offset = 0.0

    def evaluate(
        self,
        t_rho: Union[
            Tuple[Any, Any],
            List[Tuple[float, float]],
            List[Tuple[int, int]],
            List[Tuple[int, float]],
            List[Tuple[float, int]],
            pd.DataFrame,
        ],
        mua: Union[int, float, None] = None,
        musp: Union[int, float, None] = None,
        # offset: Union[int, float] = 0,
        anisothropy_coeff: Union[int, float, None] = None,
        **kwargs: Any,
    ) -> Union[Tuple[Any, ...], Tuple[List[Any], ...]]:
        """
        The method evaluating parameters of the Contini model.

        :param t_rho: Variables of the model in the form (time, radial_coordinate).
        :type t_rho: Union[Tuple[float, float], List[Tuple[float, float]]]
        :param mua: Absorption coefficient of the slab in [mm^-1]. Default: None.
        :type mua: Union[int, float]
        :param musp: Reduced scattering coefficient of the slab in [mm^-1]. Default: None.
        :type musp: Union[int, float]
        :param anisothropy_coeff: The anisothropy coefficient g. musp = (1 - g) * mus. Default: 0.85
        :type anisothropy_coeff: Union[int, float, None]
        :param kwargs: Optional kwargs:
                       mode: available values: "approx", "sum"- controls G_function's computation method.
        :type kwargs: Any

        Returns:
        R_rho_t: time resolved reflectance mm^(-2) ps^(-1)
        T_rho_t: time resolved transmittance mm^(-2) ps^(-1)
        R_rho: reflectance mm^(-2)
        T_rho: transmittance mm^(-2)
        R_t: ps^(-1)
        T_t: ps^(-1)
        l_rho_R: mean free path (reflected) mm
        l_rho_T: mean free path (transmitted) mm
        R: to be added
        T: to be added
        A: A parameter
        Z: Positions of the source images

        """
        if not isinstance(t_rho, torch.Tensor):
            try:
                t_rho = torch.tensor(t_rho).reshape(2, -1)
            except Exception as exc:
                print(
                    f"There was an issue converting t_rho to torch.tensor.\n\n\
                      t_rho = {t_rho}"
                )
                raise TypeError from exc
            # mua = mua * 1e3 if self._mua is None else self._mua
            # musp = musp * 1e3 if self._musp is None else self._musp

        if mua is None:
            mua = self._mua
        else:
            mua = 1e3 * mua
        if musp is None:
            musp = self._musp
        else:
            musp = 1e3 * musp

        anisothropy_coeff = anisothropy_coeff or self.controls.get("anisothropy_coeff")

        t = t_rho[0] * 1e-12
        rho = t_rho[1] * 1e-3
        s = kwargs.get("s") or self.controls.get("_s")
        n1 = kwargs.get("n1") or self.controls.get("n1")
        n2 = kwargs.get("n2") or self.controls.get("n2")
        phantom = kwargs.get("phantom") or self.controls.get("phantom")  # noqa: F841
        DD = kwargs.get("DD") or self.controls.get("DD")
        m = kwargs.get("m") or self.controls.get("m")
        eq = kwargs.get("eq") or self.controls.get("eq")

        R_rho_t, T_rho_t = Reflectance_Transmittance_rho_t(
            rho, t, mua, musp, s, m, n1, n2, DD, eq, anisothropy_coeff, **kwargs
        )

        R_rho, T_rho = Reflectance_Transmittance_rho(
            rho, mua, musp, s, m, n1, n2, DD, eq
        )

        R_t, T_t = Reflectance_Transmittance_t(t, mua, musp, s, m, n1, n2, DD, eq)

        l_rho_R, l_rho_T = Mean_Path_T_R(rho, mua, musp, s, m, n1, n2, DD, eq)

        R, T = Reflectance_Transmittance(mua, musp, s, m, n1, n2, DD, eq)

        A = A_parameter(n1, n2)

        Z = Image_Sources_Positions(s, mua, musp, n1, n2, DD, m, eq)

        return (
            R_rho_t,
            T_rho_t,
            R_rho,
            T_rho,
            R_t,
            T_t,
            l_rho_R,
            l_rho_T,
            R,
            T,
            A,
            Z,
        )

    def __call__(self) -> None:
        """
        Invokes the forward method.
        """
        self.forward()

    def forward(
        self,
        inputs: Union[
            Tuple[Any, Any],
            List[Tuple[float, float]],
            List[Tuple[int, int]],
            List[Tuple[int, float]],
            List[Tuple[float, int]],
            pd.DataFrame,
            torch.Tensor,
        ],
        *args: Any,
        **kwargs: Any,
    ) -> Union[List[float], Dict[Any, Any], float, int, None]:
        """
        The call method returning the function used for scipy.curve_fit().

        :param t_rho_array_like: Variables of the model in the form List[(time, radial_coordinate), ...], passed as xdata to scipy.curve_fit(f, ydata, xdata, params).
        :type t_rho_array_like: Union[List[Tuple[float, float]], List[Tuple[int, int]], List[Tuple[int, float]], List[Tuple[float, int]], pd.DataFrame]
        :param normalize: Controls whether normalization is applied to the output.
        :type normalize: bool
        :param args: An iterable of the free_parameters for fitting in the order (mua, musp). The parameters have to match the free_params param. Anisothropy_coeff to be added.
        :type args: Any
        :param kwargs: Optional kwargs:
                       mode: available values: "approx", "sum", "correction", "mixed"- controls G_function's computation method.
                       kwargs supported by the scipy.curve_fit() method
        :type kwargs: Any

        Returns:
        A dictionary with keys as passed in the values_to_fit param or a Union[float, List[float]] if a single value was provided.

        """

        t_rho_array_like = inputs

        values_to_fit: Union[List[str], Any] = self.controls.get("values_to_fit") or [
            "T_rho_t"
        ]
        free_params: Union[List[str], Any] = self.controls.get("free_params") or [
            "musp",
            "offset",
            "scaling",
        ]
        normalize: Union[Any, bool] = self.controls.get("normalize") or False
        if kwargs.get("normalize") is not None:
            normalize = kwargs.get("normalize")  # noqa: F841

        log_scale: Union[Any, bool] = self.controls.get("log_scale") or False
        if kwargs.get("log_scale") is not None:
            log_scale = kwargs.get("log_scale")

        IRF: pd.DataFrame = (
            self.controls.get("IRF")
            if self.controls.get("IRF") is not None
            else kwargs.get("IRF")
        )

        available_values = [
            "R_rho_t",
            "T_rho_t",
            "R_rho",
            "T_rho",
            "R_t",
            "T_t",
            "l_rho_R",
            "l_rho_T",
        ]
        available_free_params = ["mua", "musp", "offset", "scaling"]

        if not all([param in available_free_params for param in free_params]):
            raise ValueError(
                f"Obtained params: {free_params} not in the list of params available for fitting."
            )

        if not all([value in available_values for value in values_to_fit]):
            raise ValueError(
                f"Obtained values: {values_to_fit} not in the list of values available for fitting."
            )

        ret: Union[Dict[Any, Any], Any] = None
        args_list = list(args)

        if not args:
            args_list = [self._mua, self._musp, self._offset, self._scaling]

        for param_index, param in enumerate(available_free_params):
            if (param not in free_params) and args:
                param_value: Any = None
                if param_index == 0:
                    param_value = self._mua
                if param_index == 1:
                    param_value = self._musp
                if param_index == 2:
                    param_value = self._offset
                if param_index == 3:
                    param_value = self._scaling
                try:
                    args_list.insert(param_index, param_value)
                except UnboundLocalError as exc:
                    print(UnboundLocalError)
                    print(param_index)
                    raise UnboundLocalError from exc

        index_offset = available_free_params.index("offset")
        index_scaling = available_free_params.index("scaling")
        offset = args_list[index_offset] or self._offset
        scaling = args_list[index_scaling] or self._scaling
        for arg_index, arg in enumerate(args_list):
            if arg_index in [index_offset, index_scaling]:
                continue
            arg = 0 if arg is None else arg
            args_list[arg_index] = 1e-3 * arg
        args = tuple(args_list)

        value: Any
        max_ydata = self.controls.get("ydata_info").get("_max_ydata")

        if isinstance(values_to_fit, list) and len(values_to_fit) > 1:
            ret = {}

            for value in values_to_fit:
                index = int(values_to_fit.index(str(value)))
                ret[value] = self.evaluate(t_rho_array_like, *args, **kwargs)[index]

                if IRF is not None:
                    ret[value] = convolve(ret[value], IRF, mode="same")
                if normalize:
                    max_ret = np.max(ret[value]) or 1
                    scaling = scaling or max_ydata / max_ret
                    ret[value] = scaling * np.array(ret[value]) + offset

                if log_scale:
                    ret = np.log(ret - np.min(ret[value]) + 1)

            return ret

        elif isinstance(values_to_fit, list) and len(values_to_fit) == 1:
            for value in values_to_fit:
                index = int(available_values.index(str(value)))
                ret = []
                ret = self.evaluate(t_rho_array_like, *args, **kwargs)[index]
                ret = np.array([float(ret_elem) for ret_elem in ret])

                if IRF is not None:
                    IRF = (
                        IRF
                        if (not isinstance(IRF, pd.DataFrame))
                        else [float(value) for value in IRF.values]
                    )

                    try:
                        ret = convolve(ret, IRF, mode="same")
                    except Exception as e:
                        print("---ERROR---")
                        print(e)
                        irf = [value for value in IRF.values]
                        print(type(IRF))
                        print(irf)
                        print("---END ERROR---")
                if normalize:
                    max_ret = np.max(ret) or 1
                    scaling = scaling or max_ydata / max_ret
                    print(f"scaling: {scaling}")
                    ret = scaling * np.array(ret) + offset

                if log_scale:
                    ret = np.log(ret - np.min(ret) + 1)

            return ret

        elif isinstance(values_to_fit, str):
            index = available_values.index(values_to_fit)
            ret = self.evaluate(t_rho_array_like, *args, **kwargs)[index]

            if IRF is not None:
                ret = convolve(ret, IRF, mode="same")
            if normalize:
                max_ret = np.max(ret) or 1
                scaling = scaling or max_ydata / max_ret
                ret = scaling * np.array(ret) + offset

            if log_scale:
                ret = np.log(ret - np.min(ret) + 1)

            return ret

        else:
            return None

    def fit_settings(
        self,
        values_to_fit: Union[List[str], Any] = None,
        free_params: Union[List[str], Any] = None,
        normalize: Union[bool, None] = None,
        log_scale: Union[bool, None] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Updates fit settings.
        :param values_to_fit: Values passed to scipy.curve_fit(f, ydata, xdata, params) as ydata.
        :type values_to_fit: Union[List[str], Any]
        :param free_params: A list of free parameters passed down to scipy.curve_fit(f, ydata, xdata, params) as params for fitting.
        :type free_params: Union[List[str], Any]
        :param log_scale: bool that controls whether the outputs are rescaled with log. Default: None.
        :type log_scale: Union[bool, None]
        """
        self.controls["values_to_fit"] = values_to_fit or self.controls.get(
            "values_to_fit"
        )
        self.controls["free_params"] = free_params or self.controls.get("free_params")
        self.controls["normalize"] = (
            normalize if normalize is not None else self.controls.get("normalize")
        )
        self.controls["log_scale"] = (
            log_scale if log_scale is not None else self.controls.get("log_scale")
        )

    # def fit(
    #     self,
    #     _t_rho_array_like: List[Tuple[float, float]],
    #     ydata: List[float],
    #     initial_free_params: List[Union[float, int]],
    #     IRF: Union[List[Union[float, int]], None] = None,
    #     normalize: bool = True,
    #     values_to_fit: Union[List[str], Any] = ["R_rho_t"],
    #     free_params: Union[List[str], Any] = ["musp", "offset"],
    #     plot: bool = False,
    #     show_plot: bool = False,
    #     save_path: str = "",
    #     log_scale: Union[bool, None] = None,
    #     *args: Any,
    #     **kwargs: Any,
    # ) -> Tuple[List[float], ...]:
    #     """
    #     Method used to fit the model by Contini to existing data.

    #     :param _t_rho_array_like: An array-like input with tuples of the form (time, radial_coordinate), xdata.
    #     :type _t_rho_array_like: List[Tuple[float, float]]
    #     :param ydata: The data the model's function gets fit to.
    #     :type ydata: List[float]
    #     :param initial_free_params: The initial values for the free parameters to fit.
    #     :type initial_free_params: List[Union[int, float]]
    #     :param IRF: Instrument Response Function as a list of the function's outputs. Default: None.
    #     :type IRF: Union[List[Union[float, int]]
    #     :param normalize: Decide whether to normalize the function's output to the data to fit. Default: True.
    #     :type normalize: bool
    #     :param values_to_fit: Values passed to scipy.curve_fit(f, ydata, xdata, params) as ydata.
    #     :type values_to_fit: Union[List[str], Any]
    #     :param free_params: A list of free parameters passed down to scipy.curve_fit(f, ydata, xdata, params) as params for fitting.
    #     :type free_params: Union[List[str], Any]
    #     :param plot: Controls whether to plot the results. The plots will be saved. Default: False.
    #     :type plot: bool
    #     :param show_plot: Controls whether to display the plots. Default: False.
    #     :type show_plot: bool
    #     :param save_path: The path where the plots will be saved if provided. Default: "".
    #     :type save_path: str
    #     :param log_scale: bool that controls whether the outputs are rescaled with log. Default: None.
    #     :type log_scale: Union[bool, None]
    #     :param args: A tuple of free parameters for fitting.
    #     :type args: Any
    #     :param kwargs: Supports kwargs of the scipy.curve_fit() as well as mode: "approx", "sum" of the G_function().
    #     :type kwargs: Any

    #     Returns:
    #     popt: Values of the free parameters obtained after the function has been fit to the data.
    #     pcov: Covariance matrix of the output popt.

    #     """

    #     self.IRF = IRF if IRF is not None else self.IRF
    #     IRF = self.IRF
    #     self.fit_settings(
    #         values_to_fit=values_to_fit,
    #         free_params=free_params,
    #         normalize=normalize,
    #         log_scale=log_scale,
    #     )
    #     # if IRF:
    #     #     for entry_index, entry in enumerate(IRF):
    #     #         IRF[entry_index] = entry if entry else entry + 1e-5
    #     # ydata = np.array(ydata)
    #     # if IRF is not None:
    #     #     _, ydata = deconvolve(ydata, IRF)
    #     #     print(ydata)
    #     # print(self.normalize, normalize)
    #     print("---INITIAL FREE PARAMS---\n", initial_free_params)
    #     if self.IRF is not None:
    #         # ydata = convolve(ydata, self.IRF, mode="same")
    #         pass

    #     if self.normalize:
    #         max_ydata = np.max(ydata) if np.max(ydata) != 0 else 1
    #         self._max_ydata = max_ydata

    #     _ydata = ydata
    #     if self.log_scale:
    #         _ydata = np.log(ydata - np.min(ydata) + 1)

    #     self.ydata_info = {"ydata_min": np.min(ydata), "ydata_max": np.max(ydata)}
    #     try:
    #         # t_rho_array_like = np.array(_t_rho_array_like)
    #         popt, pcov, *_ = curve_fit(
    #             self.forward,
    #             _t_rho_array_like,
    #             _ydata,
    #             initial_free_params,
    #             method="trf",
    #             bounds=([0.01, 0.01, -0.01], [0.1, 0.1, 50]),
    #             *args,
    #             **kwargs,
    #         )
    #     except TypeError:
    #         print("\n\n---ERROR---\n")
    #         print(type(_t_rho_array_like), _t_rho_array_like)
    #         # print(self.forward(_t_rho_array_like, initial_free_params[0]))
    #         print("---END ERROR---")
    #     # print(pcov[0][0], math.isinf(pcov[0][0]))
    #     # if math.isinf(pcov[0][0]):
    #     #     xdata = torch.tensor(_t_rho_array_like, requires_grad=True)
    #     #     func = self.forward
    #     #     target = torch.tensor(ydata, dtype=torch.float64)

    #     #     guess = initial_free_params
    #     #     weights_LBFGS = torch.tensor(guess, requires_grad=True)
    #     #     weights = weights_LBFGS

    #     #     optimizer = torch.optim.Adam([{"params": weights_LBFGS}], lr=0.3)
    #     #     guesses = []
    #     #     losses = []

    #     #     for epoch in range(5):
    #     #         print(f"---EPOCH {epoch}---\n")
    #     #         optimizer.zero_grad()
    #     #         output = func(xdata, weights)
    #     #         input = torch.tensor(output, requires_grad=True, dtype=torch.float64)
    #     #         loss = F.mse_loss(input, target)
    #     #         loss.backward()
    #     #         optimizer.step()
    #     #         guesses.append(weights.clone())
    #     #         losses.append(loss.clone())
    #     #         print(weights, loss, guesses)

    #     #     popt = weights

    #     if plot:
    #         rho = _t_rho_array_like[0][1]
    #         xdata_t = [t_rho[0] for t_rho in _t_rho_array_like]
    #         for value in values_to_fit:
    #             index = int(values_to_fit.index(value))
    #             params = popt.clone()
    #             offset = self._offset if ("offset" not in free_params) else params.pop()
    #             musp = self._musp if ("musp" not in free_params) else params.pop()
    #             mua = self._mua if ("mua" not in free_params) else params.pop()

    #             ydata_fit = self.forward(
    #                 _t_rho_array_like,
    #                 mua=mua,
    #                 musp=musp,
    #                 offset=offset,
    #                 normalize=True,
    #                 IRF=IRF,
    #             )[index]
    #             fit = plt.plot(  # noqa: F841
    #                 xdata_t,
    #                 ydata_fit,
    #                 color="r",
    #                 label=f"fit: mua={mua}, musp={musp}, off={offset}",
    #             )  # noqa: F841

    #             raw_data = plt.plot(  # noqa: F841
    #                 xdata_t,
    #                 ydata,
    #                 color="b",
    #                 label="raw data",
    #                 marker="o",
    #                 linestyle=" ",
    #             )

    #             plt.legend(loc="upper right")
    #             plt.xlabel("Time in ps")
    #             plt.ylabel(f"{value}(t, rho={rho}[mm])/max({value}(t, rho={rho}[mm]))")

    #             if show_plot:
    #                 plt.show()
    #             timestamp = datetime.datetime.now().isoformat()
    #             path = (
    #                 save_path
    #                 or f"{pathlib.Path(__file__).resolve().parent.parent.parent}\\plots"
    #             )
    #             plt.savefig(f"{path}\\{value}{timestamp}.pdf")
    #             plt.clf()

    #     return popt, pcov

    # def load_data(self, *args: Any, **kwargs: Any) -> None:
    #     return NotImplemented

    # def _load_IRF(self, *args: Any, **kwargs: Any) -> None:
    #     return NotImplemented

    # def load_xdata(self, *args: Any, **kwargs: Any) -> None:
    #     return NotImplemented

    # def _convolve(self, *args: Any, **kwargs: Any) -> None:
    #     return NotImplemented
