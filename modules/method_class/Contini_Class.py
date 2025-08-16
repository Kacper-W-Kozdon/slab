import datetime
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import convolve

from ..other.utils import A_parameter, Image_Sources_Positions, Mean_Path_T_R
from ..reflect_transmit.reflect_transmit import (
    Reflectance_Transmittance,
    Reflectance_Transmittance_rho,
    Reflectance_Transmittance_rho_t,
    Reflectance_Transmittance_t,
)


class Contini:
    def __init__(
        self,
        s: Union[int, float] = 0,
        mua: Union[int, float, None] = None,
        musp: Union[int, float, None] = None,
        n1: Union[int, float] = 0,
        n2: Union[int, float] = 0,
        anisothropy_coeff: Union[int, float, None] = 0.85,
        phantom: Optional[str] = "",
        DD: Optional[str] = "Dmus",
        m: int = 100,
        eq: str = "RTE",
        IRF: Union[List[Union[float, int]], None] = None,
        normalize: bool = True,
        values_to_fit: Union[List[str], Any] = ["R_rho_t"],
        free_params: Union[List[str], Any] = ["musp", "offset"],
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
        """

        self._s = s * 1e-3
        self._mua = None if not mua else mua * 1e3
        self._musp = None if not musp else musp * 1e3
        self.n1 = n1
        self.n2 = n2
        self.phantom = phantom
        self.DD = DD
        self.m = m
        self.eq = eq
        self.anisothropy_coeff = anisothropy_coeff
        self.IRF = IRF
        self.normalize = normalize
        self.values_to_fit = values_to_fit
        self.free_params = free_params
        self.ydata_info = {}
        self._offset = 0

        # print(f"---INIT---\n{self._mua, self._musp, self._offset}")
        self.err = 1e-6  # noqa: F841

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
    def s(self) -> float:
        return self._s

    @s.setter
    def s(self, value: float) -> None:
        self._s = value * 1e-3

    @s.deleter
    def s(self) -> None:
        self._s = 0

    @property
    def offset(self) -> float:
        return self._offset

    @offset.setter
    def offset(self, value: float) -> None:
        self._offset = value

    @offset.deleter
    def offset(self) -> None:
        self._offset = 0

    def __call__(
        self,
        t_rho: Union[Tuple[float, float], List[Tuple[float, float],]],
        mua: Union[int, float, None] = None,
        musp: Union[int, float, None] = None,
        offset: Union[int, float] = 0,
        anisothropy_coeff: Union[int, float, None] = None,
        **kwargs: Any,
    ) -> Union[Tuple[Any, ...], Tuple[List[Any], ...]]:
        """
        The call method evaluating parameters of the Contini model.

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
        if isinstance(t_rho, tuple):
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

            anisothropy_coeff = anisothropy_coeff or self.anisothropy_coeff

            t = t_rho[0] * 1e-12
            rho = t_rho[1] * 1e-3
            s = self._s
            n1 = self.n1
            n2 = self.n2
            phantom = self.phantom  # noqa: F841
            DD = self.DD
            m = self.m
            eq = self.eq
            # print(mua, musp)

            R_rho_t, T_rho_t = Reflectance_Transmittance_rho_t(
                rho, t, mua, musp, s, m, n1, n2, DD, eq, anisothropy_coeff, **kwargs
            )
            # print(rho, t, mua, musp, R_rho_t)

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

        else:
            R_rho_t = []
            T_rho_t = []
            R_rho = []
            T_rho = []
            R_t = []
            T_t = []
            l_rho_R = []
            l_rho_T = []
            R = []
            T = []
            A = []
            Z = []
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
            anisothropy_coeff = anisothropy_coeff or self.anisothropy_coeff

            for value in t_rho:
                t = value[0] * 1e-12
                rho = value[1] * 1e-3
                s = self._s
                n1 = self.n1
                n2 = self.n2
                phantom = self.phantom  # noqa: F841
                DD = self.DD
                m = self.m
                eq = self.eq

                R_rho_t_, T_rho_t_ = Reflectance_Transmittance_rho_t(
                    rho, t, mua, musp, s, m, n1, n2, DD, eq, anisothropy_coeff
                )
                # print(R_rho_t)
                R_rho_t.append(R_rho_t_)
                T_rho_t.append(T_rho_t_)

                R_rho_, T_rho_ = Reflectance_Transmittance_rho(
                    rho, mua, musp, s, m, n1, n2, DD, eq
                )
                R_rho.append(R_rho_)
                T_rho.append(T_rho_)

                R_t_, T_t_ = Reflectance_Transmittance_t(
                    t, mua, musp, s, m, n1, n2, DD, eq
                )
                R_t.append(R_t_)
                T_t.append(T_t_)

                l_rho_R_, l_rho_T_ = Mean_Path_T_R(rho, mua, musp, s, m, n1, n2, DD, eq)
                l_rho_R.append(l_rho_R_)
                l_rho_T.append(l_rho_T_)

                R_, T_ = Reflectance_Transmittance(mua, musp, s, m, n1, n2, DD, eq)
                R.append(R_)
                T.append(T_)

                A_ = A_parameter(n1, n2)
                A.append(A_)

                Z_ = Image_Sources_Positions(s, mua, musp, n1, n2, DD, m, eq)
                Z.append(Z_)

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

    def fit_settings(
        self,
        values_to_fit: Union[List[str], Any] = None,
        free_params: Union[List[str], Any] = None,
        normalize: Union[bool, None] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Updates fit settings.
        :param values_to_fit: Values passed to scipy.curve_fit(f, ydata, xdata, params) as ydata.
        :type values_to_fit: Union[List[str], Any]
        :param free_params: A list of free parameters passed down to scipy.curve_fit(f, ydata, xdata, params) as params for fitting.
        :type free_params: Union[List[str], Any]
        """
        self.values_to_fit = values_to_fit or self.values_to_fit
        self.free_params = free_params or self.free_params
        self.normalize = normalize if normalize is not None else self.normalize

    def forward(
        self,
        t_rho_array_like: List[Tuple[float, float]],
        *args: Any,
        **kwargs: Any,
    ) -> Union[float, int, None, List[float], Dict[Any, Any]]:
        """
        The call method returning the function used for scipy.curve_fit().

        :param t_rho_array_like: Variables of the model in the form List[(time, radial_coordinate), ...], passed as xdata to scipy.curve_fit(f, ydata, xdata, params).
        :type t_rho_array_like: List[Tuple[float, float]]
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

        values_to_fit: Union[List[str], Any] = self.values_to_fit or ["R_rho_t"]
        free_params: Union[List[str], Any] = self.free_params or ["musp", "offset"]
        normalize = self.normalize or False
        if kwargs.get("normalize") is not None:
            normalize = kwargs.get("normalize")  # noqa: F841

        IRF = self.IRF

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
        available_free_params = ["mua", "musp", "offset"]

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
            args_list = [self._mua, self._musp, self._offset]

        for param_index, param in enumerate(available_free_params):
            # print((param not in free_params) and args, args_list, param, param_index)
            if (param not in free_params) and args:
                # param_value = self._mua if param_index == 0 else self._musp
                if param_index == 0:
                    param_value = self._mua
                if param_index == 1:
                    param_value = self._musp
                if param_index == 3:
                    param_value = self._offset

                args_list.insert(param_index, param_value)
                # print(args_list)
        # print(args, kwargs, args_list, free_params, available_free_params)

        for param_index, param in enumerate(available_free_params):
            if (param in free_params) and args:
                # try:
                args_list[param_index] = 1e3 * args_list[param_index]
                # except:
                #     print("---ERROR---")
                #     print(args_list, param_index, args)

        index = available_free_params.index("offset")
        offset = args_list[index] or self._offset
        for arg_index, arg in enumerate(args_list):
            if arg_index == available_free_params.index("offset"):
                continue
            args_list[arg_index] = 1e-3 * arg
        args = tuple(args_list)

        # print(args)

        value: Any
        # print(values_to_fit, isinstance(values_to_fit, list) and len(values_to_fit) == 1, t_rho_array_like)

        if isinstance(values_to_fit, list) and len(values_to_fit) > 1:
            ret = {}
            for value in values_to_fit:
                index = int(values_to_fit.index(str(value)))
                ret[value] = self(t_rho_array_like, *args, **kwargs)[index]

                if IRF is not None:
                    ret[value] = convolve(ret[value], IRF, mode="same")
                if normalize:
                    max_ret = np.max(ret[value]) or 1
                    ret[value] = np.array(ret[value]) / max_ret

            # ret = np.log(ret + 1)
            return ret + offset

        elif isinstance(values_to_fit, list) and len(values_to_fit) == 1:
            for value in values_to_fit:
                index = int(available_values.index(str(value)))
                ret = []
                # for elem in t_rho_array_like:
                #     ret.append(self(elem, *args, **kwargs)[index])
                ret = self(t_rho_array_like, *args, **kwargs)[index]
                ret = np.array([float(ret_elem) for ret_elem in ret])
                # print(ret)
                # print("---TEST RETURN---")
                # print(index, ret, args)
                # print(t_rho_array_like)
                # print()

                if IRF is not None:
                    try:
                        ret = convolve(ret, IRF, mode="same")
                    except Exception as e:
                        print("---ERROR---")
                        print(e)
                        print(args, type(t_rho_array_like))
                        print(ret)
                        print("---END ERROR---")
                if normalize:
                    max_ret = np.max(ret) or 1
                    ret = np.array(ret) / max_ret

                    # except Exception:
                    #     print(free_params, values_to_fit)
                    #     print(
                    #         t_rho_array_like[0][0],
                    #         t_rho_array_like[1][0],
                    #         t_rho_array_like[2][0],
                    #     )
                    #     print(
                    #         self(t_rho_array_like, *args, **kwargs)[0],
                    #         self(t_rho_array_like, *args, **kwargs)[1],
                    #     )
            # ret = np.log(ret + 1)
            # print(ret)
            return ret + offset

        elif isinstance(values_to_fit, str):
            index = available_values.index(values_to_fit)
            ret = self(t_rho_array_like, *args, **kwargs)[index]

            if IRF is not None:
                ret = convolve(ret, IRF, mode="same")
            if normalize:
                max_ret = np.max(ret) or 1
                ret = np.array(ret) / max_ret

            # ret = np.log(ret + 1)
            return ret + offset

        else:
            return None

    def fit(
        self,
        _t_rho_array_like: List[Tuple[float, float]],
        ydata: List[float],
        initial_free_params: List[Union[float, int]],
        IRF: Union[List[Union[float, int]], None] = None,
        normalize: bool = True,
        values_to_fit: Union[List[str], Any] = ["R_rho_t"],
        free_params: Union[List[str], Any] = ["musp", "offset"],
        plot: bool = False,
        show_plot: bool = False,
        save_path: str = "",
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[List[float], ...]:
        """
        Method used to fit the model by Contini to existing data.

        :param _t_rho_array_like: An array-like input with tuples of the form (time, radial_coordinate), xdata.
        :type _t_rho_array_like: List[Tuple[float, float]]
        :param ydata: The data the model's function gets fit to.
        :type ydata: List[float]
        :param initial_free_params: The initial values for the free parameters to fit.
        :type initial_free_params: List[Union[int, float]]
        :param IRF: Instrument Response Function as a list of the function's outputs. Default: None.
        :type IRF: Union[List[Union[float, int]]
        :param normalize: Decide whether to normalize the function's output to the data to fit. Default: True.
        :type normalize: bool
        :param values_to_fit: Values passed to scipy.curve_fit(f, ydata, xdata, params) as ydata.
        :type values_to_fit: Union[List[str], Any]
        :param free_params: A list of free parameters passed down to scipy.curve_fit(f, ydata, xdata, params) as params for fitting.
        :type free_params: Union[List[str], Any]
        :param plot: Controls whether to plot the results. The plots will be saved. Default: False.
        :type plot: bool
        :param show_plot: Controls whether to display the plots. Default: False.
        :type show_plot: bool
        :param save_path: The path where the plots will be saved if provided. Default: "".
        :type save_path: str
        :param args: A tuple of free parameters for fitting.
        :type args: Any
        :param kwargs: Supports kwargs of the scipy.curve_fit() as well as mode: "approx", "sum" of the G_function().
        :type kwargs: Any

        Returns:
        popt: Values of the free parameters obtained after the function has been fit to the data.
        pcov: Covariance matrix of the output popt.

        """

        self.IRF = IRF if IRF is not None else self.IRF
        IRF = self.IRF
        self.fit_settings(
            values_to_fit=values_to_fit, free_params=free_params, normalize=normalize
        )
        # if IRF:
        #     for entry_index, entry in enumerate(IRF):
        #         IRF[entry_index] = entry if entry else entry + 1e-5
        # ydata = np.array(ydata)
        # if IRF is not None:
        #     _, ydata = deconvolve(ydata, IRF)
        #     print(ydata)
        # print(self.normalize, normalize)
        print("---INITIAL FREE PARAMS---\n", initial_free_params)
        if self.normalize:
            ydata = ydata - np.min(ydata)
            max_ydata = np.max(ydata) if np.max(ydata) != 0 else 1
            ydata = ydata / max_ydata
        self.ydata_info = {"ydata_min": np.min(ydata), "ydata_max": np.max(ydata)}
        try:
            popt, pcov, *_ = curve_fit(
                self.forward,
                _t_rho_array_like,
                ydata,
                initial_free_params,
                *args,
                **kwargs,
            )
        except ValueError:
            print("\n\n---ERROR---\n")
            print(initial_free_params)
            print(self.forward(_t_rho_array_like, initial_free_params[0]))
            print("---END ERROR---")
        # print(pcov[0][0], math.isinf(pcov[0][0]))
        # if math.isinf(pcov[0][0]):
        #     xdata = torch.tensor(_t_rho_array_like, requires_grad=True)
        #     func = self.forward
        #     target = torch.tensor(ydata, dtype=torch.float64)

        #     guess = initial_free_params
        #     weights_LBFGS = torch.tensor(guess, requires_grad=True)
        #     weights = weights_LBFGS

        #     optimizer = torch.optim.Adam([{"params": weights_LBFGS}], lr=0.3)
        #     guesses = []
        #     losses = []

        #     for epoch in range(5):
        #         print(f"---EPOCH {epoch}---\n")
        #         optimizer.zero_grad()
        #         output = func(xdata, weights)
        #         input = torch.tensor(output, requires_grad=True, dtype=torch.float64)
        #         loss = F.mse_loss(input, target)
        #         loss.backward()
        #         optimizer.step()
        #         guesses.append(weights.clone())
        #         losses.append(loss.clone())
        #         print(weights, loss, guesses)

        #     popt = weights

        if plot:
            rho = _t_rho_array_like[0][1]
            xdata_t = [t_rho[0] for t_rho in _t_rho_array_like]
            for value in values_to_fit:
                index = int(values_to_fit.index(value))
                params = popt.clone()
                offset = self._offset if ("offset" not in free_params) else params.pop()
                musp = self._musp if ("musp" not in free_params) else params.pop()
                mua = self._mua if ("mua" not in free_params) else params.pop()

                ydata_fit = self.forward(
                    _t_rho_array_like,
                    mua=mua,
                    musp=musp,
                    offset=offset,
                    normalize=True,
                    IRF=IRF,
                )[index]
                fit = plt.plot(  # noqa: F841
                    xdata_t,
                    ydata_fit,
                    color="r",
                    label=f"fit: mua={mua}, musp={musp}, off={offset}",
                )  # noqa: F841

                if not self.normalize:
                    ydata = ydata - np.min(ydata)
                    ydata_max = np.max(ydata)
                    ydata = ydata / ydata_max

                raw_data = plt.plot(  # noqa: F841
                    xdata_t,
                    ydata,
                    color="b",
                    label="raw data",
                    marker="o",
                    linestyle=" ",
                )

                plt.legend(loc="upper right")
                plt.xlabel("Time in ps")
                plt.ylabel(f"{value}(t, rho={rho}[mm])/max({value}(t, rho={rho}[mm]))")

                if show_plot:
                    plt.show()
                timestamp = datetime.datetime.now().isoformat()
                path = (
                    save_path
                    or f"{pathlib.Path(__file__).resolve().parent.parent.parent}\\plots"
                )
                plt.savefig(f"{path}\\{value}{timestamp}.pdf")
                plt.clf()

        return popt, pcov

    def load_data(self, *args: Any, **kwargs: Any) -> None:
        return NotImplemented

    def _load_IRF(self, *args: Any, **kwargs: Any) -> None:
        return NotImplemented

    def load_xdata(self, *args: Any, **kwargs: Any) -> None:
        return NotImplemented

    def _convolve(self, *args: Any, **kwargs: Any) -> None:
        return NotImplemented
