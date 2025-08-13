from typing import List, Optional, Tuple, Union

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
        phantom: Optional[str] = "",
        DD: Optional[str] = "Dmus",
        m: int = 200,
        eq: str = "RTE",
    ):
        self.s = s * 1e-3
        self.mua = None if not mua else mua * 1e3
        self.musp = None if not musp else musp * 1e3
        self.n1 = n1
        self.n2 = n2
        self.phantom = phantom
        self.DD = DD
        self.m = m
        self.eq = eq

        self.err = 1e-6  # noqa: F841

    def __call__(self, t_rho, mua=0, musp=0):
        if isinstance(t_rho, tuple):
            mua = mua if self.mua is None else self.mua
            musp = musp if self.musp is None else self.musp
            mua *= 1e3
            musp *= 1e3
            t = t_rho[0] * 1e-9
            rho = t_rho[1] * 1e-3
            s = self.s
            n1 = self.n1
            n2 = self.n2
            phantom = self.phantom  # noqa: F841
            DD = self.DD
            m = self.m
            eq = self.eq

            R_rho_t, T_rho_t = Reflectance_Transmittance_rho_t(
                rho, t, mua, musp, s, m, n1, n2, DD, eq
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
            mua = mua if self.mua is None else self.mua
            musp = musp if self.musp is None else self.musp
            mua *= 1e3
            musp *= 1e3

            for value in t_rho:
                t = value[0] * 1e-9
                rho = value[1] * 1e-3
                s = self.s
                n1 = self.n1
                n2 = self.n2
                phantom = self.phantom  # noqa: F841
                DD = self.DD
                m = self.m
                eq = self.eq

                R_rho_t_, T_rho_t_ = Reflectance_Transmittance_rho_t(
                    rho, t, mua, musp, s, m, n1, n2, DD, eq
                )
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

    def _fit(
        self,
        t_rho_array_like: List[Tuple],
        *args,
        values_to_fit: List[str] = ["R_rho_t"],
        free_params: List[str] = ["musp"],
    ):
        available_values = [
            "R_rho_t",
            "T_rho_t",
            "R_rho",
            "T_rho",
            "R_t",
            "T_t",
            "l_rho_R",
            "l_rho_T",
            "R",
            "T",
            "A",
            "Z",
        ]
        available_free_params = ["mua", "musp"]

        if not all([param in available_free_params for param in free_params]):
            raise ValueError(
                f"Obtained params: {free_params} not in the list of params available for fitting."
            )

        if not all([value in available_values for value in values_to_fit]):
            raise ValueError(
                f"Obtained values: {values_to_fit} not in the list of values available for fitting."
            )

        ret = None
        args_list = list(args)

        for param_index, param in enumerate(available_free_params):
            if param not in free_params:
                value = self.mua if param_index == 0 else self.musp
                args_list.insert(param_index, value)

        args = tuple(args_list)
        # print(f"---MUA--- : {args[0]}\n")
        # print(f"---MUSP--- : {args[1]}\n\n")

        if isinstance(values_to_fit, list) and len(values_to_fit) > 1:
            ret = {}
            for value in values_to_fit:
                index = values_to_fit.index(value)
                ret[value] = self(t_rho_array_like, *args)[index]

            return ret

        elif isinstance(values_to_fit, list) and len(values_to_fit) == 1:
            for value in values_to_fit:
                index = available_values.index(value)
                ret = self(t_rho_array_like, *args)[index]

            return ret

        elif isinstance(values_to_fit, str):
            index = available_values.index(values_to_fit)
            ret = self(t_rho_array_like, *args)[index]

            return ret

    def fit(self, _t_rho_array_like: List[Tuple], *args):
        # print(args)
        return self._fit(_t_rho_array_like, *args)
