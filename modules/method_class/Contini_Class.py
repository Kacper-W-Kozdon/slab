from typing import Optional, Union

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
        self.mua = mua * 1e3
        self.musp = musp * 1e3
        self.n1 = n1
        self.n2 = n2
        self.phantom = phantom
        self.DD = DD
        self.m = m
        self.eq = eq

        self.err = 1e-6  # noqa: F841

    def __call__(self, t_rho=(0, 0), mua=None, musp=None):
        mua = mua or self.mua
        musp = musp or self.musp
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

        return R_rho_t, T_rho_t, R_rho, T_rho, R_t, T_t, l_rho_R, l_rho_T, R, T, A, Z
