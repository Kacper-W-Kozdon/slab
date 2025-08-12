from collections import OrderedDict
from typing import Optional, Union

from numpy import exp, pi


def Contini(
    rho: Union[int, float],
    t: Union[int, float],
    s: Union[int, float],
    mua: Union[int, float],
    musp: Union[int, float],
    n1: Union[int, float],
    n2: Union[int, float],
    phantom: Optional[str],
    DD: Optional[str],
    m: int = 200,
):
    t = t * 1e-9
    rho = rho * 1e-3
    s = s * 1e-3
    mua = mua * 1e3
    musp = musp * 1e3

    err = 1e-6  # noqa: F841

    R_rho_t, T_rho_t = Reflectance_Transmittance_rho_t(
        rho, t, s, m, mua, musp, n1, n2, DD
    )

    R_rho, T_rho = Reflectance_Transmittance_rho(rho, s, m, mua, musp, n1, n2, DD)

    R_t, T_t = Reflectance_Transmittance_t(t, s, m, mua, musp, n1, n2, DD)

    l_rho_R, l_rho_T = Mean_Path_T_R(rho, s, m, mua, musp, n1, n2, DD)

    R, T = Reflectance_Transmittance(s, m, mua, musp, n1, n2, DD)

    A = A_param(n1, n2)

    Z = Image_Sources_Positions(s, mua, musp, n1, n2, DD, m)

    return R_rho_t, T_rho_t, R_rho, T_rho, R_t, T_t, l_rho_R, l_rho_T, R, T, A, Z


def D_parameter(DD, mua, musp):
    D = None

    return D


def Reflectance_Transmittance_rho_t(rho, t, s, m, mua, musp, n1, n2, DD):
    c = 299792458
    v = c / n2

    D = D_parameter(DD, mua, musp)

    R_rho_t = 0.0
    T_rho_t = 0.0

    R_rho_t_source_sum = 0.0
    T_rho_t_source_sum = 0.0

    Z = Image_Sources_Positions(s, mua, musp, n1, n2, DD, m)

    for index in range(-m, m + 1):
        z1, z2, z3, z4 = Z[f"Z_{index}"]

        R_rho_t_source_sum += z3 * exp(-(z3**2) / (4 * D * v * t)) - z4 * exp(
            -(z4**2) / (4 * D * v * t)
        )

        T_rho_t_source_sum += z1 * exp(-(z1**2) / (4 * D * v * t)) - z2 * exp(
            -(z2**2) / (4 * D * v * t)
        )

    R_rho_t = (
        -exp(-(mua * v * t - rho**2) / (4 * D * v * t))
        / (2 * ((4 * pi * D * v) ** (3 / 2)) * t ** (5 / 2))
        * R_rho_t_source_sum
    )
    T_rho_t = (
        exp(-(mua * v * t - rho**2) / (4 * D * v * t))
        / (2 * ((4 * pi * D * v) ** (3 / 2)) * t ** (5 / 2))
        * T_rho_t_source_sum
    )

    R_rho_t *= 1e-6 * 1e-12
    T_rho_t *= 1e-6 * 1e-12

    R_rho_t = R_rho_t if t > 0 else 0
    T_rho_t = T_rho_t if t > 0 else 0

    return R_rho_t, T_rho_t


def Reflectance_Transmittance_rho(rho, s, m, mua, musp, n1, n2, DD):
    R_rho, T_rho = None

    D = D_parameter(DD, mua, musp)

    R_rho_source_sum = 0
    T_rho_source_sum = 0

    Z = Image_Sources_Positions(s, mua, musp, n1, n2, DD, m)

    for index in range(-m, m + 1):
        z1, z2, z3, z4 = Z[f"Z_{index}"]

        R_rho_source_sum += z3 * (
            D ** (-1 / 2) * mua ** (1 / 2) * (rho**2 + z3**2) ** (-1)
            + (rho**2 + z3**2) ** (-3 / 2)
        ) * exp(-((mua * (rho**2 + z3**2) / D) ** (1 / 2))) - z4 * (
            D ** (-1 / 2) * mua
            ^ (1 / 2) * (rho**2 + z4**2) ** (-1) + (rho**2 + z4**2) ** (-3 / 2)
        ) * exp(-((mua * (rho**2 + z4**2) / D) ** (1 / 2)))
        T_rho_source_sum += z1 * (
            D ** (-1 / 2) * mua ** (1 / 2) * (rho**2 + z1**2) ** (-1)
            + (rho**2 + z1**2) ** (-3 / 2)
        ) * exp(-((mua * (rho**2 + z1**2) / D) ** (1 / 2))) - z2 * (
            D ** (-1 / 2) * mua
            ^ (1 / 2) * (rho**2 + z2**2) ** (-1) + (rho**2 + z2**2) ** (-3 / 2)
        ) * exp(-((mua * (rho**2 + z2**2) / D) ** (1 / 2)))

    R_rho = -1 / (4 * pi) * R_rho_source_sum
    T_rho = 1 / (4 * pi) * T_rho_source_sum

    return R_rho, T_rho


def Reflectance_Transmittance_t(t, s, m, mua, musp, n1, n2, DD):
    R_t, T_t = None
    return R_t, T_t


def Mean_Path_T_R(rho, s, m, mua, musp, n1, n2, DD):
    l_rho_R, l_rho_T = None
    return l_rho_R, l_rho_T


def Reflectance_Transmittance(s, m, mua, musp, n1, n2, DD):
    R, T = None
    return R, T


def A_param(n1, n2):
    A = None
    return A


def Image_Sources_Positions(s, mua, musp, n1, n2, DD, m):
    Z = OrderedDict()
    for index in range(-m, m + 1):
        z1, z2, z3, z4 = None
        Z[f"Z_{index}"] = z1, z2, z3, z4
    return Z
