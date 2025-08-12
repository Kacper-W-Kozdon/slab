from collections import OrderedDict
from typing import Optional, Union

from numpy import exp, log, pi


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
    eq: str = "RTE",
):
    t = t * 1e-9
    rho = rho * 1e-3
    s = s * 1e-3
    mua = mua * 1e3
    musp = musp * 1e3

    err = 1e-6  # noqa: F841

    R_rho_t, T_rho_t = Reflectance_Transmittance_rho_t(
        rho, t, s, m, mua, musp, n1, n2, DD, eq
    )

    R_rho, T_rho = Reflectance_Transmittance_rho(rho, s, m, mua, musp, n1, n2, DD, eq)

    R_t, T_t = Reflectance_Transmittance_t(t, s, m, mua, musp, n1, n2, DD, eq)

    l_rho_R, l_rho_T = Mean_Path_T_R(rho, s, m, mua, musp, n1, n2, DD, eq)

    R, T = Reflectance_Transmittance(s, m, mua, musp, n1, n2, DD, eq)

    A = A_param(n1, n2)

    Z = Image_Sources_Positions(s, mua, musp, n1, n2, DD, m, eq)

    return R_rho_t, T_rho_t, R_rho, T_rho, R_t, T_t, l_rho_R, l_rho_T, R, T, A, Z


def D_parameter(DD, mua, musp, eq):
    D = None

    return D


def Reflectance_Transmittance_rho_t(rho, t, s, m, mua, musp, n1, n2, DD, eq):
    c = 299792458
    v = c / n2

    D = D_parameter(DD, mua, musp, eq)

    R_rho_t = 0.0
    T_rho_t = 0.0

    R_rho_t_source_sum = 0.0
    T_rho_t_source_sum = 0.0

    Z = Image_Sources_Positions(s, mua, musp, n1, n2, DD, m, eq)

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


def Reflectance_Transmittance_rho(rho, s, m, mua, musp, n1, n2, DD, eq):
    R_rho, T_rho = None, None

    D = D_parameter(DD, mua, musp, eq)

    R_rho_source_sum = 0
    T_rho_source_sum = 0

    Z = Image_Sources_Positions(s, mua, musp, n1, n2, DD, m, eq)

    for index in range(-m, m + 1):
        z1, z2, z3, z4 = Z[f"Z_{index}"]

        R_rho_source_sum += z3 * (
            D ** (-1 / 2) * mua ** (1 / 2) * (rho**2 + z3**2) ** (-1)
            + (rho**2 + z3**2) ** (-3 / 2)
        ) * exp(-((mua * (rho**2 + z3**2) / D) ** (1 / 2))) - z4 * (
            D ** (-1 / 2) * mua ** (1 / 2) * (rho**2 + z4**2) ** (-1)
            + (rho**2 + z4**2) ** (-3 / 2)
        ) * exp(-((mua * (rho**2 + z4**2) / D) ** (1 / 2)))

        if m == 0:
            continue
        else:
            T_rho_source_sum += z1 * (
                D ** (-1 / 2) * mua ** (1 / 2) * (rho**2 + z1**2) ** (-1)
                + (rho**2 + z1**2) ** (-3 / 2)
            ) * exp(-((mua * (rho**2 + z1**2) / D) ** (1 / 2))) - z2 * (
                D ** (-1 / 2) * mua ** (1 / 2) * (rho**2 + z2**2) ** (-1)
                + (rho**2 + z2**2) ** (-3 / 2)
            ) * exp(-((mua * (rho**2 + z2**2) / D) ** (1 / 2)))

    R_rho = -1 / (4 * pi) * R_rho_source_sum
    T_rho = 1 / (4 * pi) * T_rho_source_sum

    R_rho *= 1e-6
    T_rho *= 1e-6

    return R_rho, T_rho


def Reflectance_Transmittance_t(t, s, m, mua, musp, n1, n2, DD, eq):
    R_t, T_t = None, None

    c = 299792458
    v = c / n2
    D = D_parameter(DD, mua, musp, eq)

    R_t_source_sum = 0
    T_t_source_sum = 0

    Z = Image_Sources_Positions(s, mua, musp, n1, n2, DD, m, eq)

    for index in range(-m, m + 1):
        z1, z2, z3, z4 = Z[f"Z_{index}"]
        R_t_source_sum += z3 * exp(-(z3**2) / (4 * D * v * t)) - z4 * exp(
            -(z4**2) / (4 * D * v * t)
        )
        T_t_source_sum += z1 * exp(-(z1**2) / (4 * D * v * t)) - z2 * exp(
            -(z2**2) / (4 * D * v * t)
        )

    R_t = (
        -exp(-mua * v * t)
        / (2 * (4 * pi * D * v) ** (1 / 2) * t ** (3 / 2))
        * R_t_source_sum
    )
    T_t = (
        exp(-mua * v * t)
        / (2 * (4 * pi * D * v) ** (1 / 2) * t ** (3 / 2))
        * T_t_source_sum
    )
    return R_t, T_t


def Mean_Path_T_R(rho, s, m, mua, musp, n1, n2, DD):
    l_rho_R, l_rho_T = None
    return l_rho_R, l_rho_T


def Reflectance_Transmittance(s, m, mua, musp, n1, n2, DD, eq):
    R, T = None, None
    return R, T


def A_param_approx(n1, n2):
    A = None
    A = (
        504.332889
        - 2641.0021 * (n2 / n1)
        + 5923.699064 * (n2 / n1) ** 2
        - 7376.355814 * (n2 / n1) ** 3
        + 5507.53041 * (n2 / n1) ** 4
        - 2463.357945 * (n2 / n1) ** 5
        + 610.956547 * (n2 / n1) ** 6
        - 64.8047 * (n2 / n1) ** 7
    )
    return A


def A_param(n1, n2):
    A = 0
    n = n2 / n1

    if n == 1:
        A = 1
        return A

    if n > 1:
        t1 = (
            4
            * (
                -1
                - n**2
                + 6 * n**3
                - 10 * n**4
                - 3 * n**5
                + 2 * n**6
                + 6 * n**7
                - 3 * n**8
                - (6 * n**2 + 9 * n**6) * (n**2 - 1) ** (1 / 2)
            )
            / (3 * n * (n**2 - 1) ** 2 * (n**2 + 1) ** 3)
        )
        t2 = (
            -8
            + 28 * n**2
            + 35 * n**3
            - 140 * n**4
            + 98 * n**5
            - 13 * n**7
            + 13 * n * (n**2 - 1) ** 3 * (1 - (1 / n**2)) ** (1 / 2)
        ) / (105 * n**3 * (n**2 - 1) ** 2)
        t3 = (
            2
            * n**3
            * (3 + 2 * n**4)
            * log(
                (
                    (n - (1 + n**2) ** (1 / 2))
                    * (2 + n**2 + 2 * (1 + n**2) ** (1 / 2))
                    * (n**2 + (n**4 - 1) ** (1 / 2))
                )
                / (
                    n**2
                    * (n + (1 + n**2) ** (1 / 2))
                    * (-(n**2) + (n**4 - 1) ** (1 / 2))
                )
            )
            / ((n**2 - 1) ** 2 * (n**2 + 1) ** (7 / 2))
        )
        t4 = (
            (1 + 6 * n**4 + n**8) * log((-1 + n) / (1 + n))
            + 4 * (n**2 + n**6) * log((n**2 * (1 + n)) / (n - 1))
        ) / ((n**2 - 1) ** 2 * (n**2 + 1) ** 3)

        B = 1 + (3 / 2) * (
            2 * (1 - 1 / n**2) ** (3 / 2) / 3
            + t1
            + t2
            + ((1 + 6 * n**4 + n**8) * (1 - (n**2 - 1) ** (3 / 2) / n**3))
            / (3 * (n**4 - 1) ** 2)
            + t3
        )
        C = (
            1
            - (
                (
                    2
                    + 2 * n
                    - 3 * n**2
                    + 7 * n**3
                    - 15 * n**4
                    - 19 * n**5
                    - 7 * n**6
                    + 3 * n**7
                    + 3 * n**8
                    + 3 * n**9
                )
                / (3 * n**2 * (n - 1) * (n + 1) ** 2 * (n**2 + 1) ** 2)
            )
            - t4
        )
        A = B / C

        return A

    r1 = (
        -4
        + n
        - 4 * n**2
        + 25 * n**3
        - 40 * n**4
        - 6 * n**5
        + 8 * n**6
        + 30 * n**7
        - 12 * n**8
        + n**9
        + n**11
    ) / (3 * n * (n**2 - 1) ** 2 * (n**2 + 1) ** 3)
    r2 = (
        (2 * n**3 * (3 + 2 * n**4))
        / ((n**2 - 1) ** 2 * (n**2 + 1) ** (7 / 2))
        * log(
            (n**2 * (n - (1 + n**2) ** (1 / 2)))
            * (2 + n**2 + 2 * (1 + n**2) ** (1 / 2))
            / (n + (1 + n**2) ** (1 / 2))
            / (-2 + n**4 - 2 * (1 - n**4) ** (1 / 2))
        )
    )
    r3 = (4 * (1 - n**2) ** (1 / 2) * (1 + 12 * n**4 + n**8)) / (
        3 * n * (n**2 - 1) ** 2 * (n**2 + 1) ** 3
    )
    r4 = (
        (1 + 6 * n**4 + n**8) * log((1 - n) / (1 + n))
        + 4 * (n**2 + n**6) * log((1 + n) / (n**2 * (1 - n)))
    ) / ((n**2 - 1) ** 2 * (n**2 + 1) ** 3)

    A = (
        1
        + (3 / 2) * (8 * (1 - n**2) ** (3 / 2) / (105 * n**3))
        - (
            ((n - 1) ** 2 * (8 + 32 * n + 52 * n**2 + 13 * n**3))
            / (105 * n**3 * (1 + n) ** 2)
            + r1
            + r2
            + r3
        )
    ) / (
        1
        - (-3 + 7 * n + 13 * n**2 + 9 * n**3 - 7 * n**4 + 3 * n**5 + n**6 + n**7)
        / (3 * (n - 1) * (n + 1) ** 2 * (n**2 + 1) ** 2)
        - r4
    )

    return A


def Image_Sources_Positions(s, mua, musp, n1, n2, DD, m, eq):
    Z = OrderedDict()

    z0 = 1 / musp  # noqa: F841
    # z0 = 1 / (mua + musp)

    A = A_param(n1, n2)  # noqa: F841

    for index in range(-m, m + 1):
        z1, z2, z3, z4 = None, None, None, None

        Z[f"Z_{index}"] = z1, z2, z3, z4
    return Z
