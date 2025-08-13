from collections import OrderedDict

from numpy import exp, log, sqrt
from scipy.special import factorial, gamma


def D_parameter(DD, mua, musp, eq):
    D = None
    if DD == "Dmuas":
        D = 1 / (3 * (musp + mua))
    elif DD == "Dmus":
        D = 1 / (3 * (musp))

    return D


def Mean_Path_T_R(rho, mua, musp, s, m, n1, n2, DD, eq):
    l_rho_R, l_rho_T = None, None
    return l_rho_R, l_rho_T


def A_parameter_approx(n1, n2):
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


def A_parameter(n1, n2):
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

    A = A_parameter(n1, n2)  # noqa: F841
    D = D_parameter(DD, mua, musp, eq)

    if eq == "DE":
        for index in range(-m, m + 1):
            ze = 2 * A * D

            z1 = s * (1 - 2 * m) - 4 * m * ze - z0
            z2 = s * (1 - 2 * m) - (4 * m - 2) * ze + z0
            z3 = -2 * m * s - 4 * m * ze - z0
            z4 = -2 * m * s - (4 * m - 2) * ze + z0

            Z[f"Z_{index}"] = z1, z2, z3, z4
        return Z

    if eq == "RTE":
        for index in range(-m, m + 1):
            ze = 2 * A * D

            z_plus = 2 * m * (s + 2 * ze) + z0
            z_minus = 2 * m * (s + 2 * ze) - 2 * ze - z0

            Z[f"Z_{index}"] = z_plus, z_minus
        return Z


def G_func(x, N_scatter=200, mode: str = "approx"):
    G = 0
    if mode == "sum":
        factor = 8 * (3 * x) ** (-3 / 2)

        for N in range(1, N_scatter + 1):
            G += (
                factor
                * gamma(3 / 4 * N + 3 / 2)
                / gamma(3 / 4 * N)
                * x**N
                / factorial(N)
            )

        return G
    if mode == "approx":
        G += exp(x) * sqrt(1 + 2.026 / x)
        return G
