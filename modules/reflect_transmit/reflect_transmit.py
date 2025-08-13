from numpy import exp, pi, sqrt

from ..other.utils import A_parameter, D_parameter, G_func, Image_Sources_Positions


def Reflectance_Transmittance_rho_t(rho, t, mua, musp, s, m, n1, n2, DD, eq):
    c = 299792458
    v = c / n2
    D = D_parameter(DD, mua, musp, eq)
    A = A_parameter(n1, n2)
    ze = 2 * A * D  # noqa: F841

    R_rho_t = 0.0
    T_rho_t = 0.0

    R_rho_t_source_sum = 0.0
    T_rho_t_source_sum = 0.0

    Z = Image_Sources_Positions(s, mua, musp, n1, n2, DD, m, eq)
    if eq == "DE":
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

    if eq == "RTE":
        mean_free_path = 1 / (mua + musp)

        for index in range(-m, m + 1):
            z_plus, z_minus = Z[f"Z_{index}"]

            r_plus = sqrt(rho**2 + (z_plus) ** 2)
            r_minus = sqrt(rho**2 + (z_minus) ** 2)

            Delta_plus = 1 if r_plus == c * t else 0
            Delta_minus = 1 if r_minus == c * t else 0
            Theta_plus = 1 if r_plus >= c * t else 0
            Theta_minus = 1 if r_minus >= c * t else 0

            G_plus = G_func(
                c * t / mean_free_path * (1 - r_plus**2 / (c**2 * t) ** 2) ** (3 / 4)
            )
            G_minus = G_func(
                c * t / mean_free_path * (1 - r_minus**2 / (c**2 * t) ** 2) ** (3 / 4)
            )

            R_rho_t_source_sum += (
                exp(-c * t / mean_free_path) / (4 * pi * r_plus**2) * Delta_plus
                + exp(-c * t / mean_free_path) / (4 * pi * r_minus**2) * Delta_minus
                + G_plus * Theta_plus
                + G_minus * Theta_minus
            )

        for index in range(-m, m + 1):
            z_plus, z_minus = Z[f"Z_{index}"]

            r_plus = sqrt(rho**2 + (s - z_plus) ** 2)
            r_minus = sqrt(rho**2 + (s - z_minus) ** 2)

            Delta_plus = 1 if r_plus == c * t else 0
            Delta_minus = 1 if r_minus == c * t else 0
            Theta_plus = 1 if r_plus >= c * t else 0
            Theta_minus = 1 if r_minus >= c * t else 0

            G_plus = G_func(
                c * t / mean_free_path * (1 - r_plus**2 / (c**2 * t) ** 2) ** (3 / 4)
            )
            G_minus = G_func(
                c * t / mean_free_path * (1 - r_minus**2 / (c**2 * t) ** 2) ** (3 / 4)
            )

            T_rho_t_source_sum += (
                exp(-c * t / mean_free_path) / (4 * pi * r_plus**2) * Delta_plus
                + exp(-c * t / mean_free_path) / (4 * pi * r_minus**2) * Delta_minus
                + G_plus * Theta_plus
                + G_minus * Theta_minus
            )

        R_rho_t = 1 / (2 * A) * R_rho_t_source_sum
        T_rho_t = 1 / (2 * A) * T_rho_t_source_sum

    R_rho_t *= 1e-6 * 1e-12
    T_rho_t *= 1e-6 * 1e-12

    R_rho_t = R_rho_t if t > 0 else 0
    T_rho_t = T_rho_t if t > 0 else 0

    return R_rho_t, T_rho_t


def Reflectance_Transmittance_rho(rho, mua, musp, s, m, n1, n2, DD, eq):
    R_rho, T_rho = None, None

    D = D_parameter(DD, mua, musp, eq)

    R_rho_source_sum = 0
    T_rho_source_sum = 0

    Z = Image_Sources_Positions(s, mua, musp, n1, n2, DD, m, eq)

    if eq == "DE":
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

    if eq == "RTE":
        pass

    R_rho = -1 / (4 * pi) * R_rho_source_sum
    T_rho = 1 / (4 * pi) * T_rho_source_sum

    R_rho *= 1e-6
    T_rho *= 1e-6

    return R_rho, T_rho


def Reflectance_Transmittance_t(t, mua, musp, s, m, n1, n2, DD, eq):
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


def Reflectance_Transmittance(mua, musp, s, m, n1, n2, DD, eq):
    R, T = None, None
    return R, T
