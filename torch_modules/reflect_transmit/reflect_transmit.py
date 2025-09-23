import sys

import torch
from numpy import exp, pi, sqrt

from ..other.utils import A_parameter, D_parameter, G_func, Image_Sources_Positions

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


def Reflectance_Transmittance_rho_t(
    rho, t, mua, musp, s, m, n1, n2, DD, eq, anisothropy_coeff, **kwargs
):
    c = 299792458
    v = c / n2
    D = D_parameter(DD, mua, musp, eq)
    A = A_parameter(n1, n2)
    ze = 2 * A * D  # noqa: F841
    # print(D, A, rho, t, mua, musp)
    R_rho_t = 0.0
    T_rho_t = 0.0

    R_rho_t_source_sum = 0.0
    T_rho_t_source_sum = 0.0
    Z = Image_Sources_Positions(s, mua, musp, n1, n2, DD, m, eq)

    if not isinstance(t, torch.Tensor):
        try:
            t = torch.tensor(t)
            if t[0] == 0:
                t = t[1:]
        except Exception as exc:
            print(f"Failed to convert t = {t}, {type(t)} to torch.Tensor")
            raise exc

    if not isinstance(rho, torch.Tensor):
        try:
            rho = torch.full_like(t, rho)
        except Exception as exc:
            print(f"Failed to convert rho = {rho}, {type(rho)} to torch.Tensor")
            raise exc

    # if not isinstance(s, torch.Tensor):
    #     try:
    #         s = torch.full_like(t, s)
    #     except Exception as exc:
    #         print(f"Failed to convert rho = {s}, {type(s)} to torch.Tensor")
    #         raise exc

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
            -exp(-mua * v * t - (rho**2) / (4 * D * v * t))
            / (2 * ((4 * pi * D * v) ** (3 / 2)) * t ** (5 / 2))
            * R_rho_t_source_sum
        )
        T_rho_t = (
            exp(-mua * v * t - (rho**2) / (4 * D * v * t))
            / (2 * ((4 * pi * D * v) ** (3 / 2)) * t ** (5 / 2))
            * T_rho_t_source_sum
        )

        R_rho_t /= 2 * A
        T_rho_t /= 2 * A

    if eq == "RTE":
        mus = musp / (1 - anisothropy_coeff)  # noqa: F841
        mean_free_path = 1 / (mua + musp)
        # print(mean_free_path, mua, musp)
        # if type(rho) is torch.Tensor:
        # rho = rho.detach().numpy()

        for index in range(-m, m + 1):
            z_plus, z_minus = Z[f"Z_{index}"]

            r_plus = sqrt(rho**2 + (z_plus) ** 2)
            r_minus = sqrt(rho**2 + (z_minus) ** 2)

            Delta_plus = 1.0 * torch.tensor(r_plus == c * t)
            Delta_minus = 1.0 * torch.tensor(r_minus == c * t)
            Theta_plus = 1.0 * torch.tensor(r_plus < c * t)
            Theta_minus = 1.0 * torch.tensor(
                r_minus < c * t
            )  # TODO: Fix the source of nan in the argument of G_func(). \endtodo

            G_plus = Theta_plus * G_func(
                c * t / mean_free_path * (1 - r_plus**2 / (c**2 * t**2)) ** (3 / 4),
                **kwargs,
            )
            G_minus = Theta_minus * G_func(
                c * t / mean_free_path * (1 - r_minus**2 / (c**2 * t**2)) ** (3 / 4),
                **kwargs,
            )
            factor_plus_unfiltered = Theta_plus * (1 - r_plus**2 / (c**2 * t**2)) ** (
                1 / 8
            )
            factor_minus_unfiltered = Theta_minus * (
                1 - r_minus**2 / (c**2 * t**2)
            ) ** (1 / 8)
            factor_plus = torch.where(
                ~torch.isnan(factor_plus_unfiltered), factor_plus_unfiltered, 0.0
            )
            factor_minus = torch.where(
                ~torch.isnan(factor_minus_unfiltered), factor_minus_unfiltered, 0.0
            )

            R_rho_t_source_sum += exp(-c * t / mean_free_path) * (
                1 / (4 * pi * r_plus**2) * Delta_plus
                - 1 / (4 * pi * r_minus**2) * Delta_minus
                + (
                    G_plus * Theta_plus * factor_plus
                    - G_minus * Theta_minus * factor_minus
                )
                / ((1 / 3 * 4 * pi * mean_free_path * c * t) ** (3 / 2))
            )
            if any(torch.isnan(R_rho_t_source_sum)):
                print(f"{G_plus=}")
                print(f"{Delta_plus=}")
                print(f"{Theta_plus=}")
                print(f"{factor_plus=}")
                raise ValueError(
                    f"Found nan values in partial sum. {R_rho_t_source_sum=}"
                )

        for index in range(-m, m + 1):
            z_plus, z_minus = Z[f"Z_{index}"]

            r_plus = sqrt(rho**2 + (s - z_plus) ** 2)
            r_minus = sqrt(rho**2 + (s - z_minus) ** 2)

            Delta_plus = 1.0 * torch.tensor(r_plus == c * t)
            Delta_minus = 1.0 * torch.tensor(r_minus == c * t)
            Theta_plus = 1.0 * torch.tensor(r_plus < c * t)
            Theta_minus = 1 * torch.tensor(r_minus < c * t)
            # print(f"{r_plus=}")
            # print(f"{Delta_plus=}")
            G_plus = Theta_plus * G_func(
                c * t / mean_free_path * (1 - r_plus**2 / (c**2 * t**2)) ** (3 / 4),
                **kwargs,
            )
            G_minus = Theta_minus * G_func(
                c * t / mean_free_path * (1 - r_minus**2 / (c**2 * t**2)) ** (3 / 4),
                **kwargs,
            )

            factor_plus_unfiltered = Theta_plus * (1 - r_plus**2 / (c**2 * t**2)) ** (
                1 / 8
            )
            # print(f"{c * t / mean_free_path * (1 - r_plus**2 / (c**2 * t**2)) ** (3 / 4)=}")
            factor_minus_unfiltered = Theta_minus * (
                1 - r_minus**2 / (c**2 * t**2)
            ) ** (1 / 8)
            factor_plus = torch.where(
                ~torch.isnan(factor_plus_unfiltered), factor_plus_unfiltered, 0.0
            )
            factor_minus = torch.where(
                ~torch.isnan(factor_minus_unfiltered), factor_minus_unfiltered, 0.0
            )

            T_rho_t_source_sum += exp(-c * t / mean_free_path) * (
                1 / (4 * pi * r_plus**2) * Delta_plus
                - 1 / (4 * pi * r_minus**2) * Delta_minus
                + (
                    G_plus * Theta_plus * factor_plus
                    - G_minus * Theta_minus * factor_minus
                )
                / ((1 / 3 * 4 * pi * mean_free_path * c * t) ** (3 / 2))
            )
            if any(torch.isnan(T_rho_t_source_sum)):
                print(f"{G_plus=}")
                print(f"{Delta_plus=}")
                print(f"{Theta_plus=}")
                print(f"{factor_plus=}")
                raise ValueError(
                    f"Found nan values in partial sum. {T_rho_t_source_sum=}"
                )

        R_rho_t = 1 / (2 * A) * R_rho_t_source_sum
        T_rho_t = 1 / (2 * A) * T_rho_t_source_sum
        # R_rho_t = R_rho_t if R_rho_t > 0 else -R_rho_t
        # T_rho_t = T_rho_t if T_rho_t > 0 else -T_rho_t

        # if T_rho_t < 0:
        #     print(t, r_plus, r_minus, T_rho_t)
        # print(R_rho_t)

    # print(R_rho_t)
    R_rho_t *= 1e-6 * 1e-12
    T_rho_t *= 1e-6 * 1e-12
    R_rho_t = R_rho_t * (R_rho_t > 0)
    T_rho_t = T_rho_t * (T_rho_t > 0)

    R_rho_t = R_rho_t * (t > 0)
    T_rho_t = T_rho_t * (t > 0)

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

    if eq == "DE":
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

    if eq == "RTE":
        pass

    return R_t, T_t


def Reflectance_Transmittance(mua, musp, s, m, n1, n2, DD, eq):
    R, T = None, None
    return R, T
