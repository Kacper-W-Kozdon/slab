from typing import Union, Optional
from collections import OrderedDict

def Contini(rho: Union[int, float], t: Union[int, float], s: Union[int, float], mua: Union[int, float], musp: Union[int, float], n1: Union[int, float], n2: Union[int, float], phantom: Optional[str], DD: Optional[str], m: int = 200):
    
    t = t * 1e-9
    rho = rho * 1e-3
    s = s * 1e-3
    mua = mua * 1e3
    musp = musp * 1e3

    err = 1e-6
    
    R_rho_t, T_rho_t = Reflectance_Transmittance_rho_t(rho, t, s, m, mua, musp, n1, n2, DD)

    R_rho, T_rho = Reflectance_Transmittance_rho(rho, s, m, mua, musp, n1, n2, DD)

    R_t, T_t = Reflectance_Transmittance_t(t, s, m, mua, musp, n1, n2, DD)

    l_rho_R, l_rho_T = Mean_Path_T_R(rho, s, m, mua, musp, n1, n2, DD)

    R, T = Reflectance_Transmittance(s, m, mua, musp, n1, n2, DD)

    A = A_param(n1, n2)

    Z = Image_Sources_Positions(s, mua, musp, n1, n2, DD, m)

    return R_rho_t, T_rho_t, R_rho, T_rho, R_t, T_t, l_rho_R, l_rho_T, R, T, A, Z

    

def Reflectance_Transmittance_rho_t(rho, t, s, m, mua, musp, n1, n2, DD):
    R_rho_t, T_rho_t = None
    return R_rho_t, T_rho_t


def Reflectance_Transmittance_rho(rho, s, m, mua, musp, n1, n2, DD):
    R_rho, T_rho = None
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
    for index in range(-m, m+1):
        z1, z2, z3, z4 = None
        Z[f"Z_{i}"] = z1, z2, z3, z4
    return Z