import pathlib

import matplotlib.pyplot as plt
import pandas as pd
from modules import Contini

# def func(x, k, s, u):
#     x=np.array(x)
#     print(k, s, u)
#     return k * (1 / (x*s*np.sqrt(2*np.pi)))*np.exp(-np.power((np.log(x)-u), 2)/(2*np.power(s, 2)))


# xdata = list(range(100, 40000, 100))
# rng = np.random.default_rng()
# noise = rng.normal(size=len(xdata))
# ydata = [func(x, 1000, 1, 10) * (1 + 0.05*n) for x, n in zip(xdata, noise)]

# p0 = [1100, 2, 10]
# popt, pcov = curve_fit(func, xdata, ydata, p0)

# pyplot.figure()
# pyplot.plot(xdata, ydata, label='Data', marker='o')
# pyplot.plot(xdata,  func(xdata, popt[0], popt[1], popt[2]), 'g--')
# pyplot.show()

# print (popt)
# path = f"{pathlib.Path(__file__).parent.resolve()}\\test_data\\all_raw_data_combined.xlsx"
# if pathlib.Path(path).exists():

#     df = pd.read_excel(path, engine='openpyxl')
#     df_time = df.iloc[:,0]
#     df_ydata = df.iloc[:,1]

print("\n\n")
print(pathlib.Path(__file__).resolve(), pathlib.Path(__file__).resolve().parent)
plt.ion()


def test_plot() -> None:
    """
    Test plots.
    """

    rho = 5
    s = 3

    xdata = []
    for t_index, t in enumerate(range(1, 311, 2)):
        xdata.append((t, rho))

    inputs = pd.DataFrame(xdata, columns=["t", "rho"])

    initial_params = {
        "mua": 0.05,
        "musp": 0.05,
        "offset": 40,
        "lower_bounds": [0, 0, 20],
        "upper_bounds": [1, 1, 80],
    }

    contini = Contini(
        s=s, mua=initial_params["mua"], musp=initial_params["musp"], n2=1, n1=1
    )

    contini.values_to_fit = ["R_rho_t"]

    outputs = contini.forward(xdata)

    plot = plt.plot(  # noqa: F841
        inputs,
        outputs,
        color="b",
        label="test data",
        # marker="o",
        # linestyle=" ",
    )

    plt.legend(loc="upper right")
    plt.xlabel("Time in ps")
    plt.ylabel("T(t, rho=5[mm])/max(R(t, rho=5[mm])), s=3[mm]")

    plt.show(block=True)
    plt.clf()
