# placeholder
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
from modules import Contini

if __name__ == "__main__":
    contini = Contini(s=40, mua=0.05, musp=0.3, n1=1, n2=1)

    rho = 4

    ydata = []
    xdata = []
    ydata_noisy = []

    # for t_index, t in enumerate(range(1, 311, 2)):
    #     picot = t
    #     subresult = contini((picot, rho))

    #     ydata.append(subresult[0])
    #     xdata.append(tuple([picot, rho]))

    # rng = np.random.default_rng()
    # noise = rng.normal(size=len(xdata))
    # # print(noise)

    # for index in range(len(ydata)):
    #     ydata_noisy.append(ydata[index] + 0.05 * ydata[index] * noise[index])

    # # print(ydata)
    # print(ydata_noisy)
    # contini.mua = 0.05
    # contini.musp = None

    # # popt, pcov = curve_fit(contini._fit, xdata, ydata_noisy, [0.9])
    # popt, pcov = contini.fit(xdata, ydata_noisy, [0.25])

    # print(popt)
    # print(pcov)

    # xdata_t = []
    # for coord in xdata:
    #     xdata_t.append(coord[0])

    # # print(xdata_t)
    # plot1 = plt.plot(xdata_t, ydata_noisy, color="r", label="noisy")
    # plot0 = plt.plot(xdata_t, ydata, color="b", label="control")

    # contini2 = Contini(s=40, musp=popt[0], n1=1, n2=1)

    # ydata = []
    # xdata = []
    # ydata_noisy = []

    # for t_index, t in enumerate(range(1, 311, 2)):
    #     picot = t
    #     subresult = contini2((picot, rho))

    #     ydata.append(subresult[0])
    #     xdata.append(tuple([picot, rho]))

    # plot2 = plt.plot(xdata_t, ydata, color="g", label="fit")
    # plt.yscale("log")

    path = f"{pathlib.Path(__file__).parent.resolve()}\\test_data\\all_raw_data_combined.xlsx"
    if pathlib.Path(path).exists():
        df = pd.read_excel(path, engine="openpyxl")
        print(df.head())

        df = pd.read_excel(path, engine="openpyxl")
        df_time = df.iloc[:, 0]
        df_ydata = df.iloc[:, 1]

        raw_data = plt.plot(
            df_time, df_ydata, color="b", label="raw data", marker="o", linestyle=" "
        )

        xdata = [tuple([time, rho]) for time in df_time]
        popt, pcov = contini.fit(xdata, df_ydata, [0.25])
        print(popt, pcov)

        plt.show()
