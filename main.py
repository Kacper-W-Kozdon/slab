# placeholder
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from modules import Contini

if __name__ == "__main__":
    contini = Contini(s=40, mua=0.05, musp=0.3, n1=1, n2=1)

    rho = 4

    ydata = []
    xdata = []
    ydata_noisy = []
    IRF = [2, 1]

    for t_index, t in enumerate(range(1, 211, 2)):
        picot = t
        subresult = contini((picot, rho))

        ydata.append(subresult[0])
        xdata.append(tuple([picot, rho]))

    rng = np.random.default_rng()
    noise = rng.normal(size=len(xdata))
    # print(noise)

    for index in range(len(ydata)):
        ydata_noisy.append(ydata[index] + 0.05 * ydata[index] * noise[index])

    # print(ydata)
    print(ydata_noisy)
    contini.mua = 0.05
    contini.musp = None
    ydata_conv = scipy.signal.convolve(ydata_noisy, IRF, mode="same")
    # popt, pcov = curve_fit(contini._fit, xdata, ydata_noisy, [0.9])
    popt, pcov = contini.fit(xdata, ydata_conv, [0.35], normalize=True, IRF=IRF)
    contini.mua = 0.05

    print(popt)
    print(pcov)

    xdata_t = []
    for coord in xdata:
        xdata_t.append(coord[0])

    # print(xdata_t)
    plot1 = plt.plot(xdata_t, ydata_noisy, color="r", label="noisy")
    plot0 = plt.plot(xdata_t, ydata, color="b", label="control")

    contini2 = Contini(s=40, musp=popt[0], n1=1, n2=1)

    ydata = []
    xdata = []
    ydata_noisy = []

    for t_index, t in enumerate(range(1, 211, 2)):
        picot = t
        subresult = contini2((picot, rho))

        ydata.append(subresult[0])
        xdata.append(tuple([picot, rho]))

    plot2 = plt.plot(xdata_t, ydata, color="g", label="fit")
    plt.show()

    path = f"{pathlib.Path(__file__).parent.resolve()}\\test_data\\all_raw_data_combined.xlsx"
    if pathlib.Path(path).exists():
        df = pd.read_excel(path, engine="openpyxl")
        print(df.head())

        df = pd.read_excel(path, engine="openpyxl")
        df = df.fillna(0.0)
        column_names = df.columns.values.tolist()
        xdata_column_name = column_names[0]
        # df_clean = df.loc[df[xdata_column_name] != 0.0]
        df_clean = df
        # df_time = df.iloc[:, 0].fillna(0)
        # df_ydata = df.iloc[:, 1].fillna(0)
        df_time = df_clean.iloc[1:, 0]
        df_ydata = df_clean.iloc[1:, 3]
        df_irf = df_clean.iloc[1:, 1]
        # df_time = df_time.loc[(df[xdata_column_name] != 0.0)]
        # df_ydata = df_ydata.loc[(df[xdata_column_name] != 0.0)]
        # ~df['column_name'].isin(some_values)
        print(xdata_column_name)
        print(df_time)

        raw_data = plt.plot(
            df_time, df_ydata, color="b", label="raw data", marker="o", linestyle=" "
        )

        xdata = [tuple([time, rho]) for time in df_time]
        # print(xdata)
        # print(np.max(df_ydata))
        popt, pcov = contini.fit(xdata, df_ydata, [0.35], IRF=df_irf)
        print(popt, pcov)

        ydata = []
        for t in df_time:
            subresult = contini((t, rho))

            ydata.append(subresult[0])

        # fit = plt.plot(df_time, ydata, color="r", label="raw data")

        plt.show()
