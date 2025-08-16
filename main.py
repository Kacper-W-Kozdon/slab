# placeholder
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from modules import Contini

if __name__ == "__main__":
    contini = Contini(s=40, mua=0.05, musp=0.3, n1=1, n2=1)

    rho = 4

    ydata = []
    xdata = []
    ydata_conv_noisy = []
    IRF = [2, 1]

    for t_index, t in enumerate(range(1, 211, 2)):
        picot = t
        subresult = contini((picot, rho))

        xdata.append(tuple([picot, rho]))
    print("Computing ydata_conv.")
    ydata_conv = contini.forward(xdata, IRF=IRF, normalize=True)
    print(ydata_conv)

    rng = np.random.default_rng()
    noise = rng.normal(size=len(xdata))
    # print(noise)

    for index in range(len(ydata_conv)):
        ydata_conv_noisy.append(
            ydata_conv[index] + 0.05 * ydata_conv[index] * noise[index]
        )

    # print(ydata)
    print(ydata_conv_noisy)
    contini.mua = 0.05
    contini.musp = None
    # ydata_conv = scipy.signal.convolve(ydata_noisy, IRF, mode="same")
    # popt, pcov = curve_fit(contini._fit, xdata, ydata_noisy, [0.9])
    popt, pcov = contini.fit(xdata, ydata_conv_noisy, [0.35], normalize=True, IRF=IRF)
    contini.mua = 0.05

    print(popt[0])
    print(pcov)

    xdata_t = []
    for coord in xdata:
        xdata_t.append(coord[0])

    # print(xdata_t)
    ydata_conv_norm = ydata_conv / np.max(ydata_conv)
    ydata_conv_noisy_norm = ydata_conv_noisy / np.max(ydata_conv_noisy)

    plot1 = plt.plot(xdata_t, ydata_conv_noisy_norm, color="r", label="noisy")
    plot0 = plt.plot(xdata_t, ydata_conv_norm, color="b", label="control")

    contini2 = Contini(s=40, mua=0.05, musp=popt, n1=1, n2=1)
    contini.musp = popt[0] * 1e3
    # print(xdata, IRF, contini.musp, contini.mua)
    ydata_fit = contini.forward(xdata, normalize=True, IRF=IRF)

    ydata = []
    xdata = []
    ydata_noisy = []

    for t_index, t in enumerate(range(1, 211, 2)):
        picot = t
        subresult = contini2((picot, rho))

        ydata.append(subresult[0])
        xdata.append(tuple([picot, rho]))

    plot2 = plt.plot(xdata_t, ydata_fit, color="g", label="fit")
    plt.xlabel("Time in ps")
    plt.ylabel("R(t, rho=40[mm])/max(R(t, rho=40[mm]))")
    plt.legend(loc="upper right")
    plt.show()
    plt.clf()

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
        df_ydata_raw = df_ydata / np.max(df_ydata)
        raw_data = plt.plot(
            df_time,
            df_ydata_raw,
            color="b",
            label="raw data",
            marker="o",
            linestyle=" ",
        )

        xdata = [tuple([time, rho]) for time in df_time]
        # print(xdata)
        # print(np.max(df_ydata))
        popt, pcov = contini2.fit(xdata, df_ydata, [0.35], IRF=df_irf, normalize=True)
        print(popt, pcov)
        contini.musp = popt[0]
        # ydata = []
        # for t in df_time:
        #     subresult = contini((t, rho))

        #     ydata.append(subresult[0])

        # contini.IRF = None
        ydata_fit = None
        if not contini.normalize:
            contini.normalize = True
        ydata_fit = contini.forward(xdata)

        fit = plt.plot(df_time, ydata_fit, color="r", label="raw data")
        plt.legend(loc="upper right")
        plt.xlabel("Time in ps")
        plt.ylabel("R(t, rho=40[mm])/max(R(t, rho=40[mm]))")
        plt.show()
