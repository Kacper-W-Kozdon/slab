# placeholder

import copy
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from modules import Contini

if __name__ == "__main__":
    # contini = Contini(s=5, mua=0.05, musp=0.05, n1=1, n2=1)

    rho = 5

    ydata = []
    xdata = []
    ydata_conv_noisy = []
    # IRF = [2, 1]

    # for t_index, t in enumerate(range(1, 211, 2)):
    #     picot = t
    #     subresult = contini((picot, rho))
    #     # print(subresult[0])
    #     ydata.append(subresult[0])

    #     xdata.append(tuple([picot, rho]))

    # xdata_t = []
    # for coord in xdata:
    #     xdata_t.append(coord[0])

    # print("Computing ydata_conv.")
    # ydata_test = contini.forward(xdata, normalize=False)
    # plot_test0 = plt.plot(xdata_t, ydata, color="b", label="control")

    # plot_test1 = plt.plot(
    #     xdata_t, ydata_test, color="g", label="control", marker="o", linestyle=" "
    # )
    # plt.show()
    # plt.clf()
    # ydata_conv = contini.forward(xdata, IRF=IRF, normalize=True)
    # print(ydata_conv)
    # plot_test2 = plt.plot(xdata_t, ydata_conv, color="b", label="control")
    # plt.show()
    # plt.clf()

    # rng = np.random.default_rng()
    # noise = rng.normal(size=len(xdata))
    # # print(noise)

    # for index in range(len(ydata_conv)):
    #     ydata_conv_noisy.append(
    #         ydata_conv[index] + 0.05 * ydata_conv[index] * noise[index]
    #     )

    # # print(ydata)
    # print(ydata_conv_noisy)
    # contini.mua = 0.05
    # contini.musp = None
    # # ydata_conv = scipy.signal.convolve(ydata_noisy, IRF, mode="same")
    # # popt, pcov = curve_fit(contini._fit, xdata, ydata_noisy, [0.9])
    # popt, pcov = contini.fit(
    #     xdata, ydata_conv_noisy, [0.35, 0], normalize=True, IRF=IRF
    # )
    # contini.mua = 0.05

    # print(popt[0])
    # print(pcov)

    # # print(xdata_t)
    # ydata_conv_norm = ydata_conv / np.max(ydata_conv)
    # ydata_conv_noisy_norm = ydata_conv_noisy / np.max(ydata_conv_noisy)

    # plot1 = plt.plot(xdata_t, ydata_conv_noisy_norm, color="r", label="noisy")
    # plot0 = plt.plot(xdata_t, ydata_conv_norm, color="b", label="control")

    # contini.musp = popt[0]
    # # print(xdata, IRF, contini.musp, contini.mua)
    # ydata_fit = contini.forward(xdata, normalize=True, IRF=IRF)

    # ydata = []
    # xdata = []
    # ydata_noisy = []

    # for t_index, t in enumerate(range(1, 211, 2)):
    #     picot = t
    #     subresult = contini((picot, rho))

    #     ydata.append(subresult[0])
    #     xdata.append(tuple([picot, rho]))

    # plot2 = plt.plot(
    #     xdata_t,
    #     ydata_fit,
    #     color="g",
    #     label=f"fit: mua={contini.mua}, musp={contini.musp}, off={contini.offset}",
    # )
    # plt.xlabel("Time in ps")
    # plt.ylabel("R(t, rho=40[mm])/max(R(t, rho=40[mm]))")
    # plt.legend(loc="upper right")
    # plt.show()
    # plt.clf()

    path = f"{pathlib.Path(__file__).parent.resolve()}\\test_data\\all_raw_data_combined.xlsx"
    if pathlib.Path(path).exists():
        initial_params = {
            "mua": 0.065,
            "musp": 0.05,
            "offset": 40,
            "lower_bounds": [0.01, 0.01, 0.0],
            "upper_bounds": [0.1, 0.1, 50],
        }

        rho = 5

        ydata = []
        xdata = []
        ydata_conv_noisy = []

        contini2 = Contini(
            s=3, mua=initial_params["mua"], musp=initial_params["musp"], n1=1, n2=1
        )
        contini2.offset = initial_params["offset"]

        df = pd.read_excel(path, engine="openpyxl")
        print(df.head())

        df = pd.read_excel(path, engine="openpyxl")
        df = df.fillna(0.0)
        column_names = df.columns.values.tolist()
        xdata_column_name = column_names[0]
        df_clean = df.loc[df[xdata_column_name] != 0.0]
        # df_clean = df

        # print(df_clean)
        # df_time = df.iloc[:, 0].fillna(0)
        # df_ydata = df.iloc[:, 1].fillna(0)
        df_time_raw = df_clean.iloc[:, [0]]
        df_ydata_raw = df_clean.iloc[:, [3]]
        df_irf_raw = df_clean.iloc[:, [1]]
        # df_time = df_time.loc[(df[xdata_column_name] != 0.0)]
        # df_ydata = df_ydata.loc[(df[xdata_column_name] != 0.0)]
        # ~df['column_name'].isin(some_values)
        print(xdata_column_name)
        # print(df_time)

        df_time_ = copy.copy(df_time_raw)
        df_ydata_ = copy.copy(df_ydata_raw)
        df_irf_ = copy.copy(df_irf_raw)

        print(df_time_.columns, df_ydata_.columns, df_irf_.columns)
        print(df_irf_.columns[0])
        print(df_irf_raw["Intensity_tw"])

        irf_max = np.max(df_irf_raw)
        irf_max_head = np.max(
            df_irf_raw.head(10)
        )  # if np.max(df_irf_raw.head(10)) < 0.1 * irf_max else 0.1 * irf_max

        print(irf_max_head, irf_max)
        y_max = np.max(df_ydata_raw)
        y_max_head = np.max(
            df_ydata_raw.head(10)
        )  # if np.max(df_ydata_raw.head(10)) < 0.1 * y_max else 0.1 * y_max
        df_irf = df_irf_.loc[
            (df_irf_[df_irf_.columns[0]] >= irf_max_head + 0.01 * irf_max)
            | (df_ydata_[df_ydata_.columns[0]] >= y_max_head + 0.01 * y_max)
        ]
        df_time = df_time_.loc[
            (df_irf_[df_irf_.columns[0]] >= irf_max_head + 0.01 * irf_max)
            | (df_ydata_[df_ydata_.columns[0]] >= y_max_head + 0.01 * y_max)
        ]
        df_ydata = df_ydata_.loc[
            (df_irf[df_irf_.columns[0]] >= irf_max_head + 0.01 * irf_max)
            | (df_ydata_[df_ydata_.columns[0]] >= y_max_head + 0.01 * y_max)
        ]
        print(len(df_time), len(df_ydata), len(df_irf))

        # # df_irf = df_irf_raw.loc[df_irf_raw != 0]
        # df_irf = copy.copy(df_irf_raw)
        # # df_ydata_raw = scipy.signal.convolve(df_ydata_raw, df_irf, mode="same")

        raw_data = plt.plot(
            df_time,
            df_ydata,
            color="b",
            label="raw data",
            marker="o",
            linestyle=" ",
        )

        raw_irf = plt.plot(
            df_time,
            df_irf,
            color="g",
            label="irf",
        )
        plt.legend(loc="upper right")
        plt.xlabel("Time in ps")
        plt.ylabel("I(t, rho=40[mm])")
        # plt.show()
        plt.clf()

        xdata = pd.DataFrame([tuple([float(time), rho]) for time in df_time_raw.values])
        contini2.offset = initial_params["offset"]
        contini2._max_ydata = np.max(df_ydata_raw)
        # ydata_fit = contini2.forward(xdata, normalize=True, IRF=df_irf_raw)
        # fit = plt.plot(
        #     df_time,
        #     ydata_fit,
        #     color="r",
        #     label=f"fit: mua={contini2.mua * 1e-3}, musp={contini2.musp * 1e-3}, off={contini2.offset}",
        # )
        # plt.legend(loc="upper right")
        # plt.xlabel("Time in ps")
        # plt.ylabel("R(t, rho=40[mm])/max(R(t, rho=40[mm]))")
        # # plt.show()
        # plt.clf()

        # df_ydata_raw = df_ydata
        # raw_data = plt.plot(
        #     df_time,
        #     df_ydata_raw,
        #     color="b",
        #     label="raw data",
        #     marker="o",
        #     linestyle=" ",
        # )

        # xdata = [tuple([time, rho]) for time in df_time]
        # contini2.offset = initial_params["offset"]
        # contini2._max_ydata = np.max(df_ydata_raw)
        # ydata_fit = contini2.forward(xdata, normalize=True, IRF=df_irf)
        # fit = plt.plot(
        #     df_time,
        #     ydata_fit,
        #     color="r",
        #     label=f"fit: mua={contini2.mua * 1e-3}, musp={contini2.musp * 1e-3}, off={contini2.offset}",
        # )
        # plt.legend(loc="upper right")
        # plt.xlabel("Time in ps")
        # plt.ylabel("R(t, rho=40[mm])/max(R(t, rho=40[mm]))")
        # plt.show()
        plt.clf()

        print("---TEST DATA FIT---")
        print(df_ydata)
        print(xdata)
        # print(xdata)
        # print(np.max(df_ydata))
        popt, pcov = contini2.fit(
            xdata,
            df_ydata_raw,
            [initial_params["mua"], initial_params["musp"], initial_params["offset"]],
            IRF=df_irf_raw,
            free_params=["mua", "musp", "offset"],
            bounds=[initial_params["lower_bounds"], initial_params["upper_bounds"]],
            normalize=True,
            log_scale=False,
        )
        print(popt, pcov)
        contini2.mua = popt[0]
        contini2.musp = popt[1]
        contini2.offset = popt[2]
        # ydata = []
        # for t in df_time:
        #     subresult = contini((t, rho))

        #     ydata.append(subresult[0])

        # contini.IRF = None
        # ydata_fit = None
        if not contini2.normalize:
            contini2.normalize = True
        contini2.IRF = None
        contini2._max_ydata = np.max(df_ydata_raw)
        contini2.log_scale = None

        ydata_fit = contini2.forward(xdata, normalize=True, IRF=df_irf_raw)
        raw_data = plt.plot(
            df_time_raw,
            df_ydata_raw,
            color="b",
            label="raw data",
            marker="o",
            linestyle=" ",
        )

        fit = plt.plot(
            df_time_raw,
            ydata_fit,
            color="r",
            label=f"fit: mua={contini2.mua * 1e-3}, musp={contini2.musp * 1e-3}, off={contini2.offset}",
        )
        plt.legend(loc="upper right")
        plt.xlabel("Time in ps")
        plt.ylabel("T(t, rho=40[mm])/max(R(t, rho=40[mm]))")

        path = pathlib.Path(__file__).resolve().parent
        plt.savefig(
            f"{pathlib.Path(__file__).resolve().parent}\\plots\\fit_convolved.pdf"
        )
        plt.show()
        plt.clf()

        df_ydata_raw = scipy.signal.convolve(df_ydata_raw, df_irf, mode="same")
        contini2._max_ydata = np.max(df_ydata_raw)
        contini2.log_scale = None
        ydata_fit = contini2.forward(xdata, normalize=True, IRF=df_irf)

        # raw_data = plt.plot(
        #     df_time_raw,
        #     df_ydata_raw,
        #     color="b",
        #     label="raw data",
        #     marker="o",
        #     linestyle=" ",
        # )

        # fit = plt.plot(
        #     df_time,
        #     ydata_fit,
        #     color="r",
        #     label=f"fit: mua={contini2.mua * 1e-3}, musp={contini2.musp * 1e-3}, off={contini2.offset}",
        # )
        # plt.legend(loc="upper right")
        # plt.xlabel("Time in ps")
        # plt.ylabel("R(t, rho=40[mm])/max(R(t, rho=40[mm]))")
        # plt.show()
        # path = pathlib.Path(__file__).resolve().parent
        # plt.savefig(f"{pathlib.Path(__file__).resolve().parent}\\plots\\convolved.pdf")
        plt.clf()
