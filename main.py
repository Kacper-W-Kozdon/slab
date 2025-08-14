# placeholder
import matplotlib.pyplot as plt
import numpy as np

from modules import Contini

if __name__ == "__main__":
    contini = Contini(s=40, mua=0.05, musp=0.05, n1=1, n2=1)

    rho = 1

    ydata = []
    xdata = []
    ydata_noisy = []

    for t_index, t in enumerate(range(1, 203, 2)):
        picot = t * 1e-3
        subresult = contini((picot, rho))

        ydata.append(subresult[0])
        xdata.append(tuple([picot, rho]))

    rng = np.random.default_rng()
    noise = rng.normal(size=len(xdata))
    # print(noise)

    for index in range(len(ydata)):
        ydata_noisy.append(ydata[index] + 0.05 * ydata[index] * noise[index])

    # print(ydata)

    contini.mua = 0.05
    contini.musp = None

    # popt, pcov = curve_fit(contini._fit, xdata, ydata_noisy, [0.9])
    popt, pcov = contini.fit(xdata, ydata_noisy, [0.04])

    print(popt)
    print(pcov)

    xdata_t = []
    for coord in xdata:
        xdata_t.append(coord[0])

    # print(xdata_t)
    plot1 = plt.plot(xdata_t, ydata_noisy, color="r", label="noisy")

    contini2 = Contini(s=40, musp=popt[0], n1=1, n2=1)

    ydata = []
    xdata = []
    ydata_noisy = []

    for t_index, t in enumerate(range(1, 203, 2)):
        picot = t * 1e-3
        subresult = contini2((picot, rho))

        ydata.append(subresult[0])
        xdata.append(tuple([picot, rho]))

    plot2 = plt.plot(xdata_t, ydata, color="g", label="fit")
    # plt.yscale("log")
    plt.show()
