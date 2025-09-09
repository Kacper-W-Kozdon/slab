import pathlib
import unittest

import matplotlib.pyplot as plt
import pandas as pd
from modules import Contini
from torch_modules import tContini

print("\n\n")
print(pathlib.Path(__file__).resolve(), pathlib.Path(__file__).resolve().parent)
plt.ion()


def test_plot() -> None:
    """
    Test plots.
    """

    assertions = unittest.TestCase("__init__")

    rho = 5
    s = 3

    xdata = []
    for t_index, t in enumerate(range(1, 311, 2)):
        xdata.append((t, rho))

    inputs = pd.DataFrame(xdata, columns=["t", "rho"])

    # initial_params = {
    #     "mua": 0.05,
    #     "musp": 0.05,
    #     "offset": 40,
    #     "scaling": 0.9,
    #     "lower_bounds": [0, 0, 20],
    #     "upper_bounds": [1, 1, 80],
    # }

    initial_params = {
        "mua": 0.0,
        "musp": 0.5,
        "offset": 40,
        "scaling": 0.9,
        "lower_bounds": [0, 0, 20],
        "upper_bounds": [1, 1, 80],
    }

    contini = Contini(
        s=s, mua=initial_params["mua"], musp=initial_params["musp"], n2=1, n1=1
    )

    torch_contini = tContini(
        s=s, mua=initial_params["mua"], musp=initial_params["musp"], n2=1, n1=1
    )

    contini.values_to_fit = ["R_rho_t"]
    torch_contini.controls["values_to_fit"] = ["R_rho_t"]

    outputs_R = contini.forward(xdata)
    assert outputs_R is not None

    plt.plot(  # noqa: F841
        inputs,
        outputs_R,
        color="b",
        label="test data R_RTE",
        # marker="o",
        # linestyle=" ",
    )

    contini.values_to_fit = ["T_rho_t"]

    outputs_T_RTE = contini.forward(xdata, eq="RTE")
    torch_outputs_T_RTE = torch_contini.forward(xdata, eq="RTE")
    contini.eq = "DE"
    outputs_T_DE = contini.forward(xdata, eq="DE")
    assert outputs_T_RTE is not None, "forward function returned None for eq='RTE'"
    assert outputs_T_DE is not None, "forward function returned None for eq='DE'"

    for torch_output, output in zip(list(torch_outputs_T_RTE), list(outputs_T_RTE)):
        assertions.assertAlmostEqual(torch_output, output)

    plt.plot(  # noqa: F841
        inputs,
        outputs_T_RTE,
        color="g",
        label="test data T_RTE",
        # marker="o",
        linestyle="--",
    )

    plt.plot(  # noqa: F841
        inputs,
        torch_outputs_T_RTE,
        color="o",
        label="test data T_RTE",
        marker="o",
        linestyle=" ",
    )

    plt.plot(  # noqa: F841
        inputs,
        outputs_T_DE,
        color="r",
        label="test data T_DE",
        # marker="o",
        linestyle="-.",
    )

    plt.legend(loc="upper right")
    plt.xlabel("Time in ps")
    plt.ylabel("Intensity(t, rho=5[mm])/max(R(t, rho=5[mm])), s=3[mm]")

    plt.show(block=False)
    path = f"{pathlib.Path(__file__).resolve().parent.parent}\\plots\\pytestplot.pdf"
    plt.savefig(path)
    plt.clf()

    for output_index, output in enumerate(zip(outputs_T_RTE, outputs_T_DE)):
        if output_index < 60:
            continue
        try:
            assertions.assertAlmostEqual(output[0], output[1])
        except Exception as exc:
            raise ValueError(
                f"Mismatch in outputs for index {output_index}. T_RTE = {output[0]}, T_DE = {output[1]}"
            ) from exc
