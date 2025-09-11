import pathlib
import unittest

import matplotlib.pyplot as plt
import pandas as pd
from modules import Contini
from torch_modules import tContini

print("\n\n")
print(pathlib.Path(__file__).resolve(), pathlib.Path(__file__).resolve().parent)
plt.ion()


def test_init_methods() -> None:
    """
    Test __init__ for regular and torch variant.
    """
    contini = Contini()
    torch_contini = tContini()
    methods = set(
        [
            attr if not (callable(getattr(contini, attr))) else None
            for attr in dir(contini)
        ]
    )
    torch_methods = set(
        [
            attr if not (callable(getattr(torch_contini, attr))) else None
            for attr in dir(torch_contini)
        ]
    )

    test = [
        (
            str(attr) in str(torch_methods)
            or str(attr) in str(list(torch_contini.controls.keys()))
            or str(attr) in str(list(torch_contini.controls.get("ydata_info").keys()))
        )
        for attr in methods
    ]
    assert all(test), "Attributes mismatch between tContini and Contini in test_init()."


def test_init_defaults() -> None:
    """
    Tests default values for tContini and Contini.
    """

    contini = Contini()
    torch_contini = tContini()
    torch_contini_dict = torch_contini.__dict__
    contini_dict = contini.__dict__

    test_dict = [
        (
            item in torch_contini_dict.items()
            or item in torch_contini_dict["controls"].items()
            or item in torch_contini_dict["controls"]["ydata_info"].items()
        )
        for item in contini_dict.items()
    ]

    assert all(
        test_dict
    ), "Mismatch in default __init__ attributes in test_init_defaults()."


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
