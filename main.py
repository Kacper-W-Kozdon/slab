# placeholder
from modules import Contini

if __name__ == "__main__":
    contini = Contini(s=40, mua=0, musp=1, n1=1, n2=1)

    rho = 1

    result = []

    for t_index, t in enumerate(range(5, 105, 5)):
        picot = 0.001 * t
        subresult = contini((picot, rho))

        result.append(tuple([subresult[0], subresult[1]]))

    print(result)
