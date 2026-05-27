import math
import numpy as np
import matplotlib.pyplot as plt


def main():
    x_1 = np.linspace(0,1,100)
    x_2 = np.linspace(0,1,100)

    X_1, X_2 = np.meshgrid(x_1,x_2)
    f = (X_2 - ((5.1/(4*(math.pi)**2)))*X_2 + (5/math.pi)*X_1 - 6)**2 + 10*((1-(1/8*math.pi))*np.cos(X_1) + 1) + 5*X_1

    plt.contourf(X_1,X_2,f)
    plt.savefig("example-2.png")


if __name__ == "__main__":
    main()