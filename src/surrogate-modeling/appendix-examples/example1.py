import numpy as np
import matplotlib.pyplot as plt


def main():
    x = np.linspace(0,1,100)
    y = ((6*x - 2)**2) * np.sin((12*x) - 4)

    plt.plot(x,y)
    plt.savefig("example-1.png")


if __name__ == "__main__":
    main()
