
import matplotlib.pyplot as plt


def quickplot_scatter(x, y, title = "out"):
    plt.figure(figsize=(6, 5))
    plt.subplot(1, 1, 1)
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig(title+'.png')


def quickplot_1d(x, y, title = "out", xscatter=None, yscatter=None):
    plt.figure(figsize=(6, 5))
    plt.subplot(1, 1, 1)
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    if xscatter is not None:
        plt.scatter(xscatter,yscatter)
    plt.tight_layout()
    plt.savefig(title+'.png')


def quickplot_hist(samples, title = "out"):
    plt.figure(figsize=(6, 5))
    plt.subplot(1, 1, 1)
    plt.hist(samples, bins=30, density=True, alpha=0.6, color='g')
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.savefig(title+'.png')


