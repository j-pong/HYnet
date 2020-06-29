import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def grapher(x, y):
    fig = plt.figure(figsize=(10, 4))
    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

    ax = fig.add_subplot()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    markers = ['o', 's']
    colors = ['r', 'b']
    print(np.shape(y))
    for i in range(2):
        ax.plot(x, y[i], marker=markers[i], markerfacecolor="None", ms=10, color=colors[i])
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.legend(['$\it{test-clean}$', '$\it{test-other}$'], ncol=1, loc='upper left')
    ax.grid()
    ax.set_xlabel("Label Corruption Ratio ($\zeta$)",fontsize=20)
    ax.set_ylabel("WER (%)",fontsize=20)

    plt.savefig('../degrade.pdf', transparent=True, bbox_inches='tight', pad_inches=0)

    # plt.show()


if __name__ == '__main__':
    x = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    y = np.array([[10.3, 15.0, 18.6, 24.4, 31.5, 48.2],
                  [21.0, 26.4, 31.1, 36.2, 43.3, 58.3]])
    grapher(x, y)
