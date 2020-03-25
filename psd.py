from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np 
import math
from scipy.signal import welch
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['axes.labelweight'] = 'normal'


def plot_psd_v2(X, fs, unit, scaling = 'density', dpi=None, log=False, fmax=100):
    n, l = X.shape 
    # nsec = l / fs 
    # t_axis = np.linspace(0, nsec, l)
    x_tick = np.linspace(0, fmax, 11)
    nperseg = 4 * fs
    # test_sig = np.sin(2*math.pi*10*fs*t_axis)

    if dpi:
        fig = plt.figure(dpi = dpi)
    else:
        fig = plt.figure()

    gs = GridSpec(nrows = n, ncols = 11, figure = fig)
    
    pxxs = []

    for i in range(n):
        # calculate psd
        f, pxx = welch(X[i,:], fs=fs, nperseg=nperseg, scaling=scaling, average='median')

        if log:
            pxx = np.log(pxx)

        pxxs.append(pxx)

    d1, d2 = len(pxxs), len(pxx)
    X_pss = np.zeros([d1, d2])
    
    for i in range(n):
        X_pss[i,:] = np.asarray(pxxs[i])

    min_, max_ = np.min(X_pss), np.max(X_pss)
    mid_ = (min_ + max_)/2
    for i in range(n):
        ax = fig.add_subplot(gs[i, 1:])
        ax.set(xlim=(0, fmax), ylim=(min_, max_))
        
        ax.plot(
            f, X_pss[i,:],
            linewidth = .5,
            color='black'
        )

        ax.text(
            fmax, mid_, #(max_ - min_)/2,
            f" Comp {i}",
            verticalalignment="center",
            horizontalalignment="left"
        )
        
        # ax.set_ylabel("Power(db)")

        # Hide borders
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # ax.spines["bottom"].set_visible(False)
        # ax.get_xaxis().set_ticks([])
        ax.get_xaxis().set_ticks(x_tick)
        ax.set_xticklabels([])
    
    # ax.spines["bottom"].set_visible(True)
    ax.set_xticklabels(x_tick)
    ax.set_xlabel("Frequency (Hz)")
    # plt.ylabel('ylabel')
    
    # y label
    ax0 = fig.add_subplot(gs[:, 0])
    ax0.axis("off")
    ax0.text(
        0, .5,
        f"{unit}",
        verticalalignment="center",
        horizontalalignment="right",
        rotation=90
    )
    return fig

if __name__ == "__main__":
    X = np.random.randn(5, 1000)
    plot_psd_v2(X, 500, log=True, unit="unit: Shared ylabel")
    plt.show()
    