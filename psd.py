from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np 
import math
from scipy.signal import welch


from config import fontsize_label, fontsize_axis, dpi, figsize, n_tick
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['figure.figsize'] = figsize

def plot_psd_v2(X, fs, unit, ch_names,fname = None, scaling = 'density', dpi=None, log=False, fmax=100):
    n, _  = X.shape 
    x_tick = np.linspace(0, fmax, n_tick + 1)
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
        # ax.patch.set_alpha(0.5)
        ax.set(xlim=(0, fmax), ylim=(min_, max_))
        
        ax.plot(
            f, X_pss[i,:],
            linewidth = 1,
            color='black'
        )

        ax.text(
            fmax, mid_, #(max_ - min_)/2,
            ch_names[i],
            verticalalignment="center",
            horizontalalignment="left",
            fontdict={"fontsize":fontsize_label}
        )
       

        # Hide borders
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        # ax.spines["left"].set_alpha(.1)
        ax.spines["bottom"].set_alpha(.1)
        ax.get_xaxis().set_ticks(x_tick)
        ax.set_xticklabels([])
    
    # ax.spines["bottom"].set_visible(True)
    ax.set_xticklabels(x_tick)
    
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize_label) 
    
    ax.set_xlabel("Frequency (Hz)",fontdict={"fontsize":18})
    # plt.ylabel('ylabel')
    
    # y label
    ax0 = fig.add_subplot(gs[:, 0])
    ax0.axis("off")
    ax0.text(
        .5, .5,
        f"{unit}",
        verticalalignment="center",
        horizontalalignment="right",
        rotation=90,
        fontdict={"fontsize":fontsize_label}
    )
    if fname:
        fig.savefig(fname,bbox_inches='tight')
        return fig 
    else:
        return fig

if __name__ == "__main__":
    X = np.zeros([10, 10000])
    
    for i in range(10):
        X[i,:] = np.sin(2*math.pi*i * 100 *np.linspace(0, 2, 10000))

    plot_psd_v2(X, 500, log=False, ch_names = [f"CH [{i}]" for i in range(10)],unit="unit: Shared ylabel")
    plt.show()