from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

def plot_sig(X, X_EOG, **kwargs):
    """
    """

    # Number of component(s)/Channel(s), length of data:
    n_comp, l = X.shape
    # EOG
    n_eog, l = X_EOG.shape

    # Required params:
    fs = kwargs.get("fs")
    ch_name = kwargs.get("ch_names")
    eog_ch_name = kwargs.get("eog_ch_names")

    
    # assert fs==None or ch_name==None or eog_ch_name==None, "Missing Params!!"

    # Timespan of signal
    nsec = l/fs
    t_axis = np.linspace(0, nsec, l)

    # Transforming signal to plot:
    comp_scale = kwargs.get("scale")

    if not comp_scale:
        comp_scale = np.max(X) - np.min(X)
    
    X_ = np.zeros([n_comp, l])

    for i in range(n_comp):
        baseline_X = X[i,:] - np.mean(X[i,:])
        X_[i,:] = baseline_X + comp_scale * i

    comp_min, comp_max = np.min(X_), np.max(X_)
    X_mean = np.mean(X_, axis = 1)

    # figure & grid spacing
    dpi = kwargs.get("dpi")
    if dpi:
        fig = plt.figure(dpi = dpi)
    else:
        fig = plt.figure()


    # Outlining the plot
    n_ = n_comp + 1 + n_eog + 1

    # Grid
    gs = GridSpec(n_, 11, figure=fig)

    # EEG/ICA components >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    ## Comp's name/ch_name
    ax0 = fig.add_subplot(gs[:n_comp, 0])
    ax0.set(xlim = (0, 1), ylim=(comp_min, comp_max))
    ax0.axis("off")
    
    assert len(ch_name) == n_comp, "len(ch_name) != n_comp"
    
    for idx, ch in enumerate(ch_name):    
        ax0.text(
            1, X_mean[idx],
            ch,
            verticalalignment="center",
            horizontalalignment="right",
            fontdict={"fontsize":18}
        )


    ## Comp/channel's Signal
    ax1 = fig.add_subplot(gs[:n_comp, 1:])
    ax1.axis("off")
    ax1.set(xlim = (0, nsec), ylim=(comp_min, comp_max))

    for i in range(n_comp):
        ax1.plot(
            t_axis, 
            X_[i,:],
            color="black",
            linewidth=.25,
            
        )

    ## Signal Scale:
    ax2 = fig.add_subplot(gs[n_comp:n_comp+1, 1:])
    ax2.set(xlim = (0, nsec), ylim=(0, comp_scale))
    ax2.axis("off")

    ## x-axis scale
    ax2.plot([nsec-1, nsec], [0, 0], color="black")
    ax2.text(
            nsec, -.05 * comp_scale,
            "1 sec",
            verticalalignment="top",
            horizontalalignment="right",
            fontdict={"fontsize":14}
        )

    ## y-axis scale
    unit = kwargs.get("unit")
    
    if unit:
        ax2.plot([nsec-1, nsec-1], [0, comp_scale], \
            color="black",
            # linewidth = .5
            )
        
        ax2.text(
            nsec-1, comp_scale,
            f" {np.round(comp_scale, 1)} {unit}",
            verticalalignment="top",
            horizontalalignment="left",
            fontdict={"fontsize":14}
        )
    
    # EOG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    # Transforming signal to plot:
    eog_scale = np.max(X_EOG) - np.min(X_EOG)
    
    X_EOG_ = np.zeros([n_comp, l])

    for i in range(n_eog):
        baseline_X_EOG = X_EOG[i,:] - np.mean(X_EOG[i,:])
        X_EOG_[i,:] = baseline_X_EOG + eog_scale * i

    eog_min, eog_max = np.min(X_EOG_), np.max(X_EOG_)
    X_EOG_mean = np.mean(X_EOG_, axis = 1)
    
    ## Comp's name/ch_name
    ax0 = fig.add_subplot(gs[-n_eog-1:-1, 0])
    ax0.set(xlim = (0, 1), ylim=(eog_min, eog_max))
    ax0.axis("off")
    
    assert len(eog_ch_name) == n_eog, "len(ch_name) != n_comp"
    
    for idx, ch in enumerate(eog_ch_name):    
        ax0.text(
            1, X_EOG_mean[idx],
            ch,
            verticalalignment="center",
            horizontalalignment="right",
            fontdict={"fontsize":18}
        )

    ## Comp/channel's Signal
    ax1 = fig.add_subplot(gs[-n_eog-1:-1, 1:])
    ax1.axis("off")
    ax1.set(xlim = (0, nsec), ylim=(eog_min, eog_max))

    for i in range(n_eog):
        ax1.plot(
            t_axis, 
            X_EOG_[i,:],
            color="black",
            linewidth=.25
        )
    
    ## Signal Scale:
    ax2 = fig.add_subplot(gs[-1, 1:])
    ax2.set(xlim = (0, nsec), ylim=(0, eog_scale))
    ax2.axis("off")

    ## x-axis scale
    ax2.plot([nsec-1, nsec], [0, 0], color="black")
    ax2.text(
            nsec, -.05 * eog_scale,
            "1 sec",
            verticalalignment="top",
            horizontalalignment="right",
            fontdict={"fontsize":14}
        )

    ## y-axis scale
    eog_unit = kwargs.get("eog_unit")
    
    if eog_unit:
        ax2.plot([nsec-1, nsec-1], [0, eog_scale], \
            color="black",
            # linewidth = .5
            )
        
        ax2.text(
            nsec-1, eog_scale,
            f" {np.round(eog_scale, 1)} {eog_unit}",
            verticalalignment="top",
            horizontalalignment="left",
            fontdict={"fontsize":14}
        )


    fname = kwargs.get("fname")
    if fname:
        fname = f"{fname}.png"
        fig.savefig(fname)
        return fig

    return fig

if __name__ == "__main__":
    X = np.random.randn(5, 10000)
    X_EOG = np.random.randn(2, 10000) * 100

    plot_sig(
        X = X,
        X_EOG = X_EOG,
        fs = 500,
        ch_names = [f"EEG {i}" for i in range(5)],
        eog_ch_names = [f"EOG {i}" for i in range(2)],
        unit="μV"
    )

    plt.show()