from matplotlib import pyplot as plt
from matplotlib import figure
import matplotlib
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter
from matplotlib import markers as _markers

plt.rcParams["figure.figsize"] = [15, 5]
# matplotlib.rcParams['text.usetex'] = True

def plot_sig_3(sig, **kwargs):
    """
    
    ===
    Example:
        plot_sig_2(
            X,
            fs = 500,
            dpi = 150,
            scale = _scale1,
            unit = "1e-06V",
            ch_names = sample_raw_bandpass.ch_names,
            eog_sig = sample_raw_eog.get_data(),
            eog_ch_names = sample_raw_eog.ch_names,
            markers = [.3,2.2,3.3,4.9]
        );    
    """
    
    # Shape
    n, l = sig.shape
    _min, _max = np.min(sig),np.max(sig)
    
    # Sampling frequency
    fs = kwargs.get("fs")
    assert fs, "Missing fs!!!"
    
    nsec= l/fs
    taxis = np.linspace(0, nsec, l)
    
    # Scale
    if kwargs.get("scale"):
        _scale = kwargs.get("scale")
    else:
        _scale =_max - _min
    
    ## Number of scales in figures:
    n_scale = 1


    # DPI
    dpi = kwargs.get("dpi")

    if dpi:
        fig = plt.figure(dpi=dpi)
    else:
        fig = plt.figure()
    
    n_ = n
    
    # EOG signal
    eog_sig = kwargs.get("eog_sig")
    
    try:
        n_eog, _ = eog_sig.shape
    except Exception as e:
        n_eog = 0
        print(e)

    if n_eog:
        eog_ch_names = kwargs.get("eog_ch_names")
        assert eog_ch_names, "Missing EOG ch_names!!!"

        eog_scale = np.max(eog_sig) - np.min(eog_sig)
        # assert eog_scale, "Missing EOG Scale!!!"
        
        n_ = n + n_eog
        n_scale += 1


        X_eog = np.zeros([n_eog, l])
        for i in range(n_eog):
            X_eog[i,:] = eog_sig[i,:] - i  * eog_scale

        eog_min = np.mean(X_eog[-1,:]) - eog_scale
        eog_max = np.mean(X_eog[0,:]) + eog_scale
    
    gs = GridSpec(n_ + n_scale,11,figure=fig) # Ch_names / Comp_name + Signal; Scale bellow signal block

    X = np.zeros([n,l])
    
    for i in range(n):
        X[i,:] = sig[i,:] - i * _scale
    
    _min = np.mean(X[-1,:]) - _scale
    _max = np.mean(X[0,:]) + _scale
    
    
    ## Comp/Chan name
    ch_names = kwargs.get("ch_names")
    assert ch_names, "Missing ch_names!!!"
    
    ax0 = fig.add_subplot(gs[:n,0])
    ax0.axis("off")
    for i in range(n):
        ax0.text(
            1, (n-i)/(n+1),
            ch_names[i],
            verticalalignment="center",
            horizontalalignment="right",
        )
    
    
    ## Sig
    ax1 = fig.add_subplot(gs[:n ,1:]) # n for comp/ch; last one for scale
    ax1.set(xlim=(0,nsec),ylim=(_min,_max))
    ax1.axis("off")
    
    for i in range(n):
        ax1.plot(taxis, X[i,:], color="black",linewidth=.25)
    
    # MARKER
    markers = kwargs.get("markers")
    # [TODO]: move closer to signal
    
    # Sig
    
    if markers:
        for m in markers:
            ax1.scatter(m, _max - _scale / 2, marker=_markers.CARETDOWN, color="black")
    
    # SIGNAL EEG

    ## Scale
    ax2 = fig.add_subplot(gs[n:n+1,1:])
    ax2.axis("off")
    ax2.set(xlim=(0, nsec),ylim=(0,_scale))
    
    ## Temporal scale
    ax2.plot([nsec-2,nsec-1],[0,0], color="black")
    ax2.text(
        nsec - 1.5, -.1 * _scale,
        "1 sec",
        verticalalignment="top",
        horizontalalignment="center"
    )
    
    unit = kwargs.get("unit")
    if unit:
        ax2.plot([nsec - 1,nsec - 1],[0, _scale], color="black", linewidth=.5)
        ax2.text(
            nsec - 1, _scale/2,
            f" {np.round(_scale * 1e6,2)} {unit}",
            verticalalignment="center",
            horizontalalignment="left"
        )
        
    # EOG SIG
    ## Channel name
    ax0 = fig.add_subplot(gs[n+1:n+1+n_eog, 0])
    ax0.axis("off")
    for i in range(n_eog):
        ax0.text(
            1, (n_eog-i)/(n_eog+1),
            eog_ch_names[i],
            verticalalignment="center",
            horizontalalignment="right",
        )

    ## Signal
    ax1 = fig.add_subplot(gs[n+1:n+1+n_eog, 1:])
    ax1.set(xlim=(0,nsec),ylim=(eog_min,eog_max))
    ax1.axis("off")
    
    for i in range(n_eog):
        ax1.plot(taxis, X_eog[i,:], color="black",linewidth=.25)

    if markers:
        for m in markers:
            ax1.scatter(m, eog_max - eog_scale / 4, marker=_markers.CARETDOWN, color="gray")

    ## Scale
    ax2 = fig.add_subplot(gs[n+1+n_eog:n_ + n_scale, 1:])

    ax2.axis("off")
    ax2.set(xlim=(0, nsec),ylim=(0,eog_scale))
    
    ## Temporal scale
    ax2.plot([nsec-2,nsec-1],[0,0], color="black")
    ax2.text(
        nsec - 1.5, -.1 * eog_scale,
        "1 sec",
        verticalalignment="top",
        horizontalalignment="center"
    )
    
    eog_unit = kwargs.get("eog_unit")
    if unit:
        ax2.plot([nsec - 1,nsec - 1],[0, eog_scale], color="black", linewidth=.5)
        ax2.text(
            nsec - 1, eog_scale/2,
            f" {np.round(eog_scale * 1e6,2)} {eog_unit}",
            verticalalignment="center",
            horizontalalignment="left"
        )

    fname = kwargs.get("fname")
    
    if fname:
        fig.savefig(fname=f"{fname}.png", pad_inches=0, dpi=dpi if dpi else None)
        plt.close(fig)
        pass
    else:
        return fig

# https://matplotlib.org/3.2.0/api/_as_gen/matplotlib.pyplot.savefig.html
# savefig(fname, dpi=None, facecolor='w', edgecolor='w',
#     orientation='portrait', papertype=None, format=None,
#     transparent=False, bbox_inches=None, pad_inches=0.1,
#     frameon=None, metadata=None)

if __name__ == "__main__":
    X = np.random.randn(10, 750)/1e6
    plt.rcParams["figure.figsize"] = [15,10]
    X_eog = np.random.randn(2, 750)/1e6
    
    plot_sig_3(
        X,
        fs = 50,
        ch_names = [f"CH {i}" for i in range(10)],
        unit = "microV",
        markers = [1, 2, 3],
        eog_sig = X_eog,
        eog_ch_names = [f"EOG {i}" for i in range(2)],
        eog_unit = "microV",
        fname = "Test"
    )
    plt.show()


