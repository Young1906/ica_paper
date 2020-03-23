from matplotlib import pyplot as plt
from matplotlib import figure
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter

plt.rcParams["figure.figsize"] = [15, 5]


def plot_sig_(sig=None, **kwargs):
    """
    plot_sig_(
        X,
        eog = X_eog,
        eog_ch_names=[f"EOG [{i}]" for i in range(2)],
        fs=500,
        ch_names=[f"CH [{i}]" for i in range(10)],
        unit="microV",
        fname="test",
        markers=[{"t":0,"c":"black"},
                {"t":1,"c":"black"},
                {"t":2,"c":"red"}]
    );
    """
    
    # Getting specs    
    n, l = sig.shape
    
    dpi = kwargs.get("dpi")
    
    if dpi:
        fig = plt.figure(dpi=dpi)
    else:
        fig = plt.figure()
    
    eog_sig = kwargs.get("eog")
    
    try:
        n_eog, _ = eog_sig.shape
    except Exception as e:
        n_eog = False
    
    # Get markers
    markers = kwargs.get("markers")
    
    n_ = n
    
    if n_eog:
        n_ = n + n_eog
        
    if markers:
        n_= n_ + 1
    
    gs = GridSpec(n_, 1)
            
    fs = kwargs.get("fs")
    ch_names = kwargs.get("ch_names")
    
    nsec = l/fs
    assert nsec >= 10, "Temporal scale is wrong!!!"
    taxis = np.linspace(0,nsec, l)
    
    if kwargs.get("ylim"):
        _min, _max = kwargs.get("ylim")
    else:
        _min, _max = np.min(sig),np.max(sig)
    
    _unit = kwargs.get("unit")
    
    # Plotting EEG signal
    for i in range(n):
        inner = GridSpecFromSubplotSpec(1,12, subplot_spec=gs[i])
        
        ## Component/Channel name
        ax0 = fig.add_subplot(inner[:,0])
        ax0.axis("off")
        ax0.text(1,.5,
                 ch_names[i],
                 verticalalignment="center",
                 horizontalalignment="right",)
        
        ## Signal
        ax1 = fig.add_subplot(inner[:,1:-1])
        ax1.plot(
            taxis,
            sig[i,:],
            linewidth=.5,
            color="black"
        )
        ax1.set(ylim=[_min, _max])
        ax1.set(xlim=(0, nsec))
        ax1.axis("off")
        
        if i == 0:
            # Save a copy for marking artefacts
            maker_ax = ax1
        
        ## Scale
        ax2 = fig.add_subplot(inner[:,-1])
        ax2.axis("off")
        
        if i == n-1:
            ax2.set(xlim=[-1,0], ylim=[0,1])
            
            ax2.plot([0,-10/nsec],[0,0], color="black")
            
            if _unit:
                ax2.plot([0,0],[0,1], color="black")
                ax2.text(0.1,0.5,
                         f"{np.round(_max - _min, 2)} {_unit}",
                        verticalalignment="center",
                        horizontalalignment="left",)
            
            ax2.text(-10/nsec/2,-.1,
                     "1 sec",
                    verticalalignment="top",
                    horizontalalignment="center",)
    
    # Plotting EOG signal (if given)
    if n_eog:
        eog_ch_names = kwargs.get("eog_ch_names")
        eog_min, eog_max = np.min(eog_sig),np.max(eog_sig)
        
        eog_unit = kwargs.get("eog_unit")
        
        for i in range(n, n_ - 1 if markers else n_):
            inner = GridSpecFromSubplotSpec(1,12, subplot_spec=gs[i])
        
            ## Component/Channel name
            ax0 = fig.add_subplot(inner[:,0])
            ax0.axis("off")
            ax0.text(1,.5,
                     eog_ch_names[i-n],
                     verticalalignment="center",
                     horizontalalignment="right",)

            ## Signal
            ax1 = fig.add_subplot(inner[:,1:-1])
            ax1.plot(
                taxis,
                eog_sig[i-n,:],
                linewidth=.5,
                color="black"
            )
            ax1.set(ylim=[eog_min, eog_max])
            ax1.set(xlim=(0, nsec))
            ax1.axis("off")


            ## Scale
            ax2 = fig.add_subplot(inner[:,-1])
            ax2.axis("off")

            if i == n + n_eog -1:
                ax2.set(xlim=[-1,0], ylim=[0,1])
                ax2.plot([0,0],[0,1], color="black")
                ax2.plot([0,-10/nsec],[0,0], color="black")
                
                ax2.text(0.1,0.5,
                         f"{np.round(eog_max - eog_min, 2)} {eog_unit}",
                        verticalalignment="center",
                        horizontalalignment="left",)

                ax2.text(-10/nsec/2,-.1,
                         "1 sec",
                        verticalalignment="top",
                        horizontalalignment="center",)
    fname = kwargs.get("fname")
    
    # Makers
    # [TODO]
    markers = kwargs.get("markers")
    
    if markers:
        inner = GridSpecFromSubplotSpec(1,12, subplot_spec=gs[-1])
        
        # Label 
        ax0 = fig.add_subplot(inner[:,0])
        ax0.axis("off")
        
        # Makers
        ax1 = fig.add_subplot(inner[:,1:-1])
        ax1.set(xlim=(0, nsec))
        ax1.axis("off")
        
        for m in markers:
            ax1.scatter(m["t"], 0, marker=6,c=m["c"])
        
    
    if fname:
        fig.savefig(f"{fname}.png")
        plt.close(fig)
        pass
    else:
        plt.show()
        return fig

    
    
def plot_sig_2(sig, **kwargs):
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
        pass
    
    if n_eog:
        eog_ch_names = kwargs.get("eog_ch_names")
        assert eog_ch_names, "Missing EOG ch_names!!!"
        n_ = n + n_eog
    
    gs = GridSpec(n_,12,figure=fig)

    X = np.zeros([n,l])
    
    for i in range(n):
        X[i,:] = sig[i,:] - i * _scale
    
    _min = np.mean(X[-1,:]) - _scale
    _max = np.mean(X[0,:]) + _scale
    
    
    # MARKER
    markers = kwargs.get("markers")
    
    ## Sig
    ax1 = fig.add_subplot(gs[0,1:-1])
    ax1.set(xlim=(0, nsec))
    ax1.axis("off")
    
    if markers:
        for m in markers:
            ax1.scatter(m, 0, marker=11, color="black")
    
    # SIGNAL EEG
    
    
    ## Comp/Chan name
    ch_names = kwargs.get("ch_names")
    assert ch_names, "Missing ch_names!!!"
    
    ax0 = fig.add_subplot(gs[1:-2,0])
    ax0.axis("off")
    for i in range(n):
        ax0.text(
            1, (n-i)/(n+1),
            ch_names[i],
            verticalalignment="center",
            horizontalalignment="right",
        )
    
    
    ## Sig
    ax1 = fig.add_subplot(gs[1:-2,1:-1])
    ax1.set(xlim=(0,nsec),ylim=(_min,_max))
    ax1.axis("off")
    
    for i in range(n):
        ax1.plot(taxis, X[i,:], color="black",linewidth=.25)
    
    ## Scale
    ax2 = fig.add_subplot(gs[1:-2,-1])
    ax2.axis("off")
    ax2.set(xlim=(-1,0),ylim=(_min,_max))
    
    ### Temporal scale
    ax2.plot([-10/nsec,0],[_min,_min], color="black")
    ax2.text(
        -5/nsec, _min*1.01,
        "1 sec",
        verticalalignment="top",
        horizontalalignment="center"
    )
    
    unit = kwargs.get("unit")
    if unit:
        ax2.plot([0,0],[_min, _min + _scale], color="black")
        ax2.text(
            0, _min + _scale/2,
            f" {np.round(_scale * 1e6,2)} {unit}",
            verticalalignment="center"
        )
        
    # EOG SIG
    
    ## Comp / ch_name
    ax0 = fig.add_subplot(gs[-2:,0])
    ax0.axis("off")
    for i in range(n_eog):
        ax0.text(
            1, (n_eog-i)/(n_eog+1),
            eog_ch_names[i],
            verticalalignment="center",
            horizontalalignment="right",
        )
    ## Sig
    
    
    X_eog = np.zeros([n_eog,l])
    eog_scale = np.max(eog_sig) - np.min(eog_sig)
    
    for i in range(n_eog):
        X_eog[i,:] = eog_sig[i,:] - i * eog_scale
    
    eog_min = np.mean(X_eog[-1,:]) - eog_scale
    eog_max = np.mean(X_eog[0,:]) + eog_scale
    
    
    ax1 = fig.add_subplot(gs[-2:,1:-1])
    ax1.set(xlim=(0, nsec), ylim=(eog_min, eog_max))
    ax1.axis("off")
    
    for i in range(n_eog):
        ax1.plot(taxis, X_eog[i,:], color="black",linewidth=.25)
    
    
    ## Scale
    ax2 = fig.add_subplot(gs[-2:,-1])
    ax2.axis("off")
    ax2.set(xlim=(-1,0),ylim=(eog_min,eog_max))
    
    ax2.plot([-10/nsec,0],[eog_min,eog_min], color="black")
    
    ax2.text(
        -5/nsec, eog_min*1.05,
        "1 sec",
        verticalalignment="top",
        horizontalalignment="center"
    )
    
    eog_unit = kwargs.get("eog_unit")
    
    ax2.plot([0,0],[eog_min, eog_min + eog_scale], color="black")
    ax2.text(
        0, eog_min + eog_scale/2,
        f" {np.round(eog_scale * 1e6,2)} {eog_unit}",
        verticalalignment="center"
    )
    
    return fig


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
        pass
    
    if n_eog:
        eog_ch_names = kwargs.get("eog_ch_names")
        assert eog_ch_names, "Missing EOG ch_names!!!"
        n_ = n + n_eog
    
    gs = GridSpec(n_,11,figure=fig)

    X = np.zeros([n,l])
    
    for i in range(n):
        X[i,:] = sig[i,:] - i * _scale
    
    _min = np.mean(X[-1,:]) - _scale
    _max = np.mean(X[0,:]) + _scale
    
    
    # MARKER
    # markers = kwargs.get("markers")
    # [TODO]: move closer to signal
    
    ## Sig
    # ax1 = fig.add_subplot(gs[0,1:0])
    # ax1.set(xlim=(0, nsec))
    # ax1.axis("off")
    
    # if markers:
    #     for m in markers:
    #         ax1.scatter(m, 0, marker=11, color="black")
    
    # SIGNAL EEG
    
    
    ## Comp/Chan name
    ch_names = kwargs.get("ch_names")
    assert ch_names, "Missing ch_names!!!"
    
    ax0 = fig.add_subplot(gs[1:-2,0])
    ax0.axis("off")
    for i in range(n):
        ax0.text(
            1, (n-i)/(n+1),
            ch_names[i],
            verticalalignment="center",
            horizontalalignment="right",
        )
    
    
    ## Sig
    ax1 = fig.add_subplot(gs[1:-2,1:])
    ax1.set(xlim=(0,nsec),ylim=(_min,_max))
    ax1.axis("off")
    
    for i in range(n):
        ax1.plot(taxis, X[i,:], color="black",linewidth=.25)
    
    ## Scale
    ax2 = fig.add_subplot(gs[1:-2,-1])
    ax2.axis("off")
    ax2.set(xlim=(-1,0),ylim=(_min,_max))
    
    ### Temporal scale
    ax2.plot([-10/nsec,0],[_min,_min], color="black")
    ax2.text(
        -5/nsec, _min*1.01,
        "1 sec",
        verticalalignment="top",
        horizontalalignment="center"
    )
    
    unit = kwargs.get("unit")
    if unit:
        ax2.plot([0,0],[_min, _min + _scale], color="black")
        ax2.text(
            0, _min + _scale/2,
            f" {np.round(_scale * 1e6,2)} {unit}",
            verticalalignment="center"
        )
        
    # EOG SIG
    
    ## Comp / ch_name
    ax0 = fig.add_subplot(gs[-2:,0])
    ax0.axis("off")
    for i in range(n_eog):
        ax0.text(
            1, (n_eog-i)/(n_eog+1),
            eog_ch_names[i],
            verticalalignment="center",
            horizontalalignment="right",
        )
    ## Sig
    
    
    X_eog = np.zeros([n_eog,l])
    eog_scale = np.max(eog_sig) - np.min(eog_sig)
    
    for i in range(n_eog):
        X_eog[i,:] = eog_sig[i,:] - i * eog_scale
    
    eog_min = np.mean(X_eog[-1,:]) - eog_scale
    eog_max = np.mean(X_eog[0,:]) + eog_scale
    
    
    ax1 = fig.add_subplot(gs[-2:,1:-1])
    ax1.set(xlim=(0, nsec), ylim=(eog_min, eog_max))
    ax1.axis("off")
    
    for i in range(n_eog):
        ax1.plot(taxis, X_eog[i,:], color="black",linewidth=.25)
    
    
    ## Scale
    ax2 = fig.add_subplot(gs[-2:,-1])
    ax2.axis("off")
    ax2.set(xlim=(-1,0),ylim=(eog_min,eog_max))
    
    ax2.plot([-10/nsec,0],[eog_min,eog_min], color="black")
    
    ax2.text(
        -5/nsec, eog_min*1.05,
        "1 sec",
        verticalalignment="top",
        horizontalalignment="center"
    )
    
    eog_unit = kwargs.get("eog_unit")
    
    ax2.plot([0,0],[eog_min, eog_min + eog_scale], color="black")
    ax2.text(
        0, eog_min + eog_scale/2,
        f" {np.round(eog_scale * 1e6,2)} {eog_unit}",
        verticalalignment="center"
    )
    
    return fig

if __name__ == "__main__":
    X = np.random.randn(10, 750)
    plt.rcParams["figure.figsize"] = [15,10]

    plot_sig_(
        X,
        fs = 50,
        ch_names = [f"CH [i]" for i in range(10)]
    )
    plt.show()


