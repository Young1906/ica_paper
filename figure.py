from matplotlib import pyplot as plt
from matplotlib import figure
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter

plt.rcParams["figure.figsize"] = [15, 5]
plt.rcParams["font.family"] = "Times New Roman"

def plot_eeg(_sig, fs, markers):
    sig = _sig.get_data()*1e6
    fs = fs
    ch_names = _sig.ch_names
    
    return _plot_eeg(sig, fs=fs, ch_names=ch_names, markers=markers)

def _plot_eeg(sig, **kwargs):
    """
    fs,
    ch_names,
    markers,
    
    """
    
    # Sampling frequency
    assert kwargs.get("fs"), "_plot: Missing sample frequency"
    fs = kwargs.get("fs")
    
    # Shape of data:
    n, _l = sig.shape
    t = np.linspace(0, _l/fs, _l)
    
    # Global min and max of data
    _min, _max = np.min(sig), np.max(sig)
    
    # Create figure
    dpi = kwargs.get("dpi")
    if dpi:
        fig = plt.figure(dpi=dpi)
    else:
        fig = plt.figure()
    
    # MARKER >>>
    
    # first axis is used for marker(s)
    l, b, w, h = 0, 1 - 1/(n+2), 1, 1/(n+1)
    ax0 = fig.add_axes([l,b,w,h])
    ax0.set(xlim=(0, np.max(t)))
    # Get marker
    markers = kwargs.get("markers")
    """
    Marker:
    ["t":1, "c":"black"]
    - t in seconds
    - c plt color pallete
    """
    
    if markers:
        for i in markers:
            _t = i["t"]
            c = i["c"]
            ax0.scatter(_t , 0, marker=11, s=100, color=c)
    
    ax0.axis("off")    

    # <<< MARKER
    
    
    # DATA >>>
    # Get channels / comp names:
    ch_names = kwargs.get("ch_names")
    
    # Function to clean channel_names
    
    def f(t):
        t = str(t)
        t = t.upper()
        return t
    
    ch_names = list(map(f, ch_names))
    
    for i in range(n):
        
        b = b - h
        ax1 = fig.add_axes([l,b,w,h], sharex=ax0, frameon=False)
        
        # option
        ax1.set(ylim=(_min, _max))
        ax1.axis("off")

        # Plotting the data
        ax1.plot(
            t,
            sig[i,:],
            color="black",
            linewidth = .5,)
        
        # Display x-axis if i == n - 1 (last channel)
        ax1.axis(option="tight")
        ax1.text(-.01, .5, ch_names[i], 
                horizontalalignment="right",
                verticalalignment="center",
                fontdict =
                 {
                     'color':  'black',
                     'size': 12,
                 }
                ) 
        
        
    # <<< DATA 
    
    # SCALE >>>
    
    # Temporal scale on last axis
    _len = 1
    _smax = max(t)
    _smin = _smax - _len
    _smid = (_smin + _smax) / 2
    
    ax1.scatter(_smin, _min, marker="|", s=500, color="black", alpha=.1)
    ax1.scatter(_smax, _min, marker="|", s=500, color="black", alpha=.1)
    ax1.hlines(_min, _smin, _smax, alpha=.1)
    
    ax1.text(_smid, _min, "1s",
             horizontalalignment="center",
             verticalalignment="center",
             fontdict =
             {
                 'color':  'darkred',
                 'size': 12,
             })
    
    # Vertical scale
    l, b, w = 1.025, 0, .1
    ax2 = fig.add_axes([l,b,w,h], sharex=ax0, frameon=False)
    ax2.set(ylim=(_min, _max))
    ax2.axvline(x = 0, linewidth = 5, color="darkred")
    
    ax2.axis("off")
    _s = np.round(_max - _min, 2)
    
    ax2.text(.1, .5, f" {np.round((_s)*1e-3,2)} 10^-6V",
             horizontalalignment="left",
             verticalalignment="center",
             fontdict =
             {
                 'color':  'darkred',
                 'size': 12,
             })
    # <<< SCALE
    # plt.savefig("fig.png")
    return fig


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
    n_eog, _ = eog_sig.shape
    
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
        return fig



if __name__ == "__main__":
    X = np.random.randn(10, 750)
    plt.rcParams["figure.figsize"] = [15,10]

    _plot(
        X, fs=500,\
        markers=[{"t":.25, "c":"black"},{"t":1.25, "c":"black"}], \
        ch_names=[f"CH {i}" for i in range (10)]
    )
    plt.show()