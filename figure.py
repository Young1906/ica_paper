from matplotlib import pyplot as plt
from matplotlib import figure
import numpy as np

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


if __name__ == "__main__":
    X = np.random.randn(10, 750)
    plt.rcParams["figure.figsize"] = [15,10]

    _plot(
        X, fs=500,\
        markers=[{"t":.25, "c":"black"},{"t":1.25, "c":"black"}], \
        ch_names=[f"CH {i}" for i in range (10)]
    )
    plt.show()