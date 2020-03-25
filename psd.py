from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np 
import math

def plotly_psd(data, fs=128, unit='mV^2/Hz', max_f=60,  cols=1, height=1000, width=1024, title="PSD plots"):
    """
    Plot PSD of the data (interactive version)
    Args:
    - data (DataFrame): Raw data of shape (n_samples, n_channels)
    - fs (int): sample rate
    - unit (str): Unit of y-axis
    - max_f (float): Maximum frequency to show in x-axis
    - cols (int): Number of columns in plot
    - height (int): Figure height
    - width (int): Figure width
    - title (str): Figure title
    """
    if cols == 1:
        rows = data.shape[1]
    else:
        rows = math.ceil(data.shape[1] / cols)
    name_list = ["Channel: " + str(c) for c in data.columns]
    fig = make_subplots(rows=rows, cols=cols,\
                       subplot_titles=name_list)
    # Column & row indices
    r_id, c_id = 1, 1
    # Loop through all channels and plot
    for i, ch in enumerate(data.columns):
        x, y = welch(data.iloc[:,i], fs=fs, average='median', nperseg=fs*4)
        x, y = (x[np.where(x<=max_f)], y.T[np.where(x<=max_f)])
        if unit == 'V^2/Hz' or unit == 'mV^2/Hz':
            x_, y_ = x, y
        elif unit == 'V^2' or unit == 'mV^2':
            x_, y_ = x, y*x
        elif unit == 'V/Hz' or unit == 'mV/Hz': # V or mV
            x_, y_ = x, np.sqrt(y*x)/x
        else: # V or mV
            x_, y_ = x, np.sqrt(y*x)
        fig.add_trace(
            go.Scatter(x=x_, y=y_),
            row=r_id, col=c_id
        )
        fig.update_xaxes(title_text="Frequency (Hz)", row=r_id, col=c_id)
        fig.update_yaxes(title_text=unit, row=r_id, col=c_id)
        if cols == 1:
            r_id += 1
        else:
            if (i + 1) % cols == 0:
                r_id += 1
                c_id = 1
            else:
                c_id += 1

    fig.update_layout(height=height, width=width, title_text=title)
    return fig

def plot_psd_v2(X, fs, unit, dpi=None):
    n, l = X.shape 
    nsec = l / fs 
    t_axis = np.linspace(0, nsec, l)
    # test_sig = np.sin(2*math.pi*10*fs*t_axis)

    if dpi:
        fig = plt.figure(dpi = dpi)
    else:
        fig = plt.figure()

    gs = GridSpec(nrows = n, ncols = 1, figure = fig)
    for i in range(n):
        
        ax = fig.add_subplot(gs[i, 0], sharex = None if i==0 else ax)

        
        ax.plot(t_axis, X[i,:])
    
    return fig

if __name__ == "__main__":
    X = np.random.randn(5, 1000)
    plot_psd_v2(X, 500, unit=None, dpi=None)
    plt.show()
    