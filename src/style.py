import matplotlib.font_manager as fm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

fe = fm.FontEntry(
    fname='fonts/Poppins/Poppins-Regular.ttf',
    name='poppins-regular')
fm.fontManager.ttflist.insert(0, fe)
mpl.rcParams['font.family'] = fe.name

colors = {
    "green": "#03ef62",
    "navy": "#05192d",
    "blue": "#06bdfc",
    "red": "#ff5400",
    "orange": "#ff931e",
    "purple": "#7933ff",
    "pink": "#ff6ea9",
    "yellow": "#fcce0d"
}

cmap = mpl.colors.ListedColormap(list(colors.values()))


def plot_examples(colormaps):
    """
    Helper function to plot data with associated colormap.
    """
    np.random.seed(19680801)
    data = np.random.randn(30, 30)
    n = len(colormaps)
    fig, axs = plt.subplots(1, n, figsize=(n * 2 + 2, 3),
                            constrained_layout=True, squeeze=False)
    for [ax, cmap] in zip(axs.flat, colormaps):
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-4, vmax=4)
        fig.colorbar(psm, ax=ax)
    plt.show()
