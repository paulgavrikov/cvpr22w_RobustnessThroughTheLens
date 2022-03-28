import matplotlib.pyplot as plt
import numpy as np

def set_box_color(bp, color):
    for patch in bp["boxes"]:
        patch.set(facecolor=color)

    plt.setp(bp["medians"], color="black")
    plt.setp(bp["means"], color="black")


def boxplots(selectors, labels=None, ax=None, hide_ticks=False):
    plt.rcParams['xtick.major.size'] = 3.5
    if ax is None:
        plt.figure(facecolor="white")
        ax = plt.gca()
        
    n = len(selectors)
    w = 1 / n
    p = 0.1
    
    for i, selector in enumerate(selectors):
        df = selector
        pos = np.array(range(len(df.values)))*n - w/2 - p + (2*p+w)*i
            
        bp = ax.boxplot(df.values, positions=pos, widths=w, showmeans=True, showfliers=True, patch_artist=True, meanprops={"markerfacecolor": "black", "markeredgecolor":"black", "markersize":4}, flierprops = dict(marker='.', markersize=4))
        color = f"C{i%9}"
        set_box_color(bp, color)
        if labels:
            ax.plot([], c=color, linewidth=5, label=labels[i])
            
    if not hide_ticks:
        ax.set_xticks(np.arange(0, len(df.index) * n, n), df.index, rotation=22.5)
        ax.set_xlabel("convolution depth decile")
    else:
        ax.set_xticks(np.arange(0, len(df.index) * n, n), [], rotation=22.5)
        ax.tick_params(length=0)
        
    ax.grid() #axis="y"
    return df