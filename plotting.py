import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle as pkl
import seaborn as sns
import sys

from scipy.stats import pearsonr

sys.path.append("./statannot/statannot")
import statannot
add_stat_annotation = statannot.add_stat_annotation


def plot_trajectories(Y, T, output_dir, outfile):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    def plot_bar(ax, y, time, unique_color_id, remaining_ids):
        T = y.shape[0]
        cm = plt.get_cmap("tab20c")
        colors = [cm(i) for i in range(20)]
        #time = np.array([t for t in range(T)])
        widths = np.concatenate((time[1:] - time[:-1], [1])).astype(float)
        widths[widths > 1] = 1

        widths -= 1e-1

        y_colors = y[:,unique_color_id]
        ax.bar(time, y_colors[:,0], width=widths, color=colors[0], align="edge")
        for j in range(1, y_colors.shape[1]):
            ax.bar(time, y_colors[:,j], bottom=y_colors[:,:j].sum(axis=1), width=widths, color=colors[j], align="edge")
        
        ax.bar(time, y[:,remaining_ids].sum(axis=1), bottom=y_colors.sum(axis=1), width=widths, color=colors[19], align="edge")
        #ax.set_title("Relative Abundances", fontsize=10)
        #ax.legend(prop={"size" : 4}, bbox_to_anchor=[-0.1,1.225], loc="upper left", ncol=4)

    def find_top_ids(Y, n):
        ntaxa = Y[0].shape[1]
        rel_abun = np.zeros(ntaxa)
        for y in Y:
            tpts = y.shape[0]
            denom = y.sum(axis=1,keepdims=True)
            denom[denom == 0] = 1
            p = y / denom
            rel_abun += p.sum(axis=0) / tpts
        ids = np.argsort(-rel_abun)
        return np.sort(ids[:n]), np.sort(ids[n:])

    N = len(Y)
    top19_ids, remaining_ids = find_top_ids(Y, 19)
    fig, ax = plt.subplots(nrows=N,ncols=1,figsize=(7*N,5*N))
    for i in range(N):
        ax = ax[i] if N > 1 else ax
        denom = Y[i].sum(axis=1)
        denom[denom == 0] = 1
        plot_bar(ax, (Y[i].T / denom).T, T[i], top19_ids, remaining_ids)


    outfile = os.path.splitext(outfile)[0]
    plt.tight_layout()
    plt.savefig(output_dir + "/" + outfile + ".pdf")
    plt.close()


def plot_bar(ax, y, time):
    cm = plt.get_cmap("tab20c")
    colors = [cm(i) for i in range(20)]
    width = 1
    ax.bar(time, y[:,0], width=width, color=colors[0])
    for j in range(1, y.shape[1]):
        sign = np.sign(y[:,j])
        bottom = np.copy(y[:,:j])
        #bottom[ (np.sign(bottom).T != sign).T ] = 0
        #ax.bar(time, y[:,j], bottom=(1-sign)*y[:,:j].sum(axis=1), width=width, color=colors[j % 20])
        ax.bar(time, y[:,j], bottom=bottom.sum(axis=1), width=width, color=colors[j % 20])