import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
import seaborn as sns

from compositional_lotka_volterra import CompositionalLotkaVolterra
from generalized_lotka_volterra import GeneralizedLotkaVolterra

def swap_denom(A, g, B, new_denom, column_names):
    numer = np.array([i for i in range(A.shape[0]) if i != new_denom])
    A_tmp = A[new_denom,:]
    A_new = np.zeros(A.shape)
    A_new[numer,:] = A[numer,:] - A[new_denom,:]
    A_new[new_denom,:] = - A_tmp

    g_tmp = g[new_denom]
    g_new = np.zeros(g.shape)
    g_new[numer] = g[numer] - g[new_denom]
    g[new_denom] = - g_tmp

    B_tmp = B[new_denom,:]
    B_new = np.zeros(B.shape)
    B_new[numer,:] = B[numer,:] - B[new_denom,:]
    B_new[new_denom,:] = - B_tmp

    numer = np.array([i for i in range(A.shape[1]) if i != new_denom])
    new_dimensions = np.array(column_names)[numer]

    return A_new, g_new, B_new, new_dimensions

def choose_perturb_denom(P):
    """Pick a denominator for additive log-ratio transformation.
    """
    np.seterr(divide="ignore", invalid="ignore")
    log_change = None
    for p in P:
        s = p.sum(axis=1,keepdims=True)
        s[s==0] = 1
        deltas = np.log( (p/s)[1] ) - np.log( (p/s)[0] )
        if log_change is None:
            log_change = deltas
        else:
            log_change = np.vstack((log_change, deltas))
    
    np.seterr(divide="warn", invalid="warn")
    # pick taxon with smallest change in log proportion
    min_idx = -1
    min_var = np.inf
    ntaxa = log_change.shape[1]
    for i in range(ntaxa):
        if not np.all(np.isfinite(log_change[:,i])):
            continue
        var = np.var(log_change[:,i])
        if var < min_var:
            min_idx = i
            min_var = var

    if min_idx == -1:
        print("Error: no valid denominator found", file=sys.stderr)
        exit(1)

    return min_idx


def plot_heatmaps(A, B, taxon_row_names, taxon_col_names, antibiotic_row_names, antibiotic_col_names, suffix, which_plot="A"):

    if which_plot == "A":
        m = np.abs(A).max()
        ax = sns.heatmap(A, xticklabels=taxon_col_names, annot=True, annot_kws={"size": 7}, yticklabels=taxon_row_names, vmin=-m, vmax=m, cmap="coolwarm")
        plt.setp(ax.get_xticklabels(),fontsize=8)
        plt.setp(ax.get_yticklabels(),fontsize=8)
        plt.tight_layout()
        plt.tight_layout()
        plt.savefig("plots/A-heatmap-{}.pdf".format(suffix))
        plt.close()

    if which_plot == "B":
        fig, ax = plt.subplots(figsize=(3, 4))
        m = np.abs(B).max()
        sns.heatmap(B, ax=ax, xticklabels=antibiotic_col_names, annot=True, annot_kws={"size": 7}, yticklabels=antibiotic_row_names, vmin=-m, vmax=m, cmap="coolwarm")
        plt.setp(ax.get_xticklabels(),fontsize=8)
        plt.setp(ax.get_yticklabels(),fontsize=8)
        plt.tight_layout()
        plt.tight_layout()
        plt.savefig("plots/B-heatmap-{}.pdf".format(suffix))
        plt.close()



if __name__ == "__main__":
    Y = pkl.load(open("data/stein/Y.pkl", "rb"))
    U = pkl.load(open("data/stein/U.pkl", "rb"))
    T = pkl.load(open("data/stein/T.pkl", "rb"))

    col_names = np.array(['und. Enterobacteriaceae', 'Blautia', 'Barnesiella',
                 'und. uncl. Mollicutes',
                 'und. Lachnospiraceae', 'Akkermansia',
                 'C. difficile', 'uncl. Lachnospiraceae', 'Coprobacillus',
                 'Enterococcus', 'Other'])

    P = []
    held_out_rel_abun = []

    for y in Y:
        P.append(y / y.sum(axis=1,keepdims=True))

    clv = CompositionalLotkaVolterra(P, T, U)
    A = np.loadtxt("pub-results/stein_A")
    g = np.loadtxt("pub-results/stein_g")
    B = np.loadtxt("pub-results/stein_B")
    B = np.expand_dims(B, axis=1)

    glv = GeneralizedLotkaVolterra(P, T, U)
    A_glv = np.loadtxt("pub-results/stein_A_glv")
    g_glv = np.loadtxt("pub-results/stein_g_glv")
    B_glv = np.loadtxt("pub-results/stein_B_glv")
    B_glv = np.expand_dims(B_glv, axis=1)

    ntaxa = Y[0].shape[1]

    old_denom = clv.denom
    taxon_row_names = col_names[np.array([i for i in range(ntaxa) if i != old_denom])]
    antibiotics = ["Clindamycin"]
    plot_heatmaps(A, B, taxon_row_names, col_names, antibiotics, antibiotics, "clv-A", "A")

    perturb_denom = choose_perturb_denom(P)
    A_new, g_new, B_new, antibiotic_row_names = swap_denom(A, g, B, perturb_denom, col_names)
    plot_heatmaps(A_new, B_new, taxon_row_names, col_names, antibiotic_row_names, antibiotics, "clv-B", "B")

    plot_heatmaps(A_glv, B_glv, col_names, col_names, col_names, antibiotics, "glv-A", "A")
    plot_heatmaps(A_glv, B_glv, col_names, col_names, col_names, antibiotics, "glv-B", "B")

