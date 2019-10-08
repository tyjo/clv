import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl

from scipy.special import logsumexp
from scipy.stats import entropy
from scipy.stats import pearsonr
from scipy.stats import norm

from optimizers import elastic_net, elastic_net_lotka_volterra, least_squares_latent
from stein_cross_validation import *

pseudo_count = 1e-6


def plot_corr(A_stein, g_stein, B_stein, A_est, g_est, B_est, filename):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3))

    A = np.array([A_stein.flatten(), A_est.flatten()]).T
    B = np.array([B_stein.flatten(), B_est.flatten()]).T
    g = np.array([g_stein.flatten(), g_est.flatten()]).T

    df_A = pd.DataFrame(A, columns=["gLV $a_{ij} - a_{iD}$", r"cLV $\overline{a}_{ij}$"])
    df_B = pd.DataFrame(B, columns=["gLV $b_{ip} - b_{iD}$", r"cLV $\overline{b}_{ip}$"])
    df_g = pd.DataFrame(g, columns=["gLV $g_i - g_D$", r"cLV $\overline{g}_i$"])

    df_A.plot.scatter(ax=ax[0], x="gLV $a_{ij} - a_{iD}$", y=r"cLV $\overline{a}_{ij}$")
    ax[0].set_xlim(-3, 3)
    ax[0].set_xticks([-2, 0, 2])
    ax[0].set_ylim(-3, 3)
    ax[0].set_yticks([-2, 0, 2])
    ax[0].set_title("Interactions")
    m,b = np.polyfit(A[:,0], A[:,1], deg=1)
    x = np.linspace(-4, 3, 3)
    y = m*x + b
    handle = ax[0].plot(x, y, label="R = {}".format(np.round(pearsonr(A[:,0], A[:,1])[0], 3)), linestyle="--", color="C3")
    ax[0].legend(handles=handle)

    df_B.plot.scatter(ax=ax[1], x="gLV $b_{ip} - b_{iD}$", y=r"cLV $\overline{b}_{ip}$")
    ax[1].set_xlim(-2.2, 6.2)
    ax[1].set_xticks([-2, 0, 2, 4, 6])
    ax[1].set_ylim(-2.2, 6.2)
    ax[1].set_yticks([-2, 0, 2, 4, 6])
    ax[1].set_title("Antibiotics")
    m,b = np.polyfit(B[:,0], B[:,1], deg=1)
    x = np.linspace(-2, 9, 3)
    y = m*x + b
    handle = ax[1].plot(x, y, label="R = {}".format(np.round(pearsonr(B[:,0], B[:,1])[0], 3)), linestyle="--", color="C3")
    ax[1].legend(handles=handle, loc="upper left")


    df_g.plot.scatter(ax=ax[2], x="gLV $g_i - g_D$", y=r"cLV $\overline{g}_i$")
    ax[2].set_xlim(-0.5, 0.5)
    ax[2].set_xticks([-0.4, -0.2, 0, 0.2, 0.4])
    ax[2].set_ylim(-0.5, 0.5)
    ax[2].set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
    ax[2].set_title("Growth")
    m,b = np.polyfit(g[:,0], g[:,1], deg=1)
    x = np.linspace(-2, 9, 3)
    y = m*x + b
    handle = ax[2].plot(x, y, label="R = {}".format(np.round(pearsonr(g[:,0], g[:,1])[0], 3)), linestyle="--", color="C3")
    ax[2].legend(handles=handle, loc="upper left")

    #plt.tight_layout()
    plt.subplots_adjust(left=0.075, right=0.95, bottom=0.2, wspace=0.35, hspace=0.35)
    plt.savefig(filename)


def compute_relative_parameters(A_abs, g_abs, B_abs):
    last_dim = A_abs.shape[1]-1
    A_rel = A_abs[:last_dim,] - A_abs[last_dim]
    B_rel = B_abs[:last_dim,] - B_abs[last_dim]
    g_rel = g_abs[:last_dim] - g_abs[last_dim]

    return A_rel, g_rel, B_rel


if __name__ == "__main__":
    Y = pkl.load(open("data/stein/Y.pkl", "rb"))
    U = pkl.load(open("data/stein/U.pkl", "rb"))
    T = pkl.load(open("data/stein/T.pkl", "rb"))
    
    alpha = 1
    r_A = 0
    r_g = 0
    r_B = 0

    X = estimate_log_space_from_observations(Y)
    A_abs, g_abs, B_abs = elastic_net_lotka_volterra(X, U, T, np.eye(Y[0].shape[1]), r_A, r_g, r_B, alpha)
    X = estimate_latent_from_observations(Y)
    A_en, g_en, B_en = elastic_net(X, U, T, np.eye(Y[0].shape[1]-1), r_A, r_g, r_B)

    A_rel, g_rel, B_rel = compute_relative_parameters(A_abs, g_abs, B_abs)

    plot_corr(A_rel, g_rel, B_rel, A_en, g_en, B_en, "plots/stein_correlation.pdf")
