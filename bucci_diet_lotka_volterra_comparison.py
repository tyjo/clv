import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl

from scipy.special import logsumexp
from scipy.stats import entropy
from scipy.stats import pearsonr
from scipy.stats import norm

from compositional_lotka_volterra import CompositionalLotkaVolterra
from generalized_lotka_volterra import GeneralizedLotkaVolterra, ridge_regression_glv


def compute_relative_parameters(A_abs, g_abs, B_abs, denom):
    numer = np.array([i for i in range(A_abs.shape[0]) if i != denom])
    A_rel = A_abs[numer,:] - A_abs[denom,:]
    B_rel = B_abs[numer,:] - B_abs[denom,:]
    g_rel = g_abs[numer] - g_abs[denom]

    return A_rel, g_rel, B_rel



def plot_corr(A_rel, g_rel, B_rel, A_est, g_est, B_est, filename):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3))

    A_nonzero = np.logical_and(A_rel != 0, A_est != 0)
    B_nonzero = np.logical_and(B_rel != 0, B_est != 0)
    g_nonzero = np.logical_and(g_rel != 0, g_est != 0)
    A_rel = A_rel[A_nonzero]
    B_rel = B_rel[B_nonzero]
    g_rel = g_rel[g_nonzero]

    A_est = A_est[A_nonzero]
    B_est = B_est[B_nonzero]
    g_est = g_est[g_nonzero]

    A = np.array([A_rel.flatten(), A_est.flatten()]).T
    B = np.array([B_rel.flatten(), B_est.flatten()]).T
    g = np.array([g_rel.flatten(), g_est.flatten()]).T

    df_A = pd.DataFrame(A, columns=["gLV $a_{ij} - a_{iD}$", r"cLV $\overline{a}_{ij}$"])
    df_B = pd.DataFrame(B, columns=["gLV $b_{ip} - b_{iD}$", r"cLV $\overline{b}_{ip}$"])
    df_g = pd.DataFrame(g, columns=["gLV $g_i - g_D$", r"cLV $\overline{g}_i$"])

    df_A.plot.scatter(ax=ax[0], x="gLV $a_{ij} - a_{iD}$", y=r"cLV $\overline{a}_{ij}$")
    ax[0].set_xlim(-8, 7)
    ax[0].set_xticks([-5, 0, 5])
    ax[0].set_ylim(-8, 7)
    ax[0].set_yticks([-5, 0, 5])
    ax[0].set_title("Interactions")
    m,b = np.polyfit(A[:,0], A[:,1], deg=1)
    x = np.linspace(-7, 7, 3)
    y = m*x + b
    handle = ax[0].plot(x, y, label="R = {}".format(np.round(pearsonr(A[:,0], A[:,1])[0], 3)), linestyle="--", color="C3")
    ax[0].legend(handles=handle, loc="upper left")

    df_B.plot.scatter(ax=ax[1], x="gLV $b_{ip} - b_{iD}$", y=r"cLV $\overline{b}_{ip}$")
    ax[1].set_xlim(-1.25, 1.5)
    ax[1].set_xticks([-1, 0, 1])
    ax[1].set_ylim(-1.25, 1.5)
    ax[1].set_yticks([-1, 0, 1])
    ax[1].set_title("Change in Diet")
    m,b = np.polyfit(B[:,0], B[:,1], deg=1)
    x = np.linspace(-2.5, 9, 3)
    y = m*x + b
    handle = ax[1].plot(x, y, label="R = {}".format(np.round(pearsonr(B[:,0], B[:,1])[0], 3)), linestyle="--", color="C3")
    ax[1].legend(handles=handle, loc="upper left")


    df_g.plot.scatter(ax=ax[2], x="gLV $g_i - g_D$", y=r"cLV $\overline{g}_i$")
    ax[2].set_xlim(-1, 1.1)
    ax[2].set_xticks([-0.75, 0, 0.75])
    ax[2].set_ylim(-1, 1.1)
    ax[2].set_yticks([-0.75, 0, 0.75])
    ax[2].set_title("Growth")
    m,b = np.polyfit(g[:,0], g[:,1], deg=1)
    x = np.linspace(-1.5, 1.5, 3)
    y = m*x + b
    handle = ax[2].plot(x, y, label="R = {}".format(np.round(pearsonr(g[:,0], g[:,1])[0], 3)), linestyle="--", color="C3")
    ax[2].legend(handles=handle, loc="upper left")

    #plt.tight_layout()
    plt.subplots_adjust(left=0.075, right=0.95, bottom=0.2, wspace=0.35, hspace=0.35)
    plt.savefig(filename)


def adjust_concentrations(Y):
    con =  []
    for y in Y:
        con += y.sum(axis=1).tolist()
    con = np.array(con)
    C = 1 / np.mean(con)

    Y_adjusted = []
    for y in Y:
        Y_adjusted.append(C*y)

    return Y_adjusted


if __name__ == "__main__":
    Y = pkl.load(open("data/bucci/Y_diet.pkl", "rb"))
    U = pkl.load(open("data/bucci/U_diet.pkl", "rb"))
    T = pkl.load(open("data/bucci/T_diet.pkl", "rb"))

    Y = adjust_concentrations(Y)

    # estimated previously
    r_A = 0.5
    r_g = 0.5
    r_B = 4

    P = []
    Y_pc = []
    log_Y = []
    for y in Y:
        mass = y.sum(axis=1)
        p = y / y.sum(axis=1,keepdims=True)
        p = (p + 1e-5) / (p + 1e-5).sum(axis=1,keepdims=True)
        P.append(p)
        Y_pc.append((mass.T*p.T).T)
        log_Y.append(np.log(mass.T*p.T).T)

    clv = CompositionalLotkaVolterra(P, T, U, pseudo_count=1e-5)
    clv.r_A = r_A
    clv.r_g = r_g
    clv.r_B = r_B
    clv.train_ridge()
    A_clv, g_clv, B_clv = clv.get_params()

    A_glv, g_glv, B_glv = ridge_regression_glv(log_Y, U, T, clv.r_A, clv.r_g, clv.r_B)

    A_rel, g_rel, B_rel = compute_relative_parameters(A_glv, g_glv, B_glv, clv.denom)
    plot_corr(A_rel, g_rel, B_rel, A_clv, g_clv, B_clv, "plots/bucci-diet_correlation.pdf")
