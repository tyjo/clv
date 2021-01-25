import matplotlib
matplotlib.use('agg')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
import sys

import matplotlib.gridspec as gridspec

def plot_error_by_time(ax, y_true, y_pred_clv, y_pred2, time):
    errors_clv = np.zeros(time.size)
    errors_pred2 = np.zeros(time.size)
    y_true = np.copy(y_true) / y_true.sum(axis=1,keepdims=True)

    y_pred_clv = np.copy(y_pred_clv) / y_pred_clv.sum(axis=1,keepdims=True)
    errors_clv[1:] = np.square(y_true[1:] - y_pred_clv[1:]).sum(axis=1)

    y_pred2 = np.copy(y_pred2) / y_pred2.sum(axis=1,keepdims=True)
    errors_pred2[1:] = np.square(y_true[1:] - y_pred2[1:]).sum(axis=1)

    ax.scatter(time[1:], errors_pred2[1:] - errors_clv[1:])
    ax.axhline(y=0, linestyle=":", color="black", linewidth=0.5)



def plot_trajectory(ax, y, time):
    T = y.shape[0]
    cm = plt.get_cmap("tab20c")
    colors = [cm(i) for i in range(20)]
    widths = np.concatenate((time[1:] - time[:-1], [1])).astype(float)
    widths[widths > 1] = 1

    y = np.copy(y)
    y[1:] /= y[1:].sum(axis=1, keepdims=True)
    # for predictions, initial time is set to zero
    if y[0].sum() != 0:
        y[0] /= y[0].sum()
    ax.bar(time, y[:,0], width=widths, color=colors[0], align="edge")
    for j in range(1, y.shape[1]):
        ax.bar(time, y[:,j], bottom=y[:,:j].sum(axis=1), width=widths, color=colors[j], align="edge")
    

def plot_bucci_cdiff():
    models = ["clv", "alr", "lra", "glv", "glv-ra"]

    true_Y = pkl.load(open("data/bucci/Y_cdiff-denoised.pkl", "rb"))
    true_T = pkl.load(open("data/bucci/T_cdiff.pkl", "rb"))

    sample_size = len(true_Y)
    scale_x = 3
    scale_y = 2
    #fig,ax = plt.subplots(nrows=6,ncols=sample_size, figsize=(6*scale_x, sample_size*scale_y))
    fig = plt.figure(figsize=(6*scale_x, sample_size*scale_y))
    gs = gridspec.GridSpec(10, sample_size,
                          height_ratios=[1, 1, 0.75, 1, 0.75, 1, 0.75, 1, 0.75, 1])

    for i in range(sample_size):
        parameter_filename = "pub-results/bucci_cdiff_predictions-{}".format(i)
        pred_clv = pkl.load(open(parameter_filename + "-clv", "rb"))
        pred_alr = pkl.load(open(parameter_filename + "-alr", "rb"))
        pred_lra = pkl.load(open(parameter_filename + "-lra", "rb"))
        pred_glv = pkl.load(open(parameter_filename + "-glv", "rb"))
        pred_glv_ra = pkl.load(open(parameter_filename + "-glv-ra", "rb"))


        ax_true = plt.subplot(gs[0,i])
        plot_trajectory(ax_true, true_Y[i], true_T[i])
        plt.subplot(gs[0,0]).set_ylabel("Truth")

        ax_clv = plt.subplot(gs[1,i])
        plot_trajectory(ax_clv, pred_clv[0], true_T[i])
        plt.subplot(gs[1,0]).set_ylabel("cLV")

        ax_glv = plt.subplot(gs[3,i])
        plot_trajectory(ax_glv, pred_glv[0], true_T[i])
        plt.subplot(gs[3,0]).set_ylabel("gLV$_{abs}$")

        ax_glv_ra = plt.subplot(gs[5,i])
        plot_trajectory(ax_glv_ra, pred_glv_ra[0], true_T[i])
        plt.subplot(gs[5,0]).set_ylabel("gLV$_{rel}$")

        ax_alr = plt.subplot(gs[7,i])
        plot_trajectory(ax_alr, pred_alr[0], true_T[i])
        plt.subplot(gs[7,0]).set_ylabel("ALR")

        ax_lra = plt.subplot(gs[9,i])
        plot_trajectory(ax_lra, pred_lra[0], true_T[i])
        plt.subplot(gs[9,0]).set_ylabel("linear")


        ax_clv_glv = plt.subplot(gs[2,i])
        plot_error_by_time(ax_clv_glv, true_Y[i], pred_clv[0], pred_glv[0], true_T[i])
        plt.subplot(gs[2,0]).set_ylabel("SE(gLV$_{abs})$ - \nSE(cLV)", fontsize=6)

        ax_clv_glv_ra = plt.subplot(gs[4,i])
        plot_error_by_time(ax_clv_glv_ra, true_Y[i], pred_clv[0], pred_glv_ra[0], true_T[i])
        plt.subplot(gs[4,0]).set_ylabel("SE(gLV$_{rel})$ - \nSE(cLV)", fontsize=6)

        ax_clv_alr = plt.subplot(gs[6,i])
        plot_error_by_time(ax_clv_alr, true_Y[i], pred_clv[0], pred_alr[0], true_T[i])
        plt.subplot(gs[6,0]).set_ylabel("SE(ALR) - \nSE(cLV)", fontsize=6)

        ax_clv_lra = plt.subplot(gs[8,i])
        plot_error_by_time(ax_clv_lra, true_Y[i], pred_clv[0], pred_lra[0], true_T[i])
        plt.subplot(gs[8,0]).set_ylabel("SE(linear) - \nSE(cLV)", fontsize=6)


    plt.tight_layout()
    plt.savefig("plots/cdiff-preds.pdf")


def plot_bucci_diet():
    models = ["clv", "alr", "lra", "glv", "glv-ra"]

    true_Y = pkl.load(open("data/bucci/Y_diet.pkl", "rb"))
    true_T = pkl.load(open("data/bucci/T_diet.pkl", "rb"))

    sample_size = len(true_Y)
    scale_x = 3
    scale_y = 2
    #fig,ax = plt.subplots(nrows=6,ncols=sample_size, figsize=(6*scale_x, sample_size*scale_y))
    fig = plt.figure(figsize=(6*scale_x, sample_size*scale_y))
    gs = gridspec.GridSpec(10, sample_size,
                          height_ratios=[1, 1, 0.75, 1, 0.75, 1, 0.75, 1, 0.75, 1])

    for i in range(sample_size):
        parameter_filename = "pub-results/bucci_diet_predictions-{}".format(i)
        pred_clv = pkl.load(open(parameter_filename + "-clv", "rb"))
        pred_alr = pkl.load(open(parameter_filename + "-alr", "rb"))
        pred_lra = pkl.load(open(parameter_filename + "-lra", "rb"))
        pred_glv = pkl.load(open(parameter_filename + "-glv", "rb"))
        pred_glv_ra = pkl.load(open(parameter_filename + "-glv-ra", "rb"))

        ax_true = plt.subplot(gs[0,i])
        plot_trajectory(ax_true, true_Y[i], true_T[i])
        plt.subplot(gs[0,0]).set_ylabel("Truth")

        ax_clv = plt.subplot(gs[1,i])
        plot_trajectory(ax_clv, pred_clv[0], true_T[i])
        plt.subplot(gs[1,0]).set_ylabel("cLV")

        ax_glv = plt.subplot(gs[3,i])
        plot_trajectory(ax_glv, pred_glv[0], true_T[i])
        plt.subplot(gs[3,0]).set_ylabel("gLV$_{abs}$")

        ax_glv_ra = plt.subplot(gs[5,i])
        plot_trajectory(ax_glv_ra, pred_glv_ra[0], true_T[i])
        plt.subplot(gs[5,0]).set_ylabel("gLV$_{rel}$")

        ax_alr = plt.subplot(gs[7,i])
        plot_trajectory(ax_alr, pred_alr[0], true_T[i])
        plt.subplot(gs[7,0]).set_ylabel("ALR")

        ax_lra = plt.subplot(gs[9,i])
        plot_trajectory(ax_lra, pred_lra[0], true_T[i])
        plt.subplot(gs[9,0]).set_ylabel("linear")


        ax_clv_glv = plt.subplot(gs[2,i])
        plot_error_by_time(ax_clv_glv, true_Y[i], pred_clv[0], pred_glv[0], true_T[i])
        plt.subplot(gs[2,0]).set_ylabel("SE(gLV$_{abs})$ - \nSE(cLV)", fontsize=6)

        ax_clv_glv_ra = plt.subplot(gs[4,i])
        plot_error_by_time(ax_clv_glv_ra, true_Y[i], pred_clv[0], pred_glv_ra[0], true_T[i])
        plt.subplot(gs[4,0]).set_ylabel("SE(gLV$_{rel})$ - \nSE(cLV)", fontsize=6)

        ax_clv_alr = plt.subplot(gs[6,i])
        plot_error_by_time(ax_clv_alr, true_Y[i], pred_clv[0], pred_alr[0], true_T[i])
        plt.subplot(gs[6,0]).set_ylabel("SE(ALR) - \nSE(cLV)", fontsize=6)

        ax_clv_lra = plt.subplot(gs[8,i])
        plot_error_by_time(ax_clv_lra, true_Y[i], pred_clv[0], pred_lra[0], true_T[i])
        plt.subplot(gs[8,0]).set_ylabel("SE(linear) - \nSE(cLV)", fontsize=6)


    plt.tight_layout()
    plt.savefig("plots/diet-preds.pdf")


def plot_stein_antibiotic():
    models = ["clv", "alr", "lra", "glv", "glv-ra"]

    true_Y = pkl.load(open("pub-results/stein/Y.pkl", "rb"))
    true_T = pkl.load(open("pub-results/stein/T.pkl", "rb"))

    sample_size = len(true_Y)
    scale_x = 4
    scale_y = 2
    #fig,ax = plt.subplots(nrows=6,ncols=sample_size, figsize=(6*scale_x, sample_size*scale_y))
    fig = plt.figure(figsize=(6*scale_x, sample_size*scale_y))
    gs = gridspec.GridSpec(10, sample_size,
                          height_ratios=[1, 1, 0.75, 1, 0.75, 1, 0.75, 1, 0.75, 1])

    for i in range(sample_size):
        parameter_filename = "pub-results/stein_prediction_parameters-{}".format(i)
        pred_clv = pkl.load(open(parameter_filename + "-clv", "rb"))
        pred_alr = pkl.load(open(parameter_filename + "-alr", "rb"))
        pred_lra = pkl.load(open(parameter_filename + "-lra", "rb"))
        pred_glv = pkl.load(open(parameter_filename + "-glv", "rb"))
        pred_glv_ra = pkl.load(open(parameter_filename + "-glv-ra", "rb"))

        ax_true = plt.subplot(gs[0,i])
        plot_trajectory(ax_true, true_Y[i], true_T[i])
        plt.subplot(gs[0,0]).set_ylabel("Truth")

        ax_clv = plt.subplot(gs[1,i])
        plot_trajectory(ax_clv, pred_clv[0], true_T[i])
        plt.subplot(gs[1,0]).set_ylabel("cLV")

        ax_glv = plt.subplot(gs[3,i])
        plot_trajectory(ax_glv, pred_glv[0], true_T[i])
        plt.subplot(gs[3,0]).set_ylabel("gLV$_{abs}$")

        ax_glv_ra = plt.subplot(gs[5,i])
        plot_trajectory(ax_glv_ra, pred_glv_ra[0], true_T[i])
        plt.subplot(gs[5,0]).set_ylabel("gLV$_{rel}$")

        ax_alr = plt.subplot(gs[7,i])
        plot_trajectory(ax_alr, pred_alr[0], true_T[i])
        plt.subplot(gs[7,0]).set_ylabel("ALR")

        ax_lra = plt.subplot(gs[9,i])
        plot_trajectory(ax_lra, pred_lra[0], true_T[i])
        plt.subplot(gs[9,0]).set_ylabel("linear")


        ax_clv_glv = plt.subplot(gs[2,i])
        plot_error_by_time(ax_clv_glv, true_Y[i], pred_clv[0], pred_glv[0], true_T[i])
        plt.subplot(gs[2,0]).set_ylabel("SE(gLV$_{abs})$ - \nSE(cLV)", fontsize=6)

        ax_clv_glv_ra = plt.subplot(gs[4,i])
        plot_error_by_time(ax_clv_glv_ra, true_Y[i], pred_clv[0], pred_glv_ra[0], true_T[i])
        plt.subplot(gs[4,0]).set_ylabel("SE(gLV$_{rel})$ - \nSE(cLV)", fontsize=6)

        ax_clv_alr = plt.subplot(gs[6,i])
        plot_error_by_time(ax_clv_alr, true_Y[i], pred_clv[0], pred_alr[0], true_T[i])
        plt.subplot(gs[6,0]).set_ylabel("SE(ALR) - \nSE(cLV)", fontsize=6)

        ax_clv_lra = plt.subplot(gs[8,i])
        plot_error_by_time(ax_clv_lra, true_Y[i], pred_clv[0], pred_lra[0], true_T[i])
        plt.subplot(gs[8,0]).set_ylabel("SE(linear) - \nSE(cLV)", fontsize=6)


    plt.tight_layout()
    plt.savefig("plots/antibiotic-preds.pdf")


if __name__ == "__main__":
    plot_bucci_cdiff()

    plot_bucci_diet()

    plot_stein_antibiotic()