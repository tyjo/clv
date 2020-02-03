import matplotlib
matplotlib.use('agg')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
import sys

from scipy.special import logsumexp
from wilcoxon_exact import wilcoxon_exact

from fit_models import fit_clv, fit_glv, fit_linear_alr, fit_linear_rel_abun
from plotting import plot_bar, plot_trajectories


def compute_errors(Y, Y_pred):
    def compute_square_errors(y, y_pred):
        err = []
        for yt, ypt in zip(y[1:], y_pred[1:]):
            err.append(np.square(yt - ypt).sum())
        return err
    err = []
    for y, y_pred in zip(Y, Y_pred):
        err += compute_square_errors(y, y_pred)
    return np.array(err)


def compute_baseline_errors(Y):
    Y_pred = []
    for y in Y:
        p0 = y[0] / y[0].sum()
        y_pred = np.array([p0 for t in range(y.shape[0])])
        Y_pred.append(y_pred)
    return compute_errors(Y, Y_pred)


def compute_errors_by_time(Y, Y_pred):
    error_by_time = []
    for y, y_pred in zip(Y, Y_pred):
        y_error = [ [0,np.nan] + (y[0]/y[0].sum()).tolist() + np.zeros(Y[0].shape[1]).tolist() ]
        for t in range(1, y_pred.shape[0]):
            err = np.square(y[t] -  y_pred[t]).sum()
            t_err = [t, err] + (y[t] / y[t].sum()).tolist() + y_pred[t].tolist()
            y_error.append(t_err)
        error_by_time.append(np.array(y_error))
    return error_by_time


def fit_model(Y, U, T, model):
    models = ["clv", "alr", "lra", "glv", "glv-ra"]
    if model not in ["clv", "alr", "lra", "glv", "glv-ra"]:
        print("model", (model),  "must be one of", models, file=sys.stderr)
        exit(1)

    folds = 7
    for fold in range(folds):
        print("running fold", fold)
        train_Y = []
        train_U = []
        train_T = []

        test_Y = []
        test_U = []
        test_T = []
        for i in range(len(Y)):
            if i % folds == fold:
                test_Y.append(Y[i])
                test_U.append(U[i])
                test_T.append(T[i])

            else:
                train_Y.append(Y[i])
                train_U.append(U[i])
                train_T.append(T[i])

        parameter_filename = "tmp/bucci_diet_predictions-{}".format(fold)


        if model == "clv":
            try:
                pred_clv = pkl.load(open(parameter_filename + "-clv", "rb"))
            except FileNotFoundError:
                pred_clv = fit_clv(train_Y, train_T, train_U, test_Y, test_T, test_U, folds=3)
                pkl.dump(pred_clv, open(parameter_filename + "-clv", "wb"))

        if model == "alr":
            try:
                pred_alr = pkl.load(open(parameter_filename + "-alr", "rb"))
            except FileNotFoundError:
                pred_alr = fit_linear_alr(train_Y, train_T, train_U, test_Y, test_T, test_U, folds=3)
                pkl.dump(pred_alr, open(parameter_filename + "-alr", "wb"))

        if model == "lra":
            try:
                pred_lra = pkl.load(open(parameter_filename + "-lra", "rb"))
            except FileNotFoundError:
                pred_lra = fit_linear_rel_abun(train_Y, train_T, train_U, test_Y, test_T, test_U, folds=3)
                pkl.dump(pred_lra, open(parameter_filename + "-lra", "wb"))

        if model == "glv":
            try:
                pred_glv = pkl.load(open(parameter_filename + "-glv", "rb"))
            except FileNotFoundError:
                pred_glv = fit_glv(train_Y, train_T, train_U, test_Y, test_T, test_U, folds=3)
                pkl.dump(pred_glv, open(parameter_filename + "-glv", "wb"))


        if model == "glv-ra":
            try:
                pred_glv_ra = pkl.load(open(parameter_filename + "-glv-ra", "rb"))
            except FileNotFoundError:
                pred_glv_ra = fit_glv(train_Y, train_T, train_U, test_Y, test_T, test_U, use_rel_abun=True, folds=3)
                pkl.dump(pred_glv_ra, open(parameter_filename + "-glv-ra", "wb"))


   
def prediction_experiment(Y, U, T):
    # plot fit on test data with cross validation
    baseline_err_cv = []
    en_err_cv = []
    linear_err_cv = []
    rel_abun_err_cv = []
    glv_err_cv = []
    glv_rel_abun_err_cv = []

    en_err_stratified_cv = []
    linear_err_stratified_cv = []
    rel_abun_err_stratified_cv = []
    glv_err_stratified_cv = []
    glv_rel_abun_err_stratified_cv = []

    folds = 7
    for fold in range(folds):
        print("running fold", fold)
        train_Y = []
        train_U = []
        train_T = []

        test_Y = []
        test_U = []
        test_T = []
        for i in range(len(Y)):
            if i % folds == fold:
                test_Y.append(Y[i])
                test_U.append(U[i])
                test_T.append(T[i])

            else:
                train_Y.append(Y[i])
                train_U.append(U[i])
                train_T.append(T[i])

        parameter_filename = "tmp_c2b2/bucci_diet_predictions-{}".format(fold)


        print("cLV")
        try:
            pred_clv = pkl.load(open(parameter_filename + "-clv", "rb"))
        except FileNotFoundError:
            pred_clv = fit_clv(train_Y, train_T, train_U, test_Y, test_T, test_U, folds=3)
            pkl.dump(pred_clv, open(parameter_filename + "-clv", "wb"))

        #plot_trajectories(pred_clv, test_T, "tmp_plots", "diet-clv-{}".format(fold))

        print("Linear ALR")
        try:
            pred_alr = pkl.load(open(parameter_filename + "-alr", "rb"))
        except FileNotFoundError:
            pred_alr = fit_linear_alr(train_Y, train_T, train_U, test_Y, test_T, test_U, folds=3)
            pkl.dump(pred_alr, open(parameter_filename + "-alr", "wb"))

        print("Linear Rel Abun")
        try:
            pred_lra = pkl.load(open(parameter_filename + "-lra", "rb"))
        except FileNotFoundError:
            pred_lra = fit_linear_rel_abun(train_Y, train_T, train_U, test_Y, test_T, test_U, folds=3)
            pkl.dump(pred_lra, open(parameter_filename + "-lra", "wb"))

        print("gLV")
        try:
            pred_glv = pkl.load(open(parameter_filename + "-glv", "rb"))
        except FileNotFoundError:
            pred_glv = fit_glv(train_Y, train_T, train_U, test_Y, test_T, test_U, folds=3)
            pkl.dump(pred_glv, open(parameter_filename + "-glv", "wb"))

        # print("gLV Rel Abun")
        # try:
        #     pred_glv_ra = pkl.load(open(parameter_filename + "-glv-ra", "rb"))
        # except FileNotFoundError:
        #     pred_glv_ra = fit_glv(train_Y, train_T, train_U, test_Y, test_T, test_U, use_rel_abun=True, folds=3)
        #     pkl.dump(pred_glv_ra, open(parameter_filename + "-glv-ra", "wb"))


        baseline_err_cv += [compute_baseline_errors(test_Y)]
        en_err_cv += [compute_errors(test_Y, pred_clv)]
        linear_err_cv += [compute_errors(test_Y, pred_alr)]
        rel_abun_err_cv += [compute_errors(test_Y, pred_lra)]
        glv_err_cv += [compute_errors(test_Y, pred_glv)]
        #glv_rel_abun_err_cv = [compute_errors(test_Y, pred_glv_ra)]


        en_err_stratified_cv += compute_errors_by_time(test_Y, pred_clv)
        linear_err_stratified_cv += compute_errors_by_time(test_Y, pred_alr)
        rel_abun_err_stratified_cv += compute_errors_by_time(test_Y, pred_lra)
        glv_err_stratified_cv += compute_errors_by_time(test_Y, pred_glv)
        #glv_rel_abun_err_stratified_cv += compute_errors_by_time(test_Y, pred_glv_ra)
    
    baseline = []
    linear = []
    rel_abun = []
    glv = []

    # compute p-values for difference in total error per sample
    baseline_sum = []
    linear_sum = []
    rel_abun_sum = []
    glv_sum = []
    clv_sum = []
    for cl,bl,ln,ra,gl in zip(en_err_cv, baseline_err_cv, linear_err_cv, rel_abun_err_cv, glv_err_cv):
        baseline += (bl - cl).tolist()
        linear += (ln - cl).tolist()
        rel_abun += (ra - cl).tolist()
        glv += (gl - cl).tolist()

        baseline_sum += [np.sum(bl)]
        linear_sum += [np.sum(ln)]
        rel_abun_sum += [np.sum(ra)]
        glv_sum += [np.sum(gl)]
        clv_sum += [np.sum(cl)]


    baseline = np.array(baseline)
    linear = np.array(linear)
    rel_abun = np.array(rel_abun)
    glv = np.array(glv)

    baseline_p = wilcoxon_exact(baseline_sum, clv_sum, alternative="greater")[1]
    linear_p = wilcoxon_exact(linear_sum, clv_sum, alternative="greater")[1]
    rel_abun_p = wilcoxon_exact(rel_abun_sum, clv_sum, alternative="greater")[1]
    glv_p = wilcoxon_exact(glv_sum, clv_sum, alternative="greater")[1]


    df = pd.DataFrame(np.array([baseline, glv, linear, rel_abun]).T,
                      columns=["baseline\n$p={:.3f}$".format(baseline_p),
                               "gLV\n$p={:.3f}$".format(glv_p),
                               "alr-linear\n$p={:.3f}$".format(linear_p),
                               "ra-linear\n$p={:.3f}$".format(rel_abun_p)])
    ax = df.boxplot(showmeans=True)
    ax.set_ylabel("Square Error(X) $-$ Square Error(cLV)")
    ax.set_title("Diet Dataset")

    plt.savefig("plots/bucci-diet_prediction-comparison.pdf")


    for idx, en_glv_linear_rel in enumerate(zip(en_err_stratified_cv, glv_err_stratified_cv, linear_err_stratified_cv, rel_abun_err_stratified_cv)):
        obs_dim = Y[0].shape[1]
        en, glv, linear, rel = en_glv_linear_rel
        fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(8,10))
        plot_bar(ax[0][0], en[:,2:(2+obs_dim)], en[:,0])
        ax[0][0].set_xticks(en[:0].tolist())
        ax[0][0].set_xticklabels(en[:0].tolist())
        ax[0][0].set_title("Truth")

        plot_bar(ax[0][1], en[:,(2+obs_dim):(2+2*obs_dim)], en[:,0])
        ax[0][1].set_xticks(en[:0].tolist())
        ax[0][1].set_xticklabels(en[:0].tolist())
        ax[0][1].set_title("cLV")

        plot_bar(ax[1][0], glv[:,(2+obs_dim):(2+2*obs_dim)], glv[:,0])
        ax[1][0].set_xticks(en[:0].tolist())
        ax[1][0].set_xticklabels(en[:0].tolist())
        ax[1][0].set_title("gLV")

        plot_bar(ax[2][0], linear[:,(2+obs_dim):(2+2*obs_dim)], linear[:,0])
        ax[2][0].set_xticks(en[:0].tolist())
        ax[2][0].set_xticklabels(en[:0].tolist())
        ax[2][0].set_title("alr-linear")

        plot_bar(ax[3][0], rel[:,(2+obs_dim):(2+2*obs_dim)], rel[:,0])
        ax[3][0].set_xticks(en[:0].tolist())
        ax[3][0].set_xticklabels(en[:0].tolist())
        ax[3][0].set_title("ra-linear")

        ax[1][1].scatter(glv[:,0], glv[:,1] - en[:,1] )
        ax[2][1].scatter(linear[:,0], linear[:,1] - en[:,1])
        ax[3][1].scatter(rel[:,0], rel[:,1] - en[:,1])

        ymin = np.min( np.array([glv[1:,1] - en[1:,1], linear[1:,1] - en[1:,1], rel[1:,1] - en[1:,1]]) ) - 0.15
        ymax = np.max( np.array([glv[1:,1] - en[1:,1], linear[1:,1] - en[1:,1], rel[1:,1] - en[1:,1]]) ) + 0.15
        for i in range(1, 4):
            ax[i][1].set_ylim(ymin, ymax)
            ax[i][1].set_xlim(ax[i][0].get_xlim())
            ax[i][1].axhline(y=0, linestyle=":", color="black", linewidth=0.5)
            ax[i][0].set_yticklabels([])

        ax[0][0].set_yticklabels([])
        ax[0][1].set_yticklabels([])

        ax[1][1].set_ylabel("Sqr Err(gLV) $-$ Sqr Err(cLV)", fontsize=9)
        ax[2][1].set_ylabel("Sqr Err(alr-linear) $-$ Sqr Err(cLV)", fontsize=9)
        ax[3][1].set_ylabel("Sqr Err(ra-linear) $-$ Sqr Err(cLV)", fontsize=9)

        plt.tight_layout()

        plt.savefig("plots/bucci-diet_prediction_comparison-test-{}.pdf".format(idx))

    return (baseline, baseline_p), (linear, linear_p), (rel_abun, rel_abun_p), (glv, glv_p)


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

    Y_adj = adjust_concentrations(Y)

    if len(sys.argv) < 2:
      print("USAGE: python bucci_diet_prediction_experiments.py [MODEL]")
    model = sys.argv[1]
    fit_model(Y_adj, U, T, model)
    #prediction_experiment(Y_adj, U, T)