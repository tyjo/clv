import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl

from scipy.special import logsumexp
from wilcoxon_exact import wilcoxon_exact

from bucci_cross_validation import *


def compute_errors(Y, U, T, A, g, B):
    def compute_square_errors(y, y_pred):
        err = []
        for yt, ypt in zip(y[1:], y_pred[1:]):
            err.append(compute_square_error(yt, ypt))
        return err
    err = []
    for y, u, t in zip(Y, U, T):
        y_pred = predict(y[0], u, t, A, g, B)
        err += compute_square_errors(y, y_pred)
    return np.array(err)


def compute_errors_linear(Y, U, T, A, g, B):
    def compute_square_errors(y, y_pred):
        err = []
        for yt, ypt in zip(y[1:], y_pred[1:]):
            err.append(compute_square_error(yt, ypt))
        return err
    err = []
    for y, u, t in zip(Y, U, T):
        y_pred = predict_linear(y[0], u, t, A, g, B)
        err += compute_square_errors(y, y_pred)
    return np.array(err)


def compute_errors_rel_abun(Y, U, T, A, g, B):
    def compute_square_errors(y, y_pred):
        err = []
        for yt, ypt in zip(y[1:], y_pred[1:]):
            err.append(compute_square_error(yt, ypt))
        return err
    err = []
    for y, u, t in zip(Y, U, T):
        y_pred = predict_rel_abun(y[0], u, t, A, g, B)
        err += compute_square_errors(y, y_pred)
    return np.array(err)


def compute_baseline_errors(Y):
    def compute_square_errors(y, y_pred):
        err = []
        for yt in y[1:]:
            err.append(compute_square_error(yt, y_pred))
        return err
    err = []
    for y in Y:
        y_pred = y[0] / y[0].sum()
        err += compute_square_errors(y, y_pred)
    return np.array(err)


def compute_errors_glv(Y, U, T, A, g, B):
    def compute_square_errors(y, y_pred):
        err = []
        for yt, ypt in zip(y[1:], y_pred[1:]):
            err.append(compute_square_error(yt, ypt))
        return err
    err = []
    for y, u, t in zip(Y, U, T):
        y_pred = predict_glv(y[0], u, t, A, g, B)

        if not np.all(np.isfinite(y_pred)):
            return np.inf
        err += compute_square_errors(y, y_pred)
    return np.array(err)


def compute_errors_by_time(Y, U, T, A, g, B):
    error_by_time = []
    for y, u, t in zip(Y, U, T):
        y_pred = predict(y[0], u, t, A, g, B)
        y_error = [ [0,np.nan] + (y[0]/y[0].sum()).tolist() + np.zeros(Y[0].shape[1]).tolist() ]
        for t in range(1, y_pred.shape[0]):
            err = compute_square_error(y[t], y_pred[t])
            t_err = [t, err] + (y[t] / y[t].sum()).tolist() + y_pred[t].tolist()
            y_error.append(t_err)
        error_by_time.append(np.array(y_error))
    return error_by_time


def compute_errors_by_time_linear(Y, U, T, A, g, B):
    error_by_time = []
    for y, u, t in zip(Y, U, T):
        y_pred = predict_linear(y[0], u, t, A, g, B)
        y_error = [ [0,np.nan] + (y[0]/y[0].sum()).tolist() + np.zeros(Y[0].shape[1]).tolist() ]
        for t in range(1, y_pred.shape[0]):
            err = compute_square_error(y[t], y_pred[t])
            t_err = [t, err] + (y[t] / y[t].sum()).tolist() + y_pred[t].tolist()
            y_error.append(t_err)
        error_by_time.append(np.array(y_error))
    return error_by_time


def compute_errors_by_time_rel_abun(Y, U, T, A, g, B):
    error_by_time = []
    for y, u, t in zip(Y, U, T):
        y_pred = predict_rel_abun(y[0], u, t, A, g, B)
        y_error = [ [0,np.nan] + (y[0]/y[0].sum()).tolist() + np.zeros(Y[0].shape[1]).tolist() ]
        for t in range(1, y_pred.shape[0]):
            err = compute_square_error(y[t], y_pred[t])
            t_err = [t, err] + (y[t] / y[t].sum()).tolist() + y_pred[t].tolist()
            y_error.append(t_err)
        error_by_time.append(np.array(y_error))
    return error_by_time


def compute_errors_by_time_glv(Y, U, T, A, g, B):
    error_by_time = []
    for y, u, t in zip(Y, U, T):
        y_pred = predict_glv(y[0], u, t, A, g, B)
        y_error = [ [0,np.nan] + (y[0]/y[0].sum()).tolist() + np.zeros(Y[0].shape[1]).tolist() ]
        for t in range(1, y_pred.shape[0]):
            err = compute_square_error(y[t], y_pred[t])
            t_err = [t, err] + (y[t] / y[t].sum()).tolist() + (y_pred[t]).tolist()
            y_error.append(t_err)
        error_by_time.append(np.array(y_error))
    return error_by_time



def prediction_experiment(Y, U, T):
    # plot fit on test data with cross validation
    baseline_err_cv = []
    en_err_cv = []
    linear_err_cv = []
    rel_abun_err_cv = []
    glv_err_cv = []

    en_err_stratified_cv = []
    linear_err_stratified_cv = []
    rel_abun_err_stratified_cv = []
    glv_err_stratified_cv = []

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

        parameter_filename = "tmp/bucci_diet_prediction_parameters-{}".format(fold)

        try:
            A_e, g_e, B_e, A_ln, g_ln, B_ln, A_ra, g_ra, B_ra = pkl.load(open(parameter_filename, "rb"))
        
        except FileNotFoundError:
            train_X = estimate_latent_from_observations(train_Y)
            Q_inv = np.eye(train_X[0].shape[1])

            alpha, r_A, r_g, r_B = estimate_elastic_net_regularizers_cv(train_Y, train_U, train_T, folds=6)
            A_e, g_e, B_e = elastic_net(train_X, train_U, train_T, Q_inv, r_A, r_g, r_B, alpha, tol=1e-3)
            
            alpha, r_A, r_g, r_B = estimate_elastic_net_regularizers_cv_linear(train_Y, train_U, train_T, folds=6)
            A_ln, g_ln, B_ln = elastic_net_linear(train_X, train_U, train_T, Q_inv, r_A, r_g, r_B, alpha, tol=1e-3)

            train_X_rel_abun = estimate_rel_abun(train_Y)
            Q_inv = np.eye(train_X_rel_abun[0].shape[1])
            alpha, r_A, r_g, r_B = estimate_elastic_net_regularizers_cv_rel_abun(train_Y, train_U, train_T, folds=6)
            A_ra, g_ra, B_ra = elastic_net_linear(train_X_rel_abun, train_U, train_T, Q_inv, r_A, r_g, r_B, alpha, tol=1e-3)

            pkl.dump((A_e, g_e, B_e, A_ln, g_ln, B_ln, A_ra, g_ra, B_ra), open(parameter_filename, "wb"))

        baseline_err_cv += [compute_baseline_errors(test_Y)]
        en_err_cv += [compute_errors(test_Y, test_U, test_T, A_e, g_e, B_e)]
        linear_err_cv += [compute_errors_linear(test_Y, test_U, test_T, A_ln, g_ln, B_ln)]
        rel_abun_err_cv += [compute_errors_rel_abun(test_Y, test_U, test_T, A_ra, g_ra, B_ra)]

        en_err_stratified_cv += compute_errors_by_time(test_Y, test_U, test_T, A_e, g_e, B_e)
        linear_err_stratified_cv += compute_errors_by_time_linear(test_Y, test_U, test_T, A_ln, g_ln, B_ln)
        rel_abun_err_stratified_cv += compute_errors_by_time_rel_abun(test_Y, test_U, test_T, A_ra, g_ra, B_ra)

        parameter_filename = "tmp/bucci_diet_lotka_volterra_parameters-{}".format(fold)

        try:
            A_s, g_s, B_s = pkl.load(open(parameter_filename, "rb"))
        
        except FileNotFoundError:
            train_X_log = estimate_log_space_from_observations(train_Y)
            Q_inv = np.eye(train_X_log[0].shape[1])
            alpha, r_A2, r_g2, r_B2 = estimate_elastic_net_regularizers_cv_lotka_volterra(train_Y, train_U, train_T, folds=6)
            A_s, g_s, B_s = elastic_net_lotka_volterra(train_X_log, train_U, train_T, Q_inv, r_A2, r_g2, r_B2, alpha, tol=1e-3)

            pkl.dump((A_s, g_s, B_s), open(parameter_filename, "wb"))

        glv_err_cv += [compute_errors_glv(test_Y, test_U, test_T, A_s, g_s, B_s)]
        glv_err_stratified_cv += compute_errors_by_time_glv(test_Y, test_U, test_T, A_s, g_s, B_s)

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


if __name__ == "__main__":
    Y = pkl.load(open("data/bucci/Y_diet.pkl", "rb"))
    U = pkl.load(open("data/bucci/U_diet.pkl", "rb"))
    T = pkl.load(open("data/bucci/T_diet.pkl", "rb"))

    prediction_experiment(Y, U, T)



