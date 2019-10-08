import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl

from scipy.special import logsumexp
from wilcoxon_exact import wilcoxon_exact

from taur_cross_validation import *
from stein_cross_validation import plot_bar
from optimizers import elastic_net, elastic_net_linear
from util import load_observations



def compute_errors(Y, U_normalized, T, A, g, B):
    def compute_square_errors(y, y_pred):
        err = []
        for yt, ypt in zip(y[1:], y_pred[1:]):
            err.append(np.square(yt / yt.sum() - ypt).sum())
        return err
    err = []
    for y, u, t in zip(Y, U_normalized, T):
        y_pred = predict(y[0], u, t, A, g, B)
        err += compute_square_errors(y, y_pred)
    return np.array(err)


def compute_errors_linear(Y, U_normalized, T, A, g, B):
    def compute_square_errors(y, y_pred):
        err = []
        for yt, ypt in zip(y[1:], y_pred[1:]):
            err.append(np.square(yt / yt.sum() - ypt).sum())
        return err
    err = []
    for y, u, t in zip(Y, U_normalized, T):
        y_pred = predict_linear(y[0], u, t, A, g, B)
        err += compute_square_errors(y, y_pred)
    return np.array(err)


def compute_errors_rel_abun(Y, U_normalized, T, A, g, B):
    def compute_square_errors(y, y_pred):
        err = []
        for yt, ypt in zip(y[1:], y_pred[1:]):
            err.append(np.square(yt / yt.sum() - ypt).sum())
        return err
    err = []
    for y, u, t in zip(Y, U_normalized, T):
        y_pred = predict_rel_abun(y[0], u, t, A, g, B)
        err += compute_square_errors(y, y_pred)
    return np.array(err)


def compute_baseline_errors(Y):
    def compute_square_errors(y, y_pred):
        err = []
        for yt in y[1:]:
            err.append(np.square(yt / yt.sum() - y_pred).sum())
        return err
    err = []
    for y in Y:
        y_pred = y[0] / y[0].sum()
        err += compute_square_errors(y, y_pred)
    return np.array(err)


def compute_errors_by_time(Y, U, T, A, g, B):
    error_by_time = []
    for y, u, t in zip(Y, U, T):
        y_pred = predict(y[0], u, t, A, g, B)
        y_error = [ [0,np.nan] + (y[0]/y[0].sum()).tolist() + np.zeros(Y[0].shape[1]).tolist() ]
        for t in range(1, y_pred.shape[0]):
            err = np.square(y_pred[t] - y[t] / y[t].sum()).sum()
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
            err = np.square(y_pred[t] - y[t] / y[t].sum()).sum()
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
            err = np.square(y_pred[t] - y[t] / y[t].sum()).sum()
            t_err = [t, err] + (y[t] / y[t].sum()).tolist() + y_pred[t].tolist()
            y_error.append(t_err)
        error_by_time.append(np.array(y_error))
    return error_by_time


def prediction_experiment(Y, U, T, U_normalized):
    # plot fit on test data with cross validation
    baseline_err_cv = []
    en_err_cv = []
    linear_err_cv = []
    rel_abun_err_cv = []

    en_err_stratified_cv = []
    linear_err_stratified_cv = []
    rel_abun_err_stratified_cv = []

    folds = 10
    for fold in range(folds):
        print("running fold", fold)
        train_Y = []
        train_U = []
        train_T = []
        train_U_normalized = []

        test_Y = []
        test_U = []
        test_T = []
        test_U_normalized = []
        for i in range(len(Y)):
            if i % folds == fold:
                test_Y.append(Y[i])
                test_U.append(U[i])
                test_T.append(T[i])

                test_U_normalized.append(U_normalized[i])
            else:
                train_Y.append(Y[i])
                train_U.append(U[i])
                train_T.append(T[i])

                train_U_normalized.append(U_normalized[i])

        try:
            A_e, g_e, B_e, A_ln, g_ln, B_ln, A_ra, g_ra, B_ra = pkl.load(open("tmp/taur_prediction_parameters-{}".format(fold), "rb"))
        
        except FileNotFoundError:
            train_X = estimate_latent_from_observations(train_Y)
            Q_inv = np.eye(train_X[0].shape[1])
            A_e, g_e, B_e = elastic_net(train_X, train_U, train_T, Q_inv, r_A=0.7, r_g=0.9, r_B=0, alpha=0.1, tol=1e-3)
            A_ln, g_ln, B_ln = elastic_net_linear(train_X, train_U, train_T, Q_inv, r_A=0, r_g=0, r_B=0.9, alpha=10, tol=1e-3)

            train_X_rel_abun = estimate_rel_abun(train_Y)
            Q_inv = np.eye(train_X_rel_abun[0].shape[1])
            A_ra, g_ra, B_ra = elastic_net_linear(train_X_rel_abun, train_U, train_T, Q_inv, r_A=0, r_g=0,r_B=0.1,alpha=0.1, tol=1e-3)

            pkl.dump((A_e, g_e, B_e, A_ln, g_ln, B_ln, A_ra, g_ra, B_ra), open("tmp/taur_prediction_parameters-{}".format(fold), "wb"))

        baseline_err_cv += [compute_baseline_errors(test_Y)]
        en_err_cv += [compute_errors(test_Y, test_U_normalized, test_T, A_e, g_e, B_e)]
        linear_err_cv += [compute_errors_linear(test_Y, test_U_normalized, test_T, A_ln, g_ln, B_ln)]
        rel_abun_err_cv += [compute_errors_rel_abun(test_Y, test_U_normalized, test_T, A_ra, g_ra, B_ra)]

        en_err_stratified_cv += compute_errors_by_time(test_Y, test_U_normalized, test_T, A_e, g_e, B_e)
        linear_err_stratified_cv += compute_errors_by_time_linear(test_Y, test_U_normalized, test_T, A_ln, g_ln, B_ln)
        rel_abun_err_stratified_cv += compute_errors_by_time_rel_abun(test_Y, test_U_normalized, test_T, A_ra, g_ra, B_ra)


    baseline = []
    linear = []
    rel_abun = []

    # compute p-values for difference in total error per sample
    baseline_sum = []
    linear_sum = []
    rel_abun_sum = []
    clv_sum = []
    for cl,bl,ln,ra in zip(en_err_cv, baseline_err_cv, linear_err_cv, rel_abun_err_cv):
        baseline += (bl - cl).tolist()
        linear += (ln - cl).tolist()
        rel_abun += (ra - cl).tolist()

        baseline_sum += [np.sum(bl)]
        linear_sum += [np.sum(ln)]
        rel_abun_sum += [np.sum(ra)]
        clv_sum += [np.sum(cl)]


    baseline = np.array(baseline)
    linear = np.array(linear)
    rel_abun = np.array(rel_abun)

    baseline_p = wilcoxon_exact(baseline_sum, clv_sum, alternative="greater")[1]
    linear_p = wilcoxon_exact(linear_sum, clv_sum, alternative="greater")[1]
    rel_abun_p = wilcoxon_exact(rel_abun_sum, clv_sum, alternative="greater")[1]

    df = pd.DataFrame(np.array([baseline, linear, rel_abun]).T,
                      columns=["baseline\n$p={:.3f}$".format(baseline_p),
                               "alr-linear\n$p={:.3f}$".format(linear_p),
                               "ra-linear\n$p={:.3f}$".format(rel_abun_p)])
    ax = df.boxplot(showmeans=True)
    ax.set_ylabel("Square Error(X) $-$ Square Error(cLV)")
    ax.set_title("Patient Dataset")
    plt.savefig("plots/taur_prediction-comparison.pdf")

    # for idx, en_linear_rel in enumerate(zip(en_err_stratified_cv, linear_err_stratified_cv, rel_abun_err_stratified_cv)):
    #     obs_dim = Y[0].shape[1]
    #     en, linear, rel = en_linear_rel
    #     fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(8,10))
    #     plot_bar(ax[0][0], en[:,2:(2+obs_dim)], en[:,0])
    #     ax[0][0].set_xticks(en[:0].tolist())
    #     ax[0][0].set_xticklabels(en[:0].tolist())
    #     ax[0][0].set_title("Truth")

    #     plot_bar(ax[0][1], en[:,(2+obs_dim):(2+2*obs_dim)], en[:,0])
    #     ax[0][1].set_xticks(en[:0].tolist())
    #     ax[0][1].set_xticklabels(en[:0].tolist())
    #     ax[0][1].set_title("cLV")

    #     plot_bar(ax[1][0], linear[:,(2+obs_dim):(2+2*obs_dim)], linear[:,0])
    #     ax[1][0].set_xticks(en[:0].tolist())
    #     ax[1][0].set_xticklabels(en[:0].tolist())
    #     ax[1][0].set_title("$\eta$-linear")

    #     plot_bar(ax[2][0], rel[:,(2+obs_dim):(2+2*obs_dim)], rel[:,0])
    #     ax[2][0].set_xticks(en[:0].tolist())
    #     ax[2][0].set_xticklabels(en[:0].tolist())
    #     ax[2][0].set_title("$\pi$-linear")

    #     ax[1][1].scatter(linear[:,0], linear[:,1] - en[:,1])
    #     ax[2][1].scatter(rel[:,0], rel[:,1] - en[:,1])

    #     ymin = np.min( np.array([linear[1:,1] - en[1:,1], rel[1:,1] - en[1:,1]]) ) - 0.15
    #     ymax = np.max( np.array([linear[1:,1] - en[1:,1], rel[1:,1] - en[1:,1]]) ) + 0.15
    #     for i in range(1, 3):
    #         ax[i][1].set_ylim(ymin, ymax)
    #         ax[i][1].set_xlim(ax[i][0].get_xlim())

    #     plt.tight_layout()

    #     plt.tight_layout()
    #     plt.savefig("plots/taur_prediction_comparison-test-{}.pdf".format(idx))
    #     plt.close()

    return (baseline, baseline_p), (linear, linear_p), (rel_abun, rel_abun_p)

if __name__ == "__main__":
    Y, U, T, U_normalized, T_normalized = load_observations("data/taur/taur-otu-table-top10+dom.csv", "data/taur/taur-events.csv")

    # last 20 observations were used to estimate
    # regularization coefficients and should be
    # omitted.
    Y_train = Y[:-20]
    U_train = U[:-20]
    T_train = T[:-20]
    U_train_normalized = U_normalized[:-20]
    prediction_experiment(Y_train, U_train, T_train, U_train_normalized)