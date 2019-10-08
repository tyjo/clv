import matplotlib.pyplot as plt
import numpy as np

from scipy.special import logsumexp
from scipy.stats import pearsonr

from clv import CompositionalLV
from optimizers import least_squares_observed, \
                       elastic_net, \
                       elastic_net_linear, \
                       estimate_latent_from_observations, \
                       elastic_net_lotka_volterra

pseudo_count = 1e-5

def estimate_elastic_net_regularizers_cv(Y, U, T, folds=5):
    ratio = [0, 0.1, 0.5, 0.7, 0.9]
    alphas = [0.1, 1, 10]
    alpha_rA_rg_rB = []
    for alpha in alphas:
        for r_A in ratio:
            for r_g in ratio:
                for r_B in ratio:
                    alpha_rA_rg_rB.append( (alpha, r_A, r_g, r_B ) )
    
    best_r = 0
    best_sqr_err = np.inf
    for alpha, r_A, r_g, r_B in alpha_rA_rg_rB:
            sqr_err = 0
            for fold in range(folds):
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

                train_X = estimate_latent_from_observations(train_Y)
                Q_inv = np.eye(train_X[0].shape[1])
                A, g, B = elastic_net(train_X, train_U, train_T, Q_inv, r_A, r_g, r_B, alpha=alpha, tol=1e-3)
                np.set_printoptions(suppress=True)
                sqr_err += compute_prediction_error(test_Y, test_U, test_T, A, g, B)

            if sqr_err < best_sqr_err:
                best_r = (alpha, r_A, r_g, r_B)
                best_sqr_err = sqr_err
                print("\tr", (alpha, r_A, r_g, r_B), "sqr error", sqr_err)
    return best_r


def estimate_elastic_net_regularizers_cv_linear(Y, U, T, folds=5):
    ratio = [0, 0.1, 0.5, 0.7, 0.9]
    alphas = [0.1, 1, 10]
    alpha_rA_rg_rB = []
    for alpha in alphas:
        for r_A in ratio:
            for r_g in ratio:
                for r_B in ratio:
                    alpha_rA_rg_rB.append( (alpha, r_A, r_g, r_B ) )
    
    best_r = 0
    best_sqr_err = np.inf
    for alpha, r_A, r_g, r_B in alpha_rA_rg_rB:
            sqr_err = 0
            for fold in range(folds):
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
                train_X = estimate_latent_from_observations(train_Y)
                Q_inv = np.eye(train_X[0].shape[1])
                A, g, B = elastic_net_linear(train_X, train_U, train_T, Q_inv, r_A, r_g, r_B, alpha=alpha, tol=1e-3)
                np.set_printoptions(suppress=True)
                sqr_err += compute_prediction_error_linear(test_Y, test_U, test_T, A, g, B)

            if sqr_err < best_sqr_err:
                best_r = (alpha, r_A, r_g, r_B)
                best_sqr_err = sqr_err
                print("\tr", (alpha, r_A, r_g, r_B), "sqr error", sqr_err)
    return best_r


def estimate_elastic_net_regularizers_cv_rel_abun(Y, U, T, folds=5):
    ratio = [0, 0.1, 0.5, 0.7, 0.9]
    alphas = [0.1, 1, 10]
    alpha_rA_rg_rB = []
    for alpha in alphas:
        for r_A in ratio:
            for r_g in ratio:
                for r_B in ratio:
                    alpha_rA_rg_rB.append( (alpha, r_A, r_g, r_B ) )
    
    best_r = 0
    best_sqr_err = np.inf
    for alpha, r_A, r_g, r_B in alpha_rA_rg_rB:
            sqr_err = 0
            for fold in range(folds):
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
                train_X = estimate_rel_abun(train_Y)
                Q_inv = np.eye(train_X[0].shape[1])
                A, g, B = elastic_net_linear(train_X, train_U, train_T, Q_inv, r_A, r_g, r_B, alpha=alpha, tol=1e-3)
                np.set_printoptions(suppress=True)
                sqr_err += compute_prediction_error_rel_abun(test_Y, test_U, test_T, A, g, B)

            if sqr_err < best_sqr_err:
                best_r = (alpha, r_A, r_g, r_B)
                best_sqr_err = sqr_err
                print("\tr", (alpha, r_A, r_g, r_B), "sqr error", sqr_err)
    return best_r


def estimate_elastic_net_regularizers_cv_lotka_volterra(Y, U, T, folds=5):
    ratio = [0, 0.1, 0.5, 0.7, 0.9]
    alphas = [0.1, 1, 10]
    alpha_rA_rg_rB = []
    for alpha in alphas:
        for r_A in ratio:
            for r_g in ratio:
                for r_B in ratio:
                    alpha_rA_rg_rB.append( (alpha, r_A, r_g, r_B ) )
    
    best_r = 0
    best_sqr_err = np.inf
    for alpha, r_A, r_g, r_B in alpha_rA_rg_rB:
            sqr_err = 0
            for fold in range(folds):
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
                train_X = estimate_log_space_from_observations(train_Y)
                Q_inv = np.eye(train_X[0].shape[1])
                A, g, B = elastic_net_lotka_volterra(train_X, train_U, train_T, Q_inv, r_A, r_g, r_B, alpha=alpha, tol=1e-3)
                np.set_printoptions(suppress=True)
                sqr_err += compute_prediction_error_glv(test_Y, test_U, test_T, A, g, B)

                if np.isnan(sqr_err):
                    continue
                    
            if sqr_err < best_sqr_err:
                best_r = (alpha, r_A, r_g, r_B)
                best_sqr_err = sqr_err
                print("\tr", (alpha, r_A, r_g, r_B), "sqr error", sqr_err)
    return best_r


def compute_square_error(true, est):
    true = np.copy(true)
    est = np.copy(est)
    true /= true.sum()
    est /= est.sum()
    true = (true + pseudo_count) / (true + pseudo_count).sum()
    est = (est + pseudo_count) / (est + pseudo_count).sum()

    return np.square(true - est).sum()


def compute_corr(y, y_pred):
    y = np.copy(y[1:])
    y /= y.sum(axis=1, keepdims=True)

    y_pred = np.copy(y_pred[1:])

    corr = []
    for yt, yp in zip(y.T, y_pred.T):
        if np.median(yt) < 0.01:
            continue

        c = pearsonr(yt, yp)[0]
        if np.isnan(c):
            c = 0
        corr.append(c)
    return corr


def compute_prediction_error(Y, U, T, A, g, B):
    def compute_total_square_error(y, y_pred):
        err = 0
        for yt, ypt in zip(y[1:], y_pred[1:]):
            err += compute_square_error(yt, ypt)
        return err
    err = 0
    for y, u, t in zip(Y, U, T):
        y_pred = predict(y[0], u, t, A, g, B)
        err += compute_total_square_error(y, y_pred)
    return err


def compute_prediction_error_linear(Y, U, T, A, g, B):
    def compute_total_square_error(y, y_pred):
        err = 0
        for yt, ypt in zip(y[1:], y_pred[1:]):
            err += compute_square_error(yt, ypt)
        return err
    err = 0
    for y, u, t in zip(Y, U, T):
        y_pred = predict_linear(y[0], u, t, A, g, B)
        err += compute_total_square_error(y, y_pred)
    return np.array(err)


def compute_prediction_error_rel_abun(Y, U, T, A, g, B):
    def compute_total_square_error(y, y_pred):
        err = 0
        for yt, ypt in zip(y[1:], y_pred[1:]):
            err += compute_square_error(yt, ypt)
        return err
    err = 0
    for y, u, t in zip(Y, U, T):
        y_pred = predict_rel_abun(y[0], u, t, A, g, B)
        err += compute_total_square_error(y, y_pred)
    return np.array(err)


def compute_prediction_error_glv(Y, U, T, A, g, B):
    def compute_total_square_error(y, y_pred):
        err = 0
        for yt, ypt in zip(y[1:], y_pred[1:]):
            err += compute_square_error(yt, ypt)
        return err
    err = 0
    for y, u, t in zip(Y, U, T):
        y_pred = predict_glv(y[0], u, t, A, g, B)
        err += compute_total_square_error(y, y_pred)
    return np.array(err)


def predict(y0, u, times, A, g, B):
    y0 = np.copy(y0)
    y0 = y0 / y0.sum()
    y0 = (y0 + pseudo_count) / (y0 + pseudo_count).sum()
    mu = np.log( y0[:-1] / y0[-1] )
    zt = mu
    xt  = np.concatenate((zt, np.array([0])))
    pt  = np.exp(xt - logsumexp(xt))
    y_pred = np.zeros((times.shape[0], y0.size))
    for i in range(1,times.shape[0]):
        dt = times[i] - times[i-1]
        zt = zt + dt*(g + A.dot(pt) + B.dot(u[i-1]))
        xt  = np.concatenate((zt, np.array([0])))
        pt  = np.exp(xt - logsumexp(xt))
        y_pred[i] = pt        
    return y_pred


def predict_linear(y0, u, times, A, g, B):
    y0 = np.copy(y0)
    y0 = y0 / y0.sum()
    y0 = (y0 + pseudo_count) / (y0 + pseudo_count).sum()
    mu = np.log( y0[:-1] / y0[-1] )
    zt = mu
    y_pred = np.zeros((times.shape[0], y0.size))
    y_idx = 0
    for i in range(1,times.shape[0]):
        dt = times[i] - times[i-1]
        zt = zt + dt*(g + A.dot(zt) + B.dot(u[i-1]))
        xt  = np.concatenate((zt, np.array([0])))
        pt  = np.exp(xt - logsumexp(xt))
        y_pred[i] = pt   

    return y_pred


def predict_rel_abun(y0, u, times, A, g, B):
    y0 = np.copy(y0)
    y0 /= y0.sum()
    y_pred = np.zeros((times.shape[0], y0.size))
    y_idx = 0
    zt = y0
    for i in range(1,times.shape[0]):
        dt = times[i] - times[i-1]
        zt = zt + dt*(g + A.dot(zt) + B.dot(u[i-1]))
        zt[zt < 0] = 0
        zt /= zt.sum()
        y_pred[i] = np.copy(zt)
    return y_pred


def predict_glv(y0, u, times, A, g, B):
    y0 = np.copy(y0)
    mass = y0.sum()
    y0 /= y0.sum()
    y0 = (y0 + pseudo_count) / (y0 + pseudo_count).sum()
    y0 = mass*y0
    mu = np.log(y0)
    zt = mu
    y_pred = np.zeros((times.shape[0], y0.size))
    y_idx = 0
    for i in range(1,times.shape[0]):
        dt = times[i] - times[i-1]
        zt = zt + dt*(g + A.dot(np.exp(zt)) + B.dot(u[i-1]))
        pt  = np.exp(zt - logsumexp(zt))
        y_pred[i] = pt  
        y_idx += 1
    return y_pred


def estimate_rel_abun(Y):
    X = []
    for y in Y:
        x = np.zeros((y.shape[0], y.shape[1]))
        for t in range(y.shape[0]):
            yt = np.copy(y[t])
            x[t] = yt / yt.sum()
            assert np.all(np.isfinite(x[t]))
        X.append(x)
    return X


def estimate_latent_from_observations(Y):
    X = []
    for y in Y:
        x = np.zeros((y.shape[0], y.shape[1]-1))
        for t in range(y.shape[0]):
            pt = y[t] / y[t].sum()
            pt = (pt + pseudo_count) / (pt + pseudo_count).sum()
            xt  = np.log(pt[:-1]) - np.log(pt[-1])
            x[t] = xt
        X.append(x)
    return X


def estimate_log_space_from_observations(Y):
    X = []
    for y in Y:
        x = np.zeros((y.shape[0], y.shape[1]))
        for t in range(y.shape[0]):
            mass = y[t].sum()
            pt = y[t] / y[t].sum()
            pt = (pt + pseudo_count) / (pt + pseudo_count).sum()
            yt = mass*pt
            x[t] = np.log(yt)
            assert np.all(np.isfinite(x[t]))
        X.append(x)
    return X


def plot_bar(ax, y, time):
    cm = plt.get_cmap("tab20c")
    colors = [cm(i) for i in range(20)]
    width = 1
    ax.bar(time, y[:,0], width=width, color=colors[0])
    for j in range(1, y.shape[1]):
        sign = np.sign(y[:,j])
        bottom = np.copy(y[:,:j])
        ax.bar(time, y[:,j], bottom=bottom.sum(axis=1), width=width, color=colors[j % 20])