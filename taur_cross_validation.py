import numpy as np

from scipy.special import logsumexp

from clv import CompositionalLV
from optimizers import least_squares_observed, \
                       elastic_net, \
                       elastic_net_linear

pseudo_count = 1e-5

def estimate_elastic_net_regularizers_cv(Y, U, T, U_normalized):
    ratio = [0, 0.1, 0.5, 0.7, 0.9]
    alphas = [0.1, 1, 10]
    alpha_rA_rg_rB = []
    for alpha in alphas:
        for r_A in ratio:
            for r_g in ratio:
                for r_B in ratio:
                    alpha_rA_rg_rB.append( (alpha, r_A, r_g, r_B ) )
    
    folds = 3
    best_r = 0
    best_sqr_err = np.inf
    for alpha, r_A, r_g, r_B in alpha_rA_rg_rB:
            sqr_err = 0
            for fold in range(folds):
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
                train_X = estimate_latent_from_observations(train_Y)
                Q_inv = np.eye(train_X[0].shape[1])
                A, g, B = elastic_net(train_X, train_U, train_T, Q_inv, r_A, r_g, r_B, alpha=alpha, tol=1e-3)
                np.set_printoptions(suppress=True)
                sqr_err += compute_prediction_error(test_Y, test_U_normalized, test_T, A, g, B)


            if sqr_err < best_sqr_err:
                best_r = (alpha, r_A, r_g, r_B)
                best_sqr_err = sqr_err
                print("\tr", (alpha, r_A, r_g, r_B), "sqr error", sqr_err)
    return best_r


def estimate_elastic_net_regularizers_cv_linear(Y, U, T, U_normalized):
    ratio = [0, 0.1, 0.5, 0.7, 0.9]
    alphas = [0.1, 1, 10]
    alpha_rA_rg_rB = []
    for alpha in alphas:
        for r_A in ratio:
            for r_g in ratio:
                for r_B in ratio:
                    alpha_rA_rg_rB.append( (alpha, r_A, r_g, r_B ) )
    
    folds = 3
    best_r = 0
    best_sqr_err = np.inf
    for alpha, r_A, r_g, r_B in alpha_rA_rg_rB:
            sqr_err = 0
            for fold in range(folds):
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
                train_X = estimate_latent_from_observations(train_Y)
                Q_inv = np.eye(train_X[0].shape[1])
                A, g, B = elastic_net_linear(train_X, train_U, train_T, Q_inv, r_A, r_g, r_B, alpha=alpha, tol=1e-3)
                np.set_printoptions(suppress=True)
                sqr_err += compute_prediction_error_linear(test_Y, test_U_normalized, test_T, A, g, B)

            if sqr_err < best_sqr_err:
                best_r = (alpha, r_A, r_g, r_B)
                best_sqr_err = sqr_err
                print("\tr", (alpha, r_A, r_g, r_B), "sqr error", sqr_err)
    return best_r


def estimate_elastic_net_regularizers_cv_rel_abun(Y, U, T, U_normalized):
    ratio = [0, 0.1, 0.5, 0.7, 0.9]
    alphas = [0.1, 1, 10]
    alpha_rA_rg_rB = []
    for alpha in alphas:
        for r_A in ratio:
            for r_g in ratio:
                for r_B in ratio:
                    alpha_rA_rg_rB.append( (alpha, r_A, r_g, r_B ) )
    
    folds = 3
    best_r = 0
    best_sqr_err = np.inf
    for alpha, r_A, r_g, r_B in alpha_rA_rg_rB:
            sqr_err = 0
            for fold in range(folds):
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
                train_X = estimate_rel_abun(train_Y)
                Q_inv = np.eye(train_X[0].shape[1])
                A, g, B = elastic_net_linear(train_X, train_U, train_T, Q_inv, r_A, r_g, r_B, alpha=alpha, tol=1e-3)
                np.set_printoptions(suppress=True)
                sqr_err += compute_prediction_error_rel_abun(test_Y, test_U_normalized, test_T, A, g, B)

            if sqr_err < best_sqr_err:
                best_r = (alpha, r_A, r_g, r_B)
                best_sqr_err = sqr_err
                print("\tr", (alpha, r_A, r_g, r_B), "sqr error", sqr_err)
    return best_r


def compute_prediction_error(Y, U_normalized, T, A, g, B):
    def compute_total_square_error(y, y_pred):
        err = 0
        for yt, ypt in zip(y[1:], y_pred[1:]):
            err += np.square(yt / yt.sum() - ypt).sum()
        return err
    err = 0
    for y, u, t in zip(Y, U_normalized, T):
        y_pred = predict(y[0], u, t, A, g, B)
        err += compute_total_square_error(y, y_pred)
    return err


def compute_prediction_error_linear(Y, U_normalized, T, A, g, B):
    def compute_total_square_error(y, y_pred):
        err = 0
        for yt, ypt in zip(y[1:], y_pred[1:]):
            err += np.square(yt / yt.sum() - ypt).sum()
        return err
    err = 0
    for y, u, t in zip(Y, U_normalized, T):
        y_pred = predict_linear(y[0], u, t, A, g, B)
        err += compute_total_square_error(y, y_pred)
    return np.array(err)


def compute_prediction_error_rel_abun(Y, U_normalized, T, A, g, B):
    def compute_total_square_error(y, y_pred):
        err = 0
        for yt, ypt in zip(y[1:], y_pred[1:]):
            err += np.square(yt / yt.sum() - ypt).sum()
        return err
    err = 0
    for y, u, t in zip(Y, U_normalized, T):
        y_pred = predict_rel_abun(y[0], u, t, A, g, B)
        err += compute_total_square_error(y, y_pred)
    return np.array(err)


def predict(y0, u_normalized, times, A, g, B):
    y0 = np.copy(y0)
    y0 = y0 / y0.sum()
    y0 = (y0 + pseudo_count) / (y0 + pseudo_count).sum()
    mu = np.log( y0[:-1] / y0[-1] )
    zt = mu
    y_pred = np.zeros((times.shape[0], y0.size))
    y_idx = 0
    for time in range(times.min(), times.max()+1):
        t = time - times.min()
        xt  = np.concatenate((zt, np.array([0])))
        pt  = np.exp(xt - logsumexp(xt))
        if time in times:
            y_pred[y_idx] = pt  
            y_idx += 1
        if t < times.shape[0] - 1:
            zt = zt + g + A.dot(pt) + B.dot(u_normalized[t])
    return y_pred


def predict_linear(y0, u_normalized, times, A, g, B):
    y0 = np.copy(y0)
    y0 = y0 / y0.sum()
    y0 = (y0 + pseudo_count) / (y0 + pseudo_count).sum()
    mu = np.log( y0[:-1] / y0[-1] )
    zt = mu
    y_pred = np.zeros((times.shape[0], y0.size))
    y_idx = 0
    for time in range(times.min(), times.max()+1):
        t = time - times.min()
        xt  = np.concatenate((zt, np.array([0])))
        pt  = np.exp(xt - logsumexp(xt))
        if time in times:
            y_pred[y_idx] = pt  
            y_idx += 1
        if t < times.shape[0] - 1:
            zt = zt + g + A.dot(zt) + B.dot(u_normalized[t])
    return y_pred


def predict_rel_abun(y0, u_normalized, times, A, g, B):
    y0 = np.copy(y0)
    y0 /= y0.sum()
    y_pred = np.zeros((times.shape[0], y0.size))
    y_idx = 0
    zt = y0
    for time in range(times.min(), times.max()+1):
        t = time - times.min()
        pt = np.copy(zt)
        pt[pt < 0] = 0
        pt /= pt.sum()
        zt = np.copy(pt)
        if time in times:
            y_pred[y_idx] = pt  
            y_idx += 1
        if t < times.shape[0] - 1:
            zt = zt + g + A.dot(zt) + B.dot(u_normalized[t])
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
