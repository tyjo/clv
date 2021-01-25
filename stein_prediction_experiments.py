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
        print("model must be one of", models, file=sys.stderr)
        exit(1)

    folds = 9
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

        parameter_filename = "pkl/stein_prediction_parameters-{}".format(fold)

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
    Y = pkl.load(open("data/stein/Y.pkl", "rb"))
    U = pkl.load(open("data/stein/U.pkl", "rb"))
    T = pkl.load(open("data/stein/T.pkl", "rb"))

    Y_adj = adjust_concentrations(Y)

    if len(sys.argv) < 2:
       print("USAGE: python stein_prediction_experiments.py [MODEL]")
       exit(1)
    model = sys.argv[1]
    fit_model(Y_adj, U, T, model)
