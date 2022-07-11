"""
Run the simulations on healthy data

Usage: python scripts/clv_code_repo/healthy_prediction_experiments.py \
    -m "glv" \
    -r "ridge" \
    -i "scripts/clv_code_repo/data/gibson_agg/top_10_otus"\
    -o "output/lv_sims/top_10_otus"
    -t ""
"""

import matplotlib
matplotlib.use('agg')

import logging
import numpy as np
import pickle as pkl
import argparse
import copy
import time
from pathlib import Path

from fit_models import fit_clv, fit_glv, fit_linear_alr, fit_linear_rel_abun, fit_glv_ridge


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='Name of the model')
    parser.add_argument("-r", "--regression", type=str, help="Type of regression to use")
    parser.add_argument("-l", "--limit_of_detection", type=float,
                        help="The limit of detection. Any initial abundance <= this value gets set to this value.")
    parser.add_argument("-o", "--output_loc",
        help="Location of the folder where the output is saved")
    parser.add_argument("-i", "--input_loc",
        help="location of the input files")

    return parser.parse_args()


def compute_errors(Y, Y_pred):
    def compute_square_errors(y, y_pred):
        err = []
        for yt, ypt in zip(y[1:], y_pred[1:]):
            err.append(np.sqrt(np.square(yt - ypt)).sum())
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


def fit_model(Y, U, T, model, savepath, scale, n_subjects, regression_type="elastic-net"):

    models = ["clv", "alr", "lra", "glv", "glv-ra"]
    if model not in models:
        print("model", (model), "must be one of", models)#, file = sys.stderr)
        exit(1)
    folds = 1
    for fold in range(folds):
        print("running fold", fold)
        train_Y = []
        train_U = []
        train_T = []

        test_Y = []
        test_U = []
        test_T = []
        for i in range(len(Y)):
            train_Y.append(Y[i])
            train_U.append(U[i])
            train_T.append(T[i])

        savepath.mkdir(parents=True, exist_ok=True)
        parameter_filename = savepath
        logging.warning(parameter_filename)

        if model == "clv":
            try:
                pred_clv = pkl.load(open(parameter_filename / "clv", "rb"))
            except FileNotFoundError:
                pred_clv = fit_clv(train_Y, train_T, train_U, test_Y, test_T,
                           test_U,
                           save_name=parameter_filename / "clv-model.pkl")
                #pkl.dump(pred_clv, open(parameter_filename + "-clv", "wb"))
                #print("saved: " + parameter_filename + " " + model)
        elif model == "alr":
            try:
                pred_alr = pkl.load(open(parameter_filename / "alr", "rb"))
            except FileNotFoundError:
                pred_alr = fit_linear_alr(train_Y, train_T, train_U, test_Y,
                    test_T, test_U,
                  save_name=parameter_filename / "alr-model.pkl")
                #pkl.dump(pred_alr, open(parameter_filename + "-alr", "wb"))
                #print("saved: " + parameter_filename + " " + model)

        elif model == "lra":
            try:
                pred_lra = pkl.load(open(parameter_filename / "lra", "rb"))
            except FileNotFoundError:
                pred_lra = fit_linear_rel_abun(train_Y, train_T, train_U, test_Y,
                   test_T, test_U,
                   save_name=parameter_filename / "lra-model.pkl")
                #pkl.dump(pred_lra, open(parameter_filename + "-lra", "wb"))
                #print("saved: " + parameter_filename + " " + model)

        elif model == "glv":
            if regression_type=="ridge":
                try:
                    pred_glv = pkl.load(open(parameter_filename / "glv-ridge", "rb"))
                except FileNotFoundError:
                    pred_glv = fit_glv_ridge(train_Y, train_T, train_U, test_Y,
                        test_T, test_U, scale=scale,
                        save_name=parameter_filename / "glv-ridge-model.pkl")
                #pkl.dump(pred_glv, open(parameter_filename + "-glv-ridge", "wb"))
                #print("saved: " + parameter_filename + " " + model)
            else:
                try:
                    pred_glv = pkl.load(open(parameter_filename / "glv-elastic-net", "rb"))
                except FileNotFoundError:
                    pred_glv = fit_glv(train_Y, train_T, train_U, test_Y, test_T,
                        test_U, scale=scale,
                        save_name=parameter_filename / "glv-elastic-net-model.pkl")
        elif model == "glv-ra":
            if regression_type == "ridge":
                try:
                    pred_glv_ra = pkl.load(open(parameter_filename / "glv-ra-ridge", "rb"))
                except FileNotFoundError:
                    pred_glv_ra = fit_glv_ridge(train_Y, train_T, train_U, test_Y,
                        test_T, test_U, use_rel_abun=True,
                        save_name=parameter_filename / "glv-ra-model.pkl")
            else:
                try:
                    pred_glv_ra = pkl.load(open(parameter_filename / "glv-ra-elastic-net", "rb"))
                except FileNotFoundError:
                    pred_glv_ra = fit_glv(train_Y, train_T, train_U, test_Y, test_T,
                        test_U, use_rel_abun=True,
                        save_name=parameter_filename / "glv-ra-elastic-net-model.pkl")
                    #pkl.dump(pred_glv_ra, open(parameter_filename + "-glv-ra", "wb"))
                    #print("saved: " + parameter_filename + " " + model)


def adjust_concentrations(Y):
    """adjust the concentrations for stability purposes"""

    con =  []
    for y in Y:
        con += y.sum(axis=1).tolist()
    con = np.array(con)
    C = 1 / np.mean(con)

    Y_adjusted = []
    for y in Y:
        Y_adjusted.append(C*y)

    return Y_adjusted, C


def add_limit_detection(X, lim=1e5):
    """adjust the 0 concentration values"""
    print(f"Using limit of detection value `{lim}`.")

    X_ = copy.deepcopy(X)
    new_X =[]

    for x in X_:
        new_X.append(np.where(x <= lim, lim, x))

    return new_X


if __name__ == "__main__":
    args = parse_arguments()
    input_files_path = Path(args.input_loc)
    Y = pkl.load(open(input_files_path / "Y.pkl", "rb"))
    U = pkl.load(open(input_files_path / "U.pkl", "rb"))
    T = pkl.load(open(input_files_path / "T.pkl", "rb"))

    Y_corrected = add_limit_detection(Y, args.limit_of_detection)
    Y_adj, scale = adjust_concentrations(Y_corrected)  # denoise Y

    model = args.model
    fit_model(Y_adj, U, T, model, Path(args.output_loc), scale, n_subjects=len(Y),
              regression_type=args.regression)
