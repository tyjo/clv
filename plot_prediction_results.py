import matplotlib
matplotlib.use('agg')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
import seaborn as sns

from wilcoxon_exact import wilcoxon_exact

import sys
sys.path.append("./statannot/statannot")
import statannot
add_stat_annotation = statannot.add_stat_annotation

def compute_rmses(dataset, model, truth):
    folder = "pub-results"

    rmses = []
    for i, tr in enumerate(truth):
        tr /= tr.sum(axis=1,keepdims=True)
        if dataset=="stein":
            pred = pkl.load(open(folder + "/{}_prediction_parameters-{}-{}".format(dataset, i, model), "rb"))
        else:
            pred = pkl.load(open(folder + "/{}_predictions-{}-{}".format(dataset, i, model), "rb"))
        pred = pred[0]
        rmses.append(np.sqrt(np.mean(np.square(tr[1:] - pred[1:]))))
    return rmses


def compute_rmse_static(truth):
    rmses = []
    for tr in truth:
        pred = np.array([tr[0] for t in range(tr.shape[0])])
        rmses.append(np.sqrt(np.mean(np.square(tr[1:] - pred[1:]))))
    return rmses

def plot_model_comparison():
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    # bucci diet
    Y = pkl.load(open("pub-results/bucci/Y_diet.pkl", "rb"))
    U = pkl.load(open("pub-results/bucci/U_diet.pkl", "rb"))
    T = pkl.load(open("pub-results/bucci/T_diet.pkl", "rb"))

    models = ["clv", "glv",  "glv-ra", "alr", "lra"]
    results = []
    results_dict = {}
    results_dict["clv"] = []
    results_dict["glv"] = []
    results_dict["alr"] = []
    results_dict["lra"] = []
    results_dict["glv-ra"] = []
    results_dict["initial"] = []
    for model in models:
        for rmse in compute_rmses("bucci_diet", model, Y):
            results.append([model, rmse])
            results_dict[model].append(rmse)

    for rmse in compute_rmse_static(Y):
        results.append(["initial", rmse])
        results_dict["initial"].append(rmse)


    rmse = pd.DataFrame(results, columns=["Model", "RMSE"])
    sns.boxplot(ax=ax[0], x="Model", y="RMSE", data=rmse)
    ax[0].set_xticklabels(["cLV",
                           "gLV$_{abs}$",
                           "gLV$_{rel}$",
                           "ALR",
                           "linear",
                           "constant"],
                           fontsize=9)
    ax[0].set_title("Diet")

    box_pairs = [("clv", "glv"), ("clv", "glv-ra"), ("clv", "alr"), ("clv", "lra"), ("clv", "initial")]
    add_stat_annotation(ax[0], data=rmse, x="Model", y="RMSE",
                            loc='inside', verbose=2, text_format='star', box_pairs=box_pairs,
                            test='wilcoxon-exact')


    # bucci C. diff
    Y = pkl.load(open("pub-results/bucci/Y_cdiff-denoised.pkl", "rb"))
    U = pkl.load(open("pub-results/bucci/U_cdiff.pkl", "rb"))
    T = pkl.load(open("pub-results/bucci/T_cdiff.pkl", "rb"))

    models = ["clv", "glv", "glv-ra", "alr", "lra"]
    results = []
    results_dict = {}
    results_dict["clv"] = []
    results_dict["glv"] = []
    results_dict["alr"] = []
    results_dict["lra"] = []
    results_dict["glv-ra"] = []
    results_dict["initial"] = []
    for model in models:
        for rmse in compute_rmses("bucci_cdiff", model, Y):
            results.append([model, rmse])
            results_dict[model].append(rmse)

    for rmse in compute_rmse_static(Y):
        results.append(["initial", rmse])
        results_dict["initial"].append(rmse)

    rmse = pd.DataFrame(results, columns=["Model", "RMSE"])
    sns.boxplot(ax=ax[1], x="Model", y="RMSE", data=rmse)
    ax[1].set_xticklabels(["cLV",
                           "gLV$_{abs}$",
                           "gLV$_{rel}$",
                           "ALR",
                           "linear",
                           "constant"],
                           fontsize=9)
    ax[1].set_title("C. diff")

    box_pairs = [("clv", "glv"), ("clv", "glv-ra"), ("clv", "alr"), ("clv", "lra"), ("clv", "initial")]
    add_stat_annotation(ax[1], data=rmse, x="Model", y="RMSE",
                            loc='inside', verbose=2, text_format='star', box_pairs=box_pairs,
                            test='wilcoxon-exact')


    # stein
    Y = pkl.load(open("pub-results/stein/Y.pkl", "rb"))
    U = pkl.load(open("pub-results/stein/U.pkl", "rb"))
    T = pkl.load(open("pub-results/stein/T.pkl", "rb"))

    models = ["clv", "glv", "glv-ra", "alr", "lra"]
    results = []
    results_dict = {}
    results_dict["clv"] = []
    results_dict["glv"] = []
    results_dict["alr"] = []
    results_dict["lra"] = []
    results_dict["glv-ra"] = []
    results_dict["initial"] = []
    for model in models:
        for rmse in compute_rmses("stein", model, Y):
            results.append([model, rmse])
            results_dict[model].append(rmse)

    for rmse in compute_rmse_static(Y):
        results.append(["initial", rmse])
        results_dict["initial"].append(rmse)


    rmse = pd.DataFrame(results, columns=["Model", "RMSE"])
    sns.boxplot(ax=ax[2], x="Model", y="RMSE", data=rmse)
    ax[2].set_xticklabels(["cLV",
                       "gLV$_{abs}$",
                        "gLV$_{rel}$",
                        "ALR\n",
                       "linear",
                       "constant"],
                       fontsize=8)
    ax[2].set_title("Antibiotic")


    box_pairs = [("clv", "glv"), ("clv", "glv-ra"), ("clv", "alr"), ("clv", "lra"), ("clv", "initial")]
    add_stat_annotation(ax[2], data=rmse, x="Model", y="RMSE",
                            loc='inside', verbose=2, text_format='star', box_pairs=box_pairs,
                            test='wilcoxon-exact')


    ylim_lower = 0
    ylim_upper = np.max([ ax[0].get_ylim()[1], ax[1].get_ylim()[1]])
    ax[0].set_ylim((ylim_lower, ylim_upper))
    ax[1].set_ylim((ylim_lower, ylim_upper))
    ax[2].set_ylim((ylim_lower, ax[2].get_ylim()[1]))


    plt.tight_layout()
    plt.savefig("plots/model_comparison.pdf")

if __name__ == "__main__":
    plot_model_comparison()