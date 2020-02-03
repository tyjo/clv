import matplotlib
matplotlib.use('agg')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
import seaborn as sns
import util

from wilcoxon_exact import wilcoxon_exact

import sys
sys.path.append("./statannot/statannot")
import statannot
add_stat_annotation = statannot.add_stat_annotation


def compute_rmses(dataset, model, truth):
    rmses = []
    for i, tr in enumerate(truth):
        tr /= tr.sum(axis=1,keepdims=True)
        if dataset=="stein":
            pred = pkl.load(open("tmp/{}_prediction_parameters-{}-{}".format(dataset, i, model), "rb"))
        else:
            pred = pkl.load(open("tmp/{}_predictions-{}-{}".format(dataset, i, model), "rb"))
        pred = pred[0]
        rmses.append(np.sqrt(np.mean(np.square(tr[-1] - pred[-1]))))
    return rmses


def compute_rmse_static(truth):
    rmses = []
    for tr in truth:
        pred = np.array([tr[0] for t in range(tr.shape[0])])
        rmses.append(np.sqrt(np.mean(np.square(tr[1:] - pred[1:]))))
    return rmses


def plot_model_comparison():
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

    # stein
    Y = pkl.load(open("data/stein/Y.pkl", "rb"))
    U = pkl.load(open("data/stein/U.pkl", "rb"))
    T = pkl.load(open("data/stein/T.pkl", "rb"))

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

    glv_p = np.round(wilcoxon_exact(results_dict["clv"], results_dict["glv"], alternative="less")[1], 3)
    alr_p = np.round(wilcoxon_exact(results_dict["clv"], results_dict["alr"], alternative="less")[1], 3)
    lra_p = np.round(wilcoxon_exact(results_dict["clv"], results_dict["lra"], alternative="less")[1], 3)
    glv_ra_p = np.round(wilcoxon_exact(results_dict["clv"], results_dict["glv-ra"], alternative="less")[1], 3)
    static_p = np.round(wilcoxon_exact(results_dict["clv"], results_dict["initial"], alternative="less")[1], 3)

    glv_p = "p=" + str(glv_p) if glv_p < 0.05 else "n.s."
    alr_p = "p=" + str(alr_p) if alr_p < 0.05 else "n.s."
    lra_p = "p=" + str(lra_p) if lra_p < 0.05 else "n.s."
    glv_ra_p = "p=" + str(glv_ra_p) if glv_ra_p < 0.05 else "n.s."
    static_p = "p=" + str(static_p) if static_p < 0.05 else "n.s."

    rmse = pd.DataFrame(results, columns=["Model", "RMSE"])
    sns.boxplot(ax=ax, x="Model", y="RMSE", data=rmse, palette="colorblind")
    ax.set_xticklabels(["cLV",
                       "gLV$_{abs}$",
                        "gLV$_{rel}$",
                        "ALR\n",
                       "linear",
                       "constant"],
                       fontsize=8)
    ax.set_title("Antibiotic")


    box_pairs = [("clv", "glv"), ("clv", "glv-ra"), ("clv", "alr"), ("clv", "lra"), ("clv", "initial")]
    add_stat_annotation(ax, data=rmse, x="Model", y="RMSE",
                            loc='inside', verbose=2, text_format='star', box_pairs=box_pairs,
                            test='wilcoxon-exact')


    ylim_lower = 0
    ax.set_ylim((ylim_lower, ax.get_ylim()[1]))


    plt.tight_layout()
    plt.savefig("plots/antibiotic_last.pdf")

if __name__ == "__main__":
    plot_model_comparison()