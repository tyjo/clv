import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle as pkl
import seaborn as sns
import sys

sys.path.append("./statannot/statannot")
import statannot
add_stat_annotation = statannot.add_stat_annotation

#sns.set_style("whitegrid")

def plot_trajectories(Y, T, output_dir, outfile):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    def plot_bar(ax, y, time, unique_color_id, remaining_ids):
        T = y.shape[0]
        cm = plt.get_cmap("tab20c")
        colors = [cm(i) for i in range(20)]
        #time = np.array([t for t in range(T)])
        widths = np.concatenate((time[1:] - time[:-1], [1])).astype(float)
        widths[widths > 1] = 1

        widths -= 1e-1

        y_colors = y[:,unique_color_id]
        ax.bar(time, y_colors[:,0], width=widths, color=colors[0], align="edge")
        for j in range(1, y_colors.shape[1]):
            ax.bar(time, y_colors[:,j], bottom=y_colors[:,:j].sum(axis=1), width=widths, color=colors[j], align="edge")
        
        ax.bar(time, y[:,remaining_ids].sum(axis=1), bottom=y_colors.sum(axis=1), width=widths, color=colors[19], align="edge")
        #ax.set_title("Relative Abundances", fontsize=10)
        #ax.legend(prop={"size" : 4}, bbox_to_anchor=[-0.1,1.225], loc="upper left", ncol=4)

    def find_top_ids(Y, n):
        ntaxa = Y[0].shape[1]
        rel_abun = np.zeros(ntaxa)
        for y in Y:
            tpts = y.shape[0]
            denom = y.sum(axis=1,keepdims=True)
            denom[denom == 0] = 1
            p = y / denom
            rel_abun += p.sum(axis=0) / tpts
        ids = np.argsort(-rel_abun)
        return np.sort(ids[:n]), np.sort(ids[n:])

    N = len(Y)
    top19_ids, remaining_ids = find_top_ids(Y, 19)
    fig, ax = plt.subplots(nrows=N,ncols=1,figsize=(N,2*N))
    for i in range(N):
        denom = Y[i].sum(axis=1)
        denom[denom == 0] = 1
        plot_bar(ax[i], (Y[i].T / denom).T, T[i], top19_ids, remaining_ids)


    outfile = os.path.splitext(outfile)[0]
    plt.tight_layout()
    plt.savefig(output_dir + "/" + outfile + ".pdf")
    plt.close()

def compute_rmse_params(param_true, param_est):
    return np.sqrt(np.mean(np.square(param_true - param_est)))

def compute_corr_params(param_true, param_est):
    return pearsonr(param_true.flatten(), param_est.flatten())[0]

def compute_rmse_trajectories(traj_true, traj_est):
    rmse = 0
    total = 0
    for tr, est in zip(traj_true, traj_est):
        total += tr[1:].shape[0]
        rmse += np.sum(np.square(tr[1:] - est[1:]))
    return np.sqrt(rmse/total)

def compute_nonzero_corr(A_true, A_est):
    nonzero = A_true != 0
    return pearsonr(A_true[nonzero].flatten(), A_est[nonzero].flatten())[0]

def compute_zero_rmse(A_true, A_est):
    zero = A_true == 0
    return np.sqrt(np.mean(np.square(A_est[zero])))

def plot_sample_size_results(days_btwn, seq_depth):
    seq_depth = "inf" if seq_depth == -1 else seq_depth
    sample_size = [5, 10, 25, 50]

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    rmse_A = []
    rmse_g = []
    rmse_prediction = []

    for i in range(50):
        for ss in sample_size:

            truth_filename = "tmp/clv-sim-set-{}-{}-{}-{}.pkl".format(seq_depth, days_btwn, ss, i)
            try:
                train_counts, train_rel_abn, train_t_pts, hold_out_counts, hold_out_rel_abn, hold_out_t_pts, A, g, denom = \
                    pkl.load(open(truth_filename, "rb"))
            except FileNotFoundError:
                print("missing file", truth_filename)

            clv_en_filename = "tmp/clv-sim-en-{}-{}-{}-{}.pkl".format(seq_depth, days_btwn, ss, i)
            try:
                A_clv_en, g_clv_en, pred_clv_en = pkl.load(open(clv_en_filename, "rb"))
                rmse_A.append(["cLV Elastic Net", ss, compute_rmse_params(A, A_clv_en)])
                rmse_g.append(["cLV Elastic Net", ss, compute_rmse_params(g, g_clv_en)])
                rmse_prediction.append(["cLV Elastic Net", ss, compute_rmse_trajectories(hold_out_rel_abn, pred_clv_en)])
            except FileNotFoundError:
                print("missing file", clv_en_filename)

            clv_rg_filename = "tmp_sim/clv-sim-rg-{}-{}-{}-{}.pkl".format(seq_depth, days_btwn, ss, i)
            try:
                 A_clv_rg, g_clv_rg, pred_clv_rg = pkl.load(open(clv_rg_filename, "rb"))
                 rmse_A.append(["cLV Ridge", ss, compute_rmse_params(A, A_clv_rg)])
                 rmse_g.append(["cLV Ridge", ss, compute_rmse_params(g, g_clv_rg)])
                 rmse_prediction.append(["cLV Ridge", ss, compute_rmse_trajectories(hold_out_rel_abn, pred_clv_rg)])
            except FileNotFoundError:
                print("missing file", clv_rg_filename)



    rmse_A = pd.DataFrame(rmse_A, columns=["Model", "Sample Size", "RMSE"])
    rmse_g = pd.DataFrame(rmse_g, columns=["Model", "Sample Size", "RMSE"])
    rmse_prediction = pd.DataFrame(rmse_prediction, columns=["Model", "Sample Size", "RMSE"])

    sns.boxplot(ax=ax[0], x="Sample Size", y="RMSE", hue="Model", data=rmse_A, palette="colorblind")
    sns.boxplot(ax=ax[1], x="Sample Size", y="RMSE", hue="Model", data=rmse_g, palette="colorblind")
    sns.boxplot(ax=ax[2], x="Sample Size", y="RMSE", hue="Model", data=rmse_prediction, palette="colorblind")

    ax[0].set_title("Interactions")
    ax[1].set_title("Growth")
    ax[2].set_title("Predictions")
    ax[0].legend().remove()
    ax[1].legend().remove()

    box_pairs = [((5, "cLV Elastic Net"), (5, "cLV Ridge")),
                 ((10, "cLV Elastic Net"), (10, "cLV Ridge")),
                 ((25, "cLV Elastic Net"), (25, "cLV Ridge")),
                 ((50, "cLV Elastic Net"), (50, "cLV Ridge"))]
    add_stat_annotation(ax[0], data=rmse_A, x="Sample Size", y="RMSE", hue="Model",
                            loc='inside', verbose=2, text_format='star', box_pairs=box_pairs[::-1],
                            test='Wilcoxon', line_offset=0.02, line_offset_to_box=0.05,
                            line_height=0.0, linewidth=1, text_offset=0.01
                            )

    add_stat_annotation(ax[1], data=rmse_g, x="Sample Size", y="RMSE", hue="Model",
                        loc='inside', verbose=2, text_format='star', box_pairs=box_pairs[::-1],
                        test='Wilcoxon', line_offset=0.02, line_offset_to_box=0.05,
                        line_height=0.0, linewidth=1, text_offset=0.01
                        )

    add_stat_annotation(ax[2], data=rmse_prediction, x="Sample Size", y="RMSE", hue="Model",
                        loc='inside', verbose=2, text_format='star', box_pairs=box_pairs[::-1],
                        test='Wilcoxon', line_offset=0.02, line_offset_to_box=0.05,
                        line_height=0.0, linewidth=1, text_offset=0.01
                        )


    ax[0].set_ylim((0,ax[0].get_ylim()[1]))
    ax[1].set_ylim((0,ax[0].get_ylim()[1]))
    ax[2].set_ylim((0,ax[2].get_ylim()[1]))

    plt.tight_layout()
    plt.savefig("plots/sim-{}-{}.pdf".format(days_btwn, seq_depth))


def plot_seq_depth_results(days_btwn, sample_size):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    rmse_A = []
    rmse_g = []
    rmse_prediction = []

    ss = sample_size
    depths = [1000, 2000, 5000, 10000, 25000, "inf"]
    for i in range(50):
        for seq_depth in depths:

            truth_filename = "tmp_sim/clv-sim-set-{}-{}-{}-{}.pkl".format(seq_depth, days_btwn, ss, i)
            try:
                train_counts, train_rel_abn, train_t_pts, hold_out_counts, hold_out_rel_abn, hold_out_t_pts, A, g, denom = \
                    pkl.load(open(truth_filename, "rb"))
            except FileNotFoundError:
                print("missing file", truth_filename)

            clv_en_filename = "tmp_sim/clv-sim-en-{}-{}-{}-{}.pkl".format(seq_depth, days_btwn, ss, i)
            try:
                A_clv_en, g_clv_en, pred_clv_en = pkl.load(open(clv_en_filename, "rb"))
                rmse_A.append(["cLV Elastic Net", seq_depth, compute_rmse_params(A, A_clv_en)])
                rmse_g.append(["cLV Elastic Net", seq_depth, compute_rmse_params(g, g_clv_en)])
                rmse_prediction.append(["cLV Elastic Net", seq_depth, compute_rmse_trajectories(hold_out_rel_abn, pred_clv_en)])
            except FileNotFoundError:
                print("missing file", clv_en_filename)


            # clv_rg_filename = "tmp_sim/clv-sim-rg-{}-{}-{}-{}.pkl".format(seq_depth, days_btwn, ss, i)
            # try:
            #      A_clv_rg, g_clv_rg, pred_clv_rg = pkl.load(open(clv_rg_filename, "rb"))
            #      rmse_A.append(["cLV Ridge", seq_depth, compute_rmse_params(A, A_clv_rg)])
            #      rmse_g.append(["cLV Ridge", seq_depth, compute_rmse_params(g, g_clv_rg)])
            #      rmse_prediction.append(["cLV Ridge", seq_depth, compute_rmse_trajectories(hold_out_rel_abn, pred_clv_rg)])
            # except FileNotFoundError:
            #     print("missing file", clv_rg_filename)



    rmse_A = pd.DataFrame(rmse_A, columns=["Model", "Depth", "RMSE"])
    rmse_g = pd.DataFrame(rmse_g, columns=["Model", "Depth", "RMSE"])
    rmse_prediction = pd.DataFrame(rmse_prediction, columns=["Model", "Depth", "RMSE"])

    sns.boxplot(ax=ax[0], x="Depth", y="RMSE", hue="Model", data=rmse_A, palette="colorblind")
    sns.boxplot(ax=ax[1], x="Depth", y="RMSE", hue="Model", data=rmse_g, palette="colorblind")
    sns.boxplot(ax=ax[2], x="Depth", y="RMSE", hue="Model", data=rmse_prediction, palette="colorblind")

    ax[0].set_title("Interactions")
    ax[1].set_title("Growth")
    ax[2].set_title("Predictions")
    ax[0].legend().remove()
    ax[1].legend().remove()

    ax[0].set_xticklabels(["1000", "2000", "5000", "10000", "25000", "no noise"])
    ax[1].set_xticklabels(["1000", "2000", "5000", "10000", "25000", "no noise"])
    ax[2].set_xticklabels(["1000", "2000", "5000", "10000", "25000", "no noise"])

    ax[0].set_ylim((0,ax[0].get_ylim()[1]))
    ax[1].set_ylim((0,ax[0].get_ylim()[1]))
    ax[2].set_ylim((0,ax[0].get_ylim()[1]))

    plt.tight_layout()
    plt.savefig("plots/sim-depths-{}-{}.pdf".format(days_btwn, sample_size))



def plot_density_results(ax, seq_depth, sample_size, legend=False):
    #fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    rmse_A = []
    rmse_g = []
    rmse_prediction = []

    ss = sample_size
    days = [1, 2, 4, 6]
    for i in range(50):
        for days_btwn in days:

            truth_filename = "tmp_sim/clv-sim-set-{}-{}-{}-{}.pkl".format(seq_depth, days_btwn, ss, i)
            try:
                train_counts, train_rel_abn, train_t_pts, hold_out_counts, hold_out_rel_abn, hold_out_t_pts, A, g, denom = \
                    pkl.load(open(truth_filename, "rb"))
            except FileNotFoundError:
                print("missing file", truth_filename)

            clv_en_filename = "tmp_sim/clv-sim-en-{}-{}-{}-{}.pkl".format(seq_depth, days_btwn, ss, i)
            try:
                A_clv_en, g_clv_en, pred_clv_en = pkl.load(open(clv_en_filename, "rb"))
                rmse_A.append(["cLV Elastic Net", days_btwn, compute_rmse_params(A, A_clv_en)])
                rmse_g.append(["cLV Elastic Net", days_btwn, compute_rmse_params(g, g_clv_en)])
                rmse_prediction.append(["cLV Elastic Net", days_btwn, compute_rmse_trajectories(hold_out_rel_abn, pred_clv_en)])
            except FileNotFoundError:
                print("missing file", clv_en_filename)

            #plot_trajectories(pred_clv_en[:5], hold_out_t_pts[:5], "tmp_plots", "pred-en-{}-{}".format(ss, days_btwn))

            # clv_rg_filename = "tmp_sim/clv-sim-rg-{}-{}-{}-{}.pkl".format(seq_depth, days_btwn, ss, i)
            # try:
            #      A_clv_rg, g_clv_rg, pred_clv_rg = pkl.load(open(clv_rg_filename, "rb"))
            #      rmse_A.append(["cLV Ridge", days_btwn, compute_rmse_params(A, A_clv_rg)])
            #      rmse_g.append(["cLV Ridge", days_btwn, compute_rmse_params(g, g_clv_rg)])
            #      rmse_prediction.append(["cLV Ridge", days_btwn, compute_rmse_trajectories(hold_out_rel_abn, pred_clv_rg)])
            # except FileNotFoundError:
            #     print("missing file", clv_rg_filename)

            #plot_trajectories(pred_clv_rg[:5], hold_out_t_pts[:5], "tmp_plots", "pred-rg-{}-{}".format(ss, days_btwn))

    rmse_A = pd.DataFrame(rmse_A, columns=["Model", "Days Between Obs", "RMSE"])
    rmse_g = pd.DataFrame(rmse_g, columns=["Model", "Days Between Obs", "RMSE"])
    rmse_prediction = pd.DataFrame(rmse_prediction, columns=["Model", "Days Between Obs", "RMSE"])

    sns.boxplot(ax=ax[0], x="Days Between Obs", y="RMSE", hue="Model", data=rmse_A, palette="colorblind")
    sns.boxplot(ax=ax[1], x="Days Between Obs", y="RMSE", hue="Model", data=rmse_g, palette="colorblind")
    sns.boxplot(ax=ax[2], x="Days Between Obs", y="RMSE", hue="Model", data=rmse_prediction, palette="colorblind")

    ax[0].set_title("Interactions")
    ax[1].set_title("Growth")
    ax[2].set_title("Predictions")

    if legend:
        ax[0].legend().remove()
        ax[1].legend().remove()
    else:
        ax[0].legend().remove()
        ax[1].legend().remove()
        ax[2].legend().remove()

    ymin = 0
    ymax = 0.75
    #ax[0].set_ylim((0,ax[0].get_ylim()[1]))
    #ax[1].set_ylim((0,ax[0].get_ylim()[1]))
    #ax[2].set_ylim((0,ax[2].get_ylim()[1]))

    ax[0].set_ylim((0,ymax))
    ax[1].set_ylim((0,ymax))
    ax[2].set_ylim((0,ymax))

    ax[1].set_title("N = {}".format(ss))

    #plt.tight_layout()
    #plt.savefig("tmp_plots/sim-density-{}-{}.pdf".format(ss, seq_depth))



if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    plot_sample_size_results(1, -1)
    plot_sample_size_results(1, 25000)
    plot_seq_depth_results(1, 50)