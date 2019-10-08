import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import roc_curve, auc

from taur_cross_validation import predict, predict_linear, predict_rel_abun, estimate_rel_abun
from optimizers import elastic_net, elastic_net_linear, estimate_latent_from_observations
from util import load_observations


def make_train_test(Y, U, T, U_normalized, fold, nfolds, taxon_idx):
    has_dom_idx = []
    no_dom_idx = []

    for idx,y in enumerate(Y):

        # only want samples that develop domination
        p = ( y.T / y.sum(axis=1) ).T
        if p[0, taxon_idx] > 0.3:
            continue

        if np.any(p[1:,taxon_idx] > 0.3):
            has_dom_idx.append(idx)
            continue
        else:
            no_dom_idx.append(idx)
            continue

    Y_train = []
    U_train = []
    T_train = []
    U_normalized_train = []

    Y_test = []
    U_test = []
    T_test = []
    U_normalized_test = []

    for j,idx in enumerate(has_dom_idx):
        if j % nfolds == fold:
            Y_test.append(Y[idx])
            U_test.append(U[idx])
            T_test.append(T[idx])
            U_normalized_test.append(U_normalized[idx])
        else:
            Y_train.append(Y[idx])
            U_train.append(U[idx])
            T_train.append(T[idx])
            U_normalized_train.append(U_normalized[idx])

    for j,idx in enumerate(no_dom_idx):
        if j % nfolds == fold:
            Y_test.append(Y[idx])
            U_test.append(U[idx])
            T_test.append(T[idx])
            U_normalized_test.append(U_normalized[idx])
        else:
            Y_train.append(Y[idx])
            U_train.append(U[idx])
            T_train.append(T[idx])
            U_normalized_train.append(U_normalized[idx])

    return Y_train, U_train, T_train, U_normalized_train, \
           Y_test, U_test, T_test, U_normalized_test


def compute_tpr_fpr(Y, U_normalized, T, A, g, B, taxon_idx):
    tp = 0
    fp = 0

    n_tp = 0
    n_fp = 0

    for idx, y in enumerate(Y):

        # only want samples that develop domination
        p = ( y.T / y.sum(axis=1) ).T
        if p[0, taxon_idx] > 0.3:
            continue

        has_dom = np.any(p[1:, taxon_idx] > 0.3)

        y_pred = predict(y[0], U_normalized[idx], T[idx], A, g, B)
        pred_dom = np.any(y_pred[1:,taxon_idx] > 0.3) 

        if pred_dom and has_dom:
            tp += 1
        elif pred_dom and not has_dom:
            fp += 1

        if has_dom:
            n_tp += 1
        else:
            n_fp += 1

    return tp / n_tp, fp / n_fp


def compute_roc(Y, U_normalized, T, A, g, B, taxon_idx):
    true_labels = []
    scores = []

    for idx, y in enumerate(Y):

        # only want samples that develop domination
        p = ( y.T / y.sum(axis=1) ).T
        if p[0, taxon_idx] > 0.3:
            continue

        has_dom = np.any(p[1:, taxon_idx] > 0.3)
        true_labels.append(1 if has_dom else 0)

        u = U_normalized[idx]
        times = T[idx]
        y_pred = predict(y[0], U_normalized[idx], T[idx], A, g, B)
        score = y_pred[1:,taxon_idx].max()
        scores.append(score)

    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc, np.array(true_labels).sum()


def compute_roc_linear(Y, U_normalized, T, A, g, B, taxon_idx):
    true_labels = []
    scores = []

    for idx, y in enumerate(Y):

        # only want samples that develop domination
        p = ( y.T / y.sum(axis=1) ).T
        if p[0, taxon_idx] > 0.3:
            continue

        has_dom = np.any(p[1:, taxon_idx] > 0.3)
        true_labels.append(1 if has_dom else 0)

        u = U_normalized[idx]
        times = T[idx]
        y_pred = predict_linear(y[0], U_normalized[idx], T[idx], A, g, B)
        score = y_pred[1:,taxon_idx].max()
        scores.append(score)

    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc, np.array(true_labels).sum()



def predict_enterococcus_domination_cv(Y, U, T, U_normalized):
    ent_idx = 4
    nfolds = 5

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for fold in range(nfolds):
        print("fold", fold)
        Y_train, U_train, T_train, U_normalized_train, Y_test, U_test, T_test, U_normalized_test = make_train_test(Y, U, T, U_normalized, fold, nfolds, ent_idx)
        X_train = estimate_latent_from_observations(Y_train)
        Q_inv = np.eye(Y[0].shape[1]-1)
        A, g, B = elastic_net(X_train, U_train, T_train, Q_inv, r_A=0.7, r_g=0.9, r_B=0, alpha=0.1, tol=1e-3)

        fpr, tpr, roc_auc, tp = compute_roc(Y_test, U_normalized_test, T_test, A, g, B, ent_idx)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        plt.plot(fpr, tpr, lw=1, alpha=0.3,
         label='ROC Fold %d (AUC = %0.2f; TP: %d)' % (fold, roc_auc, tp))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
     label='Random Guess', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title( 'Enterococcus Domination')
    plt.legend(loc="lower right")
    plt.savefig('plots/enterococcus_domination.pdf')
    plt.close()


def plot_heatmaps(Y, U, T):
    taxa_labels = np.loadtxt("data/taur/taur-otu-table-top10+dom.csv", delimiter=",", dtype=str)[2:,0].tolist()

    taxa_labels = [label.split(";")[-1] for label in taxa_labels[:-1]] + ["Other"]

    Q_inv = np.eye(Y[0].shape[1]-1)
    X = estimate_latent_from_observations(Y)
    A, g, B = elastic_net(X, U, T, Q_inv, r_A=0.7, r_g=0.9, r_B=0, alpha=0.1, tol=1e-3)
    
    m = np.abs(A).max()
    ax= sns.heatmap(A, xticklabels=taxa_labels, annot=True, annot_kws={"size": 3.5}, yticklabels=taxa_labels[:-1], vmin=-m, vmax=m, cmap="coolwarm")
    plt.setp(ax.get_xticklabels(),fontsize=7)
    plt.setp(ax.get_yticklabels(),fontsize=7)
    plt.tight_layout()
    plt.savefig("plots/A-heatmap.pdf")
    plt.close()

    antibiotics = {'tetracyclines': 13, 'fourth/fifth generation cephalosporins': 5, 'anti-anaerobic agent': 11, 'penicillins': 1, 'beta-lactamase inhibitors': 12, 'carbapenems': 10, 'first/second generation cephalosporins': 4, 'quinolones': 7, 'anti-VRE agents': 8, 'glycopeptide': 14, 'monobactams': 3, 'third generation cephalosporins': 6, 'surgery': 0, 'PCP prophylaxis agents': 2, 'aminoglycosides': 9}
    antibiotics = sorted(antibiotics.items(), key = lambda item: item[1])
    antibiotics = [item[0] for item in antibiotics]
    

    m = np.abs(B).max()
    ax= sns.heatmap(B, xticklabels=antibiotics, annot=True, annot_kws={"size": 3.5}, yticklabels=taxa_labels, vmin=-m, vmax=m, cmap="coolwarm")
    plt.setp(ax.get_xticklabels(),fontsize=7)
    plt.setp(ax.get_yticklabels(),fontsize=7)
    plt.tight_layout()
    plt.savefig("plots/B-heatmap.pdf")
    plt.close()


    g = g.reshape((g.size, 1))
    m = np.abs(g).max()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3,6))
    sns.heatmap(g, ax=ax, xticklabels=[], annot=True, annot_kws={"size": 3.5}, yticklabels=taxa_labels[:-1], vmin=-m, vmax=m, cmap="coolwarm")
    plt.setp(ax.get_xticklabels(),fontsize=7)
    plt.setp(ax.get_yticklabels(),fontsize=7)
    plt.tight_layout()
    plt.savefig("plots/g-heatmap.pdf")
    plt.close()


if __name__ == "__main__":
    Y, U, T, U_normalized, T_normalized = load_observations("data/taur/taur-otu-table-top10+dom.csv", "data/taur/taur-events.csv")

    # last 20 observations were used to estimate
    # regularization coefficients and should be
    # omitted.
    Y_train = Y[:-20]
    U_train = U[:-20]
    T_train = T[:-20]
    U_train_normalized = U_normalized[:-20]
    predict_enterococcus_domination_cv(Y_train, U_train, T_train, U_train_normalized)

    plot_heatmaps(Y, U, T)