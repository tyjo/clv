import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns

from scipy.special import logsumexp
from scipy.stats import spearmanr

from optimizers import estimate_latent_from_observations, \
                       least_squares_latent, \
                       least_squares_observed, \
                       elastic_net


pseudo_count = 1e-5

def estimate_elastic_net_regularizers_cv(Y, U, T, folds):
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

                train_X = estimate_latent_from_observations(train_Y, pseudo_count)
                Q_inv = np.eye(train_X[0].shape[1])
                A, g, B = elastic_net(train_X, train_U, train_T, Q_inv, r_A, r_g, r_B, alpha=alpha, tol=1e-3)
                np.set_printoptions(suppress=True)
                sqr_err += compute_prediction_error(test_Y, test_U, test_T, A, g, B)

            if sqr_err < best_sqr_err:
                best_r = (alpha, r_A, r_g, r_B)
                best_sqr_err = sqr_err
                print("\tr", (alpha, r_A, r_g, r_B), "sqr error", sqr_err)
    return best_r


def estimate_ridge_regularizers_cv(Y, U, T, folds):
    regularizers = [0.1, 1, 10, 100, 1000]
    rA_rg_rB = []
    for r_A in regularizers:
        for r_g in regularizers:
            for r_B in regularizers:
                rA_rg_rB.append( (r_A, r_g, r_B ) )
    
    best_r = 0
    best_sqr_err = np.inf
    for r_A, r_g, r_B in rA_rg_rB:
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

                train_X = estimate_latent_from_observations(train_Y, pseudo_count)
                A, g, B = least_squares_latent(train_X, train_U, train_T, r_A, r_g, r_B)
                np.set_printoptions(suppress=True)
                sqr_err += compute_prediction_error(test_Y, test_U, test_T, A, g, B)

            if sqr_err < best_sqr_err:
                best_r = (r_A, r_g, r_B)
                best_sqr_err = sqr_err
                print("\tr", (r_A, r_g, r_B), "sqr error", sqr_err)
    return best_r


def compute_square_error(true, est):
    true = np.copy(true)
    est = np.copy(est)
    true /= true.sum()
    est /= est.sum()
    true = (true + pseudo_count) / (true + pseudo_count).sum()
    est = (est + pseudo_count) / (est + pseudo_count).sum()

    return np.square(true - est).sum()


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


def simulate(days_sampled, days_btwn, ntaxa, effects, nsimulations=10):
    total_days = days_sampled*days_btwn
    u = np.zeros( (total_days, 1) )
    for i in range(3):
        idx = int( (total_days/3)*(i+1) ) - 1
        u[idx] = 1

    obs_dim = ntaxa
    effect_dim = 1

    A  = np.random.normal(loc=0,scale=0.2,size=(ntaxa-1, ntaxa))
    B  = np.random.normal(loc=0,scale=0.2, size=(ntaxa-1, effect_dim))
    g  = np.random.normal(loc=0,scale=0.1,size=ntaxa-1)


    Y = []
    U = []
    T = []
    for n in range(nsimulations):
        x,y = simulate_sequence(A, g, B, total_days, u, 10000)
        y_sparse = y[ [t for t in range(0, total_days, days_btwn)] ]
        
        # code 1 for effects it occurs in the interval between observations
        u_sparse = np.zeros( (y.shape[0], 1) )
        times = []
        for t in range(0, total_days, days_btwn):
            has_effect = u[t:(t+days_btwn)].sum() >= 1
            if has_effect:
                u_sparse[t] = 1
            times.append(t)
        times = np.array(times)

        assert u_sparse.sum() == u.sum(), "{} != {}".format(u_sparse.sum(), u.sum())

        Y.append(y_sparse)
        U.append(u_sparse)
        T.append(times)

    return Y, U, T, A, B, g



def simulate_sequence(A, g, B, T, u, N=10000):
        """Simulate a sequence of length T.

        Parameters
        ----------
            A  : species interaction matrix
            g  : species growth rates
            B  : external perturbation matrix
            T  : number of time points to simulate
            u  : T x effect_dim numpy array of external effects
            N  : Poisson parameter for number of observed reads

        Returns
        -------
            x  : T x latent_dim numpy array of latent states
            y  : T x obs_dim numpy array of multinomial observations

        """
        latent_dim = A.shape[0]
        x = []
        y = []
        mu  = np.random.multivariate_normal(mean=np.zeros(latent_dim), cov=np.eye(latent_dim))
        for t in range(T):
            xt = mu

            # increase dimension by 1
            xt1 = np.concatenate((xt, np.array([0])))
            pt = np.exp(xt1 - logsumexp(xt1))

            # simulate total number of reads with over-dispersion
            logN = np.random.normal(loc=np.log(N), scale=0.5)
            Nt = np.random.poisson(np.exp(logN))
            yt = np.random.multinomial(Nt, pt).astype(float)

            x.append(xt)
            y.append(yt)

            mu  = xt + g + A.dot(pt) + B.dot(u[t])

        x = np.array(x)
        y = np.array(y)
        return x, y


def compute_nonzero_spearmean_r(true, est, est_elastic_net):
    est = est[est_elastic_net != 0].flatten()
    true = true[est_elastic_net != 0].flatten()
    return spearmanr(true, est)


def compute_taxa_medians(Y):
    t_pts = Y[0].shape[0]
    total = len(Y)*t_pts
    obs_dim = Y[0].shape[1]
    p_all = np.zeros((total, obs_dim))
    idx = 0
    for y in Y:
        for yt in y:
            p_all[idx] = yt / yt.sum()
            idx += 1
    return np.median(p_all, axis=0)


def compute_interaction_spearman_r(true, est, Y):
    rare_rare = []
    rare_common = []
    common_rare = []
    common_common = []

    medians = compute_taxa_medians(Y)
    rare = medians < 0.01
    for i in range(true.shape[0]):
        for j in range(true.shape[1]):
            if est[i,j] == 0:
                continue

            if rare[i] and rare[j]:
                rare_rare.append( (true[i,j], est[i,j]) )
            elif rare[i] and not rare[j]:
                rare_common.append( (true[i,j], est[i,j]) )
            elif not rare[i] and rare[j]:
                common_rare.append( (true[i,j], est[i,j]) )
            else:
                common_common.append( (true[i,j], est[i,j]) )

    return (spearmanr(rare_rare)[0], len(rare_rare)), \
           (spearmanr(rare_common)[0], len(rare_common)), \
           (spearmanr(common_rare)[0], len(common_rare)), \
           (spearmanr(common_common)[0], len(common_common))


def compute_effects_spearman_r(true, est, Y):
    rare = []
    common = []

    medians = compute_taxa_medians(Y)
    rare_taxa = medians < 0.01
    for i in range(true.shape[0]):
        if est[i] == 0:
            continue
        if rare_taxa[i]:
            rare.append( (true[i, 0], est[i, 0]) )
        else:
            common.append( (true[i, 0], est[i, 0]) )
    return (spearmanr(rare)[0], len(rare)), \
           (spearmanr(common)[0], len(common))


def compute_growth_spearman_r(true, est, Y):
    rare = []
    common = []

    medians = compute_taxa_medians(Y)
    rare_taxa = medians < 0.01
    for i in range(true.shape[0]):
        if est[i] == 0:
            continue
        if rare_taxa[i]:
            rare.append( (true[i], est[i]) )
        else:
            common.append( (true[i], est[i]) )
    return (spearmanr(rare)[0], len(rare)), \
           (spearmanr(common)[0], len(common))


def compute_rmse(true, est):
    return np.sqrt(np.square(true - est).mean())


if __name__ == "__main__":
    np.random.seed(19006)

    parser = argparse.ArgumentParser(description="Run experiments to compare inference methods.")
    parser.add_argument("sample_size", metavar="sample-size", type=int, help="Number of replicates.")
    parser.add_argument("days_sampled", metavar="days-sampled", type=int, help="Number of observed time points.")
    parser.add_argument("days_between", metavar="days-between", type=int, help="Days between observations.")
    parser.add_argument("ntaxa", type=int, help="Number of taxa to simulate.")
    parser.add_argument("--effects", default=False, action="store_true", help="Simulate with external effects.")
    
    args = parser.parse_args()
    sample_size = args.sample_size
    days_sampled = args.days_sampled
    days_btwn = args.days_between
    ntaxa = args.ntaxa
    effects = args.effects

    rmse_all = []
    spearman_corr_A = []
    spearman_corr_B = []
    spearman_corr_g = []
    spearman_corr_en_A = []
    spearman_corr_en_B = []
    spearman_corr_en_g = []

    for i in range(10):
        print("running replicate", i, '...')
        np.set_printoptions(suppress=True)
        Y, U, T, A, B, g = simulate(days_sampled, days_btwn, ntaxa, effects, sample_size)
        X = estimate_latent_from_observations(Y, pseudo_count)
        
        try:
            medians, AgB, AgB_ols, AgB_en = pkl.load(open("tmp/{}taxa-{}simulations-{}days-{}btwn-{}.pkl".format(ntaxa, sample_size, days_sampled, days_btwn, i), "rb"))
            A, g, B = AgB
            A_ols, g_ols, B_ols = AgB_ols
            A_en, g_en, B_en = AgB_en

        except FileNotFoundError:
            print("\tOLS...")
            A_ols, g_ols, B_ols = least_squares_latent(X, U, T)

            print("\tElastic Net...")
            Q_inv = np.eye(ntaxa-1)
            alpha, r_A, r_g, r_B = estimate_elastic_net_regularizers_cv(Y, U, T, folds=3)
            A_en, g_en, B_en = elastic_net(X, U, T, Q_inv, r_A, r_g, r_B, alpha)


        spearmanA = compute_interaction_spearman_r(A, A_ols, Y)
        spearmanB = compute_effects_spearman_r(B, B_ols, Y)
        spearmang = compute_growth_spearman_r(g, g_ols, Y)
        rmse_all.append(["OLS", "A", compute_rmse(A, A_ols)])
        rmse_all.append(["OLS", "B", compute_rmse(B, B_ols)])
        rmse_all.append(["OLS", "g", compute_rmse(g, g_ols)])
        spearman_corr_A.append(["OLS", "rare-\n rare", spearmanA[0][0]])
        spearman_corr_A.append(["OLS", "rare-\n common", spearmanA[1][0]])
        spearman_corr_A.append(["OLS", "common-\n rare", spearmanA[2][0]])
        spearman_corr_A.append(["OLS", "common-\n common", spearmanA[3][0]])
        spearman_corr_B.append(["OLS", "rare", spearmanB[0][0]])
        spearman_corr_B.append(["OLS", "common", spearmanB[1][0]])
        spearman_corr_g.append(["OLS", "rare", spearmang[0][0]])
        spearman_corr_g.append(["OLS", "common", spearmang[1][0]])

        spearmanA = compute_interaction_spearman_r(A, A_en, Y)
        spearmanB = compute_effects_spearman_r(B, B_en, Y)
        spearmang = compute_growth_spearman_r(g, g_en, Y)
        rmse_all.append(["EN", "A", compute_rmse(A, A_en)])
        rmse_all.append(["EN", "B", compute_rmse(B, B_en)])
        rmse_all.append(["EN", "g", compute_rmse(g, g_en)])
        spearman_corr_A.append(["EN", "rare-\n rare", spearmanA[0][0]])
        spearman_corr_A.append(["EN", "rare-\n common", spearmanA[1][0]])
        spearman_corr_A.append(["EN", "common-\n rare", spearmanA[2][0]])
        spearman_corr_A.append(["EN", "common-\n common", spearmanA[3][0]])
        spearman_corr_B.append(["EN", "rare", spearmanB[0][0]])
        spearman_corr_B.append(["EN", "common", spearmanB[1][0]])
        spearman_corr_g.append(["EN", "rare", spearmang[0][0]])
        spearman_corr_g.append(["EN", "common", spearmang[1][0]])
        spearman_corr_en_A.append(["EN", "rare-\n rare", spearmanA[0][0]])
        spearman_corr_en_A.append(["EN", "rare-\n common", spearmanA[1][0]])
        spearman_corr_en_A.append(["EN", "common-\n rare", spearmanA[2][0]])
        spearman_corr_en_A.append(["EN", "common-\n common", spearmanA[3][0]])
        spearman_corr_en_B.append(["EN", "rare", spearmanB[0][0]])
        spearman_corr_en_B.append(["EN", "common", spearmanB[1][0]])
        spearman_corr_en_g.append(["EN", "rare", spearmang[0][0]])
        spearman_corr_en_g.append(["EN", "common", spearmang[1][0]])

        medians = compute_taxa_medians(Y)
        results = ( medians,
                    (A,g,B), 
                    (A_ols, g_ols, B_ols),
                    (A_en, g_en, B_en)
                  )
        pkl.dump(results, open("tmp/{}taxa-{}simulations-{}days-{}btwn-{}.pkl".format(ntaxa, sample_size, days_sampled, days_btwn, i), "wb"))

    df = pd.DataFrame(rmse_all, columns=["", "Parameters", "RMSE"])
    sns.set_context(rc={"lines.linewidth" : 0.5, "font.size" : 14})
    ax = sns.boxplot(x="Parameters", y="RMSE", hue="", data=df)
    plt.savefig("plots/{}taxa-{}simulations-{}days-{}btwn-rmse-all.pdf".format(ntaxa, sample_size, days_sampled, days_btwn))
    plt.close()

    palette = sns.color_palette()
    color = palette[1]

    df = pd.DataFrame(spearman_corr_en_A, columns=["", "Interactions", "Spearman R"])
    sns.set_context(rc={"lines.linewidth" : 0.25, "font.size" : 14})
    ax = sns.boxplot(x="Interactions", y="Spearman R", data=df, color=color)
    #ax.legend_.remove()
    plt.ylim(-1.1, 1.1)
    plt.subplots_adjust(left=0.15)
    plt.savefig("plots/{}taxa-{}simulations-{}days-{}btwn-spearmanr_A-en-stratified.pdf".format(ntaxa, sample_size, days_sampled, days_btwn))
    plt.close()

    df = pd.DataFrame(spearman_corr_en_B, columns=["", "External Effects", "Spearman R"])
    sns.set_context(rc={"lines.linewidth" : 0.25})
    ax = sns.boxplot(x="External Effects", y="Spearman R", data=df, color=color)
    #ax.legend_.remove()
    plt.ylim(-1.1, 1.1)
    plt.subplots_adjust(left=0.15)
    plt.savefig("plots/{}taxa-{}simulations-{}days-{}btwn-spearmanr_B-en-stratified.pdf".format(ntaxa, sample_size, days_sampled, days_btwn))
    plt.close()

    df = pd.DataFrame(spearman_corr_en_g, columns=["", "Growth Rates", "Spearman R"])
    sns.set_context(rc={"lines.linewidth" : 0.25})
    ax = sns.boxplot(x="Growth Rates", y="Spearman R", data=df, color=color)
    #ax.legend_.remove()
    plt.ylim(-1.1, 1.1)
    plt.subplots_adjust(left=0.15)
    plt.savefig("plots/{}taxa-{}simulations-{}days-{}btwn-spearmanr_g-en-stratified.pdf".format(ntaxa, sample_size, days_sampled, days_btwn))
    plt.close()


    df = pd.DataFrame(spearman_corr_A, columns=["", "Interactions", "Spearman R"])
    sns.set_context(rc={"lines.linewidth" : 0.25, "font.size" : 14})
    ax = sns.boxplot(x="Interactions", y="Spearman R", data=df, hue="")
    plt.ylim(-1.1, 1.1)
    plt.subplots_adjust(left=0.15)
    plt.savefig("plots/{}taxa-{}simulations-{}days-{}btwn-spearmanr_A-stratified.pdf".format(ntaxa, sample_size, days_sampled, days_btwn))
    plt.close()

    df = pd.DataFrame(spearman_corr_B, columns=["", "External Effects", "Spearman R"])
    sns.set_context(rc={"lines.linewidth" : 0.25})
    ax = sns.boxplot(x="External Effects", y="Spearman R", data=df, hue="")
    plt.ylim(-1.1, 1.1)
    plt.subplots_adjust(left=0.15)
    plt.savefig("plots/{}taxa-{}simulations-{}days-{}btwn-spearmanr_B-stratified.pdf".format(ntaxa, sample_size, days_sampled, days_btwn))
    plt.close()

    df = pd.DataFrame(spearman_corr_g, columns=["", "Growth Rates", "Spearman R"])
    sns.set_context(rc={"lines.linewidth" : 0.25})
    ax = sns.boxplot(x="Growth Rates", y="Spearman R", data=df, hue="")
    plt.ylim(-1.1, 1.1)
    plt.subplots_adjust(left=0.15)
    plt.savefig("plots/{}taxa-{}simulations-{}days-{}btwn-spearmanr_g-stratified.pdf".format(ntaxa, sample_size, days_sampled, days_btwn))
    plt.close()
