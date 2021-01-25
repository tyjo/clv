import argparse
import numpy as np
import pickle as pkl
import os

from scipy.integrate import RK45, solve_ivp
from scipy.stats import pearsonr
from scipy.special import logsumexp

from compositional_lotka_volterra import CompositionalLotkaVolterra, \
                                         estimate_relative_abundances, \
                                         choose_denom, \
                                         construct_alr, \
                                         compute_rel_abun, \
                                         ridge_regression_clv
from generalized_lotka_volterra import GeneralizedLotkaVolterra, add_pseudo_counts


def least_squares_lotka_volterra(X, U, T):
    """Computes estimates of A, g, and B using least squares. 

    Parameters
    ----------
        Y : a list of T x yDim numpy arrays
        U : a list of T x uDim numpy arrays
        T   : a list of T x 1 numpy arrays with the time of each observation

    Returns
    -------
    """
    xDim = X[0].shape[1]
    uDim = U[0].shape[1]
    ntaxa = xDim

    # design matrices
    predictors = [[] for n in range(xDim)]
    outcomes = [[] for n in range(xDim)]
    for idx, (xi, ui) in enumerate(zip(X, U)):
        for t in range(1, xi.shape[0]):
            delT = T[idx][t] - T[idx][t-1]
            xt  = xi[t]
            xt0 = xi[t-1]
            ut0 = ui[t-1]

            xt_xt_T = np.outer(xt, xt)

            for n in range(ntaxa):
                outcomes[n].append((xt - xt0)[n] / delT)

                tmp = np.concatenate( (xt_xt_T[n], [xt[n]], [ui[t-1][i]*xt[n] for i in range(uDim)]))
                predictors[n].append(tmp)

    predictors = np.array(predictors)
    outcomes = np.array(outcomes)

    A = np.zeros((ntaxa,ntaxa))
    g = np.zeros(ntaxa)
    B = np.zeros((ntaxa,uDim))

    for n in range(ntaxa):
        P = predictors[n]
        Z = np.expand_dims(outcomes[n], axis=1)
        
        parameters = np.linalg.pinv(P.T.dot(P) + 0.001*np.eye(P.shape[1])).dot(P.T).dot(Z)
        A[:,n] = parameters[:xDim].flatten()
        g[n] = parameters[xDim]
        B[n,:] = parameters[xDim+1:].flatten()

    return A, g, B


def compute_simulation_parameters():
    Y = pkl.load(open("data/bucci/Y_cdiff-denoised.pkl", "rb"))
    U = pkl.load(open("data/bucci/U_cdiff.pkl", "rb"))
    T = pkl.load(open("data/bucci/T_cdiff.pkl", "rb"))
    Y = adjust_concentrations(Y)
    A,g,B = least_squares_lotka_volterra(Y, U, T)
    ntaxa = A.shape[0]

    g = np.abs(g)
    A[np.diag_indices(ntaxa)] = -np.abs(A[np.diag_indices(ntaxa)])

    A_self_mean = np.mean(A[np.diag_indices(ntaxa)].flatten())
    A_self_var = np.var(A[np.diag_indices(ntaxa)].flatten())
    A_interact_var = np.var(A[~np.eye(ntaxa,dtype=bool)].flatten())
    g_mean = np.mean(g)
    g_var = np.var(g)


    initial_cond = np.array([y[0] for y in Y])
    initial_mean = np.mean(initial_cond)
    initial_var = np.var(initial_cond)

    return A_self_mean, A_self_var, A_interact_var, \
           g_mean, g_var, \
           initial_mean, initial_var


def simulate_clv(A, g, initial_mean_abs, initial_var_abs, ntaxa, denom):

    def grad_fn(A, g):
        def fn(t, x):
            return g + A.dot(compute_rel_abun(x, denom).flatten())
        return fn

    n_tpts = 30
    c0 = [np.random.normal(initial_mean_abs, np.sqrt(initial_var_abs)) for i in range(ntaxa)]
    c0 = np.array(c0)
    c0[c0 < 0.05*initial_mean_abs] = 0.05*initial_mean_abs
    x0 = np.log(c0[:-1]) - np.log(c0[-1])

    grad = grad_fn(A, g)
    p = [compute_rel_abun(x0, denom).flatten()]

    xt = x0
    for t in range(n_tpts - 1):
        dt = 1
        ivp = solve_ivp(grad, (0,0+dt), xt, method="RK45")
        xt = ivp.y[:,-1]
        pt = compute_rel_abun(xt, denom).flatten()
        pt /= pt.sum()
        p.append(pt)

    return np.array(p)
    


def adjust_concentrations(Y):
    """Change the scale of observed concentrations.
    """
    con =  []
    for y in Y:
        con += y.sum(axis=1).tolist()
    con = np.array(con)
    C = 1 / np.mean(con)

    Y_adjusted = []
    for y in Y:
        Y_adjusted.append(y)

    return Y_adjusted



def swap_denom(swap, denom):
    swap = np.copy(swap)
    if swap.ndim == 1 and denom < swap.size:
        tmp = np.copy(swap[denom])
        swap = swap - swap[denom]
        swap[denom] = -tmp
    elif swap.ndim == 2 and denom < swap.shape[1]:
        tmp = np.copy(swap[:,denom])
        swap = (swap.T - swap[:,denom]).T
        swap[:,denom] = -tmp
    return swap


def generate_simulation_data(sample_size, hold_out_size, days_between, seq_depth):
    A_self_mean, A_self_var, A_interact_var, \
        g_mean, g_var, initial_mean, initial_var = compute_simulation_parameters()

    ntaxa = 10
    denom = ntaxa-1
    A = np.zeros((ntaxa, ntaxa))
    g = np.array([np.abs(np.random.normal(g_mean, np.sqrt(g_var))) for i in range(ntaxa)])
    z = np.random.binomial(1, 0.2, size=A.shape)

    w = None
    while (w is None) or np.any(w > 0):
        A = np.zeros((ntaxa,ntaxa))
        for i in range(ntaxa):
            for j in range(ntaxa):
                # self interactions
                if i == j:
                    A[i,j] = -np.abs(np.random.normal(A_self_mean, np.sqrt(A_self_var)))

                # if an interaction occurs
                if z[i,j] == 1:
                    A[i,j] = np.random.normal(0, np.sqrt(A_interact_var) / z[i,].sum())

        # ensure negative definiteness
        # gives us a "stably dissipative system"
        #A_z = np.vstack((A, np.zeros(ntaxa)))
        A_s = 0.5*(A + A.T)
        w,v = np.linalg.eig(A_s)

    A_glv = np.copy(A)
    g_glv = np.copy(g)
    A = A_glv[:ntaxa-1] - A_glv[-1]
    A = A[:ntaxa-1,:]
    g = g - g[ntaxa-1]
    g = g[:ntaxa-1]

    t_pts = [days_between*i for i in range(int(30/days_between))]
    P = []
    P_stacked = None
    stds = []

    while len(P) < sample_size + hold_out_size:
        pn = simulate_clv(A, g, initial_mean, initial_var, ntaxa, denom)

        P.append(pn)
        stds.append(np.std(pn,axis=0))
        if P_stacked is None:
            P_stacked = pn
        else:
            P_stacked = np.vstack((P_stacked, pn))

    if np.any(np.mean(P_stacked[:sample_size], axis=0) < 0.001) or np.any(P_stacked > 0.8):
        return "Failed"
    elif np.any(P_stacked[:sample_size,-1] < 0.001):
        return "Failed"

    # sequencing counts
    dispersion = 286 # estimated by Bucci et al.
    Y = []
    T = []

    for pn in P:

        if np.isfinite(seq_depth):
            mean_n = (pn / pn.sum(axis=1, keepdims=True))
            prob_n = np.array([np.random.dirichlet(dispersion*p) for p in mean_n])
            yn = np.array([np.random.multinomial(seq_depth, p) for p in prob_n])
        else:
            yn = pn / pn.sum(axis=1, keepdims=True)

        Y.append(yn[t_pts])
        T.append(np.array(t_pts))

    Y_train = Y[:sample_size]
    Y_hold_out = Y[sample_size:(sample_size+hold_out_size)]
    T_train = T[:sample_size]
    T_hold_out = [np.array([i for i in range(30)]) for j in range(hold_out_size)]
    P_train = P[:sample_size]
    P_hold_out = P[sample_size:(sample_size+hold_out_size)]
    return Y_train, P_train, T_train, Y_hold_out, P_hold_out, T_hold_out, A, g, denom


def fit_clv(observations, time_points, held_out_rel_abun, held_out_time_points, denom, method):
    print("Estimating cLV parameters using", method)

    rel_abun = estimate_relative_abundances(observations)
    clv = CompositionalLotkaVolterra(rel_abun, time_points, denom=denom)

    if method == "Elastic Net":
        clv.train()
        predictions = [clv.predict(o[0],tpts) for (o,tpts) in zip(held_out_rel_abun, held_out_time_points)]
        A, g, B = clv.get_params()
        return A, g, predictions
    elif method == "Ridge":
        clv.train_ridge()
        A, g, B = clv.get_params()
        predictions = [clv.predict(o[0],tpts) for (o,tpts) in zip(held_out_rel_abun, held_out_time_points)]
        return A, g, predictions
    else:
        print("bad optimization method for cLV", file=sys.stderr)
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_size", metavar="sample-size", type=int, help="Number of replicates.")
    parser.add_argument("days_between", metavar="days-between", type=int, help="Days between observations.")
    parser.add_argument("seq_depth", metavar="seq-depth", type=int, help="Sequencing Depth (-1 for noise free).")
    parser.add_argument("random_seed", metavar="random-seed", type=int, help="Random seed for simulations.")
    n_reps = 50

    args = parser.parse_args()
    ss = args.sample_size
    db = args.days_between
    depth = args.seq_depth
    seed = args.random_seed
    hold_out_size = 5

    # noise free
    if depth == -1:
        depth = np.inf

    np.random.seed(seed)

    for i in range(n_reps):
        print("simulating dataset {}...".format(i))

        try:
            train_counts, train_rel_abn, train_t_pts, hold_out_counts, hold_out_rel_abn, hold_out_t_pts, A, g, denom = pkl.load(open("tmp_sim/clv-sim-set-{}-{}-{}-{}.pkl".format(depth, db, ss, i), "rb"))

        except FileNotFoundError:
            simulated_dataset = "Failed"
            while simulated_dataset == "Failed":
                simulated_dataset = generate_simulation_data(ss, hold_out_size, db, depth)
            train_counts, train_rel_abn, train_t_pts, hold_out_counts, hold_out_rel_abn, hold_out_t_pts, A, g, denom = simulated_dataset
            pkl.dump((train_counts, train_rel_abn, train_t_pts, hold_out_counts, hold_out_rel_abn, hold_out_t_pts, A, g, denom), open("tmp_sim/clv-sim-set-{}-{}-{}-{}.pkl".format(depth, db, ss, i), "wb"))

        try:
            A_clv_en, g_clv_en, pred_en = pkl.load(open("tmp_sim/clv-sim-en-{}-{}-{}-{}.pkl".format(depth, db, ss, i), "rb"))
        except FileNotFoundError:
            A_clv_en, g_clv_en, pred_en = fit_clv(train_counts, train_t_pts, hold_out_rel_abn, hold_out_t_pts, denom, method="Elastic Net")
            pkl.dump((A_clv_en, g_clv_en, pred_en), open("tmp_sim/clv-sim-en-{}-{}-{}-{}.pkl".format(depth, db, ss, i), "wb"))


        try:
            A_clv_rg, g_clv_rg, pred_rg = pkl.load(open("tmp_sim/clv-sim-rg-{}-{}-{}-{}.pkl".format(depth, db, ss, i), "rb"))
        except FileNotFoundError:
            A_clv_rg, g_clv_rg, pred_rg = fit_clv(train_counts, train_t_pts, hold_out_rel_abn, hold_out_t_pts, denom, method="Ridge")
            pkl.dump((A_clv_rg, g_clv_rg, pred_rg), open("tmp_sim/clv-sim-rg-{}-{}-{}-{}.pkl".format(depth, db, ss, i), "wb"))
