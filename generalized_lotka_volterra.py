import numpy as np
import sys
from scipy.special import logsumexp
from scipy.stats import linregress
from scipy.integrate import RK45, solve_ivp
from timeout import *
import time


def add_pseudo_counts(Y, pseudo_count=1e-3):
    """Adds pseudo counts to avoid zeros and compute relative abundances

    Parameters
    ----------
        Y : a list of observed concentrations per sequence
        pseudo_count : pseudo count, specific in relation to
                       the relative proportions of each taxon.

    Returns
    -------
        P : relative abundances with pseudo counts
    """
    Y_pc = []
    for y in Y:
        y_pc = np.zeros(y.shape)
        for t in range(y.shape[0]):
            pt = y[t] / y[t].sum()
            pt = (pt + pseudo_count) / (pt + pseudo_count).sum()
            y_pc[t] = y[t].sum() * pt
        Y_pc.append(y_pc)
    return Y_pc


def construct_log_concentrations(C, pseudo_count=1e-3):
    X = []
    for c in C:
        c = np.copy(c)
        x = np.zeros((c.shape[0], c.shape[1]))
        for t in range(c.shape[0]):
            ct = c[t]
            if np.any(ct == 0):
                print("Using psuedo count at time: {}".format(t))
                mass = ct.sum()
                pt = ct / ct.sum()
                pt = (pt + pseudo_count) / (pt + pseudo_count).sum()
                ct = mass*pt

            xt = np.log(ct)
            x[t] = xt
        X.append(x)

    return X


class GeneralizedLotkaVolterra:
    """Inference for compositional Lotka-Volterra.
    """

    def __init__(self, C=None, T=None, U=None, scale=1, denom_ids=None, pseudo_count=1e-3,
                 convert_to_rel=False):
        """
        Parameters
        ----------
            C : A list of T_x by D dimensional numpy arrays of
                estimated concentrations.
            T : A list of T_x by 1 dimensional numpy arrays giving
                the times of each observation x.
            U : An optional list of T_x by P numpy arrays of external
                perturbations for each x.
            denom_ids : a list of integer ids for taxa in denominator
                        of log ratio
        """
        self.C = C
        self.T = T
        self.scale = scale
        self.pseudo_count = pseudo_count
        self.convert_to_rel = convert_to_rel # whether or not to convert rel

        print("Using pseudo-count: {}".format(pseudo_count))
        if C is not None:
            self.X = construct_log_concentrations(C, pseudo_count)
        else:
            self.X = None

        if U is None and self.X is not None:
            self.U = [ np.zeros((x.shape[0], 1)) for x in self.X ]
            self.no_effects = True
        else:
            self.U = U
            self.no_effects = False

        # Parameter estimates
        self.A = None
        self.g = None
        self.B = None
        self.Q_inv = np.eye(self.C[0].shape[1]) if C is not None else None

        # Regularization parameters
        self.alpha = None
        self.r_A = None
        self.r_g = None
        self.r_B = None


    def get_regularizers(self):
        return self.alpha, self.r_A, self.r_g, self.r_B


    def set_regularizers(self, alpha, r_A, r_g, r_B):
        self.alpha = alpha
        self.r_A = r_A
        self.r_g = r_g
        self.r_B = r_B


    def train(self, verbose=False, folds=10):
        """Estimate regularization parameters and gLV model parameters.
        """
        if self.alpha is None or self.r_A is None or self.r_g is None or self.r_B is None:
            if verbose:
                print("Estimating regularizers...")
            self.alpha, self.r_A, self.r_g, self.r_B = estimate_elastic_net_regularizers_cv(self.X,
                                                      self.U, self.T, folds=folds, no_effects=self.no_effects,
                                                      verbose=verbose, convert_to_rel=self.convert_to_rel)

        if verbose:
            print("Estimating model parameters...")
        self.A, self.g, self.B = elastic_net_glv(self.X, self.U, self.T, self.Q_inv, self.alpha, self.r_A, self.r_g, self.r_B)
        self.A = self.A * self.scale

        if verbose:
            print()


    def train_ridge(self, verbose=False, folds=10):
        r_A, r_g, r_B = estimate_ridge_regularizers_cv(self.X, self.U, self.T, folds=folds,
                                no_effects=self.no_effects, verbose=verbose, convert_to_rel=self.convert_to_rel)
        self.r_A = r_A
        self.r_g = r_g
        self.r_B = r_B

        if verbose:
            print("Estimating model parameters...")
        self.A, self.g, self.B = ridge_regression_glv(self.X, self.U, self.T, r_A, r_g, r_B)
        self.A = self.A * self.scale

        if verbose:
            print()


    def predict(self, c0, times, u = None):
        """Predict relative abundances from initial conditions.

        Parameters
        ----------
            p0     : the initial observation, a D-dim numpy array
            times  : a T by 1 numpy array of sample times
            u      : a T by P numpy array of external perturbations

        Returns
        -------
            y_pred : a T by D numpy array of predicted relative
                     abundances. Since we cannot predict initial
                     conditions, the first entry is set to the array
                     of -1.
        """
        if u is None:
            u = np.zeros((times.shape[0], 1))

        if c0.ndim == 1:
            c0 = c0.reshape((1, c0.size))

        X = construct_log_concentrations([c0])
        x = X[0]

        return predict(x, u, times, self.A, self.g, self.B, self.convert_to_rel)


    def get_params(self):
        A = np.copy(self.A)
        g = np.copy(self.g)
        B = np.copy(self.B)
        return A, g, B


def elastic_net_glv(X, U, T, Q_inv, alpha, r_A, r_g, r_B, tol=1e-3, verbose=False, max_iter=10000):

    def gradient(AgB, x_stacked, pgu_stacked):
        f = x_stacked - AgB.dot(pgu_stacked.T).T
        grad = Q_inv.dot(f.T.dot(pgu_stacked))

        # l2 regularization terms
        A = AgB[:,:yDim]
        g = AgB[:,yDim:(yDim+1)]
        B = AgB[:,(yDim+1):]
        grad[:,:yDim] += -2*alpha*(1-r_A)*A
        grad[:,yDim:(yDim+1)] += -2*alpha*(1-r_g)*g
        grad[:,(yDim+1):] += -2*alpha*(1-r_B)*B
        return -grad


    def generalized_gradient(AgB, grad, step):
            nxt_AgB = prv_AgB - step*grad

            # threshold A
            A_prox = nxt_AgB[:,:yDim]
            A_prox[A_prox < -step*alpha*r_A] += step*alpha*r_A
            A_prox[A_prox > step*alpha*r_A] -= step*alpha*r_A
            A_prox[np.logical_and(A_prox >= -step*alpha*r_A, A_prox <= step*alpha*r_A)] = 0

            # threshold g
            g_prox = nxt_AgB[:,yDim:(yDim+1)]
            g_prox[g_prox < -step*alpha*r_g] += step*alpha*r_g
            g_prox[g_prox > step*alpha*r_g] -= step*alpha*r_g
            g_prox[np.logical_and(g_prox >= -step*alpha*r_g, g_prox <= step*alpha*r_g)] = 0

            # threshold B
            B_prox = nxt_AgB[:,(yDim+1):]
            B_prox[B_prox < -step*alpha*r_B] += step*alpha*r_B
            B_prox[B_prox > step*alpha*r_B] -= step*alpha*r_B
            B_prox[np.logical_and(B_prox >= -step*alpha*r_B, B_prox <= step*alpha*r_B)] = 0

            AgB_proximal = np.zeros(AgB.shape)
            AgB_proximal[:,:yDim] = A_prox
            AgB_proximal[:,yDim:(yDim+1)] = g_prox
            AgB_proximal[:,(yDim+1):] = B_prox

            return (AgB - AgB_proximal)/step


    def objective(AgB, x_stacked, pgu_stacked):
        f = x_stacked - AgB.dot(pgu_stacked.T).T
        obj = -0.5*(f.dot(Q_inv)*f).sum()

        return -obj


    def stack_observations(X, U, T):
        # number of observations by xDim
        x_stacked = None
        # number of observations by yDim + 1 + uDim
        pgu_stacked = None
        for x, u, times in zip(X, U, T):
            for t in range(1, times.size):
                dt = times[t] - times[t-1]
                pt0 = np.exp(x[t-1])
                gt0 = np.ones(1)
                ut0 = u[t-1]

                pgu = np.concatenate((pt0, gt0, ut0))

                if x_stacked is None:
                    x_stacked = x[t] - x[t-1]
                    pgu_stacked = dt*pgu

                else:
                    x_stacked = np.vstack((x_stacked, x[t] - x[t-1]))
                    pgu_stacked = np.vstack((pgu_stacked, dt*pgu))

        return x_stacked, pgu_stacked

    xDim = X[0].shape[1]
    yDim = xDim
    uDim = U[0].shape[1]
    AgB = np.zeros(( xDim, yDim + 1 + uDim ))

    AgB = np.zeros(( xDim, yDim + 1 + uDim ))
    A,g,B = ridge_regression_glv(X, U, T, np.max((alpha*(1-r_A), 0.01)), np.max((alpha*(1-r_g), 0.01)), np.max((alpha*(1-r_B), 0.01)))
    AgB[:,:yDim] = A
    AgB[:,yDim:(yDim+1)] = np.expand_dims(g,axis=1)
    AgB[:,(yDim+1):] = B

    x_stacked, pgu_stacked = stack_observations(X, U, T)
    prv_obj = np.inf
    obj = objective(AgB, x_stacked, pgu_stacked)

    it = 0
    while np.abs(obj - prv_obj) > tol:
        np.set_printoptions(suppress=True)
        prv_AgB = np.copy(AgB)
        prv_obj = obj

        # line search
        step = 0.1
        grad = gradient(prv_AgB, x_stacked, pgu_stacked)
        gen_grad = generalized_gradient(prv_AgB, grad, step)
        nxt_AgB = prv_AgB - step*gen_grad
        obj = objective(nxt_AgB, x_stacked, pgu_stacked)
        while obj > prv_obj - step*(grad*gen_grad).sum() + step*0.5*np.square(gen_grad).sum():
            step /= 2
            gen_grad = generalized_gradient(prv_AgB, grad, step)
            nxt_AgB = prv_AgB - step*gen_grad
            obj = objective(nxt_AgB, x_stacked, pgu_stacked)

        A = nxt_AgB[:,:yDim]
        g = nxt_AgB[:,yDim:(yDim+1)]
        B = nxt_AgB[:,(yDim+1):]
        AgB[:,:yDim] = A
        AgB[:,yDim:(yDim+1)] = g
        AgB[:,(yDim+1):] = B

        obj = objective(AgB, x_stacked, pgu_stacked)
        it += 1

        if verbose:
            print("\t", it, obj)

        if it > max_iter:
            print("Warning: maximum number of iterations ({}) reached".format(max_iter), file=sys.stderr)
            break

    A = AgB[:,:yDim]
    g = AgB[:,yDim:(yDim+1)].flatten()
    B = AgB[:,(yDim+1):]

    return A, g, B


def ridge_regression_glv(X, U, T, r_A, r_g, r_B):
    """Computes estimates of A, g, and B using least squares to solve
    the Composition LV differential equation.

    Parameters
    ----------
        X : a list of T x yDim numpy arrays (latent state estimates)
        P : a list of T x yDim numpy arrays giving relative abundances
        U : a list of T x uDim numpy arrays
        T   : a list of T x 1 numpy arrays with the time of each observation
        r_A : regularization parameter for the matrix A
        r_g : regularization parameter for growth rates
        r_B : regularization parameter for the matrix B

    Returns
    -------
    """
    xDim = X[0].shape[1]
    pDim = xDim
    uDim = U[0].shape[1]
    AgB_term1 = np.zeros(( xDim, pDim + 1 + uDim ))
    AgB_term2 = np.zeros(( pDim + 1 + uDim, pDim + 1 + uDim))

    for idx, (xi, ui) in enumerate(zip(X, U)):
        for t in range(1, xi.shape[0]):
            pt = np.exp(xi[t])
            pt0  = np.exp(xi[t-1])

            xt  = xi[t]
            xt0 = xi[t-1]

            gt0 = np.ones(1)
            ut0 = ui[t-1]

            pgu = np.concatenate((pt0, gt0, ut0))

            delT = T[idx][t] - T[idx][t-1]
            AgB_term1 += np.outer( (xt - xt0) / delT, pgu)
            AgB_term2 += np.outer(pgu, pgu)

    reg = np.array([r_A for i in range(pDim)] + [r_g] + [r_B for i in range(uDim)])
    reg = np.diag(reg)
    AgB = AgB_term1.dot(np.linalg.pinv(AgB_term2 + reg))
    A = AgB[:,:pDim]
    g = AgB[:,pDim:(pDim+1)].flatten()
    B = AgB[:,(pDim+1):]

    return A, g, B


def estimate_elastic_net_regularizers_cv(X, U, T, folds, no_effects=False, verbose=False,
                convert_to_rel=False):
    if len(X) == 1:
        print("Error: cannot estimate regularization parameters from single sample", file=sys.stderr)
        exit(1)
    elif len(X) < folds:
        folds = len(X)

    rs = [0.1, 0.5, 0.7, 0.9, 1]
    alphas = [0.1, 1, 10]

    alpha_rA_rg_rB = []
    for alpha in alphas:
        for r_A in rs:
            for r_g in rs:
                if no_effects:
                    alpha_rA_rg_rB.append((alpha, r_A, r_g, 0))
                else:
                    for r_B in rs:
                        alpha_rA_rg_rB.append((alpha, r_A, r_g, r_B))

    np.set_printoptions(suppress=True)
    best_r = 0
    best_sqr_err = np.inf
    for i, (alpha, r_A, r_g, r_B) in enumerate(alpha_rA_rg_rB):
        #print("\tTesting regularization parameter set", i+1, "of", len(alpha_rA_rg_rB), file=sys.stderr)
        sqr_err = 0
        for fold in range(folds):
            train_X = []
            train_U = []
            train_T = []

            test_X = []
            test_U = []
            test_T = []
            for i in range(len(X)):
                if i % folds == fold:
                    test_X.append(X[i])
                    test_U.append(U[i])
                    test_T.append(T[i])

                else:
                    train_X.append(X[i])
                    train_U.append(U[i])
                    train_T.append(T[i])

            Q_inv = np.eye(train_X[0].shape[1])
            A, g, B = elastic_net_glv(train_X, train_U, train_T, Q_inv, alpha, r_A, r_g, r_B, tol=1e-3)
            sqr_err += compute_prediction_error(test_X, test_U, test_T, A, g, B, convert_to_rel)

        if sqr_err < best_sqr_err:
            best_r = (alpha, r_A, r_g, r_B)
            best_sqr_err = sqr_err
            print("\tr", (alpha, r_A, r_g, r_B), "sqr error", sqr_err)
    np.set_printoptions(suppress=False)
    return best_r


def estimate_ridge_regularizers_cv(X, U, T, folds, no_effects=False, verbose=False,
                                  convert_to_rel=False):
    if len(X) == 1:
        print("Error: cannot estimate regularization parameters from single sample", file=sys.stderr)
        exit(1)
    elif len(X) < folds:
        folds = len(X)

    rs = [0.125, 0.25, 0.5, 1, 2, 4, 8]
    rA_rg_rB = []
    for r_A in rs:
        for r_g in rs:
            if no_effects:
                rA_rg_rB.append( (r_A, r_g, 0) )
            else:
                for r_B in rs:
                    rA_rg_rB.append(  (r_A, r_g, r_B) )

    np.set_printoptions(suppress=True)
    best_r = 0
    best_sqr_err = np.inf
    for i, (r_A, r_g, r_B) in enumerate(rA_rg_rB):
        print("\tTesting regularization parameter set", i+1, "of", len(rA_rg_rB), file=sys.stderr)
        sqr_err = 0
        for fold in range(folds):
            train_X = []
            train_U = []
            train_T = []

            test_X = []
            test_U = []
            test_T = []
            for i in range(len(X)):
                if i % folds == fold:
                    test_X.append(X[i])
                    test_U.append(U[i])
                    test_T.append(T[i])

                else:
                    train_X.append(X[i])
                    train_U.append(U[i])
                    train_T.append(T[i])

            Q_inv = np.eye(train_X[0].shape[1])
            A, g, B = ridge_regression_glv(train_X, train_U, train_T, r_A, r_g, r_B)
            sqr_err += compute_prediction_error(test_X, test_U, test_T, A, g, B, convert_to_rel)

        if sqr_err < best_sqr_err:
            best_r = (r_A, r_g, r_B)
            best_sqr_err = sqr_err
            print("\tr", (r_A, r_g, r_B), "sqr error", sqr_err)
    np.set_printoptions(suppress=False)
    return best_r


def compute_rel_abun(x):
    if x.ndim == 1:
        x = np.expand_dims(x, axis=0)
    p = np.exp(x - logsumexp(x,axis=1, keepdims=True))
    return p


@timeout(5)
def predict(x, u, times, A, g, B, convert_rel):
    """Make predictions from initial conditions
    """
    def grad_fn(A, g, B, u):
        def fn(t, x):
            return g + A.dot(np.exp(x)) + B.dot(u)
        return fn

    p_pred = np.zeros((times.shape[0], x[0].size))
    xt = x[0]

    for i in range(1, times.shape[0]):
        grad = grad_fn(A, g, B, u[i-1])
        dt = times[i] - times[i-1]
        ivp = solve_ivp(grad, (0,0+dt), xt, method="RK45")
        xt = ivp.y[:,-1]
        pt = np.exp(xt).flatten()
        if convert_rel:
            pt = compute_rel_abun(xt).flatten()
        p_pred[i] = pt
    return p_pred


def compute_prediction_error(X, U, T, A, g, B, convert_rel):
    def compute_err(p, p_pred):
        err = 0
        ntaxa = p.shape[1]
        err += np.square(p[1:] - p_pred[1:]).sum()
        return err/ntaxa
    err = 0
    for x, u, t in zip(X, U, T):
        try:
            p_pred = predict(x, u, t, A, g, B, convert_rel)
            if convert_rel:
                err += compute_err(compute_rel_abun(x), p_pred)
            else:
                err += compute_err(np.exp(x), p_pred)
        except TimeoutError:
            err += np.inf
    return err/len(X)
