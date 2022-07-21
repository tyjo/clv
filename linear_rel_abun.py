import numpy as np
import sys
from scipy.special import logsumexp
from scipy.stats import linregress
from scipy.integrate import RK45, solve_ivp
from timeout import *


def estimate_relative_abundances(Y, pseudo_count=1e-3):
    """Adds pseudo counts to avoid zeros and compute relative abundances
    
    Parameters
    ----------
        Y : a list sequencing counts or observed concentrations
            per sequence
        pseudo_count : pseudo count, specific in relation to
                       the relative proportions of each taxon.

    Returns
    -------
        P : relative abundances with pseudo counts
    """
    P = []
    for y in Y:
        p = np.zeros(y.shape)
        for t in range(y.shape[0]):
            pt = y[t] / y[t].sum()
            pt = (pt + pseudo_count) / (pt + pseudo_count).sum()
            p[t] = pt
        P.append(p)
    return P


class LinearRelAbun:
    """Inference for the linear model using relative abundances
    """

    def __init__(self, P=None, T=None, U=None, pseudo_count=1e-6):
        """
        Parameters
        ----------
            P : A list of T_x by D dimensional numpy arrays of
                estimated relative abundances.
            T : A list of T_x by 1 dimensional numpy arrays giving
                the times of each observation x.
            U : An optional list of T_x by P numpy arrays of external
                perturbations for each x.
        """
        self.P = P
        self.pseudo_count = pseudo_count
        #self.P = estimate_relative_abundances(P, self.pseudo_count)
        self.T = T

        if U is None and self.P is not None:
            self.U = [ np.zeros((p.shape[0], 1)) for p in self.P ]
            self.no_effects = True
        else:
            self.U = U
            self.no_effects = False

        # Parameter estimates
        self.A = None
        self.g = None
        self.B = None
        self.Q_inv = np.eye(self.P[0].shape[1]) if P is not None else None

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
        """Estimate regularization parameters and CLV model parameters.
        """
        if self.alpha is None or self.r_A is None or self.r_g is None or self.r_B is None:
            if verbose:
                print("Estimating regularizers...")
            self.alpha, self.r_A, self.r_g, self.r_B = estimate_elastic_net_regularizers_cv(self.P, self.U, self.T, folds=folds, no_effects=self.no_effects, verbose=verbose)
            
        if verbose:
            print("Estimating model parameters...")
        self.A, self.g, self.B = elastic_net_linear(self.P, self.U, self.T, self.Q_inv, self.alpha, self.r_A, self.r_g, self.r_B)
        
        if verbose:
            print()


    def predict(self, p0, times, u = None):
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

        if p0.ndim == 1:
            p0 = p0.reshape((1, p0.size))

        #p0 = estimate_relative_abundances([p0])[0]
        return predict(p0, u, times, self.A, self.g, self.B)


    def get_params(self):
        A = np.copy(self.A)
        g = np.copy(self.g)
        B = np.copy(self.B)
        return A, g, B


def elastic_net_linear(X, U, T, Q_inv, alpha, r_A, r_g, r_B, tol=1e-3, verbose=False, max_iter=10000):

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
        # number of observations by xDim + 1 + uDim
        pgu_stacked = None
        for x, u, times in zip(X, U, T):
            for t in range(1, times.size):
                dt = times[t] - times[t-1]
                xt0 = x[t-1]
                gt0 = np.ones(1)
                ut0 = u[t-1]
                pgu = np.concatenate((xt0, gt0, ut0))

                if x_stacked is None:
                    x_stacked = x[t] - x[t-1]
                    pgu_stacked = dt*pgu

                else:
                    x_stacked = np.vstack((x_stacked, x[t] - x[t-1]))
                    pgu_stacked = np.vstack((pgu_stacked, dt*pgu))

        return x_stacked, pgu_stacked

    xDim = X[0].shape[1]
    yDim = xDim + 1
    uDim = U[0].shape[1]

    AgB = np.zeros(( xDim, xDim + 1 + uDim ))
    A,g,B = ridge_regression_linear(X, U, T, np.max((alpha*(1-r_A), 0.01)), np.max((alpha*(1-r_g), 0.01)), np.max((alpha*(1-r_B), 0.01)))
    AgB[:,:xDim] = A
    AgB[:,xDim:(xDim+1)] = np.expand_dims(g,axis=1)
    AgB[:,(xDim+1):] = B

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

    A = AgB[:,:xDim]
    g = AgB[:,xDim:(xDim+1)].flatten()
    B = AgB[:,(xDim+1):]

    return A, g, B


def ridge_regression_linear(X, U, T, r_A=0, r_g=0, r_B=0):
    """Computes estimates of A, g, and B using least squares. 

    Parameters
    ----------
        X : a list of T x yDim-1 numpy arrays
        U : a list of T x uDim numpy arrays
        T   : a list of T x 1 numpy arrays with the time of each observation

    Returns
    -------
    """
    xDim = X[0].shape[1]
    yDim = xDim + 1
    uDim = U[0].shape[1]
    AgB_term1 = np.zeros(( xDim, xDim + 1 + uDim ))
    AgB_term2 = np.zeros(( xDim + 1 + uDim, xDim + 1 + uDim))

    for idx, (xi, ui) in enumerate(zip(X, U)):
        for t in range(1, xi.shape[0]):
            pt = xi[t]
            pt0 = xi[t-1]

            xt  = xi[t]
            xt0 = xi[t-1]
            
            gt0 = np.ones(1)
            ut0 = ui[t-1]
            
            pgu = np.concatenate((pt0, gt0, ut0))
            
            delT = T[idx][t] - T[idx][t-1]
            AgB_term1 += np.outer( (xt - xt0) / delT, pgu)
            AgB_term2 += np.outer(pgu, pgu)

    reg = np.array([r_A for i in range(xDim)] + [r_g] + [r_B for i in range(uDim)])
    reg = np.diag(reg)
    AgB = AgB_term1.dot(np.linalg.pinv(AgB_term2 + reg))
    A = AgB[:,:xDim]
    g = AgB[:,xDim:(xDim+1)].flatten()
    B = AgB[:,(xDim+1):]

    return A, g, B



def estimate_elastic_net_regularizers_cv(P, U, T, folds, no_effects=False, verbose=False):
    if len(P) == 1:
        print("Error: cannot estimate regularization parameters from single sample", file=sys.stderr)
        exit(1)
    elif len(P) < folds:
        folds = len(P)

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
            train_P = []
            train_U = []
            train_T = []

            test_P = []
            test_U = []
            test_T = []
            for i in range(len(P)):
                if i % folds == fold:
                    test_P.append(P[i])
                    test_U.append(U[i])
                    test_T.append(T[i])

                else:
                    train_P.append(P[i])
                    train_U.append(U[i])
                    train_T.append(T[i])

            Q_inv = np.eye(train_P[0].shape[1])
            A, g, B = elastic_net_linear(train_P, train_U, train_T, Q_inv, alpha, r_A, r_g, r_B, tol=1e-3)
            sqr_err += compute_prediction_error(test_P, test_U, test_T, A, g, B)

        if sqr_err < best_sqr_err:
            best_r = (alpha, r_A, r_g, r_B)
            best_sqr_err = sqr_err
            print("\tr", (alpha, r_A, r_g, r_B), "sqr error", sqr_err)
    np.set_printoptions(suppress=False)
    return best_r


@timeout(5)
def predict(p, u, times, A, g, B):
    """Make predictions from initial conditions
    """
    def grad_fn(A, g, B, u):
        def fn(t, x):
            return g + A.dot(x) + B.dot(u)
        return fn

    p_pred = np.zeros((times.shape[0], p[0].size))
    pt = p[0]

    for i in range(1, times.shape[0]):
        grad = grad_fn(A, g, B, u[i-1])
        dt = times[i] - times[i-1]
        ivp = solve_ivp(grad, (0,0+dt), pt, method="RK45")
        pt = ivp.y[:,-1]
        pt[pt < 0] = 0
        pt = pt / pt.sum()
        p_pred[i] = pt
    return p_pred


def compute_prediction_error(P, U, T, A, g, B):
    def compute_err(p, p_pred):
        err = 0
        ntaxa = p.shape[1]
        err += np.square(p[1:] - p_pred[1:]).sum()
        return err/ntaxa
    err = 0
    for p, u, t in zip(P, U, T):
        try:
            p_pred = predict(p, u, t, A, g, B)
            err += compute_err(p, p_pred)
        except TimeoutError:
            err += np.inf
    return err/len(P)
