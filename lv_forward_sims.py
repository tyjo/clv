#forward simulates the the dynamical systems model using the inferred parameters
import numpy as np
import mdsine2 as md2
import argparse
import pickle as pkl
import copy

from pathlib import Path
from scipy.special import logsumexp
from scipy.stats import linregress
from scipy.integrate import RK45, solve_ivp
from compositional_lotka_volterra import CompositionalLotkaVolterra
from generalized_lotka_volterra import GeneralizedLotkaVolterra
from linear_rel_abun import LinearRelAbun

def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--inference_location", required=True,
        help="the name of the folder containing the inference results")
    parser.add_argument("-r", "--reg_type", required=True,
        help="regression type")
    parser.add_argument("-a", "--abundance_type", required=True)
    parser.add_argument("-s", "--subject_number", required=True, type=int,
         help="subject(fold) number")
    parser.add_argument("-m", "--model_name", required=True)
    parser.add_argument("-od","--output_dir", required=True)
    parser.add_argument("-sc", "--scale", default=None, type=float,
         help="scaling applied during inference")
    parser.add_argument("-is", "--is_rescaled", type=bool, default=True,
        required=True, help="whether or not the inferred parameters are rescaled")
    parser.add_argument("-inp", "--input_files_path", required=True,
        help="directory containing the files that are used for inference")
    parser.add_argument("-p", "--pseudo_count", type=float, default=0)

    return parser.parse_args()

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
        #print(np.mean(C*y))

    return Y_adjusted, C


def add_limit_detection(X, lim=1e5):
    """adjust the 0 concentration values"""

    print("Using limit of detection value {}".format(lim))
    X_ = copy.deepcopy(X)
    new_X =[]

    for x in X_:
        new_X.append(np.where(x <= lim, lim, x))

    return new_X

def forward_sim_single_subj_glv(A, g, B, x0, u, times, rel_abund=False):
    """
    forward simulate for a single subject
    (np.ndarray) x0 : N dimensional array containing the initial abundances(log)
    (np.ndarray) A, g, B: : the coefficients for interactions, growth, perturbation
    (np.ndarray) u : the perturbation indicator
    (np.ndarray) t : the time coefficients
    """

    #print("x0:", x0)
    def grad_fn(A, g, B, u):
        def fn(t, x):
            if B is None or u is None:
                return g + A.dot(x)
            elif B is not None and u is not None:
                return g + A.dot(np.exp(x)) + B.dot(u)
        return fn

    x_pred = np.zeros((times.shape[0], x0.shape[0]))
    x_pred[0] = np.exp(x0)
    xt = x0
    if np.sum(B) == 0:
        B = None

    for t in range(1, times.shape[0]):
        #print(t)
        if u is not None:
            grad = grad_fn(A, g, B, u[t-1])
        else:
            grad = grad_fn(A, g, None, None)
        dt = times[t] - times[t-1]
        ivp = solve_ivp(grad, (0, 0+dt), xt, method="RK45")
        xt = ivp.y[:, -1]
        if rel_abund:
            x_pred[t] = np.exp(xt) / np.sum(np.exp(xt))
        else:
            x_pred[t] = np.exp(xt)

    return x_pred

def forward_sim_single_subj_lra(A, g, B, x0, u, times):
    """
    forward simulate for a single subject
    (np.ndarray) x0 : N dimensional array containing the initial abundancesm(rel)
    (np.ndarray) A, g, B: : the coefficients for interactions, growth, perturbation
    (np.ndarray) u : the perturbation indicator
    (np.ndarray) t : the time coefficients
    """

    sum_x0 = np.sum(x0)
    #print(sum_x0)
    #if sum_x0 <= 1 - 1e-5:
    #    raise Exception("Sum is not 1. Please use relative abundance")
    #else:
    #    print("Relative abundance is used")

    def grad_fn(A, g, B, u):
        def fn(t, x):
            if B is None or u is None:
                return g + A.dot(x)
            elif B is not None and u is not None:
                return g + A.dot(x) + B.dot(u)
        return fn

    x_pred = np.zeros((times.shape[0], x0.shape[0]))
    x_pred[0] = x0
    xt = x0
    if np.sum(B) == 0:
        B = None

    for t in range(1, times.shape[0]):
        if u is not None:
            grad = grad_fn(A, g, B, u[t-1])
        else:
            grad = grad_fn(A, g, None, None)
        dt = times[t] - times[t-1]
        ivp = solve_ivp(grad, (0, 0+dt), xt, method="RK45")
        xt = ivp.y[:, -1]
        xt[xt < 0] = 0
        xt = xt / np.sum(xt) #computing the relative abundance
        #print(t, np.sum(xt))
        x_pred[t] = xt

    return x_pred


def forward_sim_single_subj_clv(A, g, B, x0, u, times, denom, pc):
    """
    forward simulate for a single subject for clv
    (np.ndarray) x0 : N dimensional array containing the initial abundances(rel)
    (np.ndarray) A, g, B: : the coefficients for interactions, growth, perturbation
    (np.ndarray) u : the perturbation indicator
    (np.ndarray) t : the time coefficients
    """
    def construct_alr(P, denom, pseudo_count=1e-3):
        """Compute the additive log ratio transformation with a given
        choice of denominator. Assumes zeros have been replaced with
        nonzero values.

        P : relative abundance array
        """
        ALR = []
        if P.ndim==1:
            P = P.reshape(1, P.size)
        ntaxa = P.shape[1]

        numer = np.array([i for i in range(ntaxa) if i != denom])
        p = np.copy(P)
        p = (p + pseudo_count) / (p + pseudo_count).sum(axis=1, keepdims=True)
        p /= p.sum(axis=1, keepdims=True)
        alr = (np.log(p[:, numer]).T - np.log(p[:, denom])).T
        #print(alr)

        return alr

    def compute_rel_abund(x, d):
        """compute relative abundance for clv"""
        if x.ndim ==1:
            x = np.expand_dims(x, axis=0)
        z = np.hstack((x, np.zeros((x.shape[0], 1))))
        p = np.exp(z - logsumexp(z, axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        for i in range(p.shape[1]-1, d, -1):
            tmp = np.copy(p[:,i-1])
            p[:,i-1] = np.copy(p[:,i])
            p[:,i] = tmp
        return p.flatten()

    def grad_fn(A, g, B, u , denom):
        def fn(t, x):
            p = compute_rel_abund(x, denom)
            if B is None or u is None:
                return g + A.dot(p)
            else:
                return g + A.dot(p) + B.dot(u)
        return fn

    pred_p = np.zeros((times.shape[0], x0.shape[0]))
    pred_p[0] = x0
    xt = construct_alr(x0, denom, pseudo_count=pc).flatten()

    for t in range(1, times.shape[0]):
        grad = grad_fn(A, g, B, u[t-1], denom)
        dt = times[t] - times[t-1]
        ivp = solve_ivp(grad, (0, 0+dt), xt, method="RK45")
        xt = ivp.y[:, -1]
        pt = compute_rel_abund(xt, denom).flatten() #relative abundance
        pred_p[t] = pt

    return pred_p

def regression_forward_simulate(model_name, A, g, B, x, u, t, abund_type="rel",
     denom=None):

    sims = []
    print("denom:", denom)
    if model_name == "clv":
        print("Simulating CLV")
        sims = forward_sim_single_subj_clv(A, g, B, x[0], u, t, denom, 0)
    elif model_name =="lra":
        print("Simulating LRA")
        sims = forward_sim_single_subj_lra(A, g, B, x[0], u, t)
    elif model_name == "glv":
        if abund_type == "rel":
            print("Simulating GLV rel")
            sims = forward_sim_single_subj_glv(A, g, B, x[0], u, t, rel_abund=True)
        else:
            print("Simulating GLV abs")
            sims = forward_sim_single_subj_glv(A, g, B, x[0], u, t, rel_abund=False)
    elif model_name == "glv-ra":
        print("Simulating GLV-RA")
        sims = forward_sim_single_subj_glv(A, g, B, x[0], u, t, rel_abund=True)

    return sims

if __name__ =="__main__":

    args = parse_arguments()
    input_files_path = Path(args.input_files_path)

    Y = pkl.load(open(input_files_path / "Y.pkl", "rb"))
    U = pkl.load(open(input_files_path / "U.pkl", "rb"))
    T = pkl.load(open(input_files_path / "T.pkl", "rb"))

    Y = add_limit_detection(Y)

    scale = 1
    if args.is_rescaled:
        print("Already scaled")
    else:
        if args.scale is None:
            print("Computing scale")
            Y_adj, scale = adjust_concentrations(Y)
        else:
            scale = args.scale

    print("scale:", scale)

    model_pkl = None
    A, g, B = None, None, None
    use_rel_abund = False

    if args.model_name in ["lra", "clv", "glv-ra"]:
        use_rel_abund = True
        Y_rel = []
        for y in Y:
            Y_rel.append(y / np.sum(y, axis=1, keepdims=True))
        Y = Y_rel
    denom_clv = None

    if args.model_name == "glv" or args.model_name=="glv-ra":
        model_file = Path(args.inference_location) / "results_{}_{}".format(
            args.abundance_type, args.reg_type) / "{}-{}-{}-model.pkl".format(
            args.model_name, args.reg_type, args.subject_number)
        model_pkl = pkl.load(open(model_file, "rb"))
        A, g, B = model_pkl.get_params()
        if args.model_name == "glv":
            A = A * scale
        Y_log = []
        for y in Y:
            Y_log.append(np.log(y))
        Y = Y_log
    else:
        model_file = Path(args.inference_location) / "results_{}_{}".format(
            args.abundance_type, args.reg_type) / "{}-{}-model.pkl".format(
            args.model_name, args.subject_number)
        model_pkl = pkl.load(open(model_file, "rb"))
        A, g, B = model_pkl.get_params()
        if args.model_name == "clv":
            denom_clv = model_pkl.denom
            print("DENOM IN CLV:", denom_clv)

    print("A:{}, B:{}, g:{}, Y:{}, U:{}, T:{}".format(A.shape, B.shape,
        g.shape, Y[0].shape, U[0].shape, T[0].shape))

    subj = args.subject_number
    results = regression_forward_simulate(args.model_name, A, g, B, Y[subj],
        U[subj], T[subj], abund_type=args.abundance_type, denom=denom_clv)
    output_dir = Path(args.output_dir) / "results_{}_{}".format(args.abundance_type,
        args.reg_type)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = "{}-{}-{}.pkl".format(args.model_name, args.reg_type, args.subject_number)
    save_name = pkl.dump(results, open(output_dir/ name, "wb"))
