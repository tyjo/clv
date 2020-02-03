import numpy as np
import pickle as pkl

from compositional_lotka_volterra import CompositionalLotkaVolterra
from generalized_lotka_volterra import GeneralizedLotkaVolterra
from linear_alr import LinearALR
from linear_rel_abun import LinearRelAbun

def fit_clv(observations, time_points, effects, held_out_observations, held_out_time_points, held_out_effects, using_rel_abun=False, ret_params=False, folds=None):
    # if observations are concentrations
    rel_abun = []
    held_out_rel_abun = []

    if folds is None:
        folds = len(observations)

    if not using_rel_abun:
        for obs in observations:
            rel_abun.append(obs / obs.sum(axis=1,keepdims=True))

        for obs in held_out_observations:
            held_out_rel_abun.append(obs / obs.sum(axis=1,keepdims=True))

    else:
        rel_abun = observations
        held_out_rel_abun = held_out_observations

    clv = CompositionalLotkaVolterra(rel_abun, time_points, effects)
    clv.train(folds=folds)
 
    predictions = [clv.predict(o[0],tpts,e) for (o,tpts,e) in zip(held_out_rel_abun, held_out_time_points, held_out_effects)]

    return predictions


def fit_glv(observations, time_points, effects, held_out_observations, held_out_time_points, held_out_effects, use_rel_abun=False, folds=None):
    if folds is None:
        folds = len(observations)

    # fit on relative abundances instead of concentrations
    if use_rel_abun:
        rel_abun = []
        for obs in observations:
            rel_abun.append(obs / obs.sum(axis=1,keepdims=True))
        observations = rel_abun

        held_out_rel_abun = []
        for obs in held_out_observations:
            held_out_rel_abun.append(obs / obs.sum(axis=1,keepdims=True))
        held_out_observations = held_out_rel_abun

    glv = GeneralizedLotkaVolterra(observations, time_points, effects)
    glv.train(folds=folds)
    A, g, B = glv.get_params()

    predictions = [glv.predict(o[0],tpts,e) for (o,tpts,e) in zip(held_out_observations, held_out_time_points, held_out_effects)]

    return predictions


def fit_linear_alr(observations, time_points, effects, held_out_observations, held_out_time_points, held_out_effects, using_rel_abun=False, folds=None):
    # if observations are concentrations
    rel_abun = []
    held_out_rel_abun = []

    if folds is None:
        folds = len(observations)

    if not using_rel_abun:
        for obs in observations:
            rel_abun.append(obs / obs.sum(axis=1,keepdims=True))

        for obs in held_out_observations:
            held_out_rel_abun.append(obs / obs.sum(axis=1,keepdims=True))

    else:
        rel_abun = observations
        held_out_rel_abun = held_out_observations

    alr = LinearALR(rel_abun, time_points, effects)
    alr.train(folds=folds)
    A, g, B = alr.get_params()
    
    predictions = [alr.predict(o[0],tpts,e) for (o,tpts,e) in zip(held_out_rel_abun, held_out_time_points, held_out_effects)]

    return predictions


def fit_linear_rel_abun(observations, time_points, effects, held_out_observations, held_out_time_points, held_out_effects, using_rel_abun=False, folds=None):
    if folds is None:
        folds = len(observations)

    # if observations are concentrations
    rel_abun = []
    held_out_rel_abun = []

    if not using_rel_abun:
        for obs in observations:
            rel_abun.append(obs / obs.sum(axis=1,keepdims=True))

        for obs in held_out_observations:
            held_out_rel_abun.append(obs / obs.sum(axis=1,keepdims=True))

    else:
        rel_abun = observations
        held_out_rel_abun = held_out_observations

    lra = LinearRelAbun(rel_abun, time_points, effects)
    lra.train(folds=folds)
    A, g, B = lra.get_params()
    
    predictions = [lra.predict(o[0],tpts,e) for (o,tpts,e) in zip(held_out_rel_abun, held_out_time_points, held_out_effects)]

    return predictions