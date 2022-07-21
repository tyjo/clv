import numpy as np
import pickle as pkl
import logging

from compositional_lotka_volterra import CompositionalLotkaVolterra
from generalized_lotka_volterra import GeneralizedLotkaVolterra
from linear_alr import LinearALR
from linear_rel_abun import LinearRelAbun

def fit_clv(observations, time_points, effects, held_out_observations, held_out_time_points,
        held_out_effects, pseudo_count=1e-6, using_rel_abun=False, ret_params=False, folds=None, save_name=None):
    # if observations are concentrations
    print("Checker\nTraining Size: {}, Testing Size: {}".format(
        len(observations), len(held_out_observations)))
    logging.debug("Running cLV")
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

    clv = CompositionalLotkaVolterra(rel_abun, time_points, effects,
                     pseudo_count=pseudo_count)
    clv.train(folds=folds)
    if save_name is not None:
        print("Saving model to: {}".format(save_name))
        pkl.dump(clv, open(save_name, "wb"))

    #predictions = [clv.predict(o[0],tpts,e) for (o,tpts,e) in zip(held_out_rel_abun, held_out_time_points, held_out_effects)]

    return []


def fit_glv(observations, time_points, effects, held_out_observations, held_out_time_points, held_out_effects,
        scale=1, pseudo_count=1e-6, use_rel_abun=False, folds=None, save_name=None):

    print("Checker\nTraining Size: {}, Testing Size: {}".format(
        len(observations), len(held_out_observations)))
    logging.debug("Running gLV elastic net")
    if folds is None:
        folds = len(observations)

    # fit on relative abundances instead of concentrations
    if use_rel_abun:
        print("Computing relative abundance")
        rel_abun = []
        for obs in observations:
            rel_abun.append(obs / obs.sum(axis=1,keepdims=True))
        observations = rel_abun

        held_out_rel_abun = []
        for obs in held_out_observations:
            held_out_rel_abun.append(obs / obs.sum(axis=1,keepdims=True))
        held_out_observations = held_out_rel_abun
        scale=1

    glv = GeneralizedLotkaVolterra(observations, time_points, effects, scale=scale,
          pseudo_count=pseudo_count, convert_to_rel=use_rel_abun)
    print("scale:", scale)
    glv.train(folds=folds)
    A, g, B = glv.get_params()

    if save_name is not None:
        print("Saving model to: {}".format(save_name))
        pkl.dump(glv, open(save_name, "wb"))

    #predictions = [glv.predict(o[0],tpts,e) for (o,tpts,e) in zip(held_out_observations, held_out_time_points, held_out_effects)]

    return []

def fit_glv_ridge(observations, time_points, effects, held_out_observations, held_out_time_points,
        held_out_effects, scale=1, pseudo_count=1e-6, use_rel_abun=False, folds=None, save_name=None):

    print("Checker\nTraining Size: {}, Testing Size: {}".format(
        len(observations), len(held_out_observations)))

    logging.debug("Running gLV, ridge")
    if folds is None:
        folds = len(observations)

    # fit on relative abundances instead of concentrations
    if use_rel_abun:
        print("Computing relative abundance")
        rel_abun = []
        for obs in observations:
            rel_abun.append(obs / obs.sum(axis=1,keepdims=True))
        observations = rel_abun

        held_out_rel_abun = []
        for obs in held_out_observations:
            held_out_rel_abun.append(obs / obs.sum(axis=1,keepdims=True))
        held_out_observations = held_out_rel_abun
        scale=1

    glv = GeneralizedLotkaVolterra(observations, time_points, effects, scale=scale,
        pseudo_count=pseudo_count, convert_to_rel=use_rel_abun)
    print("scale:", scale)
    glv.train_ridge(folds=folds)
    A, g, B = glv.get_params()

    if save_name is not None:
        print("Saving model to: {}".format(save_name))
        pkl.dump(glv, open(save_name, "wb"))

    #predictions = [glv.predict(o[0],tpts,e) for (o,tpts,e) in zip(held_out_observations, held_out_time_points, held_out_effects)]

    return []


def fit_linear_alr(observations, time_points, effects, held_out_observations, held_out_time_points,
        held_out_effects, using_rel_abun=False, folds=None, save_name=None):

    print("Checker\nTraining Size: {}, Testing Size: {}".format(
        len(observations), len(held_out_observations)))
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

    if save_name is not None:
        print("Saving model to: {}".format(save_name))
        pkl.dump(alr, open(save_name, "wb"))

    #predictions = [alr.predict(o[0],tpts,e) for (o,tpts,e) in zip(held_out_rel_abun, held_out_time_points, held_out_effects)]

    return []


def fit_linear_rel_abun(observations, time_points, effects, held_out_observations, held_out_time_points,
    held_out_effects, pseudo_count=1e-6, using_rel_abun=False, folds=None, save_name=None):

    print("Checker\nTraining Size: {}, Testing Size: {}".format(
        len(observations), len(held_out_observations)))

    logging.debug("Running LRA")
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

    lra = LinearRelAbun(rel_abun, time_points, effects, pseudo_count=pseudo_count)
    lra.train(folds=folds)
    A, g, B = lra.get_params()

    if save_name is not None:
        print("Saving model to: {}".format(save_name))
        pkl.dump(lra, open(save_name, "wb"))

    #predictions = [lra.predict(o[0],tpts,e) for (o,tpts,e) in zip(held_out_rel_abun, held_out_time_points, held_out_effects)]

    return []
