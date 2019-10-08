import numpy as np

from taur_cross_validation import estimate_elastic_net_regularizers_cv, \
                                  estimate_elastic_net_regularizers_cv_linear, \
                                  estimate_elastic_net_regularizers_cv_rel_abun
from optimizers import estimate_latent_from_observations, elastic_net
from util import load_observations


Y, U, T, U_normalized, T_normalized = load_observations("data/taur/taur-otu-table-top10+dom.csv", "data/taur/taur-events.csv")
Y = Y[-20:]
U = U[-20:]
T = T[-20:]
U_normalized = U_normalized[-20:]
T_normalized = T_normalized[-20:]

en = estimate_elastic_net_regularizers_cv(Y, U, T, U_normalized)
print("\n")
linear = estimate_elastic_net_regularizers_cv_linear(Y, U, T, U_normalized)
print("\n")
rel_abun = estimate_elastic_net_regularizers_cv_rel_abun(Y, U, T, U_normalized)
print("\n")

print("EN", en)
print("Linear", linear)
print("Rel Abun", rel_abun)

# (alpha, r_A, r_g, r_B)
# EN (0.1, 0.7, 0.9, 0)
# Linear (10, 0, 0, 0.9)
# Rel Abun (0.1, 0, 0, 0.1)

#     r (0.1, 0, 0, 0) sqr error 73.79932682162575
#     r (0.1, 0, 0, 0.1) sqr error 71.56160963602494
#     r (0.1, 0, 0.1, 0.1) sqr error 71.56151474762191
#     r (0.1, 0, 0.5, 0.1) sqr error 71.56113873898404
#     r (0.1, 0, 0.7, 0.1) sqr error 71.56092512208
#     r (0.1, 0, 0.9, 0.1) sqr error 71.56071605381044
#     r (0.1, 0.7, 0, 0) sqr error 71.36581022971846
#     r (0.1, 0.7, 0.5, 0) sqr error 71.24032014611413
#     r (0.1, 0.7, 0.7, 0) sqr error 71.24028734974486
#     r (0.1, 0.7, 0.9, 0) sqr error 71.24024703095138


#     r (0.1, 0, 0, 0) sqr error 92.69710252922258
#     r (0.1, 0, 0, 0.1) sqr error 92.69113139723132
#     r (0.1, 0, 0.1, 0) sqr error 92.66782146667776
#     r (0.1, 0, 0.5, 0) sqr error 92.61037232729517
#     r (0.1, 0, 0.5, 0.1) sqr error 92.59874961153012
#     r (0.1, 0, 0.9, 0) sqr error 92.58128042000301
#     r (0.1, 0, 0.9, 0.1) sqr error 92.58100789438063
#     r (0.1, 0.1, 0.9, 0) sqr error 92.56815748368194
#     r (0.1, 0.1, 0.9, 0.1) sqr error 92.55895025811074
#     r (0.1, 0.5, 0.9, 0) sqr error 92.53700709018344
#     r (10, 0, 0, 0.1) sqr error 92.52484208636152
#     r (10, 0, 0, 0.5) sqr error 92.35717305021598
#     r (10, 0, 0, 0.7) sqr error 92.23111840640507
#     r (10, 0, 0, 0.9) sqr error 92.12133432621783


#     r (0.1, 0, 0, 0) sqr error 64.57435225808887
#     r (0.1, 0, 0, 0.1) sqr error 64.45164437213964


