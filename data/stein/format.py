import numpy as np
import pickle as pkl

# relative abundances
Y = []
# clindamycin
U = []
# time points
T = []

data = np.loadtxt("raw_data.csv", dtype=str, delimiter=",")
data = data[:,1:]

total_tpts = []

ids = data[2]
total_count = 0
zero_count = 0
for i in np.unique(ids):
    table = data[:,data[2] == i]
    times = table[3].astype(float)
    rel_abun = table[4:15].astype(float).T
    antibiotics = table[15].astype(float)
    antibiotics = antibiotics.reshape((antibiotics.size, 1))
    total_count += rel_abun.size
    zero_count += rel_abun[rel_abun == 0].size

    Y.append(rel_abun)
    U.append(antibiotics)
    T.append(times)
    total_tpts.append(rel_abun.shape[0])

print("Average Time Points", np.mean(total_tpts))
pkl.dump(Y, open("Y.pkl", "wb"))
pkl.dump(U, open("U.pkl", "wb"))
pkl.dump(T, open("T.pkl", "wb"))

print("% zero", zero_count / total_count)