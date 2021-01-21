import numpy as np
import pickle as pkl
import os

sample_id_to_subject_id = {}
subject_id_time = {}
subject_id_u = {}
subject_id_biomass = {}

f = open("data_cdiff/metadata.txt", "r")

for line in f:
    line = line.split()
    if "sampleID" in line[0]:
        continue
    sample_id = line[0]
    subject_id = int(line[2])
    day = float(line[3])
    
    if day == 28.75:
        perturb = 1
    else:
        perturb = 0

    sample_id_to_subject_id[sample_id] = subject_id
    subject_id_time[subject_id] = subject_id_time.get(subject_id, []) + [day]
    subject_id_u[subject_id] = subject_id_u.get(subject_id, []) + [perturb]

f.close()

denoised_data_files = ["denoise-conc-1.csv", "denoise-conc-2.csv", 
                        "denoise-conc-3.csv", "denoise-conc-4.csv",
                        "denoise-conc-5.csv"]

subject_id_counts = {}
for i,f in enumerate(denoised_data_files):
    matrix = np.loadtxt("data_cdiff/" + f, encoding='utf-8-sig', delimiter=",").T
    subject_id_counts[i+1] = matrix

Y_cdiff = []
U_cdiff = []
T_cdiff = []
zero_count = 0
total_count = 0
for subject_id in subject_id_counts:
    y = np.array(subject_id_counts[subject_id])
    t = np.array(subject_id_time[subject_id])
    u = np.array(subject_id_u[subject_id])
    u = u.reshape((u.size, 1))

    Y_cdiff.append(y)
    U_cdiff.append(u)
    T_cdiff.append(t)

pkl.dump(Y_cdiff, open("Y_cdiff-denoised.pkl", "wb"))
pkl.dump(U_cdiff, open("U_cdiff.pkl", "wb"))
pkl.dump(T_cdiff, open("T_cdiff.pkl", "wb"))


