import numpy as np
import pickle as pkl

sample_id_to_subject_id = {}
subject_id_time = {}
subject_id_u = {}


with open("data_diet/metadata.txt", "r") as f:
    for line in f:
        line = line.split()

        if "sampleID" in line[0]:
            continue

        sample_id = line[0]
        subject_id = line[2]
        day = float(line[3])
        perturb = float(line[4])

        sample_id_to_subject_id[sample_id] = subject_id
        subject_id_time[subject_id] = subject_id_time.get(subject_id, []) + [day]
        subject_id_u[subject_id] = subject_id_u.get(subject_id, []) + [perturb]


counts = np.loadtxt("data_diet/counts.txt", delimiter="\t", dtype=str, comments="!")

# swap last two rows since there are no zeros in the penultimate row
tmp = counts[-2]
counts[-2] = counts[-1]
counts[-1] = tmp
counts = counts[:,1:]

subject_id_counts = {}

for row in counts.T:
    sample_id = row[0]
    counts = row[1:].astype(float)
    subject_id = sample_id_to_subject_id[sample_id]

    counts /= 1000
    if subject_id in subject_id_counts:
        subject_id_counts[subject_id] = np.vstack( (subject_id_counts[subject_id], np.array(counts)) )
    else:
        subject_id_counts[subject_id] = np.array(counts)


Y_diet = []
U_diet = []
T_diet = []
zero_counts = 0
total_counts = 0
for subject_id in subject_id_counts:
    y = np.array(subject_id_counts[subject_id])
    t = np.array(subject_id_time[subject_id])
    u = np.array(subject_id_u[subject_id])
    u = u.reshape((u.size, 1))
    zero_counts += y[y == 0].size
    total_counts += y.size

    Y_diet.append(y)
    U_diet.append(u)
    T_diet.append(t)

    print(y.shape)

pkl.dump(Y_diet, open("Y_diet.pkl", "wb"))
pkl.dump(U_diet, open("U_diet.pkl", "wb"))
pkl.dump(T_diet, open("T_diet.pkl", "wb"))

print("sample size", len(Y_diet))
print("% zero", zero_counts / total_counts)