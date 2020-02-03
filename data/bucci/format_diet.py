import numpy as np
import pickle as pkl

from scipy.interpolate import UnivariateSpline

from format_cdiff import plot_trajectories

sample_id_to_subject_id = {}
subject_id_time = {}
subject_id_u = {}


def compute_breakpoints(effects):
    """Break the spline when external perturbations occur."""
    breakpoints = []
    for u in effects:
        in_perturb = False
        v = []
        for i,ut in enumerate(u):
            if i == 0 or i == u.shape[0]-1:
                v.append(1)
                continue
            if np.any(ut) > 0 and not in_perturb:
                v.append(1)
                in_perturb = True
            elif np.any(ut) > 0 and in_perturb:
                v.append(0)
            elif np.all(ut) == 0 and in_perturb:
                i = 1 if v[i-1] == 0 else 0
                v.append(i)
                in_perturb = False
            else:
                v.append(0)
        v = np.array(v)
        breakpoints.append(np.nonzero(v)[0])
    return breakpoints


def denoise(counts, t_pts, effects=None):
    """Takes a sequence of counts at t_pts, and returns denoised estimates
    of latent trajectories."""
    ntaxa = counts[0].shape[1]
    denoised_traj = []

    if effects is not None:
        breakpoints = compute_breakpoints(effects)
        for c,t,b in zip(counts,t_pts,breakpoints):
            denoised = np.zeros(c.shape)
            mass = c.sum(axis=1,keepdims=True)
            p = c / c.sum(axis=1,keepdims=True)
            p[p==0] = 1e-5
            p /= p.sum(axis=1,keepdims=True)
            c = (mass.T*p.T).T
            for i in range(ntaxa):
                for j in range(1,b.size):
                    start = b[j-1]
                    end = b[j]+1
                    k = 5 if end - start <= 3 else 5
                    #k = 3
                    f = UnivariateSpline(t[start:end],c[start:end,i],k=k)
                    denoised[start:end,i] = f(t[start:end])
            denoised[0] = c[0]
            denoised = np.clip(denoised, np.min(denoised[denoised > 0]), np.inf)
            denoised_traj.append(denoised)
    else:
        for c,t in zip(counts,t_pts):
            denoised = np.zeros(c.shape)
            k = 3 if t.shape[0] <= 5 else 5
            for i in range(ntaxa):
                f = UnivariateSpline(t,c[:,i],k=k)
                denoised[:,i] = f(t)
            denoised = np.clip(denoised, np.min(denoised[denoised > 0]), np.inf)
            denoised /= denoised.sum(axis=1,keepdims=True)
            denoised_traj.append(denoised)

    return denoised_traj


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
for subject_id in sorted(subject_id_counts):
    y = np.array(subject_id_counts[subject_id])
    t = np.array(subject_id_time[subject_id])
    u = np.array(subject_id_u[subject_id])
    u = u.reshape((u.size, 1))
    zero_counts += y[y == 0].size
    total_counts += y.size

    Y_diet.append(y)
    U_diet.append(u)
    T_diet.append(t)


Y_diet_denoised = denoise(Y_diet, T_diet, effects=U_diet)
#plot_trajectories(Y_diet_denoised, T_diet, "./", "diet-denoised")
#plot_trajectories(Y_diet, T_diet, "./", "diet-raw")

pkl.dump(Y_diet, open("Y_diet.pkl", "wb"))
pkl.dump(Y_diet_denoised, open("Y_diet_denoised.pkl", "wb"))
pkl.dump(U_diet, open("U_diet.pkl", "wb"))
pkl.dump(T_diet, open("T_diet.pkl", "wb"))

print("sample size", len(Y_diet))
print("% zero", zero_counts / total_counts)