import numpy as np
import pickle as pkl
import os


def plot_trajectories(Y, T, output_dir, outfile):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    def plot_bar(ax, y, time, unique_color_id, remaining_ids):
        T = y.shape[0]
        cm = plt.get_cmap("tab20c")
        colors = [cm(i) for i in range(20)]
        #time = np.array([t for t in range(T)])
        widths = np.concatenate((time[1:] - time[:-1], [1])).astype(float)
        widths[widths > 1] = 1

        widths -= 1e-1

        y_colors = y[:,unique_color_id]
        ax.bar(time, y_colors[:,0], width=widths, color=colors[0], align="edge")
        for j in range(1, y_colors.shape[1]):
            ax.bar(time, y_colors[:,j], bottom=y_colors[:,:j].sum(axis=1), width=widths, color=colors[j], align="edge")
        
        ax.bar(time, y[:,remaining_ids].sum(axis=1), bottom=y_colors.sum(axis=1), width=widths, color=colors[19], align="edge")
        #ax.set_title("Relative Abundances", fontsize=10)
        #ax.legend(prop={"size" : 4}, bbox_to_anchor=[-0.1,1.225], loc="upper left", ncol=4)

    def find_top_ids(Y, n):
        ntaxa = Y[0].shape[1]
        rel_abun = np.zeros(ntaxa)
        for y in Y:
            tpts = y.shape[0]
            denom = y.sum(axis=1,keepdims=True)
            denom[denom == 0] = 1
            p = y / denom
            rel_abun += p.sum(axis=0) / tpts
        ids = np.argsort(-rel_abun)
        return np.sort(ids[:n]), np.sort(ids[n:])

    N = len(Y)
    top19_ids, remaining_ids = find_top_ids(Y, 19)
    fig, ax = plt.subplots(nrows=N,ncols=1,figsize=(N,2*N))
    for i in range(N):
        denom = Y[i].sum(axis=1)
        denom[denom == 0] = 1
        plot_bar(ax[i], (Y[i].T / denom).T, T[i], top19_ids, remaining_ids)


    outfile = os.path.splitext(outfile)[0]
    plt.tight_layout()
    plt.savefig(output_dir + "/" + outfile + ".pdf")
    plt.close()


sample_id_to_subject_id = {}
subject_id_time = {}
subject_id_u = {}
subject_id_biomass = {}

f = open("data_cdiff/metadata.txt", "r")
#g = open("data_cdiff/biomass.txt", "r")

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

#plot_trajectories(Y_cdiff, T_cdiff, "./", "cdiff-denoised")
pkl.dump(Y_cdiff, open("Y_cdiff-denoised.pkl", "wb"))
pkl.dump(U_cdiff, open("U_cdiff.pkl", "wb"))
pkl.dump(T_cdiff, open("T_cdiff.pkl", "wb"))


