import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import seaborn as sns

def compute_concentrations(Y):
    con =  []
    for y in Y:
        con += y.sum(axis=1).tolist()
    con = np.array(con)
    C = 1 / np.mean(con)
    return np.array(C*con)

Y_stein = pkl.load(open("pub-results/stein/Y.pkl", "rb"))
Y_cdiff = pkl.load(open("pub-results/bucci/Y_cdiff-denoised.pkl", "rb"))
Y_diet  = pkl.load(open("pub-results/bucci/Y_diet.pkl", "rb"))

fig, ax = plt.subplots(nrows=3, ncols=1,figsize=(8,3.5))
sns.set(style="whitegrid")
sns.boxplot(ax=ax[2], x=compute_concentrations(Y_stein))
ax[2].set_title("Antibiotic Dataset", size=16)
sns.boxplot(ax=ax[0], x=compute_concentrations(Y_diet))
ax[0].set_title("Diet Dataset", size=16)
sns.boxplot(ax=ax[1], x=compute_concentrations(Y_cdiff))
ax[1].set_title("$\it{C. diff}$ Dataset", size=16)

for i in range(3):
    ax[i].set_xlabel("Concentration", fontsize=12)
    ax[i].set_xlim(-0.5, 5.5)
    ax[i].set_xticks([0, 1, 2, 3, 4, 5])
    ax[i].tick_params(axis="x", labelsize=12)
plt.tight_layout()
plt.savefig("plots/concentrations.pdf")