import numpy as np

include_taxa = ["Bacteria;Firmicutes;Bacilli;Lactobacillales;Streptococcaceae;Streptococcus",
                "Bacteria;Firmicutes;Bacilli;Bacillales;Staphylococcaceae;Staphylococcus",
                "Bacteria;Firmicutes;Bacilli;Lactobacillales;Lactobacillaceae;Lactobacillus",
                "Bacteria;Proteobacteria;Gammaproteobacteria;Enterobacterales;Enterobacteriaceae;Klebsiella",
                "Bacteria;Firmicutes;Bacilli;Lactobacillales;Enterococcaceae;Enterococcus",
                "Bacteria;Proteobacteria;Gammaproteobacteria;Enterobacterales;Enterobacteriaceae;Escherichia",
                "Bacteria;Firmicutes;Erysipelotrichia;Erysipelotrichales;Erysipelotrichaceae;Erysipelatoclostridium"]


def subset_top_otus(otu_table, N):
    """Apply a basic filter to select the top N most abundant OTUs.
    """
    data = otu_table[2:, 1:].astype(float)
    abundances = (data / data.sum(axis=0,keepdims=True)).sum(axis=1)
    sorted_idx = np.argsort(-abundances)
    top_idx = sorted_idx[:N]

    taxa = otu_table[2:,0].tolist()
    taxa_idx = [taxa.index(t) for t in include_taxa]
    
    include_idx = np.union1d(top_idx, taxa_idx)

    #print(include_idx)
    #assert np.all(np.unique(include_idx) == np.unique(top_idx))
    #include_idx = top_idx
    exclude_idx = np.array([idx for idx in range(data.shape[0]) if idx not in include_idx])
    N = include_idx.size

    new_otu_table = np.zeros((N+1, data.shape[1]))
    new_otu_table[:N,:] = data[include_idx,:]
    new_otu_table[N,:] = data[exclude_idx,:].sum(axis=0).astype(str)
    print("zeros in last column:", (new_otu_table[N,:].astype(float) == 0).sum() )

    taxa_labels = np.concatenate( (otu_table[:2,0], otu_table[include_idx+2,0], np.array(["aggregate"]) ) )
    taxa_labels = taxa_labels.reshape((taxa_labels.size, 1))
    id_days = otu_table[:2,1:]

    new_otu_table = np.vstack( (id_days, new_otu_table) )
    new_otu_table = np.hstack( (taxa_labels, new_otu_table) )
    return new_otu_table


dat = np.loadtxt("taur-otu-table-filtered.csv", delimiter=",", dtype=str)
print(dat.shape)
dat = subset_top_otus(dat, 20)
print(dat.shape)
np.savetxt("taur-otu-table-top10+dom.csv", dat, delimiter=",", fmt="%s")

# Compute fraction of reads in top taxa
otu_table = dat[2:,1:].astype(float)
reads_per_taxa = otu_table.sum(axis=1)
frac_covered = 1 - (reads_per_taxa / reads_per_taxa.sum())[-1]
print("covered", frac_covered, "of all reads")

# Fraction of zeros
print("fraction zero", otu_table[otu_table == 0].size / otu_table.size)
