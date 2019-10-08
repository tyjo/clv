import csv
import numpy as np

otus = np.loadtxt("otu_counts.csv", delimiter=",", dtype=str, encoding="utf-8-sig")[1:,0]
otus = np.unique(otus).tolist()
otuid_name = {}

with open("otu_taxonomy.csv", "r", encoding="utf-8-sig") as csvfile:
    reader = csv.DictReader(csvfile, delimiter=",")
    for row in reader:
        otu_id = row["otuId"]
        taxon = [row["blast_kingdom"],
                 row["blast_phylum"],
                 row["blast_class"],
                 row["blast_order"],
                 row["blast_family"],
                 row["blast_genus"] ]
        taxon = ";".join(taxon)
        assert otu_id not in otuid_name
        otuid_name[otu_id] = taxon



pool_patientday = {}
patientday = []
pid_days = set()
pids = set()
with open("patient_table.csv", "r", encoding="utf-8-sig") as csvfile:
    reader = csv.DictReader(csvfile, delimiter=",")
    for row in reader:
        if row["Fig1"] == '1':
            pool = row["sampleIdSequenced"]
            pid = row["patientID"]
            day = row["day"]
            assert (pid, day) not in pid_days
            pid_days.add((pid, day))
            pids.add(pid)
            pool_patientday[pool] = (pid, day)
            patientday.append((pid, day))


taxa = np.unique(list(otuid_name.values())).tolist()
nobs = len(pool_patientday.keys())
notus = len(taxa)
otu_table = np.zeros((notus, nobs))

with open("otu_counts.csv", "r", encoding="utf-8-sig") as csvfile:
    reader = csv.DictReader(csvfile, delimiter=",")
    for row in reader:
        if not row["sampleIdSequenced"] in pool_patientday: continue

        pid, day = pool_patientday[row["sampleIdSequenced"]]
        otu = row["otuId"]
        taxon = otuid_name[otu]
        count = row["counts"]

        row_idx = taxa.index(taxon)
        col_idx = patientday.index((pid, day))
        otu_table[row_idx, col_idx] += float(count)

assert np.all(otu_table.sum(axis=0) > 0)
otu_table = otu_table.astype(str)
patient_day = np.array([ [pd[0], pd[1]] for pd in patientday]).T
otus = np.array(["pid", "day"] + taxa).reshape((len(taxa) + 2, 1))


otu_table = np.vstack((patient_day, otu_table))
otu_table = np.hstack((otus, otu_table))
np.savetxt("taur-otu-table.csv", otu_table, delimiter=",", fmt="%s")

print("found", len(pids), "patients across", otu_table.shape[1] - 1, "time points")