import numpy as np

taur = np.loadtxt("taur-otu-table.csv", delimiter=",", dtype=str)
otu_table = taur[2:,1:].astype(float)
print(otu_table.shape)

print("starting with", np.unique(taur[0]).size, "patients")

# Filter by sequencing depth
seq_depth = otu_table.sum(axis=0)
keep = np.concatenate((np.array([True]), seq_depth > 1000))
taur = taur[:,keep]

p_ids = taur[0,1:].astype(int)
days = taur[1,1:].astype(int)
obs = taur[2:,1:].astype(float)
otu_names = taur[2:,0]


def average_density(days):
    days = np.array(days)
    return np.mean(days[1:] - days[:-1])


for i, p_id in enumerate(p_ids):
    if i < 2:
        continue
    assert int(p_ids[i-1]) <= int(p_ids[i])

keep = [True]
used = set()
total = 0
dom_events = {}
total_tpts = 0
t_pts = []
for p_id in np.unique(p_ids):
    patient_days = days[p_ids == p_id]
    total += patient_days.size

    # discard if there is only one observation
    if patient_days.size < 2:
        passed = [False]
        keep = keep + passed
        continue

    passed = [True for d in patient_days]
    # clip observations too far in the future or start day is too far
    k = True
    for i in range(1, patient_days.size):
        if patient_days[0] > 10:
            k = False

        if patient_days[i] - patient_days[i-1] > 10:
            k = False
        passed[i] = k

    # remove if too sparse
    if np.sum(np.array(passed).astype(int)) < 3 or average_density(patient_days[passed]) > 6:
        passed = [False for d in patient_days]
        keep = keep + passed
        continue

    # if np.sum(np.array(passed).astype(int)) < 15:# or average_density(patient_days[passed]) > 6:
    #     passed = [False for d in patient_days]
    #     keep = keep + passed
    #     continue

    keep = keep + passed

    total_tpts += np.sum(passed)
    t_pts.append(np.sum(passed))

    patient_obs = obs[:, p_ids == p_id]
    rel_abun =  patient_obs / patient_obs.sum(axis=0)
    seen_dom = set()
    if np.any(rel_abun > 0.3) and np.any(passed):
        for r in rel_abun.T:
            dom_taxa = otu_names[np.argwhere(r > 0.3)].flatten()
            for taxon in dom_taxa:
                if taxon not in seen_dom:
                    dom_events[taxon] = dom_events.get(taxon, 0) + 1
                    seen_dom.add(taxon)

total = 0
for taxon, number in sorted(dom_events.items(), key=lambda item: -item[1]):
    print(taxon, number)
    total += number
print(total)

print("Total Time Points", total_tpts)
print("Average Time Points", np.mean(t_pts))
taur = taur[:,keep]

print(np.unique(taur[0]).size - 2, "patients remaining")
np.savetxt("taur-otu-table-filtered.csv", taur, delimiter=",", fmt="%s")