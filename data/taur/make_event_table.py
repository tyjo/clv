import csv
import numpy as np

#classes = np.loadtxt("antibiotics.csv", delimiter=",", encoding="utf-8-sig", dtype=str)[1:,2]

#for idx, cl in enumerate(np.unique(classes)):
#    print('"' + str(cl) + '"', ":", '"' + str(idx) + '"', ",")

event_map = {
    "PCP prophylaxis agents" : "0" ,
    "aminoglycosides" : "1" ,
    "anti-VRE agents" : "2" ,
    "anti-anaerobic agent" : "3" ,
    "anti-tuberculous antibiotics" : "4" ,
    "beta-lactamase inhibitors" : "5" ,
    "carbapenems" : "6" ,
    "first/second generation cephalosporins" : "7" ,
    "fourth/fifth generation cephalosporins" : "8" ,
    "glycopeptide" : "9" ,
    "macrolides" : "10" ,
    "miscellaneous antibiotics" : "11" ,
    "monobactams" : "12" ,
    "penicillins" : "13" ,
    "quinolones" : "14" ,
    "tetracyclines" : "15" ,
    "third generation cephalosporins" : "16",
    "surgery" : "17"
}

patient_days = {}

# with open("patient_table.csv", "r", encoding="utf-8-sig") as csvfile:
#     reader = csv.DictReader(csvfile, delimiter=",")
#     for row in reader:
#         if row["Fig1"] == '1':
#             pid = row["patientID"]
#             day = row["day"]
#             if pid in patient_days:
#                 patient_days[pid] += [int(day)]
#             else:
#                 patient_days[pid] = [int(day)]
dat = np.loadtxt("taur-otu-table-top10+dom.csv", delimiter=",", dtype=str)
pids = dat[0,1:]
days = dat[1,1:]
for pid, day in zip(pids, days):
    if pid in patient_days:
        patient_days[pid] += [int(day)]
    else:
        patient_days[pid] = [int(day)]

events = open("taur-events.csv", "w")
events.write("patientID,eventID,startDay,endDay\n")

for pid in patient_days:
    row = [pid, "surgery", "0", "0"]
    events.write(",".join(row) + "\n")

with open("antibiotics.csv", "r", encoding="utf-8-sig") as csvfile:
    reader = csv.DictReader(csvfile, delimiter=",")
    for row in reader:
        cl = row["class"]
        start_day = int(row["startday"])
        stop_day = int(row["stopday"])
        pid = row["patientID"]
        eid = cl

        if pid not in patient_days:
            continue

        days = patient_days[pid]

        if start_day > max(days) or stop_day < min(days):
            continue
        start_day = str( max( min(days), start_day) )
        stop_day = str( min( max(days), stop_day) )

        # # only occurs when the antibiotic interval does not overlap
        # # the microbiome data interval
        # if stop_day < start_day:
        #     continue

        row = [pid, eid, start_day, stop_day]
        events.write(",".join(row) + "\n")



