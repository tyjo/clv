#!/usr/bin/env bash

DAYS_BTWN=(1 2 4 6)
SEQ_DEPTH=(-1 200 1000 5000 10000 25000)
SAMPLE_SIZE=(5 10 25 50 100)

# setting random fixes the set of random seeds
RANDOM=21789

for db in "${DAYS_BTWN[@]}"; do
    for depth in "${SEQ_DEPTH[@]}"; do
        for ss in "${SAMPLE_SIZE[@]}"; do
            python simulation_bucci_clv.py ${ss} ${db} ${depth} $RANDOM
        done
    done
done