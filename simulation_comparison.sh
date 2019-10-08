#!/usr/bin/env bash
ntaxa=(15 30 60)
nsim=(5 10 25 50)

for nt in "${ntaxa[@]}"; do
    for ns in "${nsim[@]}"; do
        python simulation_comparison.py ${ns} 25 1 ${nt} --effects
    done
done

# python simulation_comparison.py 10 25 1 15 --effects