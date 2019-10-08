Code to replicate experiments from *Compositional Lotka-Volterra describes microbial dynamics in the simplex*.

## Environment
All experiments were performed using Python 3.5.2. Packages, along with version numbers, can be found in requirements.txt. Most (if not all) of the code should run on any version of Python 3, and most versions of the packages in requirements.txt.

## Rerunning Experiments
Each experiment is organized by the dataset on which it was performed. Datasets are labeled by FIRST_AUTHOR (e.g. Bucci, Stein, Taur). Each script runs one analysis, and outputs a figure in the plots/ folder.

### Simulated Data
To rerun the analysis from Figure 2, run:

```
bash$ ./simulation_comparison.sh
```

Because model training can be slow, parameters are loaded from and saved to the tmp/ folder. Delete the files in tmp to retrain the model.

### gLV vs cLV Parameter Comparison
To rerun the analysis from Figure 3, run:

```
python plot_concentrations.py
python bucci_cdiff_lotka_volterra_comparision.py
python bucci_diet_lotka_volterra_comparison.py
python stein_lotka_volterra_comparison.py
```

### Prediction Performance
To rerun the analysis from Figures 4-6, run:

```
python bucci_cdiff_prediction_experiments.py
python bucci_diet_prediction_experiments.py
python stein_prediction_experiments.py
python taur_prediction_experiments.py
```
This loads saved parameters from tmp/. To retrain the models, delete the files in tmp.

### Domination Prediction
To rerun the analysis from Figures 7-8, run:

```
python taur_domination_prediction.py
```

## Data and Preprocessing
Data and preprocessing steps can be found under data/FIRST_AUTHOR.