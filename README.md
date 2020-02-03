Code to replicate experiments from *Compositional Lotka-Volterra describes microbial dynamics in the simplex*.

## Environment
All experiments were performed using Python 3.5.2. Packages, along with version numbers, can be found in requirements.txt. Most (if not all) of the code should run on any version of Python 3, and most versions of the packages in requirements.txt.

## Rerunning Experiments
Each experiment is organized by the dataset on which it was performed. Datasets are labeled by FIRST_AUTHOR (e.g. Bucci, Stein). Each script runs one analysis, and outputs a figure in the plots/ folder.

### Simulated Data
The script ```simulation_bucci_clv.py``` is responsible for generating simulated data and performing the comparison between elastic net and ridge regression. It uses a command line interface to set simulation parameters, an example of which can be found in ```simulation_bucci_clv.sh```. The script ```plot_simulation_bucci_clv.py``` plots the simulation results saved in ```tmp_sim```.

### cLV vs gLV Parameter Comparison
The script ```plot_concentrations.py``` plots boxplots of community size, rescaled with mean size of 1. The scripts

```
bucci_cdiff_lokta_volterra_comparison.py
bucci_diet_lokta_volterra_comparison.py
stein_lokta_volterra_comparison.py
```

generates plots of the parameter comparison between cLV and gLV.

### Prediction Performance
To rerun the prediction comparison, use the scripts

```
bucci_cdiff_prediction_experiments.py
bucci_diet_prediction_experiments.py
stein_prediction_experiments.py
```
This loads saved parameters from tmp/. To retrain the models, delete the files in tmp. Plot the results using

```
plot_prediction_results.py
```

## Data and Preprocessing
Data and preprocessing steps can be found under data/FIRST_AUTHOR.