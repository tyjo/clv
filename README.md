Code to replicate experiments from *Compositional Lotka-Volterra describes microbial dynamics in the simplex*.

## Environment
All experiments were performed using Python 3.5.2. Packages, along with version numbers, can be found in requirements.txt. Most (if not all) of the code should run on any version of Python 3, and most versions of the packages in requirements.txt.

## Generating Figures
The following scripts generate the panels for the main results of the paper.

### Figure 3
```
plot_concentrations.py
bucci_cdiff_lokta_volterra_comparison.py
bucci_diet_lokta_volterra_comparison.py
stein_lokta_volterra_comparison.py
```

### Figure 4
```
plot_prediction_results.py
```

### Figure 5 and S9
```
stein_plot_parameters.py
```

### Figure S5, S6 and S7
```
plot_prediction_trajectories.py
```



## Rerunning Experiments
Each experiment is organized by the dataset on which it was performed. Datasets are labeled by FIRST_AUTHOR (e.g. Bucci, Stein).

### Model Comparison
The following scripts run the model comparison experiments:

```
bucci_cdiff_prediction_experiments.py
bucci_diet_prediction_experiments.py
stein_prediction_experiments.py
```

### Simulated Data
The script ```simulation_bucci_clv.py``` is responsible for generating simulated data and performing the comparison between elastic net and ridge regression. It uses a command line interface to set simulation parameters, an example of which can be found in ```simulation_bucci_clv.sh```. The script ```plot_simulation_bucci_clv.py``` plots the simulation results saved in ```tmp_sim```.

## Data and Preprocessing
Data and preprocessing steps can be found under data/FIRST_AUTHOR.