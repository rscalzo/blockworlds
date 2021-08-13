# blockworlds

A sandbox module for geophysical inversions based on kinematic geological models


## Summary

This package demonstrates an anti-aliased projection from kinematic geological parameters onto discrete voxels for calculating geophysical signals, as part of an inversion workflow.  The underlying geological model uses kinematics similar to the popular package Noddy (Jessell 1981; Jessell & Valenta 1993), although it is greatly simplified and is not meant to be general-purpose in its current form.  Version 0.1.0-beta is the code used in the paper "Blockworlds 0.1.0:  A demonstration of anti-aliased geophysics for probabilistic inversions of implicit and kinematic geological models", submitted to Geoscientific Model Development as manuscript **gmd-2021-187** in June 2021.

The main difference between Blockworlds and Noddy is that the rock properties in each mesh cell are evaluated as an approximate numerical average of the high-resolution underlying geology, rather than a point estimate made at the center of the cell.  This results in a geophysical likelihood that is smooth in the _geological_ parameters, simplifying inference over geological forward models to be done using constraints from geophysical data.


## Dependencies

Core dependencies, without which the main examples and figures will not run, include:

* Python 3.6;
* ``numpy``/``scipy``/``matplotlib``;
* the [``SimPEG``](https://github.com/simpeg/simpeg) geophysics library (v0.14.0);
* the [``riemann``](https://github.com/rscalzo/riemann) MCMC sampling library (v0.1.0).

Optional dependencies include:

* [``emcee``](https://emcee.readthedocs.io/en/v2.2.1/) (v3.0.2) -- this is a popular MCMC package, but here we use only the ``autocorr`` subpackage to calculate the autocorrelation time for some of the summary tables.
* [``jax``](https://jax.readthedocs.io/en/latest/index.html) (v0.2.17) -- a convenient autodifferentiation library used in some experimental code in the examples section, not used in the initial Blockworlds paper.


## Installation

```python setup.py install```


## Reproducing Figures from the Text

All examples relevant to the GMD paper can be found under ``examples/gmd-2021-187``.  They should run normally as long as Blockworlds and its core dependencies are installed.

The code for Figure 1 and Figure 2a are found in the Jupyter notebook ``figures.ipynb``, since these figures required a lot of visual tuning that was most easily done interactively.

Figure 2b can be reproduced by running ``antialias.py`` from the command line.

The computer model elements of Figures 3-7 can be reproduced for any of the models (not just the ones shown) by running ``mcmc_figures.py``.  Models can be run individually or in batches.  For example, the command

```mcmc_figures.py --model_ids 2 4 6 8 --run_mcmc --results_table --traceplots```

will re-run MCMC sampling for all even-numbered models, writing the chain output to pickle files in the working directory, and outputting a summary table in
the same format as Table 2.  Trace plots will also be generated for all model parameters in the same format as Figure 7, with one EPS plot for each parameter of each model.

```mcmc_figures.py --model_ids 1 2 3 4 5 6 7 8 --run_slicefigs```

will re-run the posterior slice figures for all models, with one plot for each parameter of each model (i.e. a row of Figure 5 or 6).


## Demo Notebook

A separate Jupyter notebook showing an example of how Blockworlds models are set up and run interactively can be found in ```implicit.ipynb```.  The setup closely follows the production code structure.
