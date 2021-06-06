# blockworlds

A sandbox module for geophysical inversions based on kinematic geological models


## Summary

This package demonstrates an anti-aliased projection from kinematic geological parameters onto discrete voxels for calculating geophysical signals, as part of an inversion workflow.  The underlying geological model uses kinematics similar to the popular package Noddy (Jessell 1981; Jessell & Valenta 1993), although it is greatly simplified and is not meant to be general-purpose in its current form.  Version 0.1.0-beta is the code used in the paper "Anti-aliasing of geophysics for probabilistic inversions of 3-D geological forward models: a demonstration with Blockworlds 0.1.0", submitted to Geoscientific Model Development in June 2021.

The main difference between Blockworlds and Noddy is that the rock properties in each mesh cell are evaluated as an approximate numerical average of the high-resolution underlying geology, rather than a point estimate made at the center of the cell.  This results in a geophysical likelihood that is smooth in the geological parameters, enabling inference over geological forward models to be done using constraints from geophysical data.


## Dependencies

* Python 3.6;
* ``numpy``/``scipy``/``matplotlib``;
* the [``SimPEG``](https://github.com/simpeg/simpeg) geophysics library (v0.14.0);
* the [``riemann``](https://github.com/rscalzo/riemann) MCMC sampling library (v1.0.0).

The calculations of autocorrelation time in our paper use the ``autocorr`` subpackage of v3.0.2 of the [``emcee``](https://emcee.readthedocs.io/en/v2.2.1/) package (Foreman-Mackey et al. 2013), but this is only used when generating summary tables.  You won't need emcee to run our models.


## Installation

```python setup.py install```


## Reproducing Figures from the Text

The code for Figure 1 and Figure 2a are found in the Jupyter notebook ``figures.ipynb``, since these figures required a lot of visual tuning that was most easily done interactively.

Figure 2b can be reproduced by running ``antialias.py`` from the command line.

The computer model elements of Figures 3-7 can be reproduced for any of the models (not just the ones shown) by running ``mcmc_figures.py``.


## Demo Notebook

A Jupyter notebook showing an example of how blockworlds models are set up and run can be found in ```blockworlds.ipynb```.  This notebook fills the role of a manual for the code.
