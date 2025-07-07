# Stacking Variational Bayesian Monte Carlo (S-VBMC)

Stacking Variational Bayesian Monte Carlo (S-VBMC) builds on Variational Bayesian Monte Carlo (VBMC, find the python implementation [here](https://github.com/acerbilab/pyvbmc)), a sample-efficient Bayesian inference approach that performs variational inference on a Gaussian Process (GP) surrogate of the model to produce a variational posterior in the form of a mixture of Gaussians. S-VBMC consists in "stacking" a set of these variational posteriors and re-optimising the stacked ELBO w.r.t. the weights of all Gaussian components, leaving the means and covariance matrices unchanged. This is done as a simple and inexpensive post-processing step, and the model does not need to be re-evaluated at any point.

This repository is organised as follows:
- `svbmc.py` contains the `SVBMC` class, which one should use to run our main algorithm.
- `visualisation.py` contains the `overlay_corner_plot` function, used to overlay corner plots given one or more sets of samples.
- `samples` is a folder containing the "ground truth" samples from our two synthetic targets (`gmm_D_2_GT.csv` and `ring_D_2_GT.csv`) obtained via direct sampling or extensive Markov Chain Monte Carlo (MCMC) as appropriate.
- `vbmc_runs` is a folder containing the VBMC outputs from our multimodal target (in the `GMM` sub-folder) and from our ring-shaped target (in the `Ring` folder). These are `pickle` files output by PyVBMC (10 per target), the aforementioned python implementation of VBMC.
- `examples.ipynb` is a notebook showing how to use the `SVBMC` class with our two synthetic examples (multimodal target and ring target).
- `requirements.txt` contains all the dependencies necessary to run the algorithm.


### How to use S-VBMC

After creating/activating a virtual environent, one should run
```
pip install -r requirements.txt
```
to install all required dependencies.

One should already have run VBMC multiple times (see illustrative example [here](https://github.com/acerbilab/pyvbmc/blob/main/examples/pyvbmc_example_1_basic_usage.ipynb)) on the same problem before using S-VBMC, and save the resulting `VariationalPosterior` obects in `.pkl` files.

These files should be loaded and put in a list `vp_list`, which is the minimum necessary input one needs to use the `SVBMC` class. To optimise the stacked ELBO, one can simply initialise the object and run the optimisation like so:
```
vp_stacked = SVBMC(vp_list=vp_list)
vp_stacked.optimize()
```

After this, the `SVBMC` object `vp_stacked` will contained the new optimised mixture weights (`vp_stacked.w`, a `numpy` array) and the stacked ELBO (`vp_stacked.elbo`, a dictionary containing both uncorrected and debiased estimates, see paper for details).

__IMPORTANT__: Users should now use the sufficient statistics (weights, means and covariance matrices) of the stacked posterior for any application. This is because VBMC (if ran with default settings) applies different parameter transformations in each inference run. The means and covariance matrices corresponding to different VBMC posteriors therefore live in different parameter spaces, making the final mixture (i.e. the stacked posterior) tricky to interpret. Therefore, <u>one should always use samples from the stacked posterior</u>. These are available with the `sample` method within the `SVBMC` class.


The `examples.ipynb` notebook illustrates the use of the `SVBMC` class in more detail. 


__Note__: For compatibility with PyVBMC, this implementation of S-VBMC stores results in `numpy` arrays. However, it uses `pytorch` under the hood to run the stacked ELBO optimisation.