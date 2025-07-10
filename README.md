# Stacking Variational Bayesian Monte Carlo (S-VBMC)

### What is it?
Stacking Variational Bayesian Monte Carlo (S-VBMC) is a fast post-processing step for [Variational Bayesian Monte Carlo (VBMC)](https://github.com/acerbilab/pyvbmc). VBMC produces a variational posterior in the form of a Gaussian mixture. S-VBMC improves upon this by combining ("stacking") the Gaussian mixture components from several independent VBMC runs into a single, larger mixture. It then re-optimizes the weights of this combined mixture to maximize the combined Evidence Lower BOund (ELBO, a lower bound on log [model evidence](https://en.wikipedia.org/wiki/Marginal_likelihood)). 

A key advantage of S-VBMC is its efficiency: **the original model is never re-evaluated**, making it an inexpensive way to boost inference performance.

### When should it be used?



### Repository layout

This repository is organized as follows:

  - `svbmc.py` contains the `SVBMC` class, which one should use to run our main algorithm.
  - `visual.py` contains the `overlay_corner_plot` function, used to overlay corner plots given one or more sets of samples.
  - `samples` is a folder containing the "ground truth" samples from our two synthetic targets (`gmm_D_2_GT.csv` and `ring_D_2_GT.csv`) obtained via direct sampling or extensive Markov Chain Monte Carlo (MCMC) as appropriate.
  - `vbmc_runs` is a folder containing the VBMC outputs from our multimodal target (in the `GMM` sub-folder) and from our ring-shaped target (in the `Ring` folder). These are `.pkl` files output by PyVBMC (10 per target), the python implementation of VBMC.
  - `examples.ipynb` is a notebook showing how to use the `SVBMC` class with our two synthetic examples (multimodal target and ring target).
  - `requirements.txt` contains all the dependencies necessary to run the algorithm.

-----

## How to use S-VBMC

### 1\. Installation

After creating and activating a virtual environment, install all required dependencies:

```bash
pip install -r requirements.txt
```

### 2\. Running S-VBMC

You should have already run VBMC multiple times on the same problem and saved the resulting `VariationalPosterior` objects as `.pkl` files. Refer to [these notebooks](https://github.com/acerbilab/pyvbmc/tree/main/examples) for VBMC usage examples.

First, load these objects into a single list. For example, if you have your files in a folder named `vbmc_runs/`:

```python
import pickle
import glob

vp_files = glob.glob("vbmc_runs/*.pkl")
vp_list = []
for file in vp_files:
    with open(file, "rb") as f:
        vp_list.append(pickle.load(f))
```

Next, initialize the `SVBMC` object with this list and run the optimization.

```python
from svbmc import SVBMC

# Initialize the stacked object and optimize the weights
vp_stacked = SVBMC(vp_list=vp_list)
vp_stacked.optimize()

# The stacked object now contains the optimized weights and ELBO estimates
print(f"Stacked ELBO: {vp_stacked.elbo['estimated']}")
```

-----

## ⚠️ Important: how to use the final posterior

Users must use samples from the stacked posterior for any application and should **not** interpret its individual components (means and covariances).

This is because each VBMC run may use different internal parameter transformations. Consequently, the means and covariance matrices from different VBMC posteriors exist in **incompatible parameter spaces**. Combining them creates a mixture whose individual Gaussian components are not directly meaningful.

**Always use samples from the final stacked posterior**, which are correctly transformed back into the original parameter space. These are available via the `.sample()` method:

```python
# Draw 10,000 samples from the final, stacked posterior
samples = vp_stacked.sample(n_samples=10000)
```

For a detailed walkthrough, see the `examples.ipynb` notebook.

**Note**: For compatibility with PyVBMC, this implementation of S-VBMC stores results in `numpy` arrays. However, it uses `pytorch` under the hood to run the stacked ELBO optimization.




