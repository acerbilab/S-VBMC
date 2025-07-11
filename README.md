# Stacking Variational Bayesian Monte Carlo (S-VBMC)

### Overview
Stacking Variational Bayesian Monte Carlo (S-VBMC) is a fast post-processing step for [Variational Bayesian Monte Carlo (VBMC)](https://github.com/acerbilab/pyvbmc). VBMC is an approximate Bayesian inference technique that produces a variational posterior in the form of a Gaussian mixture. S-VBMC improves upon this by combining ("stacking") the Gaussian mixture components from several independent VBMC runs into a single, larger mixture, which we call "stacked posterior". It then re-optimizes the weights of this combined mixture to maximize the combined Evidence Lower BOund (ELBO, a lower bound on log [model evidence](https://en.wikipedia.org/wiki/Marginal_likelihood)). 

A key advantage of S-VBMC is its efficiency: **the original model is never re-evaluated**, making it an inexpensive way to boost inference performance. Furthermore, **no communication is needed among VBMC runs**, making it possible to run them in parallel before applying S-VBMC as a post-processing step with negligible computational overhead.

### When to use S-VBMC

S-VBMC works as a post-processing step for VBMC, so it shares its use cases (described [here](https://github.com/acerbilab/pyvbmc/tree/main?tab=readme-ov-file#when-should-i-use-pyvbmc)).

Performing several VBMC inference runs with different initialization points [is already recommended by the developers](https://github.com/acerbilab/pyvbmc/blob/main/examples/pyvbmc_example_4_validation.ipynb) for robustness and convergence diagnostics; therefore, S-VBMC naturally fits into VBMC's best practices. Because S-VBMC is inexpensive and effective, we recommend using it whenever you first perform inference with VBMC. It is especially useful when separate VBMC runs yield noticeably different variational posteriors, which might happen when the target distribution has a particularly complex shape (see the notebook `examples.ipynb` for two examples of this).

### Repository layout

This repository is organized as follows:

  - `svbmc.py` contains the `SVBMC` class, which runs the main algorithm.
  - `examples.ipynb` is a notebook showing how to use the `SVBMC` class (and, optionally, VBMC) on our two synthetic examples (multimodal target and ring target).
  - `targets.py` contains the `GMM` and `Ring` classes, our synthetic targets that are used in the `examples.ipynb` notebook.
  - `utils.py` contains handy functions that are used in the `examples.ipynb` notebook.
  - `vbmc_runs` is a folder containing 10 VBMC outputs from our multimodal target (in the `vbmc_runs/GMM` sub-folder) and 10 from our ring-shaped target (in the `vbmc_runs/Ring` sub-folder). These are `.pkl` files.
  - `requirements.txt` contains all the dependencies necessary to run the algorithm.

-----

## How to use S-VBMC

### 1. Installation

After creating and activating a virtual environment, install all required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Running S-VBMC

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

# Initialize the SVBMC object and optimize the weights
vp_stacked = SVBMC(vp_list=vp_list)
vp_stacked.optimize()

# The SVBMC object now contains the optimized weights and ELBO estimates
print(f"Stacked ELBO: {vp_stacked.elbo['estimated']}")
```

For a detailed walkthrough, see the `examples.ipynb` notebook, which optionally includes a minimal guide on how to run VBMC multiple times.

**Note**: For compatibility with VBMC, this implementation of S-VBMC stores results in `NumPy` arrays. However, it uses `PyTorch` under the hood to run the ELBO optimization.


## ⚠️ Important: how to use the final posterior

You must use samples from the stacked posterior for any application and should **not** interpret its individual components' sufficient statistics (means and covariance matrices).

This is because each VBMC run may use different internal parameter transformations. Consequently, the component means and covariance matrices from different VBMC posteriors exist in **incompatible parameter spaces**. Combining them creates a mixture whose individual Gaussian components are not directly meaningful.

**Always use samples from the final stacked posterior**, which are correctly transformed back into the original parameter space. These are available via the `.sample()` method:

```python
# Draw 10,000 samples from the final, stacked posterior
samples = vp_stacked.sample(n_samples=10000)
```




