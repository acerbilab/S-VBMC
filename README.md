# Stacking Variational Bayesian Monte Carlo (S-VBMC)

## Overview
Stacking Variational Bayesian Monte Carlo (S-VBMC)[[1](#references-and-citation)] is a fast post-processing step for [Variational Bayesian Monte Carlo (VBMC)](https://github.com/acerbilab/pyvbmc). VBMC is an approximate Bayesian inference technique that produces a variational posterior in the form of a Gaussian mixture (see the relevant papers [[2-4](#references-and-citation)] for more details). S-VBMC improves upon this by combining ("stacking") the Gaussian mixture components from several independent VBMC runs into a single, larger mixture, which we call "stacked posterior". It then re-optimizes the weights of this combined mixture to maximize the combined Evidence Lower BOund (ELBO, a lower bound on log [model evidence](https://en.wikipedia.org/wiki/Marginal_likelihood)). 

A key advantage of S-VBMC is its efficiency: **the original model is never re-evaluated**, making it an inexpensive way to boost inference performance. Furthermore, **no communication is needed among VBMC runs**, making it possible to run them in parallel before applying S-VBMC as a post-processing step with negligible computational overhead.

Refer to the S-VBMC paper for further details [[1](#references-and-citation)].

## When to use S-VBMC

S-VBMC works as a post-processing step for VBMC, so it shares its use cases (described [here](https://github.com/acerbilab/pyvbmc/tree/main?tab=readme-ov-file#when-should-i-use-pyvbmc)).

Performing several VBMC inference runs with different initialization points [is already recommended by the developers](https://github.com/acerbilab/pyvbmc/blob/main/examples/pyvbmc_example_4_validation.ipynb) for robustness and convergence diagnostics; therefore, S-VBMC naturally fits into VBMC's best practices. Because S-VBMC is inexpensive and effective, we recommend using it whenever you first perform inference with VBMC. It is especially useful when separate VBMC runs yield noticeably different variational posteriors, which might happen when the target distribution has a particularly complex shape (see [this notebook](https://github.com/acerbilab/S-VBMC/blob/main/examples/svbmc_example_1_basic_usage.ipynb) for two examples of this).

-----

## How to use S-VBMC

### 1. Installation

Create a new environment in `conda` and activate it:
   ```bash
   conda create -n svbmc-env python=3.11
   conda activate svbmc-env
   ```
Install `svbmc` with `pip`:
   ```bash
   pip install svbmc 
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
from svbmc.svbmc import SVBMC

# Initialize the SVBMC object and optimize the weights
vp_stacked = SVBMC(vp_list=vp_list)
vp_stacked.optimize()

# The SVBMC object now contains the optimized weights and ELBO estimates
print(f"Stacked ELBO: {vp_stacked.elbo['estimated']}")
```

For a detailed walkthrough, see [this notebook](https://github.com/acerbilab/S-VBMC/blob/main/examples/svbmc_example_1_basic_usage.ipynb), which optionally includes a minimal guide on how to run VBMC multiple times. Additionally, [this notebook](https://github.com/acerbilab/S-VBMC/blob/main/examples/svbmc_example_2_noisy_likelihoods.ipynb) addresses scenarios where the target log-density evaluations are noisy.

**Note**: For compatibility with VBMC, this implementation of S-VBMC stores results in `NumPy` arrays. However, it uses `PyTorch` under the hood to run the ELBO optimization.

-----

## ⚠️ Important: how to use the final posterior

You must use samples from the stacked posterior for any application and should **not** interpret its individual components' sufficient statistics (means and covariance matrices).

This is because each VBMC run may use different internal parameter transformations. Consequently, the component means and covariance matrices from different VBMC posteriors exist in **incompatible parameter spaces**. Combining them creates a mixture whose individual Gaussian components are not directly meaningful.

**Always use samples from the final stacked posterior**, which are correctly transformed back into the original parameter space. These are available via the `.sample()` method:

```python
# Draw 10,000 samples from the final, stacked posterior
samples = vp_stacked.sample(n_samples=10000)
```

## References and citation

1. Silvestrin, F., Li, C., & Acerbi, L. (2025). Stacking Variational Bayesian Monte Carlo. arXiv preprint arXiv:2504.05004. ([paper on arXiv](https://arxiv.org/abs/2504.05004))
2. Acerbi, L. (2018). Variational Bayesian Monte Carlo. In *Advances in Neural Information Processing Systems 31*: 8222-8232. ([paper + supplement on arXiv](https://arxiv.org/abs/1810.05558), [NeurIPS Proceedings](https://papers.nips.cc/paper/8043-variational-bayesian-monte-carlo))
3. Acerbi, L. (2020). Variational Bayesian Monte Carlo with Noisy Likelihoods. In *Advances in Neural Information Processing Systems 33*: 8211-8222 ([paper + supplement on arXiv](https://arxiv.org/abs/2006.08655), [NeurIPS Proceedings](https://papers.nips.cc/paper/2020/hash/5d40954183d62a82257835477ccad3d2-Abstract.html)).
4. Huggins, B., Li, C., Tobaben, M., Aarnos, M., & Acerbi, L. (2023). [PyVBMC: Efficient Bayesian inference in Python](https://joss.theoj.org/papers/10.21105/joss.05428). *Journal of Open Source Software* 8(86), 5428, https://doi.org/10.21105/joss.05428.

Please cite all four references if you use S-VBMC in your work.

## Additional references

5. Acerbi, L. (2019). An Exploration of Acquisition and Mean Functions in Variational Bayesian Monte Carlo. In *Proc. Machine Learning Research* 96: 1-10. 1st Symposium on Advances in Approximate Bayesian Inference, Montréal, Canada. ([paper in PMLR](http://proceedings.mlr.press/v96/acerbi19a.html))

## BibTeX

```BibTeX
@article{silvestrin2025stacking,
  title={{S}tacking {V}ariational {B}ayesian {M}onte Carlo},
  author={Silvestrin, Francesco and Li, Chengkun and Acerbi, Luigi},
  journal={arXiv preprint arXiv:2504.05004},
  year={2025}
}

@article{acerbi2018variational,
  title={{V}ariational {B}ayesian {M}onte {C}arlo},
  author={Acerbi, Luigi},
  journal={Advances in Neural Information Processing Systems},
  volume={31},
  pages={8222--8232},
  year={2018}
}

@article{acerbi2020variational,
  title={{V}ariational {B}ayesian {M}onte {C}arlo with noisy likelihoods},
  author={Acerbi, Luigi},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={8211--8222},
  year={2020}
}

@article{huggins2023pyvbmc,
    title = {PyVBMC: Efficient Bayesian inference in Python},
    author = {Bobby Huggins and Chengkun Li and Marlon Tobaben and Mikko J. Aarnos and Luigi Acerbi},
    publisher = {The Open Journal},
    journal = {Journal of Open Source Software},
    url = {https://doi.org/10.21105/joss.05428},
    doi = {10.21105/joss.05428},
    year = {2023},
    volume = {8},
    number = {86},
    pages = {5428}
  }

@article{acerbi2019exploration,
  title={An Exploration of Acquisition and Mean Functions in {V}ariational {B}ayesian {M}onte {C}arlo},
  author={Acerbi, Luigi},
  journal={PMLR},
  volume={96},
  pages={1--10},
  year={2019}
}
```

## License

S-VBMC is released under the terms of the [BSD 3-Clause License](LICENSE.txt).

## Acknowledgments

S-VBMC was developed by [members](https://www.helsinki.fi/en/researchgroups/machine-and-human-intelligence/people) of the [Machine and Human Intelligence Lab](https://www.helsinki.fi/en/researchgroups/machine-and-human-intelligence/) at the University of Helsinki. 
