# Manifold Sculpting

## Overview

An implementation of Manifold Sculpting algorithm as introduced in
[Iterative Non-linear Dimensionality Reduction by Manifold Sculpting (Gashler et al., 2007)](https://proceedings.neurips.cc/paper/2007/file/c06d06da9666a219db15cf575aff2824-Paper.pdf).

Manifold Sculpting is a nonlinear dimensionality reduction technique that preserves distances and angles between neighbor points, by using the loss

$$
L(p_i) = \sum_{j = 1}^{k} \omega_i \left(\left(\frac{d_{ij}^0 - d_{ij}}{2 d_{avg}}\right)^2 + \left(\frac{\theta_{ij}^0 - \theta_{ij}}{\pi}\right)^2\right)
$$

![Image not found](figs/markdown/MCN.png)

The algorithm is:

1. Compute relationship between neighbors
2. Optionally perform PCA
3. Until the stopping criterion is met
    1. Scale up dimensions to preserve by σ
    2. Scale down dimensions to scale by σ
    3. Adjust preserved dimensions by shifting the points
    4. Drop dimensions to scale

## Usage

The Setup is made using Docker. To create the container and install the needed python packages use:

```bash
docker build -t mycontainer .
```

to create a dataset, run the manifold sculpting algorithm and create a GIF of the process, use:

```bash
docker run --rm -v $(pwd)/data/:/app/data/ mycontainer
```

## Results

To validate the dataset and test its efficiency againts the theorethical result and against other methods.

### Pro

- Accurate even with a small density of points (250 in the picture) ![no pic found](figs/markdown/proj_250.png)
- Number of neighbors is not that influential on the result ![no pic found](figs/markdown/mse_vs_nn.png)

### Cons

- Slow ![no pic found](figs/markdown/time_vs_nn.png)
- Many hyperparameters to choose
- Must be rotated to align with the XY plane

The full process can be appreciated in the following gif: ![no gif found](figs/markdown/trial_1.gif).
