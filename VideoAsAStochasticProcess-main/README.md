##  Variational Inference for SDEs Driven by Fractional Noise

### Installation

The installation and code was tested for Python 3.10.

If you need CUDA support, first [install `jax==0.4.19`](https://github.com/google/jax#installation) with CUDA backend.

Install this package with
```
pip install -e .
```

Lastly, for the video experiments we use the PyTorch dataloader. We advise to install the CPU version of PyTorch, because the GPU install causes version conflicts with JAX.
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Experiments

#### Time dependent Hurst
Run `notebooks/mbm.ipynb`.

#### Bridge
```
python experiments/bridge/main.py
```

#### Latent Video Model

The video model code depends on [Weights & Biases](https://wandb.ai/), if you are logged in the run statistics and trained model parameters will be logged to a project called `jax-mmnist`.
This can be disabled by setting `WANDB_MODE=disabled`.

BM:
```
python sde/train.py --dataset=mmnist --num_latents=6 --int_sub_steps=3 --gamma_max=10. --white=true
```

fBM:
```
python sde/train.py --dataset=mmnist --num_latents=6 --int_sub_steps=3 --gamma_max=10.
```

Use `notebooks/model.ipynb` to load a trained model from Weights & Biases and generate insights or validation scores.


### Highlights

Some parts of the code that might be of particular interest:
 - Implementation of our method in Diffrax: `sde/markov_approximation.py / solve_diffrax()`
 - Simple implementation of a Euler solver for our method: `sde/markov_approximation.py / solve_vector()`
 - Implementation of our SDE model driven by MA-fBM: `sde/models / FractionalSDE()`
 - Implementation of our latent SDE video model driven by MA-fBM: `sde/models / VideoSDE()`
 - Implementation of optimized omega values: `sde/markov_approximation.py / omega_optimized_1(), omega_optimized_2()` (type 1 and 2 respectively)
 - Numerically stable implementation of Q(z,x)e^x: `sde/markov_approximation.py / gammaincc_ez()`