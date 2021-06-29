# vae-cf-pytorch

An Implementation of [Variational Autoencoders for Collaborative Filtering](https://arxiv.org/abs/1802.05814) (Liang et al. 2018) in PyTorch.

<img src="https://raw.githubusercontent.com/belepi93/vae-cf-pytorch/master/pics/vae.png" width="500">
<img src="https://raw.githubusercontent.com/belepi93/vae-cf-pytorch/master/pics/result.png" width="500">

This repo gives you an implementation of VAE for Collaborative Filtering in PyTorch. It's model is quite simple but powerful so i made a success reproducing it with PyTorch. Every data preprocessing step and code follows exactly from [Authors' Repo](https://github.com/dawenl/vae_cf).

I implemented MultiDAE too but didn't test it so MultiDAE is excluded from `main.py` but its code is available in `models.py`

# Requirements

```
PyTorch 0.4 & Python 3.6
Numpy
TensorboardX
```

# Examples

`python main.py --cuda` for full training.

# Dataset

You should execute `python data.py` first to download necessary data and preprocess MovieLens-20M dataset.

[ml-20m.zip Download](https://grouplens.org/datasets/movielens/20m/)

<img src="https://raw.githubusercontent.com/belepi93/vae-cf-pytorch/master/pics/data.png" width="500">

# Results

<img src="https://raw.githubusercontent.com/belepi93/vae-cf-pytorch/master/pics/result-experiment.png" width="800">

