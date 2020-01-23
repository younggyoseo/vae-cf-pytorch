# autencoders_cf

Experiments with auto-encoders for collaborative filtering.

This repo is based on a fork of [`vae-cf-pytorch`](https://github.com/belepi93/vae-cf-pytorch) by `belepi93`, which is based on [`vae_cf`](https://github.com/dawenl/vae_cf) by `dawenl`, author of [1].

## Models

Includes implementations of the variational and denoising auto-encoders for collaborative filtering by Dawen Liang et al. [1]:
- `models.pytorch.MultVAE`
- `models.pytorch.MultDAE`

...EASE^R, the "Embarassingly Shallow Auto-Encoder"  with closed-form solution by Harald Steck [2] (a variant of SLIM [3]):
- in its efficient form (`SLIM.closed_form_slim`)
- in PyTorch (`models.pytorch.SAE`)

and two new (sparse, full-rank, multi-layer) auto-encoders, implemented in TensorFlow:
  - `models.tf.WAE`: performs high-dim regression, squared-error loss
  - `models.tf.MultWAE`: a multinomial formulation, cross-entropy loss

## Requirements

```
Python 3.6
numpy
tensorflow==1.5, tensorboard   # for TF models
pytorch==0.4, tensorboardX   # for PyTorch models
```

## References

[1] *Variational Autoencoders for Collaborative Filtering.* Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman and Tony Jebara, WWW 2018
https://arxiv.org/abs/1802.05814

[2] *Embarrassingly shallow auto-encoders.* Harald Steck, WWW 2019
https://arxiv.org/pdf/1905.03375.pdf

[3] *SLIM: Sparse Linear Methods for Top-N Recommender Systems.* Xia Ning and George Karypis, ICDM 2011
http://glaros.dtc.umn.edu/gkhome/node/774
