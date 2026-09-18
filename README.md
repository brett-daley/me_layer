# me_layer

This repository is the official code release of our ICML 2026 spotlight paper  ***Accelerating Q-learning through Efficient Value-Sharing across Actions***. This project was done in collaboration with Brett Daley, Martha White, and Marlos C. Machado. This paper's primary algorithmic contribution is to introduce the ***mean-expansion layer***, a simple layer that has no learnable parameters and can be added at the end of a Q-network.

The PyTorch code of the mean-expansion layer is:

```
import torch
import torch.nn as nn
class MeanExpansionLayer(nn.Module):
    def __init__(self, mean_scaling_coefficient):
        super().__init__()
        self.register_buffer('scale', torch.tensor(1 + mean_scaling_coefficient))

    def forward(self, vec):
        mean = vec.mean(dim=-1, keepdim=True)
        residual = vec - mean
        output = self.scale * mean + residual
        return output
```

In pure JAX, it is
```
import jax.numpy as jnp
from jax import jit

# 1. Define the layer as a pure function
@jit
def mean_expansion_layer(vec, mean_scaling_coefficient):
    scale = 1 + mean_scaling_coefficient
    mean = jnp.mean(vec, axis=-1, keepdims=True)
    residual = vec - mean
    return scale * mean + residual
```

### Installation
To install the dependencies, run
```
pip install -e .
```

## Reproducing Results

To reproduce the DQN results, run
```
python scripts/train_dqn_atari_sticky.py --agent <agent> --env <env>  --optimizer <optimizer string> --loss <loss>.
```
The valid agents are `dqn`, `dueling_dqn`, `rdq`, `safe_ib_dqn` (which corresponds to `k=1`), and `ib_dqn`. Valid losses are `sse` (sum of squared errors), `mse` (mean squared error), and `huber`. Valid losses are `adam` and `rmsprop`. For example, 
```
python scripts/train_dqn_atari_sticky.py --agent dqn --env ALE/Breakout-v5  --optimizer adam --loss sse
```

The correct settings for the various results in the paper are:
- DQN: `python scripts/train_dqn_atari_sticky.py --agent dqn --env ALE/Breakout-v5  --optimizer adam --loss sse`
- DQN (RMSprop, Huber): `python scripts/train_dqn_atari_sticky.py --agent dqn --env ALE/Breakout-v5  --optimizer rmsprop --loss huber`.
- IB-DQN (RMSprop, Huber): `python scripts/train_dqn_atari_sticky.py --agent ib_dqn --env ALE/Breakout-v5  --optimizer rmsprop --loss huber`.
- IB-DQN(1): `python scripts/train_dqn_atari_sticky.py --agent safe_ib_dqn --env ALE/Breakout-v5  --optimizer adam --loss sse`
- IB-DQN(k): `python scripts/train_dqn_atari_sticky.py --agent ib_dqn --env ALE/Breakout-v5  --optimizer adam --loss sse`
- Dueling DQN:  `python scripts/train_dqn_atari_sticky.py --agent dueling_dqn --env ALE/Breakout-v5  --optimizer adam --loss sse`
- RDQ: `python scripts/train_dqn_atari_sticky.py --agent rdq --env ALE/Breakout-v5  --optimizer adam --loss mse`

If you do not have a GPU, run the commands above with the additional flag `--gpu -1`.

To reproduce the IQN results, run 
```
python scripts/train_iqn_sticky.py --agent <agent> --env <env>
```
The valid agents are `iqn` and `ib_iqn`. Valid environments are the Arcade Learning Environment `-v5` environments in Gymnasium.
For example,
```
python scripts/train_iqn_sticky.py --agent iqn --env ALE/Breakout-v5
```
If you do not have a GPU, run the command above with the additional flag `--gpu -1`.

To reproduce the tabular results, run
```
python scripts/tabular/ibq_tabular_camera.py
```

### Citations

If you use this repository or the mean-expansion layer, please the following bibtex:

```
@InProceedings{mean_expansion_layer,
  title =    {Accelerating {Q}-learning through Efficient Value-Sharing across Actions},
  author =   {Nagarajan, Prabhat and Daley, Brett and White, Martha and Machado, Marlos C.},
  booktitle =    {Proceedings of the 43th International Conference on Machine Learning},
  year =   {2026},
  series =   {Proceedings of Machine Learning Research},
  month =    {July},
}
```

If you use this repository for your research, please additionally cite
```
@article{JMLR:v22:20-376,
  author  = {Yasuhiro Fujita and Prabhat Nagarajan and Toshiki Kataoka and Takahiro Ishikawa},
  title   = {ChainerRL: A Deep Reinforcement Learning Library},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {77},
  pages   = {1-14},
  url     = {http://jmlr.org/papers/v22/20-376.html}
}
```
This is the PFRL library on which this repository is based.
## Third-Party Code

[`plotting/iqm.py`](plotting/iqm.py) is derived from
[google-research/rliable](https://github.com/google-research/rliable)
(Copyright 2021 The Rliable Authors), which is licensed under the Apache
License, Version 2.0. A copy of that license is included at
[`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt), and the modifications made
to the original code are described in the header of `plotting/iqm.py`. That file
remains under the Apache License 2.0; the rest of this repository is MIT
licensed.

If you use the interquartile mean / stratified bootstrap plots produced by this
repository, please also cite the rliable paper:
```
@article{agarwal2021deep,
  title={Deep Reinforcement Learning at the Edge of the Statistical Precipice},
  author={Agarwal, Rishabh and Schwarzer, Max and Castro, Pablo Samuel
          and Courville, Aaron and Bellemare, Marc G},
  journal={Advances in Neural Information Processing Systems},
  year={2021}
}
```
