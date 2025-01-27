![workflow-test-ci](https://github.com/thorben-frank/euclidean_fast_attention/actions/workflows/CI.yml/badge.svg)
[![preprint-link](https://img.shields.io/badge/paper-arxiv.org-B31B1B)](https://arxiv.org/abs/2412.08541)

![Logo](overview.png)

### Euclidean Fast Attention
Reference implementation of the Euclidean fast attention (EFA) algorithm, presented in the paper 
[*Euclidean Fast Attention: Machine Learning Global Atomic Representations at Linear Cost*](https://arxiv.org/abs/2412.08541).
#### Installation
The code in this repository can be either used with CPU or with GPU. If you want to use GPU, you have to install the 
corresponding JAX installation via 
```shell script
# On GPU
pip install --upgrade pip
pip install "jax[cuda12]"
```
If you want to run the code on CPU, e.g. for testing on your local machine which does not have a GPU, you can do
```shell script
# On CPU
pip install --upgrade pip
pip install jax
```
Note, that the code will run much fast on GPU than on CPU, so training is ideally performed on a GPU. More 
details about JAX installation can be found [here](https://jax.readthedocs.io/en/latest/installation.html).

Afterwards, you clone the EFA repository and install via
```shell script
# Euclidean fast attention installation
git clone https://github.com/thorben-frank/euclidean_fast_attention.git
cd euclidean_fast_attention
pip install .
```

#### Examples
For example usages check the `examples/` folder. It contains an examples for basic usage of the `EuclideanFastAttention` 
`flax` module. Additionally, you can find examples on how to train an O(3) equivariant MPNN with enabled / disabled 
EFA block to reproduce the results from the paper. 

#### Citation
If you find this repository useful or use the Euclidean fast attention algorithm in your research please
consider citing the corresponding paper
```
@article{frank2024euclidean,
  title={Euclidean Fast Attention: Machine Learning Global Atomic Representations at Linear Cost},
  author={Frank, J Thorben and Chmiela, Stefan and M{\"u}ller, Klaus-Robert and Unke, Oliver T},
  journal={arXiv preprint arXiv:2412.08541},
  year={2024}
}
```


