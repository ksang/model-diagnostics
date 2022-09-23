# Neural Network Diagnostics
Debugging tools for pytorch neural networks

<figure>
<img src="imgs/cat_gradients.jpg" alt="Trulli" style="width:100%">
<figcaption align = "center"><b>Fig.1 - CNN diagnostics using input gradients, from left to right: origin input; normalized input; positive gradients; negative gradients.</b></figcaption>
</figure>

### Demos

- [ConvNets](notebooks/demo_cnn.ipynb)

### Environment setup

- [Anaconda](https://www.anaconda.com/), for python package management.

```
    conda env create -f environment.yaml
    conda activate nn-diagnostics
```

- [direnv](https://direnv.net/), optional, used for automatic setup environment when changing directory.

### Contribute

- Install pre-commit hooks

```
    pre-commit install
```
