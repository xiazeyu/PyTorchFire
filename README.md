# PyTorchFire: A GPU-Accelerated Wildfire Simulator with Differentiable Cellular Automata

[![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)
[![PyPI - Version](https://img.shields.io/pypi/v/pytorchfire)](https://pypi.org/project/pytorchfire/)
[![Read the Docs](https://readthedocs.org/projects/pytorchfire/badge/)](https://pytorchfire.readthedocs.io/)
[![Code DOI](https://img.shields.io/badge/Code_DOI-10.5281%2Fzenodo.13132218-blue)](https://doi.org/10.5281/zenodo.13132218)
[![Dataset DOI](https://img.shields.io/badge/Dataset_DOI-10.17632%2Fnx2wsksp9k.1-blue)](https://doi.org/10.17632/nx2wsksp9k.1)

### Jupyter Notebook Examples

- Wildfire Prediction: [examples/prediction.ipynb](examples/prediction.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xiazeyu/PyTorchFire/blob/main/examples/prediction.ipynb)

- Parameter Calibration: [examples/calibration.ipynb](examples/calibration.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xiazeyu/PyTorchFire/blob/main/examples/calibration.ipynb)

### Installation

Install with minimal dependencies:

```shell
pip install pytorchfire
```

Install with dependencies for examples:

```shell
pip install 'pytorchfire[examples]'
```

### Quick Start

To perform wildfire prediction:

```python
from pytorchfire import WildfireModel

model = WildfireModel() # Create a model with default parameters and environment data
model = model.cuda() # Move the model to GPU
# model.reset(seed=seed) # Reset the model with a seed
for _ in range(100): # Run the model for 100 steps
    model.compute() # Compute the next state
```

To perform parameter calibration:

```python
import torch
from pytorchfire import WildfireModel, BaseTrainer

model = WildfireModel()

trainer = BaseTrainer(model)

trainer.train()
trainer.evaluate()
```

### API Documents

See at Our [Read the Docs](https://pytorchfire.readthedocs.io/).

