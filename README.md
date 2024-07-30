# PyTorchFire: A GPU-Accelerated Wildfire Simulator with Differentiable Cellular Automata

[![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)
[![Read the Docs](https://readthedocs.org/projects/pytorchfire/badge/)](https://pytorchfire.readthedocs.io/)

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

```shell
pip install pytorchfire
```

Then,

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

