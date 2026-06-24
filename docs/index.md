[![Paper doi](https://img.shields.io/badge/Paper%20DOI-10.1016%2Fj.envsoft.2025.106401-blue)](https://doi.org/10.1016/j.envsoft.2025.106401)
[![Dataset DOI](https://img.shields.io/badge/Dataset_DOI-10.17632%2Fnx2wsksp9k.1-blue)](https://doi.org/10.17632/nx2wsksp9k.1)

# Home

PyTorchFire: A GPU-Accelerated Wildfire Simulator with Differentiable Cellular Automata

## About The Project

Accurate and rapid prediction of wildfire trends is crucial for effective management and mitigation. However, the stochastic nature of fire propagation poses significant challenges in developing reliable simulators. In this paper, we introduce PyTorchFire, an open-access, PyTorch-based software that leverages GPU acceleration. With our redesigned differentiable wildfire Cellular Automata (CA) model, we achieve millisecond-level computational efficiency, significantly outperforming traditional CPU-based wildfire simulators on real-world-scale fires at high resolution. Real-time parameter calibration is made possible through gradient descent on our model, aligning simulations closely with observed wildfire behavior both temporally and spatially, thereby enhancing the realism of the simulations. Our PyTorchFire simulator, combined with real-world environmental data, demonstrates superior generalizability compared to supervised learning surrogate models. Its ability to predict and calibrate wildfire behavior in real-time ensures accuracy, stability, and efficiency. PyTorchFire has the potential to revolutionize wildfire simulation, serving as a powerful tool for wildfire prediction and management.

## Getting Started
### Notebook Examples

- Wildfire Prediction: [examples/prediction.ipynb](https://github.com/xiazeyu/PyTorchFire/blob/main/examples/prediction.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xiazeyu/PyTorchFire/blob/main/examples/prediction.ipynb)
- Parameter Calibration: [examples/calibration.ipynb](https://github.com/xiazeyu/PyTorchFire/blob/main/examples/calibration.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xiazeyu/PyTorchFire/blob/main/examples/calibration.ipynb)

### Installation

Install with minimal dependencies:

```shell
pip install pytorchfire
```

Install with dependencies for examples:

```shell
pip install 'pytorchfire[examples]'
```

Install together with the [FireDataForge](https://github.com/xiazeyu/FireDataForge) data pipeline:

```shell
pip install 'pytorchfire[firedataforge]'
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

To run on a real fire downloaded with [FireDataForge](https://github.com/xiazeyu/FireDataForge):

```python
from pytorchfire import load_event

event = load_event('output/CA3432611848120191010') # A FireDataForge event directory
model = event.build_model() # WildfireModel seeded with the real fire's layers
for _ in range(100):
    model.compute()

target = event.target() # Observed final perimeter, ready as a calibration target
```

See the [FireDataForge reference](reference/firedataforge.md) and the runnable
[`examples/firedataforge_simulate.py`](https://github.com/xiazeyu/PyTorchFire/blob/main/examples/firedataforge_simulate.py)
/ [`examples/firedataforge_calibration.py`](https://github.com/xiazeyu/PyTorchFire/blob/main/examples/firedataforge_calibration.py).

## Dataset

See at Our [Dataset](https://doi.org/10.17632/nx2wsksp9k.1).

## Reference

```bibtex
@article{xia2025pytorchfire,
 author = {Zeyu Xia and Sibo Cheng},
 copyright = {CC BY 4.0},
 doi = {10.1016/j.envsoft.2025.106401},
 issn = {1364-8152},
 journal = {Environmental Modelling & Software},
 keywords = {Wildfire simulation, Differentiable Cellular Automata, PyTorch-based software, Parallel computing techniques, GPU-acceleration},
 language = {English},
 month = {4},
 pages = {106401},
 title = {PyTorchFire: A GPU-accelerated wildfire simulator with Differentiable Cellular Automata},
 url = {https://www.sciencedirect.com/science/article/pii/S1364815225000854},
 volume = {188},
 year = {2025}
}
```

## Contact

Zeyu Xia - [zeyu.xia@virginia.edu](mailto:zeyu.xia@virginia.edu)

Sibo Cheng - [sibo.cheng@enpc.fr](mailto:sibo.cheng@enpc.fr)
