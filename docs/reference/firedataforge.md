# FireDataForge

Bridge between [FireDataForge](https://github.com/xiazeyu/FireDataForge) event
outputs and PyTorchFire. FireDataForge turns a single MTBS fire id into a
directory of harmonized raster layers (terrain, fuels, weather, observed
perimeters, ...); this module maps those layers onto the inputs
[`WildfireModel`][pytorchfire.WildfireModel] expects.

The reader only depends on `numpy` (already a PyTorchFire dependency), so the
heavyweight `firedataforge` package is **not** required to consume its outputs.
To also install the producer:

```shell
pip install 'pytorchfire[firedataforge]'
```

```python
from pytorchfire import load_event

event = load_event('output/CA3432611848120191010')
model = event.build_model()      # a WildfireModel seeded with the real fire
for _ in range(100):
    model.compute()
```

::: pytorchfire.firedataforge
