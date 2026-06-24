"""Bridge between [FireDataForge](https://github.com/xiazeyu/FireDataForge) outputs and PyTorchFire.

[FireDataForge](https://github.com/xiazeyu/FireDataForge) turns a single MTBS fire ID into a directory of
harmonized, analysis-ready raster layers (terrain, fuels, weather, observed fire
perimeters, ...), all sampled on one common grid and saved as ``<layer>.npy``
files. This module reads such an event directory and maps those layers onto the
inputs [`WildfireModel`][pytorchfire.WildfireModel] expects, so a real fire becomes a ready-to-run (or
ready-to-calibrate) simulation in a couple of lines:

```python
from pytorchfire import WildfireModel
from pytorchfire.firedataforge import load_event

event = load_event('output/CA3432611848120191010')
model = event.build_model()  # a WildfireModel seeded with the real terrain/fuel/wind
for _ in range(100):
    model.compute()
```

The reader depends only on `numpy` (already a PyTorchFire dependency), so the
heavyweight geospatial `firedataforge` package is **not** required to *consume*
its outputs. If `firedataforge` happens to be installed, its typed
``firedataforge.load_numpy`` loader is used; otherwise the same ``.npy`` files
are read directly with `numpy`. Install the producer alongside PyTorchFire with:

```bash
pip install "pytorchfire[firedataforge]"
```

Layer-to-input mapping
----------------------

| `WildfireModel` input    | FireDataForge layer(s)                  | How it is derived                                                              |
|--------------------------|-----------------------------------------|--------------------------------------------------------------------------------|
| `slope`                  | `elevation`                             | [`calculate_slope`][pytorchfire.utils.calculate_slope] at the grid resolution  |
| `p_veg`                  | `canopy_cover` (%)                      | mean-centered, then piecewise-rescaled so ``[min, mean, max] -> [p_veg_low, 0, p_veg_high]`` (defaults ``[-0.4, 0, 0.35]``) |
| `p_den`                  | `canopy_bulk_density`                   | mean-centered, then piecewise-rescaled so ``[min, mean, max] -> [p_den_low, 0, p_den_high]`` (defaults ``[-0.2, 0, 0.4]``) |
| `wind_velocity` / `wind_towards_direction` | `u10`, `v10` (HRRR, coarse) | resampled to the grid, then [`convert_wind_components_to_velocity_and_direction`][pytorchfire.utils.convert_wind_components_to_velocity_and_direction] |
| `initial_ignition`       | `burn_perimeter` (first frame)          | earliest observed perimeter as the ignition source                            |
| calibration target       | `burn_perimeter` (last / all frames)    | cumulative burned area, see [`target`][pytorchfire.firedataforge.FireDataForgeEvent.target] / [`target_series`][pytorchfire.firedataforge.FireDataForgeEvent.target_series] |

Every mapping is overridable, and any layer can be read directly with
[`frame`][pytorchfire.firedataforge.FireDataForgeEvent.frame] /
[`frames`][pytorchfire.firedataforge.FireDataForgeEvent.frames] to build your own.

Advancing time
--------------

Time-varying layers (the hourly HRRR wind, the observed perimeters) are sampled
irregularly and may have **gaps**. Rather than assume a frame per simulation
step, drive the simulation with a datetime cursor: an
[`EventClock`][pytorchfire.firedataforge.EventClock] (from
[`event.clock(dt)`][pytorchfire.firedataforge.FireDataForgeEvent.clock]) advances
a wall-clock counter by ``dt`` per step and swaps in a layer's next frame only
once the counter reaches that frame's timestamp -- so a gap simply holds the
last valid frame until the next real observation's time arrives:

```python
from datetime import timedelta

clock = event.clock(timedelta(hours=1))   # one model step == one real hour
for _ in range(steps):
    clock.apply_wind(model)               # no-op until a new wind frame is due
    model.compute()
    clock.tick()
```
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import torch

from .model import WildfireModel
from .utils import calculate_slope, convert_wind_components_to_velocity_and_direction

__all__ = [
    "FireDataForgeEvent",
    "EventClock",
    "load_event",
]


def _center_scale(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Mean-center ``values``, then piecewise-rescale around that mean.

    Maps the grid mean to ``0``, the maximum to ``high`` and the minimum to
    ``low`` with a linear scale on each side of the mean (so the two ends can
    use different slopes). Used for both ``p_veg`` and ``p_den``: the mean cell
    becomes neutral, denser-than-average fuel speeds spread (``> 0``) and
    sparser-than-average fuel slows it (``< 0``). ``NaN`` no-data is treated as
    the least-flammable end (the minimum); a constant layer maps everywhere to
    ``0``.
    """
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr)
    mn, mx = float(arr[finite].min()), float(arr[finite].max())
    mean = float(arr[finite].mean())
    centered = np.where(finite, arr, mn) - mean
    out = np.zeros_like(centered)
    if mx > mean:  # above-mean fuel -> (0, high]
        out = np.where(centered > 0, centered * (high / (mx - mean)), out)
    if mn < mean:  # below-mean fuel -> [low, 0)
        out = np.where(centered < 0, centered * (low / (mn - mean)), out)
    return out


class _DictLayer:
    """Attribute-style view over a raw FireDataForge layer ``dict``.

    A FireDataForge ``.npy`` file pickles the ``asdict()`` of a ``DataLayer``.
    Wrapping that plain ``dict`` here lets the rest of this module treat layers
    loaded with bare `numpy` and layers loaded through the optional
    `firedataforge` package (which returns typed ``DataLayer`` objects)
    identically -- both expose ``.data``, ``.timestamps``, ``.categories`` ...
    """

    def __init__(self, payload: dict):
        self._payload = payload

    def __getattr__(self, name):
        try:
            return self._payload[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc


def _load_layer(path: str):
    """Load one FireDataForge ``.npy`` layer as a uniform attribute object.

    Prefers `firedataforge.load_numpy` (typed ``DataLayer``) when the package is
    importable, and otherwise falls back to a plain `numpy` load so that
    PyTorchFire can consume FireDataForge outputs without pulling in the
    heavyweight geospatial stack.
    """
    try:  # optional dependency: only used when already installed
        import firedataforge

        return firedataforge.load_numpy(path)
    except Exception:  # ImportError, or any firedataforge-side load issue
        return _DictLayer(np.load(path, allow_pickle=True).item())


def _as_numpy(value) -> np.ndarray:
    return np.asarray(value)


def _resample_to_grid(arr: np.ndarray, shape: tuple, dtype: torch.dtype,
                      device) -> torch.Tensor:
    """Bilinearly resample a 2-D array onto the event grid as a torch tensor.

    Several FireDataForge layers (notably the HRRR ``u10``/``v10``/``r2`` weather
    fields) sit on a coarser grid than the main raster but cover the *same*
    geographic bounds, so a simple resize to ``shape`` lines them up. ``NaN``s
    are zeroed first so they do not bleed across the interpolation.
    """
    tensor = torch.as_tensor(np.nan_to_num(_as_numpy(arr), nan=0.0)).to(
        dtype=dtype, device=device)
    if tuple(tensor.shape) == tuple(shape):
        return tensor
    resized = torch.nn.functional.interpolate(
        tensor[None, None], size=tuple(shape), mode="bilinear",
        align_corners=False)
    return resized[0, 0]


class FireDataForgeEvent:
    """A loaded FireDataForge event directory, adapted for [`WildfireModel`][pytorchfire.WildfireModel].

    Wraps the ``output/<event_id>/`` directory produced by FireDataForge and
    exposes its layers both raw (via [`frame`][pytorchfire.firedataforge.FireDataForgeEvent.frame] /
    [`frames`][pytorchfire.firedataforge.FireDataForgeEvent.frames]) and pre-mapped onto the tensors
    [`WildfireModel`][pytorchfire.WildfireModel] consumes (via [`to_env_data`][pytorchfire.firedataforge.FireDataForgeEvent.to_env_data] /
    [`build_model`][pytorchfire.firedataforge.FireDataForgeEvent.build_model]).

    Attributes:
        event_dir (str): Path to the event directory.
        event_id (str): MTBS event id (e.g. ``"CA3432611848120191010"``).
        name (str): Human-readable incident name (e.g. ``"SADDLERIDGE"``).
        year (int): Year the fire occurred.
        resolution (float): Grid resolution in CRS units (meters).
        bounds (tuple): ``(minx, miny, maxx, maxy)`` in ``crs`` units.
        crs (str): Coordinate reference system identifier (e.g. ``"EPSG:5070"``).
        shape (tuple): Grid shape as ``(height, width)``.
        t_start: Start datetime of the event window.
        t_end: End datetime of the event window.

    Examples:
        ```python
        from pytorchfire.firedataforge import load_event

        event = load_event('output/CA3432611848120191010')
        print(event)                       # FireDataForgeEvent(SADDLERIDGE, ...)
        model = event.build_model()        # ready-to-run WildfireModel
        target = event.target()            # observed final burned area [H, W] bool
        ```
    """

    def __init__(self, event_dir: str):
        """Open a FireDataForge event directory.

        Parameters:
            event_dir (str): Path to an event output directory containing the
                ``<layer>.npy`` files (e.g. ``"output/CA3432611848120191010"``).

        Raises:
            FileNotFoundError: If ``event_dir`` does not exist or has no
                ``task_info.npy`` describing the grid.
        """
        if not os.path.isdir(event_dir):
            raise FileNotFoundError(f"Not a FireDataForge event directory: {event_dir}")
        self.event_dir = event_dir
        self._cache: dict = {}

        info_path = os.path.join(event_dir, "task_info.npy")
        if not os.path.exists(info_path):
            raise FileNotFoundError(
                f"{event_dir} is missing task_info.npy; is it a FireDataForge event?")
        meta = dict(_load_layer(info_path).data[0])

        self.event_id = meta.get("event_id", os.path.basename(event_dir.rstrip("/")))
        self.name = meta.get("name")
        self.year = meta.get("year")
        self.resolution = float(meta.get("resolution", 30))
        self.bounds = tuple(meta["bounds"]) if meta.get("bounds") is not None else None
        self.crs = meta.get("crs")
        self.shape = tuple(meta["shape"])
        self.t_start = meta.get("t_start")
        self.t_end = meta.get("t_end")

    # -- raw layer access ---------------------------------------------------

    def has(self, name: str) -> bool:
        """Return ``True`` if the event produced a ``<name>.npy`` layer."""
        return os.path.exists(os.path.join(self.event_dir, f"{name}.npy"))

    def layer(self, name: str):
        """Return the raw layer object for ``name`` (a ``DataLayer``-like view).

        The returned object exposes ``.data`` (list of frames), ``.timestamps``,
        ``.unit``, ``.categories``, etc. Results are cached per event.

        Raises:
            FileNotFoundError: If the layer was not produced for this event.
        """
        if name not in self._cache:
            path = os.path.join(self.event_dir, f"{name}.npy")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Layer '{name}' not found in {self.event_dir}")
            self._cache[name] = _load_layer(path)
        return self._cache[name]

    def frames(self, name: str) -> list:
        """Return every frame of layer ``name`` as a list of `numpy` arrays."""
        return [_as_numpy(frame) for frame in self.layer(name).data]

    def frame(self, name: str, index: int = 0) -> np.ndarray:
        """Return a single frame of layer ``name`` as a `numpy` array.

        Parameters:
            name (str): Layer name (file stem).
            index (int): Frame index; ``-1`` gives the most recent frame.
        """
        return _as_numpy(self.layer(name).data[index])

    def timestamps(self, name: str) -> Optional[list]:
        """Return the per-frame timestamps of layer ``name`` (or ``None``)."""
        return getattr(self.layer(name), "timestamps", None)

    def available_layers(self) -> list:
        """Return the sorted names of every ``.npy`` layer in the directory."""
        return sorted(
            f[:-4] for f in os.listdir(self.event_dir) if f.endswith(".npy"))

    # -- WildfireModel inputs ----------------------------------------------

    def slope(self, dtype: torch.dtype = torch.float32, device=None) -> torch.Tensor:
        """Terrain slope tensor ``[H, W, 3, 3]`` (degrees) from ``elevation``.

        Falls back to an all-zero (flat) slope when no ``elevation`` layer is
        available.
        """
        if not self.has("elevation"):
            return torch.zeros((*self.shape, 3, 3), dtype=dtype, device=device)
        altitude = torch.as_tensor(
            self.frame("elevation").astype(np.float32), dtype=dtype, device=device)
        cell_size = torch.tensor(self.resolution, dtype=dtype, device=device)
        return calculate_slope(altitude, cell_size)

    def p_veg(self, layer: str = "canopy_cover", low: float = -0.4,
              high: float = 0.35, dtype: torch.dtype = torch.float32,
              device=None) -> torch.Tensor:
        """Vegetation-type factor ``[H, W]`` from the ``canopy_cover`` layer.

        Canopy cover (%) is mean-centered and then piecewise-rescaled so the
        grid **mean** maps to ``0``, the densest cell to ``high`` and the barest
        to ``low`` (see ``_center_scale``).
        ``p_veg`` enters the model as ``(1 + p_veg)``, so cells with
        above-average fuel propagate faster and below-average cells slower,
        relative to the event's own mean.

        Parameters:
            layer (str): Continuous fuel layer to derive ``p_veg`` from.
            low (float): Value the minimum (barest) cell maps to.
            high (float): Value the maximum (densest) cell maps to.
        """
        if not self.has(layer):
            return torch.zeros(self.shape, dtype=dtype, device=device)
        out = _center_scale(self.frame(layer), low, high)
        return torch.as_tensor(out, dtype=dtype, device=device)

    def p_den(self, layer: str = "canopy_bulk_density", low: float = -0.2,
              high: float = 0.4, dtype: torch.dtype = torch.float32,
              device=None) -> torch.Tensor:
        """Vegetation-density factor ``[H, W]`` from a continuous fuel layer.

        The chosen ``layer`` (``canopy_bulk_density`` by default) is mean-centered
        and then piecewise-rescaled so the grid **mean** maps to ``0``, the
        densest cell to ``high`` and the sparsest to ``low`` -- the same scheme
        as [`p_veg`][pytorchfire.firedataforge.FireDataForgeEvent.p_veg] (see
        ``_center_scale``). Other fuel
        layers such as ``lai`` or ``canopy_cover`` can be substituted via
        ``layer``.

        Parameters:
            layer (str): Continuous fuel layer to derive ``p_den`` from.
            low (float): Value the sparsest cell maps to.
            high (float): Value the densest cell maps to.
        """
        if not self.has(layer):
            return torch.zeros(self.shape, dtype=dtype, device=device)
        out = _center_scale(self.frame(layer), low, high)
        return torch.as_tensor(out, dtype=dtype, device=device)

    def initial_ignition(self, layer: str = "burn_perimeter", index: int = 0,
                         device=None) -> torch.Tensor:
        """Boolean ignition mask ``[H, W]`` from the first observed perimeter.

        Uses frame ``index`` of ``burn_perimeter`` by default (the earliest
        observed fire footprint). Falls back to ``fireline`` if present, and
        finally to a small central seed so a simulation can still run.
        """
        source = layer if self.has(layer) else ("fireline" if self.has("fireline") else None)
        if source is None:
            mask = np.zeros(self.shape, dtype=bool)
            h, w = self.shape
            mask[h // 2 - 1:h // 2 + 1, w // 2 - 1:w // 2 + 1] = True
        else:
            mask = _as_numpy(self.frame(source, index)).astype(bool)
        return torch.as_tensor(mask, dtype=torch.bool, device=device)

    def wind_frame_index(self, when, layer: str = "u10") -> int:
        """Index of the wind frame nearest a datetime ``when``.

        Useful for aligning the hourly weather series with an observed perimeter
        timestamp, e.g. ``event.wind(event.wind_frame_index(event.timestamps('burn_perimeter')[3]))``.
        """
        stamps = self.timestamps(layer)
        if not stamps:
            return 0
        deltas = [abs((stamp - when).total_seconds()) for stamp in stamps]
        return int(np.argmin(deltas))

    def frame_index_at(self, layer: str, when, default: int = 0) -> int:
        """Index of the frame of ``layer`` that is *valid* at datetime ``when``.

        Returns the latest frame whose timestamp is at or before ``when`` -- the
        most recent observation the simulation clock has actually reached.
        Between two timestamps the earlier frame stays valid, so this is the
        forward-marching, gap-aware counterpart of
        [`wind_frame_index`][pytorchfire.firedataforge.FireDataForgeEvent.wind_frame_index]
        (which snaps to the nearest frame in *either* direction): as a datetime
        counter advances, the next frame is selected only once ``when`` reaches
        its timestamp, so a gap in an irregularly sampled layer simply holds the
        last frame until the next real observation's time arrives. This is the
        primitive [`EventClock`][pytorchfire.firedataforge.EventClock] steps on.

        Layers with no timestamps (static fuels/terrain) always return
        ``default``. ``when`` earlier than the first timestamp also returns
        ``default`` (the first frame).
        """
        stamps = self.timestamps(layer)
        if not stamps:
            return default
        idx = default
        for i, stamp in enumerate(stamps):
            if stamp <= when:
                idx = i
            else:
                break  # timestamps are ascending; nothing later can qualify
        return idx

    def wind(self, frame: int = 0, dtype: torch.dtype = torch.float32,
             device=None) -> dict:
        """Wind velocity & direction on the grid for one weather frame.

        Resamples the coarse HRRR ``u10``/``v10`` fields to the event grid and
        converts them with
        [`convert_wind_components_to_velocity_and_direction`][pytorchfire.utils.convert_wind_components_to_velocity_and_direction].

        Returns:
            A dict with ``wind_velocity`` (m/s) and ``wind_towards_direction``
            (degrees from East, counter-clockwise), each shaped ``[H, W]``.
        """
        if not (self.has("u10") and self.has("v10")):
            zero = torch.zeros(self.shape, dtype=dtype, device=device)
            return {"wind_velocity": zero, "wind_towards_direction": zero.clone()}
        u = _resample_to_grid(self.frame("u10", frame), self.shape, dtype, device)
        v = _resample_to_grid(self.frame("v10", frame), self.shape, dtype, device)
        return convert_wind_components_to_velocity_and_direction(u, v)

    @property
    def n_wind_frames(self) -> int:
        """Number of hourly weather frames (``0`` if no wind layer)."""
        return len(self.layer("u10").data) if self.has("u10") else 0

    # -- targets ------------------------------------------------------------

    def target(self, layer: str = "burn_perimeter", index: int = -1,
               device: Any = None) -> torch.Tensor:
        """Observed burned-area target ``[H, W]`` (bool) for calibration.

        Defaults to the last ``burn_perimeter`` frame -- the cumulative final
        fire footprint.
        """
        mask = _as_numpy(self.frame(layer, index)).astype(bool)
        return torch.as_tensor(mask, dtype=torch.bool, device=device)

    def target_series(self, layer: str = "burn_perimeter",
                      device: Any = None) -> torch.Tensor:
        """Stacked observed perimeters ``[T, H, W]`` (bool) over time.

        Parallel to ``event.timestamps(layer)``; each slice is the cumulative
        burned area observed up to that timestamp.
        """
        frames = np.stack([_as_numpy(f).astype(bool) for f in self.layer(layer).data])
        return torch.as_tensor(frames, dtype=torch.bool, device=device)

    # -- assembled model inputs --------------------------------------------

    def to_env_data(self, wind_frame: int = 0, dtype: torch.dtype = torch.float32,
                    device: Any = None, p_veg_layer: str = "canopy_cover",
                    p_den_layer: str = "canopy_bulk_density") -> dict:
        """Assemble the ``env_data`` dict accepted by [`WildfireModel`][pytorchfire.WildfireModel].

        Parameters:
            wind_frame (int): Which weather frame to freeze into the model's
                wind fields. Update them later for a time-varying simulation
                (see [`apply_wind`][pytorchfire.firedataforge.FireDataForgeEvent.apply_wind] /
                [`clock`][pytorchfire.firedataforge.FireDataForgeEvent.clock]).
            dtype (torch.dtype): Floating dtype for the returned tensors.
            device: Torch device for the tensors. Leave as ``None`` (CPU) when
                you intend to build a model and then ``.to(device)`` it, so the
                model's default scalar parameters stay on the same device.
            p_veg_layer (str): Fuel layer used to derive ``p_veg``.
            p_den_layer (str): Fuel layer used to derive ``p_den``.

        Returns:
            A dict with ``p_veg``, ``p_den``, ``wind_velocity``,
            ``wind_towards_direction``, ``slope`` and ``initial_ignition``.

        Examples:
            ```python
            from pytorchfire import WildfireModel
            from pytorchfire.firedataforge import load_event

            event = load_event('output/CA3432611848120191010')
            model = WildfireModel(env_data=event.to_env_data())
            ```
        """
        env = {
            "p_veg": self.p_veg(layer=p_veg_layer, dtype=dtype, device=device),
            "p_den": self.p_den(layer=p_den_layer, dtype=dtype, device=device),
            "slope": self.slope(dtype=dtype, device=device),
            "initial_ignition": self.initial_ignition(device=device),
        }
        env.update(self.wind(frame=wind_frame, dtype=dtype, device=device))
        return env

    def build_model(self, params: Optional[dict] = None, wind_frame: int = 0,
                    keep_acc_mask: bool = False, device: Any = None,
                    **env_kwargs: Any) -> WildfireModel:
        """Construct a [`WildfireModel`][pytorchfire.WildfireModel] seeded with this event's data.

        The model is assembled on CPU and then moved to ``device`` (if given) so
        the environment buffers and the default scalar parameters end up on the
        same device, satisfying the model's internal sanity checks.

        Parameters:
            params (dict): Optional override of the model's scalar parameters
                (``a``, ``p_h``, ``c_1``, ``c_2``, ``p_continue``).
            wind_frame (int): Weather frame to initialise the wind fields with.
            keep_acc_mask (bool): Forwarded to [`WildfireModel`][pytorchfire.WildfireModel].
            device: Device to move the finished model to.
            **env_kwargs: Forwarded to [`to_env_data`][pytorchfire.firedataforge.FireDataForgeEvent.to_env_data]
                (e.g. ``p_veg_layer``, ``p_den_layer``).

        Returns:
            A ready-to-run [`WildfireModel`][pytorchfire.WildfireModel].
        """
        env = self.to_env_data(wind_frame=wind_frame, device=None, **env_kwargs)
        model = WildfireModel(env_data=env, params=params, keep_acc_mask=keep_acc_mask)
        if device is not None:
            model = model.to(device)
        return model

    def apply_wind(self, model: WildfireModel, frame: int) -> None:
        """Update an existing model's wind fields to a different weather frame.

        Mirrors how the real-fire examples advance wind mid-simulation:

        ```python
        for f in range(event.n_wind_frames):
            event.apply_wind(model, f)
            model.compute()
        ```

        For a wall-clock-driven, gap-aware schedule use
        [`clock`][pytorchfire.firedataforge.FireDataForgeEvent.clock] instead of
        indexing frames by hand.
        """
        wind = self.wind(frame=frame, dtype=model.wind_velocity.dtype,
                         device=model.wind_velocity.device)
        model.wind_velocity = wind["wind_velocity"]
        model.wind_towards_direction = wind["wind_towards_direction"]

    def clock(self, dt: timedelta, start: Optional[datetime] = None,
              layers: Optional[list] = None) -> "EventClock":
        """Create an [`EventClock`][pytorchfire.firedataforge.EventClock] driving this event's time-varying layers.

        Parameters:
            dt: Real-world duration of one simulation step, as a
                ``datetime.timedelta``. Each [`tick`][pytorchfire.firedataforge.EventClock.tick]
                advances the counter by this much.
            start: Datetime the counter starts at. Defaults to the first wind
                timestamp (else ``t_start``), so the cursor begins on the first
                real observation.
            layers (list): Time-varying layers to track. Defaults to whichever
                of the wind layers (``u10``/``v10``) the event produced.

        Returns:
            An [`EventClock`][pytorchfire.firedataforge.EventClock].

        Examples:
            ```python
            from datetime import timedelta

            clock = event.clock(timedelta(hours=1))
            for _ in range(steps):
                clock.apply_wind(model)   # swaps frames only when one is due
                model.compute()
                clock.tick()
            ```
        """
        return EventClock(self, dt, start=start, layers=layers)

    def __repr__(self) -> str:
        return (f"FireDataForgeEvent({self.name}, id={self.event_id}, "
                f"shape={self.shape}, crs={self.crs}, year={self.year})")


class EventClock:
    """A datetime cursor that walks a simulation through an event's real time.

    Created with [`event.clock(dt)`][pytorchfire.firedataforge.FireDataForgeEvent.clock].
    Each [`tick`][pytorchfire.firedataforge.EventClock.tick] advances
    ``now`` by ``dt`` (the real-world
    duration of one simulation step) and recomputes, per tracked time-varying
    layer, the frame that is *currently valid* -- the most recent frame whose
    timestamp ``now`` has reached (via
    [`frame_index_at`][pytorchfire.firedataforge.FireDataForgeEvent.frame_index_at]).

    A frame is replaced only once a new one is actually due, so irregular
    sampling and outright **gaps** in a layer's series need no interpolation: the
    last valid frame simply persists until the next real observation's time
    arrives. This is what the FireDataForge data contract calls exposing each
    source "at its most recent valid observation".

    Attributes:
        event (FireDataForgeEvent): The event being clocked.
        dt: Real-world duration of one simulation step (``datetime.timedelta``).
        now: The current wall-clock datetime of the cursor.
        layers (list): Time-varying layers being tracked.

    Examples:
        ```python
        from datetime import timedelta

        clock = event.clock(timedelta(hours=1))   # one model step == one hour
        for _ in range(steps):
            clock.apply_wind(model)               # no-op until a new frame is due
            model.compute()
            clock.tick()
        ```
    """

    #: Wind layers an EventClock pushes into a model via :meth:`apply_wind`.
    WIND_LAYERS = ("u10", "v10")

    def __init__(self, event: FireDataForgeEvent, dt: timedelta,
                 start: Optional[datetime] = None,
                 layers: Optional[list] = None):
        self.event = event
        self.dt = dt
        if layers is None:
            layers = [name for name in self.WIND_LAYERS if event.has(name)]
        self.layers = list(layers)
        self.now = start if start is not None else self._default_start()
        # Current valid frame index per tracked layer.
        self._index = {name: event.frame_index_at(name, self.now)
                       for name in self.layers}
        # Layers whose frame changed on the latest tick. Seeded with every
        # tracked layer so the first apply_*() call actually pushes a frame.
        self._changed = set(self.layers)

    def _default_start(self):
        for name in self.layers:
            stamps = self.event.timestamps(name)
            if stamps:
                return stamps[0]
        return self.event.t_start

    def index(self, layer: str) -> int:
        """Current valid frame index for ``layer`` (tracking it if new)."""
        if layer not in self._index:
            self._index[layer] = self.event.frame_index_at(layer, self.now)
            self.layers.append(layer)
        return self._index[layer]

    def changed(self, layer: str) -> bool:
        """Whether ``layer``'s valid frame changed on the latest [`tick`][pytorchfire.firedataforge.EventClock.tick]."""
        return layer in self._changed

    def tick(self, dt: Optional[timedelta] = None) -> set:
        """Advance the counter and re-evaluate every tracked layer's frame.

        Parameters:
            dt: Step to advance by; defaults to the clock's ``dt``.

        Returns:
            The set of tracked layer names whose valid frame changed this tick
            (empty when the step stayed inside every layer's current frame --
            the common case for sub-hourly steps or during a data gap).
        """
        self.now = self.now + (self.dt if dt is None else dt)
        self._changed = set()
        for name in self.layers:
            new = self.event.frame_index_at(name, self.now)
            if new != self._index[name]:
                self._index[name] = new
                self._changed.add(name)
        return self._changed

    def apply_wind(self, model: WildfireModel, force: bool = False) -> bool:
        """Push the wind frame valid at ``now`` into ``model`` when one is due.

        Skips the resample/convert work on every step that stays inside the
        current wind frame, updating the model only when a new wind frame has
        been reached since the last apply (or when ``force`` is set, e.g. to
        seed the very first frame).

        Returns:
            ``True`` if the model's wind was updated, ``False`` if nothing was
            due. Does nothing (returns ``False``) when the event has no wind.
        """
        wind = [name for name in self.WIND_LAYERS if name in self._index]
        if not wind:
            return False
        if not (force or any(name in self._changed for name in wind)):
            return False
        self.event.apply_wind(model, self._index[wind[0]])
        self._changed -= set(wind)  # consumed; don't re-apply until next change
        return True

    def __repr__(self) -> str:
        return (f"EventClock(now={self.now}, dt={self.dt}, "
                f"layers={self.layers})")


def load_event(event_dir: str) -> FireDataForgeEvent:
    """Open a FireDataForge event output directory.

    Thin convenience wrapper around [`FireDataForgeEvent`][pytorchfire.firedataforge.FireDataForgeEvent].

    Parameters:
        event_dir (str): Path to an event directory containing ``<layer>.npy``
            files (e.g. ``"output/CA3432611848120191010"``).

    Returns:
        A [`FireDataForgeEvent`][pytorchfire.firedataforge.FireDataForgeEvent] wrapping the directory.

    Examples:
        ```python
        from pytorchfire.firedataforge import load_event

        event = load_event('output/CA3432611848120191010')
        model = event.build_model()
        ```
    """
    return FireDataForgeEvent(event_dir)
