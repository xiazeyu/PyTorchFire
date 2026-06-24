"""Run PyTorchFire on a real fire downloaded with FireDataForge.

FireDataForge (https://github.com/xiazeyu/FireDataForge) turns one MTBS fire id
into a folder of harmonized raster layers. This script loads such an event
directory, hands its terrain / fuel / wind / ignition layers to a WildfireModel
via ``pytorchfire.firedataforge``, runs the simulation forward, and scores the
simulated burn against the observed final perimeter.

Run it:

    python firedataforge_simulate.py /path/to/FireDataForge/output/CA3432611848120191010

The reader only needs numpy + torch (already PyTorchFire dependencies); the
heavyweight ``firedataforge`` package is optional and only needed to *generate*
new events.
"""

import argparse
from datetime import timedelta

import torch

from pytorchfire import load_event
from pytorchfire.utils import jaccard_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event_dir",
                        help="FireDataForge event directory (output/<event_id>)")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    # Reasonable starting parameters; calibrate them with the companion script.
    parser.add_argument("--a", type=float, default=0.1)
    parser.add_argument("--p_h", type=float, default=0.4)
    parser.add_argument("--c_1", type=float, default=0.05)
    parser.add_argument("--c_2", type=float, default=0.05)
    parser.add_argument("--plot", action="store_true",
                        help="show ignition / simulated / observed maps (needs matplotlib)")
    args = parser.parse_args()

    device = torch.device(args.device)

    # 1. Load the event and inspect what FireDataForge produced.
    event = load_event(args.event_dir)
    print(event)
    print("available layers:", ", ".join(event.available_layers()))
    print(f"grid {event.shape} @ {event.resolution:g} m, {event.n_wind_frames} hourly wind frames")

    # 2. Build a WildfireModel straight from the harmonized layers.
    params = {
        "a": torch.tensor(args.a),
        "p_h": torch.tensor(args.p_h),
        "c_1": torch.tensor(args.c_1),
        "c_2": torch.tensor(args.c_2),
    }
    model = event.build_model(params=params, device=device)
    torch.manual_seed(args.seed)
    model.reset(seed=args.seed)

    # 3. Step the fire forward, advancing the real (hourly, possibly gappy) wind
    # with a datetime cursor: map the whole observed window onto our step budget,
    # then let EventClock swap in each new wind frame only once its timestamp is
    # reached -- gaps in the HRRR series just hold the last frame.
    span = (event.t_end - event.t_start) if (event.t_start and event.t_end) else None
    dt = span / args.steps if span else timedelta(hours=1)
    clock = event.clock(dt)
    for step in range(args.steps):
        clock.apply_wind(model)   # seeds step 0, then no-op until a new frame is due
        model.compute()
        clock.tick()

    # 4. Compare the simulated footprint with the observed final perimeter.
    simulated = (model.state[0] | model.state[1]).cpu()
    observed = event.target()
    print(f"\nsimulated burned cells : {int(simulated.sum())}")
    print(f"observed  burned cells : {int(observed.sum())}")
    print(f"Jaccard index          : {jaccard_index(observed, simulated):.4f}")
    print("(uncalibrated parameters -- run firedataforge_calibration.py to fit them)")

    if args.plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        for ax, img, title in zip(
                axes,
                [event.initial_ignition().cpu(), simulated, observed],
                ["initial ignition", f"simulated (step {args.steps})", "observed final"]):
            ax.imshow(img, cmap="inferno", interpolation="nearest")
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"PyTorchFire on FireDataForge event {event.name}")
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
