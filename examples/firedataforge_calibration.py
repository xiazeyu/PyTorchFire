"""Calibrate PyTorchFire parameters against a real fire from FireDataForge.

This is the differentiable-calibration counterpart to ``firedataforge_simulate.py``.
It takes the *observed* final burn perimeter that FireDataForge ships with every
event as the ground-truth target, then uses PyTorchFire's differentiable
cellular automaton to fit the physical parameters (``a``, ``p_h``, ``c_1``,
``c_2``) by gradient descent so the simulated burn matches the real one.

Run it:

    python firedataforge_calibration.py /path/to/output/CA3432611848120191010 \
        --device cuda:0 --max_epochs 30

Then feed the printed parameters back into ``firedataforge_simulate.py`` to see
the improved fit. CPU works for a quick look; a GPU is recommended for full runs.
"""

import argparse
from datetime import timedelta

import torch
from tqdm import tqdm

from pytorchfire import BaseTrainer, load_event
from pytorchfire.utils import jaccard_index


class FireDataForgeTrainer(BaseTrainer):
    """Calibrate a WildfireModel against one FireDataForge event's perimeter."""

    def __init__(self, model, event, device, steps):
        super().__init__(model, device=device)
        self.event = event
        self.max_steps = steps
        # Map the whole observed window onto our step budget, so a datetime
        # cursor can walk the hourly (and possibly gappy) HRRR wind series as the
        # fire grows -- one EventClock per epoch, rebuilt in fit().
        span = (event.t_end - event.t_start) if (event.t_start and event.t_end) else None
        self.dt = span / steps if span else timedelta(hours=1)

    def fit(self):
        target = self.event.target(device=self.device).float()

        self.reset()
        self.model.to(self.device)
        self.model.train()

        best = {"jaccard": -1.0}
        with tqdm(range(self.max_epochs), desc="calibrating") as bar:
            for epoch in bar:
                self.model.reset()
                epoch_seed = self.model.seed

                clock = self.event.clock(self.dt)
                for step in range(self.max_steps):
                    clock.apply_wind(self.model)  # swaps frames only when one is due
                    self.model.compute(attach=self.check_if_attach(step, self.max_steps))
                    clock.tick()

                # accumulator is the differentiable soft burn map; fit it to the
                # observed final perimeter.
                loss = self.criterion(self.model.accumulator, target)

                affected = self.model.state[0] | self.model.state[1]
                jac = jaccard_index(target.bool(), affected)

                self.backward(loss)

                params = {k: round(getattr(self.model, k).item(), 4)
                          for k in ("a", "p_h", "c_1", "c_2")}
                if jac > best["jaccard"]:
                    best = {"jaccard": jac, "seed": epoch_seed, **params}
                bar.set_postfix(loss=round(loss.item(), 4), jaccard=round(jac, 4), **params)

                self.model.reset(seed=epoch_seed)

        return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event_dir",
                        help="FireDataForge event directory (output/<event_id>)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--p_h", type=float, default=0.4)
    args = parser.parse_args()

    device = torch.device(args.device)
    event = load_event(args.event_dir)
    print(event)

    model = event.build_model(params={"p_h": torch.tensor(args.p_h)},
                              keep_acc_mask=False, device=device)

    trainer = FireDataForgeTrainer(model, event, device=device, steps=args.steps)
    trainer.max_epochs = args.max_epochs
    trainer.lr = args.lr

    best = trainer.fit()
    print("\nbest calibration:")
    for key, value in best.items():
        print(f"  {key:8s}: {value}")
    print("\nreplay it with:")
    print(f"  python firedataforge_simulate.py {args.event_dir} "
          f"--a {best['a']} --p_h {best['p_h']} --c_1 {best['c_1']} "
          f"--c_2 {best['c_2']} --seed {best['seed']} --device {args.device}")


if __name__ == "__main__":
    main()
