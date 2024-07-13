import argparse
import os

import h5py
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from pytorchfire import WildfireModel, BaseTrainer


class Fig6Trainer(BaseTrainer):
    def __init__(self, model: WildfireModel, device: torch.device, run_name: str):
        super().__init__(model, device=device)
        self.run_name = run_name

    def my_train(self, ds):

        log_dir = os.path.join('/root/tf-logs', self.run_name)
        writer = SummaryWriter(log_dir=log_dir)
        wind_step_interval = ds.attrs['wind_step_interval']

        self.reset()
        self.model.to(self.device)
        self.model.train()

        max_iterations = self.max_steps // self.steps_update_interval

        postfix = {}
        with tqdm(total=self.max_steps) as progress_bar:
            for epoch in range(self.max_epoch):
                self.model.reset()
                batch_seed = self.model.seed
                running_loss = 0.0
                for iterations in range(max_iterations):
                    batch_max_steps = min(self.max_steps, (iterations + 1) * self.steps_update_interval)

                    for steps in range(batch_max_steps):
                        postfix['steps'] = f'{steps + 1}/{self.max_steps}'

                        if steps % wind_step_interval == 0:
                            self.model.wind_towards_direction = torch.tensor(
                                ds['wind_towards_direction'][steps // wind_step_interval], device=self.device)
                            self.model.wind_velocity = torch.tensor(ds['wind_velocity'][steps // wind_step_interval],
                                                                    device=self.device)

                        self.model.compute(attach=self.check_if_attach(batch_max_steps, steps))

                        postfix['burning'] = self.model.state[0].sum().detach().cpu().item()
                        postfix['burned'] = self.model.state[1].sum().detach().cpu().item()

                        progress_bar.set_postfix(postfix)
                        progress_bar.update(1)

                    outputs = self.model.accumulator
                    targets = self.model.accumulator  # replace your target here

                    loss = self.criterion(outputs, targets)
                    running_loss += loss.item()

                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    with torch.no_grad():
                        self.model.a.clamp_(min=0.0, max=1.0)
                        self.model.c_1.clamp_(min=0.0, max=1.0)
                        self.model.c_2.clamp_(min=0.0, max=1.0)
                        self.model.p_h.clamp_(min=0.2, max=1.0)

                    self.model.reset(batch_seed)

    def my_evaluate(self, ds):
        # self.model.a.data = torch.tensor(ds.attrs['a'])
        # self.model.c_1.data = torch.tensor(ds.attrs['c_1'])
        # self.model.c_2.data = torch.tensor(ds.attrs['c_2'])
        wind_step_interval = ds.attrs['wind_step_interval']

        self.reset()
        self.model.to(self.device)
        self.model.eval()

        postfix = {}
        output_list = []
        with tqdm(total=self.max_steps) as progress_bar:
            with torch.no_grad():
                for steps in range(self.max_steps):
                    postfix['steps'] = f'{steps + 1}/{self.max_steps}'

                    if steps % wind_step_interval == 0:
                        self.model.wind_towards_direction = torch.tensor(
                            ds['wind_towards_direction'][steps // wind_step_interval], device=self.device)
                        self.model.wind_velocity = torch.tensor(ds['wind_velocity'][steps // wind_step_interval],
                                                                device=self.device)

                    self.model.compute()

                    postfix['burning'] = self.model.state[0].sum().detach().cpu().item()
                    postfix['burned'] = self.model.state[1].sum().detach().cpu().item()

                    output_list.append((self.model.state[0] | self.model.state[1]).cpu().detach().numpy())

                    progress_bar.set_postfix(postfix)
                    progress_bar.update(1)

        with h5py.File(f'{self.run_name}.hdf5', 'w') as f_out:
            f_out.attrs['seed'] = self.model.seed
            f_out.create_dataset('output', data=np.stack(output_list), compression='gzip')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=int, required=True, default=0)
    parser.add_argument('--p_h', type=float, required=False, default=0.8)
    parser.add_argument('--max_epoch', type=int, required=False, default=10)
    parser.add_argument('--device', type=str, required=False, default='cuda:0')
    parser.add_argument('--run_name', type=str, required=False, default='default')
    parser.add_argument('--steps_update_interval', type=int, required=False, default=10)
    parser.add_argument('--mode', type=str, required=False, default='train')

    args = parser.parse_args()

    with h5py.File('fig6_targets.hdf5', 'r') as f_in:
        f_in_ds = f_in[str(args.exp_id)]

        trainer = Fig6Trainer(model=WildfireModel({
            'p_veg': torch.tensor(f_in_ds['p_veg'][:]),
            'p_den': torch.tensor(f_in_ds['p_den'][:]),
            'wind_towards_direction': torch.tensor(f_in_ds['wind_towards_direction'][:][0]),
            'wind_velocity': torch.tensor(f_in_ds['wind_velocity'][:][0]),
            'slope': torch.tensor(f_in_ds['slope'][:]),
            'initial_ignition': torch.tensor(f_in_ds['initial_ignition'][:], dtype=torch.bool)
        }, {
            'a': torch.tensor(0.),
            'p_h': torch.tensor(args.p_h),
            'p_continue': torch.tensor(f_in_ds.attrs['p_continue']),
            'c_1': torch.tensor(0.),
            'c_2': torch.tensor(0.),
        }), device=torch.device(args.device), run_name=args.run_name)
        trainer.max_epoch = args.max_epoch
        trainer.steps_update_interval = args.steps_update_interval
        trainer.max_steps = f_in_ds.attrs['max_steps']

        if args.mode == 'train':
            trainer.my_train(ds=f_in_ds)
        elif args.mode == 'predict':
            trainer.my_evaluate(ds=f_in_ds)
