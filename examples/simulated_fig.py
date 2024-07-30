import argparse
import os

import h5py
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pytorchfire import WildfireModel, BaseTrainer


def jaccard_index(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    # Ensure the inputs are boolean
    assert y_true.dtype == y_pred.dtype == torch.bool

    # Calculate intersection and union
    intersection = torch.sum(y_true & y_pred).float()
    union = torch.sum(y_true | y_pred).float()

    # Compute Jaccard Index
    jaccard = intersection / union

    return jaccard.item()


def manhattan_distance(tensor1: torch.Tensor, tensor2: torch.Tensor):
    # Ensure the tensors are of the same shape
    assert tensor1.shape == tensor2.shape

    # Compute the absolute differences
    abs_diff = torch.abs(tensor1 - tensor2)

    # Sum the absolute differences
    manhattan_dist = torch.sum(abs_diff)

    return manhattan_dist.item()


# noinspection DuplicatedCode
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
        with tqdm() as progress_bar:
            for epochs in range(self.max_epochs):
                postfix['epoch'] = f'{epochs + 1}/{self.max_epochs}'
                self.model.reset()
                batch_seed = self.model.seed

                accumulators = []
                accumulator_masks = []
                targets_list = []

                running_loss = []
                running_jaccard = []
                affected_cell_count_outputs = []
                affected_cell_count_targets = []

                for iterations in range(max_iterations):
                    postfix['iteration'] = f'{iterations + 1}/{max_iterations}'
                    iter_max_steps = min(self.max_steps, (iterations + 1) * self.steps_update_interval)
                    progress_bar.reset(total=iter_max_steps)

                    for steps in range(iter_max_steps):
                        postfix['step'] = f'{steps + 1}/{iter_max_steps}'

                        if steps % wind_step_interval == 0:
                            self.model.wind_towards_direction = torch.tensor(
                                ds['wind_towards_direction'][steps // wind_step_interval], device=self.device)
                            self.model.wind_velocity = torch.tensor(ds['wind_velocity'][steps // wind_step_interval],
                                                                    device=self.device)

                        self.model.compute(attach=self.check_if_attach(steps, iter_max_steps))

                        postfix['burning'] = self.model.state[0].sum().detach().cpu().item()
                        postfix['burned'] = self.model.state[1].sum().detach().cpu().item()

                        progress_bar.set_postfix(postfix)
                        progress_bar.update(1)

                    outputs = self.model.accumulator
                    targets = ds['target'][iter_max_steps - 1]
                    targets_list.append(targets)
                    targets = torch.tensor(targets, device=self.device)

                    loss = self.criterion(outputs, targets)
                    affected_cell = self.model.state[0] | self.model.state[1]

                    accumulators.append(outputs.detach().cpu())
                    accumulator_masks.append(self.model.accumulator_mask.detach().cpu())
                    affected_cell_count_outputs.append(affected_cell.sum().item())
                    affected_cell_count_targets.append(targets.sum().item())

                    jaccard_index_value = jaccard_index(targets, affected_cell)
                    manhattan_distance_value = manhattan_distance(
                        torch.tensor([affected_cell_count_targets]),
                        torch.tensor([affected_cell_count_outputs]))

                    running_loss.append(loss.item())
                    running_jaccard.append(jaccard_index_value)

                    self.backward(loss)
                    self.model.reset(seed=batch_seed)

                    global_step = epochs * max_iterations + iterations
                    writer.add_scalar('EpochMetrics/Loss', loss.item(), global_step)
                    writer.add_scalar('EpochMetrics/Jaccard', jaccard_index_value, global_step)
                    writer.add_scalar('EpochMetrics/Manhattan', manhattan_distance_value, global_step)

                    writer.add_scalar('EpochParams/a', self.model.a.item(), global_step)
                    writer.add_scalar('EpochParams/c_1', self.model.c_1.item(), global_step)
                    writer.add_scalar('EpochParams/c_2', self.model.c_2.item(), global_step)
                    writer.add_scalar('EpochParams/p_h', self.model.p_h.item(), global_step)

                writer.add_scalar('Metrics/Jaccard', np.mean(running_jaccard), epochs)
                writer.add_scalar('Metrics/Loss', np.mean(running_loss), epochs)
                writer.add_scalar('Metrics/Manhattan', manhattan_distance_value, epochs)

                writer.add_scalar('Params/a', self.model.a.item(), epochs)
                writer.add_scalar('Params/c_1', self.model.c_1.item(), epochs)
                writer.add_scalar('Params/c_2', self.model.c_2.item(), epochs)
                writer.add_scalar('Params/p_h', self.model.p_h.item(), epochs)

                os.makedirs('out', exist_ok=True)
                with h5py.File(f'out/{self.run_name}.hdf5', 'a') as f_out:
                    epoch_ds = f_out.create_group(f'epoch_{epochs}')
                    epoch_ds.attrs['seed'] = self.model.seed
                    epoch_ds.attrs['a'] = self.model.a.item()
                    epoch_ds.attrs['c_1'] = self.model.c_1.item()
                    epoch_ds.attrs['c_2'] = self.model.c_2.item()
                    epoch_ds.attrs['p_h'] = self.model.p_h.item()
                    epoch_ds.attrs['p_continue'] = self.model.p_continue.item()
                    epoch_ds.attrs['loss'] = np.mean(running_loss)
                    epoch_ds.attrs['jaccard'] = np.mean(running_jaccard)
                    epoch_ds.attrs['manhattan'] = manhattan_distance_value

                    epoch_ds.create_dataset('accumulator', data=torch.stack(accumulators).cpu().numpy(),
                                            compression='gzip')
                    epoch_ds.create_dataset('accumulator_mask', data=torch.stack(accumulator_masks).cpu().numpy(),
                                            compression='gzip')
                    epoch_ds.create_dataset('affected_cell_count_outputs', data=affected_cell_count_outputs,
                                            compression='gzip')
                    epoch_ds.create_dataset('affected_cell_count_targets', data=affected_cell_count_targets,
                                            compression='gzip')
                    epoch_ds.create_dataset('outputs', data=(torch.stack(accumulators) > 0).cpu().numpy(),
                                            compression='gzip')
                    epoch_ds.create_dataset('target', data=np.stack(targets_list, axis=0), compression='gzip')
                    epoch_ds.create_dataset('loss', data=running_loss, compression='gzip')
                    epoch_ds.create_dataset('jaccard', data=running_jaccard, compression='gzip')

        writer.close()

    def my_evaluate(self, ds, repeat=False):
        if repeat:
            self.model.a.data = torch.tensor(ds.attrs['a'])
            self.model.c_1.data = torch.tensor(ds.attrs['c_1'])
            self.model.c_2.data = torch.tensor(ds.attrs['c_2'])
            self.model.p_h.data = torch.tensor(ds.attrs['p_h'])
            self.model.p_continue.data = torch.tensor(ds.attrs['p_continue'])
        wind_step_interval = ds.attrs['wind_step_interval']

        self.reset()
        self.model.to(self.device)
        self.model.eval()

        running_jaccard = []
        affected_cell_count_outputs = []
        affected_cell_count_targets = []

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
                    outputs = self.model.state[0] | self.model.state[1]

                    if (steps + 1) % self.steps_update_interval == 0:
                        targets = ds['target'][steps]
                        targets = torch.tensor(targets, device=self.device)
                        running_jaccard.append(jaccard_index(targets, outputs))
                        affected_cell_count_outputs.append(outputs.sum().item())
                        affected_cell_count_targets.append(targets.sum().item())

                    postfix['burning'] = self.model.state[0].sum().detach().cpu().item()
                    postfix['burned'] = self.model.state[1].sum().detach().cpu().item()

                    output_list.append(outputs.cpu().detach().numpy())

                    progress_bar.set_postfix(postfix)
                    progress_bar.update(1)

        with h5py.File(f'{self.run_name}.hdf5', 'w') as f_out:
            f_out.attrs['seed'] = self.model.seed
            f_out.attrs['a'] = self.model.a.item()
            f_out.attrs['c_1'] = self.model.c_1.item()
            f_out.attrs['c_2'] = self.model.c_2.item()
            f_out.attrs['p_h'] = self.model.p_h.item()
            f_out.attrs['p_continue'] = self.model.p_continue.item()
            f_out.attrs['jaccard'] = np.mean(running_jaccard)
            f_out.attrs['manhattan'] = manhattan_distance(
                torch.tensor([affected_cell_count_targets]),
                torch.tensor([affected_cell_count_outputs]))
            f_out.create_dataset('output', data=np.stack(output_list), compression='gzip')
            f_out.create_dataset('affected_cell_count_outputs', data=affected_cell_count_outputs, compression='gzip')
            f_out.create_dataset('affected_cell_count_targets', data=affected_cell_count_targets, compression='gzip')
            f_out.create_dataset('jaccard', data=running_jaccard, compression='gzip')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=str, required=True, default=0)
    parser.add_argument('--a', type=float, required=False, default=0.0)
    parser.add_argument('--p_h', type=float, required=False, default=0.8)
    parser.add_argument('--c_1', type=float, required=False, default=0.0)
    parser.add_argument('--c_2', type=float, required=False, default=0.0)
    parser.add_argument('--seed', type=int, required=False, default=None)
    parser.add_argument('--max_epochs', type=int, required=False, default=10)
    parser.add_argument('--device', type=str, required=False, default='cuda:0')
    parser.add_argument('--run_name', type=str, required=False, default='default')
    parser.add_argument('--steps_update_interval', type=int, required=False, default=10)
    parser.add_argument('--lr', type=float, required=False, default=0.005)
    parser.add_argument('--mode', type=str, required=False, default='train')

    args = parser.parse_args()

    with h5py.File('simulated_fig_targets.hdf5', 'r') as f_in:
        f_in_ds = f_in[args.exp_id]

        trainer = Fig6Trainer(model=WildfireModel({
            'p_veg': torch.tensor(f_in_ds['p_veg'][:]),
            'p_den': torch.tensor(f_in_ds['p_den'][:]),
            'wind_towards_direction': torch.tensor(f_in_ds['wind_towards_direction'][:][0]),
            'wind_velocity': torch.tensor(f_in_ds['wind_velocity'][:][0]),
            'slope': torch.tensor(f_in_ds['slope'][:]),
            'initial_ignition': torch.tensor(f_in_ds['initial_ignition'][:], dtype=torch.bool)
        }, {
            'a': torch.tensor(args.a),
            'p_h': torch.tensor(args.p_h),
            'p_continue': torch.tensor(f_in_ds.attrs['p_continue']),
            'c_1': torch.tensor(args.c_1),
            'c_2': torch.tensor(args.c_2),
        }, keep_acc_mask=True), device=torch.device(args.device), run_name=args.run_name)
        trainer.max_epochs = args.max_epochs
        trainer.steps_update_interval = args.steps_update_interval
        trainer.max_steps = f_in_ds.attrs['max_steps']
        trainer.lr = args.lr
        trainer.seed = args.seed

        if args.mode == 'train':
            trainer.my_train(ds=f_in_ds)
        elif args.mode == 'predict':
            trainer.my_evaluate(ds=f_in_ds)
        elif args.mode == 'repeat':
            trainer.my_evaluate(ds=f_in_ds, repeat=True)
