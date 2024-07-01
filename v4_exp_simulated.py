import argparse
import os

import h5py
import numpy as np
import torch
import torchmetrics
from dtaidistance import dtw
from einops import repeat
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import WildfireModel
from trainer import Trainer


class SimulatedTrainer(Trainer):

    def train_exp(self, obs_id: int, lr: float = 0.1, max_epoch: int = 10, loss_type: int = 0, p_h: float = 0.5,
                  run_name: str = '', steps_update_interval: int = 10):
        log_dir = os.path.join('/root/tf-logs', run_name)
        writer = SummaryWriter(log_dir=log_dir)

        f_ds = h5py.File(f'sim_dataset_v4.hdf5', 'r')
        fg_ds = f_ds[str(obs_id)]

        self.model = WildfireModel({
            'wind_V': torch.tensor(fg_ds['wind_V'][0][:], dtype=self.dtype, device=self.device),
            'wind_towards_direction': torch.tensor(fg_ds['wind_towards_direction'][0][:], dtype=self.dtype,
                                                   device=self.device),
            # starting from East and going counterclockwise in degrees
            'slope': torch.tensor(fg_ds['slope'][:], dtype=self.dtype, device=self.device),  # degrees
            'canopy': torch.tensor(fg_ds['canopy'][:], dtype=self.dtype, device=self.device),  # %
            'density': torch.tensor(fg_ds['density'][:], dtype=self.dtype, device=self.device),  # kg m^{-3} * 100
            'initial_fire': torch.tensor(fg_ds['initial_fire'][:], dtype=torch.bool, device=self.device)
        }, {
            'a': nn.Parameter(torch.tensor(0.0, device=self.device)),
            'c_1': nn.Parameter(torch.tensor(0.0, device=self.device)),
            'c_2': nn.Parameter(torch.tensor(0.0, device=self.device)),
            'p_continue_burn': nn.Parameter(torch.tensor(fg_ds.attrs['p_continue_burn'], device=self.device),
                                            requires_grad=False),
            'p_h': nn.Parameter(torch.tensor(p_h, device=self.device)),  # 0.2 <= p_h <= 1
        })

        self.lr = lr
        self.max_epoch = max_epoch
        self.steps_update_interval = steps_update_interval
        self.reset()

        postfix = {}
        for epoch in range(self.max_epoch):
            postfix['epoch'] = f'{epoch + 1}/{self.max_epoch}'
            self.model.reset()
            max_iteration = self.max_steps // self.steps_update_interval

            accumulators = []
            accumulator_masks = []

            affected_cell_count_pred = []
            affected_cell_count_targ = []

            with tqdm(total=max_iteration) as progress_bar:
                batch_seed = self.model.seed
                for iteration in range(max_iteration):
                    postfix['iteration'] = f'{iteration + 1}/{max_iteration}'
                    batch_max_step = min(self.max_steps, (iteration + 1) * self.steps_update_interval)

                    # recover the situation

                    for step in range(batch_max_step):
                        postfix['step'] = f'{step + 1}/{batch_max_step}'
                        # update wind
                        self.model.wind_towards_direction = torch.tensor(fg_ds['wind_towards_direction'][step][:],
                                                                         dtype=self.dtype, device=self.device)
                        self.model.wind_V = torch.tensor(fg_ds['wind_V'][step][:], dtype=self.dtype, device=self.device)

                        # Perform a forward pass
                        self.model.compute(attach=self.check_if_attach(batch_max_step, step))
                        progress_bar.set_postfix(postfix)

                    accumulator_output = self.model.accumulator
                    target = torch.tensor(fg_ds['observation'][batch_max_step - 1], dtype=torch.bool,
                                          device=self.device)
                    loss = self.loss_fn(accumulator_output, target * 1.0)
                    binary_accumulator_output = accumulator_output > 0

                    affected_cell_count_targ.append(target.sum().detach().cpu().item())
                    affected_cell_count_pred.append(binary_accumulator_output.sum().detach().cpu().item())

                    ssim_value = torchmetrics.functional.image.structural_similarity_index_measure(
                        repeat(binary_accumulator_output * 1.0, 'h w -> 1 1 h w').float(),
                        repeat(target * 1.0, 'h w -> 1 1 h w').float(),
                        data_range=1.0)

                    jaccard_index_value = torchmetrics.functional.classification.binary_jaccard_index(
                        binary_accumulator_output * 1, target * 1)
                    dice_value = torchmetrics.functional.classification.dice(binary_accumulator_output * 1, target * 1,
                                                                             num_classes=2)
                    dtw_distance = dtw.distance_fast(np.array(affected_cell_count_pred, dtype=np.double),
                                                     np.array(affected_cell_count_targ, dtype=np.double),
                                                     use_pruning=True)
                    euclidean_distance = torchmetrics.functional.pairwise_euclidean_distance(
                        torch.tensor([affected_cell_count_pred], dtype=torch.float32),
                        torch.tensor([affected_cell_count_targ], dtype=torch.float32))
                    manhattan_distance = torchmetrics.functional.pairwise_manhattan_distance(
                        torch.tensor([affected_cell_count_pred], dtype=torch.float32),
                        torch.tensor([affected_cell_count_targ], dtype=torch.float32))

                    # Perform a backward pass
                    loss[loss_type].backward()

                    # Update the parameters
                    self.optimizer.step()

                    # Zero the gradients
                    self.optimizer.zero_grad()

                    with torch.no_grad():
                        self.model.parameter_dict['a'].clamp_(min=0.0, max=1.0)
                        self.model.parameter_dict['c_1'].clamp_(min=0.0, max=1.0)
                        self.model.parameter_dict['c_2'].clamp_(min=0.0, max=1.0)
                        self.model.parameter_dict['p_h'].clamp_(min=0.2, max=1.0)

                    accumulators.append(self.model.accumulator.detach().cpu())
                    accumulator_masks.append(self.model.accumulator_mask.detach().cpu())

                    global_step = epoch * max_iteration + iteration
                    writer.add_scalar('Loss/total', loss[0].detach().cpu().item(), global_step)
                    writer.add_scalar('Loss/bce', loss[1].detach().cpu().item(), global_step)
                    writer.add_scalar('Loss/mse', loss[2].detach().cpu().item(), global_step)

                    writer.add_scalar('Params/p_h', self.model.parameter_dict['p_h'].detach().cpu().item(), global_step)
                    writer.add_scalar('Params/a', self.model.parameter_dict['a'].detach().cpu().item(), global_step)
                    writer.add_scalar('Params/c_1', self.model.parameter_dict['c_1'].detach().cpu().item(), global_step)
                    writer.add_scalar('Params/c_2', self.model.parameter_dict['c_2'].detach().cpu().item(), global_step)

                    writer.add_scalar('Metrics/ssim', ssim_value.detach().cpu().item(), global_step)
                    writer.add_scalar('Metrics/jaccard', jaccard_index_value.detach().cpu().item(), global_step)
                    writer.add_scalar('Metrics/dice', dice_value.detach().cpu().item(), global_step)
                    writer.add_scalar('Metrics/dtw_distance', dtw_distance, global_step)
                    writer.add_scalar('Metrics/euclidean_distance', euclidean_distance, global_step)
                    writer.add_scalar('Metrics/manhattan_distance', manhattan_distance, global_step)

                    self.model.reset(batch_seed)

                    progress_bar.set_postfix(postfix)
                    progress_bar.update(1)

            writer.add_scalar('epochLoss/total', loss[0].detach().cpu().item(), epoch)
            writer.add_scalar('epochLoss/bce', loss[1].detach().cpu().item(), epoch)
            writer.add_scalar('epochLoss/mse', loss[2].detach().cpu().item(), epoch)

            writer.add_scalar('epochParams/p_h', self.model.parameter_dict['p_h'].detach().cpu().item(), epoch)
            writer.add_scalar('epochParams/a', self.model.parameter_dict['a'].detach().cpu().item(), epoch)
            writer.add_scalar('epochParams/c_1', self.model.parameter_dict['c_1'].detach().cpu().item(), epoch)
            writer.add_scalar('epochParams/c_2', self.model.parameter_dict['c_2'].detach().cpu().item(), epoch)

            writer.add_scalar('epochMetrics/ssim', ssim_value.detach().cpu().item(), epoch)
            writer.add_scalar('epochMetrics/jaccard', jaccard_index_value.detach().cpu().item(), epoch)
            writer.add_scalar('epochMetrics/dice', dice_value.detach().cpu().item(), epoch)
            writer.add_scalar('epochMetrics/dtw_distance', dtw_distance, epoch)
            writer.add_scalar('epochMetrics/euclidean_distance', euclidean_distance, epoch)
            writer.add_scalar('epochMetrics/manhattan_distance', manhattan_distance, epoch)

            os.makedirs('dump', exist_ok=True)

            with h5py.File(f'dump/{run_name}.hdf5', 'a') as f_dump:

                epoch_ds = f_dump.create_group(f'epoch_{epoch}')

                epoch_ds.attrs['seed'] = self.model.seed
                epoch_ds.attrs['p_continue_burn'] = self.model.parameter_dict['p_continue_burn'].detach().cpu().numpy()
                epoch_ds.attrs['p_h'] = self.model.parameter_dict['p_h'].detach().cpu().numpy()
                epoch_ds.attrs['a'] = self.model.parameter_dict['a'].detach().cpu().numpy()
                epoch_ds.attrs['c_1'] = self.model.parameter_dict['c_1'].detach().cpu().numpy()
                epoch_ds.attrs['c_2'] = self.model.parameter_dict['c_2'].detach().cpu().numpy()
                epoch_ds.attrs['loss'] = loss[0].detach().cpu().numpy()
                epoch_ds.attrs['bce_loss'] = loss[1].detach().cpu().numpy()
                epoch_ds.attrs['mse_loss'] = loss[2].detach().cpu().numpy()

                epoch_ds.create_dataset('accumulator', data=torch.stack(accumulators).cpu().numpy(), compression='gzip')
                epoch_ds.create_dataset('accumulator_mask', data=torch.stack(accumulator_masks).cpu().numpy(),
                                        compression='gzip')
                epoch_ds.create_dataset('target', data=target.cpu().numpy())

        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run training experiment.')
    parser.add_argument('--obs_id', type=int, required=False, default=0, help='Observation ID')
    parser.add_argument('--lr', type=float, required=False, default=0.1, help='Learning rate')
    parser.add_argument('--p_h', type=float, required=False, default=0.8, help='Probability of something')
    parser.add_argument('--max_epoch', type=int, required=False, default=10, help='Maximum number of epochs')
    parser.add_argument('--loss_type', type=int, required=False, default=0, help='Loss type')
    parser.add_argument('--device', type=str, required=False, default='cuda:0', help='Device')
    parser.add_argument('--run_name', type=str, required=False, default='default', help='Run name')
    parser.add_argument('--steps_update_interval', type=int, required=False, default=10, help='Steps update interval')

    args = parser.parse_args()
    trainer = SimulatedTrainer(device=torch.device(args.device), dtype=torch.float32)
    trainer.train_exp(obs_id=args.obs_id, lr=args.lr, max_epoch=args.max_epoch, loss_type=args.loss_type, p_h=args.p_h,
                      run_name=args.run_name, steps_update_interval=args.steps_update_interval)
