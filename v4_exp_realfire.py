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

from pytorchfire.model import WildfireModel
from trainer import Trainer


class RealFireTrainer(Trainer):

    def __init__(self, device=torch.device('cpu'), dtype=torch.float32):
        super().__init__(device=device, dtype=dtype)

    @torch.inference_mode()
    def probe_max_steps(self, fire_name: str, p_continue_burn: float = 0.3, p_h: float = 0.5):
        # Just for setting up the baseline. Not working well to directly predict the max steps.

        def get_true_boundaries(tensor):
            if not tensor.any():
                return None  # No True values in the tensor

            true_indices = tensor.nonzero(as_tuple=True)

            start_indices = [indices.min().item() for indices in true_indices]
            end_indices = [indices.max().item() for indices in true_indices]

            return start_indices, end_indices

        def is_boundary_contained(tensor1, tensor2):
            boundaries1 = get_true_boundaries(tensor1)
            boundaries2 = get_true_boundaries(tensor2)

            if boundaries1 is None:
                return True  # tensor1 has no True values, so it's trivially contained

            if boundaries2 is None:
                return False  # tensor2 has no True values, so tensor1 can't be contained

            start1, end1 = boundaries1
            start2, end2 = boundaries2

            for s1, e1, s2, e2 in zip(start1, end1, start2, end2):
                if not (s2 <= s1 <= e2 and s2 <= e1 <= e2):
                    return False

            return True

        f_ds = h5py.File(f'real_dataset_v4.hdf5', 'r')
        fg_ds = f_ds[fire_name]
        fg_shape = fg_ds['canopy'][:].shape
        self.model = WildfireModel({
            'wind_V': torch.zeros(fg_shape, dtype=self.dtype, device=self.device),
            'wind_towards_direction': torch.zeros(fg_shape, dtype=self.dtype,
                                                  device=self.device),
            # starting from East and going counterclockwise in degrees
            'slope': torch.zeros(fg_shape, dtype=self.dtype, device=self.device),  # degrees
            'canopy': torch.zeros(fg_shape, dtype=self.dtype, device=self.device),  # %
            'density': torch.zeros(fg_shape, dtype=self.dtype, device=self.device),  # kg m^{-3} * 100
            'initial_fire': torch.tensor(fg_ds['initial_fire'][:], dtype=torch.bool, device=self.device)
        }, {
            'a': nn.Parameter(torch.tensor(0.0, device=self.device)),
            'c_1': nn.Parameter(torch.tensor(0.0, device=self.device)),
            'c_2': nn.Parameter(torch.tensor(0.0, device=self.device)),
            'p_continue_burn': nn.Parameter(torch.tensor(p_continue_burn, device=self.device)),
            'p_h': nn.Parameter(torch.tensor(p_h, device=self.device)),  # 0.2 <= p_h <= 1
        })

        final_result = torch.tensor(fg_ds['target'][-1], dtype=self.dtype, device=self.device)

        self.reset()

        counter = 0
        postfix = {}

        with tqdm() as progress_bar:
            while not is_boundary_contained(final_result, self.model.fire_state[0] | self.model.fire_state[1]):
                counter += 1
                self.model.compute(attach=False)
                if self.model.fire_state[0].sum() == 0:
                    print('No more burning cells')
                    break
                postfix['step'] = f'{counter}'
                postfix['burning'] = f'{self.model.fire_state[0].sum().item()}'
                postfix['burned'] = f'{self.model.fire_state[1].sum().item()}'
                progress_bar.set_postfix(postfix)

        return counter

    def train_exp(self, fire_name: str, lr: float = 0.1, max_epoch: int = 10, loss_type: int = 0, p_h: float = 0.5,
                  p_continue_burn: float = 0.3, steps_update_interval: int = 30,
                  run_name: str = ''):
        log_dir = os.path.join('/root/tf-logs', run_name)
        writer = SummaryWriter(log_dir=log_dir)

        f_ds = h5py.File(f'real_dataset_v4.hdf5', 'r')
        fg_ds = f_ds[fire_name]

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
            'p_continue_burn': nn.Parameter(torch.tensor(p_continue_burn, device=self.device),
                                            requires_grad=False),
            'p_h': nn.Parameter(torch.tensor(p_h, device=self.device)),  # 0.2 <= p_h <= 1
        })

        self.lr = lr
        self.reset()

        postfix = {}
        max_iteration = fg_ds.attrs['day_count']
        for epoch in range(max_epoch):  # [0, max_epoch - 1]
            postfix['epoch'] = f'{epoch + 1}/{max_epoch}'
            self.model.reset()

            accumulators = []
            accumulator_masks = []
            targets = []

            affected_cell_count_pred = []
            affected_cell_count_targ = []

            with (tqdm(total=max_iteration) as progress_bar):
                batch_seed = self.model.seed
                for iteration in range(max_iteration):  # [0, day_count - 1]
                    postfix['iteration'] = f'{iteration + 1}/{max_iteration}'
                    batch_max_step = (iteration + 1) * steps_update_interval

                    for step in range(batch_max_step):
                        postfix['step'] = f'{step + 1}/{batch_max_step}'

                        # update wind
                        if step % steps_update_interval == 0:
                            self.model.wind_towards_direction = torch.tensor(
                                fg_ds['wind_towards_direction'][step // steps_update_interval][:],
                                dtype=self.dtype, device=self.device)
                            self.model.wind_V = torch.tensor(fg_ds['wind_V'][step // steps_update_interval][:],
                                                             dtype=self.dtype,
                                                             device=self.device)

                        # Perform a forward pass
                        self.model.compute(attach=self.check_if_attach(batch_max_step, step))
                        progress_bar.set_postfix(postfix)

                    accumulator_output = self.model.accumulator
                    target = torch.tensor(fg_ds['target'][iteration], dtype=torch.bool,
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
                    targets.append(target.detach().cpu())

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
                epoch_ds.create_dataset('binary_accumulator', data=(torch.stack(accumulators) > 0).cpu().numpy(),
                                        compression='gzip')
                epoch_ds.create_dataset('accumulator_mask', data=torch.stack(accumulator_masks).cpu().numpy(),
                                        compression='gzip')
                epoch_ds.create_dataset('target', data=torch.stack(targets).cpu().numpy(), compression='gzip')

        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run training experiment.')
    parser.add_argument('--fire_name', type=str, required=False, default='Bear_2020', help='Fire name')
    parser.add_argument('--lr', type=float, required=False, default=0.005, help='Learning rate')
    parser.add_argument('--p_h', type=float, required=False, default=0.8, help='Probability of something')
    parser.add_argument('--p_continue_burn', type=float, required=False, default=0.3, help='Probability of something')
    parser.add_argument('--max_epoch', type=int, required=False, default=10, help='Maximum number of epochs')
    parser.add_argument('--steps_update_interval', type=int, required=False, default=30, help='Steps update interval')
    parser.add_argument('--loss_type', type=int, required=False, default=0, help='Loss type')
    parser.add_argument('--device', type=str, required=False, default='cuda:0', help='Device')
    parser.add_argument('--run_name', type=str, required=False, default='default', help='Run name')

    args = parser.parse_args()
    trainer = RealFireTrainer(device=torch.device(args.device), dtype=torch.float32)
    trainer.train_exp(fire_name=args.fire_name, lr=args.lr, max_epoch=args.max_epoch, loss_type=args.loss_type,
                      p_h=args.p_h, p_continue_burn=args.p_continue_burn,
                      steps_update_interval=args.steps_update_interval,
                      run_name=args.run_name)
