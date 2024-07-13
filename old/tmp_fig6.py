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

    max_iteration = self.max_steps // self.steps_update_interval

    postfix = {}
    for epoch in range(self.max_epoch):
        postfix['epoch'] = f'{epoch + 1}/{self.max_epoch}'
        self.model.reset()

        accumulators = []
        accumulator_masks = []

        affected_cell_count_pred = []
        affected_cell_count_targ = []
        batch_seed = self.model.seed

        with tqdm(total=max_iteration) as progress_bar:
            for iteration in range(max_iteration):
                postfix['iteration'] = f'{iteration + 1}/{max_iteration}'
                batch_max_step = min(self.max_steps, (iteration + 1) * self.steps_update_interval)

                # recover the situation

                for step in range(batch_max_step):
                    postfix['step'] = f'{step + 1}/{batch_max_step}'
                    # update wind
                    self.model.wind_towards_direction = torch.tensor(fg_ds['wind_towards_direction'][step][:],
                                                                     dtype=self.dtype, device=self.device)
                    self.model.wind_velocity = torch.tensor(fg_ds['wind_V'][step][:], dtype=self.dtype,
                                                            device=self.device)

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

        writer.add_scalar('lastEpochLoss/total', loss[0].detach().cpu().item(), epoch)
        writer.add_scalar('lastEpochLoss/bce', loss[1].detach().cpu().item(), epoch)
        writer.add_scalar('lastEpochLoss/mse', loss[2].detach().cpu().item(), epoch)

        writer.add_scalar('lastEpochParams/p_h', self.model.parameter_dict['p_h'].detach().cpu().item(), epoch)
        writer.add_scalar('lastEpochParams/a', self.model.parameter_dict['a'].detach().cpu().item(), epoch)
        writer.add_scalar('lastEpochParams/c_1', self.model.parameter_dict['c_1'].detach().cpu().item(), epoch)
        writer.add_scalar('lastEpochParams/c_2', self.model.parameter_dict['c_2'].detach().cpu().item(), epoch)

        writer.add_scalar('lastEpochMetrics/ssim', ssim_value.detach().cpu().item(), epoch)
        writer.add_scalar('lastEpochMetrics/jaccard', jaccard_index_value.detach().cpu().item(), epoch)
        writer.add_scalar('lastEpochMetrics/dice', dice_value.detach().cpu().item(), epoch)
        writer.add_scalar('lastEpochMetrics/dtw_distance', dtw_distance, epoch)
        writer.add_scalar('lastEpochMetrics/euclidean_distance', euclidean_distance, epoch)
        writer.add_scalar('lastEpochMetrics/manhattan_distance', manhattan_distance, epoch)

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

# ------------------------------

import os

import h5py
import numpy as np
import torch
import torchmetrics
from einops import repeat
from torch import nn
from tqdm import tqdm

from pytorchfire.model import WildfireModel


class InitialGenerator:
    def __init__(self, device=torch.device('cpu'), dtype=torch.float32):
        self.model = None
        self.max_steps = 200  # [0, 199]
        self.repeat_times = 5

        self.device = device
        self.dtype = dtype
        self.folder_name = 'initial'

    def predict(self, f_ds: h5py.File):

        os.makedirs(self.folder_name, exist_ok=True)

        cond_p_h = [None, 0.2, 0.4, 0.5, 0.8]

        postfix = {}
        for obs_id in f_ds.keys():
            postfix['obs_id'] = obs_id
            fg_run = f_ds[obs_id]

            self.model = WildfireModel({
                'wind_V': torch.tensor(fg_run['wind_V'][0][:], dtype=self.dtype, device=self.device),
                'wind_towards_direction': torch.tensor(fg_run['wind_towards_direction'][0][:], dtype=self.dtype,
                                                       device=self.device),
                # starting from East and going counterclockwise in degrees
                'slope': torch.tensor(fg_run['slope'][:], dtype=self.dtype, device=self.device),  # degrees
                'canopy': torch.tensor(fg_run['canopy'][:], dtype=self.dtype, device=self.device),  # %
                'density': torch.tensor(fg_run['density'][:], dtype=self.dtype, device=self.device),  # kg m^{-3} * 100
                'initial_fire': torch.tensor(fg_run['initial_fire'][:], dtype=torch.bool, device=self.device)
            }, {
                'a': nn.Parameter(torch.tensor(fg_run.attrs['a'], device=self.device), requires_grad=False),
                'c_1': nn.Parameter(torch.tensor(fg_run.attrs['c_1'], device=self.device), requires_grad=False),
                'c_2': nn.Parameter(torch.tensor(fg_run.attrs['c_2'], device=self.device), requires_grad=False),
                'p_continue_burn': nn.Parameter(torch.tensor(fg_run.attrs['p_continue_burn'], device=self.device),
                                                requires_grad=False),
                'p_h': nn.Parameter(torch.tensor(fg_run.attrs['p_h'], device=self.device), requires_grad=False),
                # 0.2 <= p_h <= 1
            })

            with tqdm(total=len(cond_p_h) * self.repeat_times * self.max_steps) as progress_bar:
                for cond_id, p_h in enumerate(cond_p_h):

                    postfix['cond'] = f'p_h_{p_h}' if p_h is not None else 'p_h_default'
                    if p_h is not None:
                        self.model.parameter_dict['p_h'].data = torch.tensor(p_h, device=self.device)

                    for repeat_id in range(self.repeat_times):
                        postfix['repeat_id'] = f'{repeat_id + 1}/{self.repeat_times}'
                        self.model.reset()

                        accumulator_list = []

                        for step in range(self.max_steps):
                            postfix['step'] = f'{step + 1}/{self.max_steps}'
                            # update wind
                            self.model.wind_towards_direction = torch.tensor(fg_run['wind_towards_direction'][step][:],
                                                                             dtype=self.dtype, device=self.device)
                            self.model.wind_velocity = torch.tensor(fg_run['wind_V'][step][:], dtype=self.dtype,
                                                                    device=self.device)

                            # Perform a forward pass
                            self.model.compute(attach=False)

                            # Save the accumulator
                            accumulator_list.append(self.model.accumulator.detach().cpu().numpy())

                            postfix['burning'] = self.model.fire_state[0].sum().detach().cpu().numpy()
                            postfix['burned'] = self.model.fire_state[1].sum().detach().cpu().numpy()

                            progress_bar.set_postfix(postfix)
                            progress_bar.update(1)

                        accumulator_output = self.model.accumulator
                        binary_accumulator_output = accumulator_output > 0
                        target = torch.tensor(fg_run['observation'][self.max_steps - 1], dtype=torch.bool,
                                              device=self.device)

                        ssim_value = torchmetrics.functional.image.structural_similarity_index_measure(
                            repeat(binary_accumulator_output * 1.0, 'h w -> 1 1 h w').float(),
                            repeat(target * 1.0, 'h w -> 1 1 h w').float(),
                            data_range=1.0)
                        jaccard_index_value = torchmetrics.functional.classification.binary_jaccard_index(
                            binary_accumulator_output * 1, target * 1)
                        dice_value = torchmetrics.functional.classification.dice(binary_accumulator_output * 1,
                                                                                 target * 1,
                                                                                 num_classes=2)

                        with h5py.File(f'{self.folder_name}/initial_obs_{obs_id}.hdf5', 'a') as f_dump:
                            if postfix['cond'] not in f_dump:
                                f_grp = f_dump.create_group(postfix['cond'])
                            else:
                                f_grp = f_dump[postfix['cond']]
                            f_grp = f_grp.create_dataset(f'repeat_{repeat_id}', data=np.stack(accumulator_list),
                                                         compression='gzip')
                            f_grp.attrs['seed'] = self.model.seed
                            f_grp.attrs['p_continue_burn'] = self.model.parameter_dict[
                                'p_continue_burn'].detach().cpu().numpy()
                            f_grp.attrs['p_h'] = self.model.parameter_dict['p_h'].detach().cpu().numpy()
                            f_grp.attrs['a'] = self.model.parameter_dict['a'].detach().cpu().numpy()
                            f_grp.attrs['c_1'] = self.model.parameter_dict['c_1'].detach().cpu().numpy()
                            f_grp.attrs['c_2'] = self.model.parameter_dict['c_2'].detach().cpu().numpy()
                            f_grp.attrs['ssim'] = ssim_value.detach().cpu().numpy()
                            f_grp.attrs['jaccard'] = jaccard_index_value.detach().cpu().numpy()
                            f_grp.attrs['dice'] = dice_value.detach().cpu().numpy()


i_generator = InitialGenerator(device=torch.device('cuda'))
i_generator.predict(h5py.File('sim_dataset_v4.hdf5', 'r'))
