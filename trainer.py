from functools import cache

import torch
from einops import repeat
from torch import nn


class Trainer:
    def __init__(self, device=torch.device('cpu'), dtype=torch.float32):
        self.model = None
        self.lr = 0.005
        self.optimizer = None
        self.max_epoch = 10
        self.max_steps = 200  # [0, 199]
        self.steps_update_interval = 10

        self.update_steps_first_k = 3
        self.update_steps_last_k = 5
        self.update_steps_in_between = 4

        self.device = device
        self.dtype = dtype

    def reset(self):
        self.model.reset()
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    @staticmethod
    @cache
    def steps_to_attach(max_steps: int, update_steps_first_k: int, update_steps_last_k: int,
                        update_steps_in_between: int) -> list[int]:
        if max_steps <= update_steps_first_k + update_steps_last_k + update_steps_in_between:
            return sorted(range(max_steps))
        update_steps = set()

        update_steps.update(range(update_steps_first_k))
        update_steps.update(range(max_steps - update_steps_last_k, max_steps))

        start = update_steps_first_k
        end = max_steps - update_steps_last_k
        steps_in_between = end - start
        interval = steps_in_between // (update_steps_in_between + 1)
        for i in range(1, update_steps_in_between + 1):
            update_steps.add(start + i * interval)

        return sorted(update_steps)

    def check_if_attach(self, max_steps: int, current_step: int) -> bool:
        return current_step in Trainer.steps_to_attach(max_steps, self.update_steps_first_k, self.update_steps_last_k,
                                                       self.update_steps_in_between)

    @staticmethod
    def loss_fn(prediction: torch.Tensor, target: torch.Tensor):

        non_zero_indices = torch.nonzero(prediction + target)
        min_row, min_col = non_zero_indices.min(dim=0)[0]
        max_row, max_col = non_zero_indices.max(dim=0)[0]
        prediction = prediction[min_row:max_row + 1, min_col:max_col + 1]
        target = target[min_row:max_row + 1, min_col:max_col + 1]

        bce_loss_fn = nn.BCEWithLogitsLoss()
        bce_loss = bce_loss_fn(prediction, target)

        window_size = 4
        stride = window_size

        prediction = repeat(prediction, 'h w -> 1 1 h w')
        target = repeat(target, 'h w -> 1 1 h w')

        conv = nn.Conv2d(1, 1, kernel_size=window_size, stride=stride, bias=False, device=prediction.device)
        conv.weight.data.fill_(1.0 / (window_size * window_size))

        for param in conv.parameters():
            param.requires_grad = False

        prediction_avg = conv(prediction)
        target_avg = conv(target)

        mse_loss_fn = nn.MSELoss()
        mse_loss = mse_loss_fn(prediction_avg, target_avg)

        return bce_loss + mse_loss, bce_loss, mse_loss
