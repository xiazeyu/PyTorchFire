from functools import cache
from typing import TYPE_CHECKING

import torch
from einops import repeat
from torch import nn

if TYPE_CHECKING:
    from model import WildfireModel


class BaseTrainer:
    def __init__(self, model: 'WildfireModel', optimizer: torch.optim.Optimizer = None,
                 device: torch.device = torch.device('cpu')):
        self.model = model
        self.lr = 0.005
        if optimizer is None:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.optimizer = optimizer
        self.max_epochs = 10
        self.max_steps = 300  # [0, 199]
        self.steps_update_interval = 10

        self.update_steps_first = 3
        self.update_steps_last = 4
        self.update_steps_in_between = 5

        self.device = device
        self.seed = None

    def reset(self):
        self.model.reset(seed=self.seed)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    @staticmethod
    @cache
    def steps_to_attach(max_steps: int, update_steps_first: int, update_steps_last: int,
                        update_steps_in_between: int) -> list[int]:
        if max_steps <= update_steps_first + update_steps_last + update_steps_in_between:
            return sorted(range(max_steps))
        update_steps = set()

        update_steps.update(range(update_steps_first))
        update_steps.update(range(max_steps - update_steps_last, max_steps))

        start = update_steps_first
        end = max_steps - update_steps_last
        steps_in_between = end - start
        interval = steps_in_between // (update_steps_in_between + 1)
        for i in range(1, update_steps_in_between + 1):
            update_steps.add(start + i * interval)

        return sorted(update_steps)

    def check_if_attach(self, max_steps: int, current_steps: int) -> bool:
        return current_steps in self.steps_to_attach(max_steps, self.update_steps_first, self.update_steps_last,
                                                     self.update_steps_in_between)

    @staticmethod
    def criterion(inputs: torch.Tensor, target: torch.Tensor):
        inputs = inputs.float()
        target = target.float()

        non_zero_indices = torch.nonzero(inputs + target)
        min_row, min_col = non_zero_indices.min(dim=0)[0]
        max_row, max_col = non_zero_indices.max(dim=0)[0]
        inputs = inputs[min_row:max_row + 1, min_col:max_col + 1]
        target = target[min_row:max_row + 1, min_col:max_col + 1]

        # BCE Loss
        bce_loss_fn = nn.BCEWithLogitsLoss()
        bce_loss = bce_loss_fn(inputs, target)

        # MSE Loss
        window_size = 4
        stride = window_size

        inputs = repeat(inputs, 'h w -> 1 1 h w')
        target = repeat(target, 'h w -> 1 1 h w')

        conv = nn.Conv2d(1, 1, kernel_size=window_size,
                         stride=stride, bias=False, device=inputs.device, dtype=inputs.dtype)
        conv.weight.data.fill_(1.0 / (window_size * window_size))

        for param in conv.parameters():
            param.requires_grad = False

        inputs_avg = conv(inputs)
        target_avg = conv(target)

        mse_loss_fn = nn.MSELoss()
        mse_loss = mse_loss_fn(inputs_avg, target_avg)

        return bce_loss + mse_loss

    def train(self):
        print('Modify the train method to train your model')

        self.reset()
        self.model.to(self.device)
        self.model.train()

        max_iterations = self.max_steps // self.steps_update_interval

        for epochs in range(self.max_epochs):
            self.model.reset()
            batch_seed = self.model.seed
            running_loss = 0.0
            for iterations in range(max_iterations):
                batch_max_steps = min(self.max_steps, (iterations + 1) * self.steps_update_interval)
                for steps in range(batch_max_steps):
                    self.model.compute(attach=self.check_if_attach(batch_max_steps, steps))

                outputs = self.model.accumulator
                targets = self.model.accumulator  # replace your target here

                loss = self.criterion(outputs, targets)
                running_loss += loss.item()

                self.backward(loss)
                self.model.reset(batch_seed)

                print(
                    f"Epoch [{epochs + 1}/{self.max_epochs}],"
                    f"Iteration {iterations + 1}/{max_iterations},"
                    f"Epoch Loss: {running_loss / max_iterations}"
                )

    def backward(self, loss: torch.Tensor):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        with torch.no_grad():
            self.model.a.clamp_(min=0.0, max=1.0)
            self.model.c_1.clamp_(min=0.0, max=1.0)
            self.model.c_2.clamp_(min=0.0, max=1.0)
            self.model.p_h.clamp_(min=0.2, max=1.0)

    def evaluate(self):
        print('Modify the evaluate method to evaluate your model')

        self.reset()
        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            for steps in range(self.max_steps):
                self.model.compute()
                print(f"Step [{steps + 1}/{self.max_steps}]")
