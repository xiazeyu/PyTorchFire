from functools import cache
from typing import TYPE_CHECKING

import torch
from einops import repeat
from torch import nn

if TYPE_CHECKING:
    # This import is only executed by type checkers; it will not run at runtime.
    from model import WildfireModel


class BaseTrainer:
    """
    Base class for training a WildfireModel.

    Attributes:
        model (WildfireModel):
            The model to train.

        lr (float):
            The learning rate for the optimizer.

            The optimal learning rate depends on the model and the dataset.
            We recommend using a learning rate between `1e-2` to `4e-3`.

            Higher learning rates can speed up training, but might lead to instability.

        optimizer (torch.optim.Optimizer):
            The optimizer to use for training.

            We recommend using `torch.optim.AdamW` as the optimizer.

            Changing the optimizer may require changing the `lr` parameter.

        max_epochs (int):
            The maximum number of epochs to train for.

            We recommend using a minimum of 10 epochs.

        max_steps (int):
            The maximum number of steps to train for.

            This is a general upper limit for the number of steps to train for if you are not using customized epochs.

        steps_update_interval (int):
            The number of steps after which to update the model.

            This parameter depends on the number of data points.

            For example, if you have 200 data points, you can set this to 10. This will update the model every 10 steps.
            Making the number of epochs 20.

        update_steps_first (int):
            The number of steps to update the model at the beginning.

        update_steps_last (int):
            The number of steps to update the model at the end.

        update_steps_in_between (int):
            The number of steps to update the model in between the first and last steps.

        device (torch.device):
            The device to use for training.

            We recommend using `torch.device('cuda')` if you have a GPU.

        seed (int):
            The seed to use for reproducibility.

    Examples:
        ```python
        import torch
        from pytorchfire import WildfireModel, BaseTrainer

        model = WildfireModel()

        trainer = BaseTrainer(model)

        trainer.train()
        trainer.evaluate()
        ```
    """

    model: 'WildfireModel'
    lr: float
    optimizer: torch.optim.Optimizer
    max_epochs: int
    max_steps: int
    steps_update_interval: int
    update_steps_first: int
    update_steps_last: int
    update_steps_in_between: int
    device: torch.device
    seed: int | None

    def __init__(self, model: 'WildfireModel', optimizer: torch.optim.Optimizer = None,
                 device: torch.device = torch.device('cpu')):
        """
        Initialize the trainer.

        Parameters:
            model (WildfireModel):
                The model to train.

            optimizer (torch.optim.Optimizer):
                The optimizer to use for training.

            device (torch.device):
                The device to use for training.
        """
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
        """
        Reset the model and optimizer.
        """
        self.model.reset(seed=self.seed)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    @staticmethod
    @cache
    def steps_to_attach(max_steps: int, update_steps_first: int, update_steps_last: int,
                        update_steps_in_between: int) -> list[int]:
        """
        Get the steps to attach the model.

        Parameters:
            max_steps (int):
                The maximum number of steps to train for.

            update_steps_first (int):
                The number of steps to update the model at the beginning.

            update_steps_last (int):
                The number of steps to update the model at the end.

            update_steps_in_between (int):
                The number of steps to update the model in between the first and last steps.

        Returns:
            The steps to attach to the accumulator.
        """
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

    def check_if_attach(self, current_steps: int, max_steps: int) -> bool:
        """
        Check if current step should be attached to the accumulator.

        Parameters:
            current_steps (int):
                The current step.

            max_steps (int):
                The maximum number of steps to train for.

        Returns:
            Whether the current step should be attached to the accumulator.
        """
        return current_steps in self.steps_to_attach(max_steps, self.update_steps_first, self.update_steps_last,
                                                     self.update_steps_in_between)

    @staticmethod
    def criterion(inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate the loss.

        Parameters:
            inputs (torch.Tensor):
                The input tensor.

                - dtype: `torch.float`
                - shape: `[Height, Width]`

            target (torch.Tensor):
                The target tensor.

                - dtype: `torch.float`
                - shape: `[Height, Width]`

        Returns:
            The loss.

                - dtype: `torch.float`
                - shape: `[]`
        """
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

        avg_pool = nn.AvgPool2d(kernel_size=window_size, stride=stride)

        inputs_avg = avg_pool(inputs)
        target_avg = avg_pool(target)

        mse_loss_fn = nn.MSELoss()
        mse_loss = mse_loss_fn(inputs_avg, target_avg)

        return bce_loss + mse_loss

    def backward(self, loss: torch.Tensor):
        """
        Perform the backward pass.

        Parameters:
            loss (torch.Tensor):
                The loss to back-propagate.

                - dtype: `torch.float`
                - shape: `[]`
        """
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        with torch.no_grad():
            self.model.a.clamp_(min=0.0, max=1.0)
            self.model.c_1.clamp_(min=0.0, max=1.0)
            self.model.c_2.clamp_(min=0.0, max=1.0)
            self.model.p_h.clamp_(min=0.2, max=1.0)

    def train(self):
        """
        Train the model.

        This method is a placeholder. Implement this method in your subclass.

        You can do in-place operations on the model in this method.

        E.g., `model.wind_velocity = torch.rand_like(model.wind_velocity)`, or `model.a.data = torch.rand(())`
        """

        print('Modify the train method to train your model')

        # Remove this line after implementing the train method
        self.model.initial_ignition = (torch.rand_like(self.model.initial_ignition, dtype=torch.float) > .9)

        self.reset()
        self.model.to(self.device)
        self.model.train()

        max_iterations = self.max_steps // self.steps_update_interval

        for epochs in range(self.max_epochs):
            self.model.reset()
            epoch_seed = self.model.seed
            running_loss = 0.0
            for iterations in range(max_iterations):
                iter_max_steps = min(self.max_steps, (iterations + 1) * self.steps_update_interval)
                for steps in range(iter_max_steps):
                    self.model.compute(attach=self.check_if_attach(steps, iter_max_steps))

                outputs = self.model.accumulator
                targets = self.model.accumulator  # replace your target here

                loss = self.criterion(outputs, targets)
                running_loss += loss.item()

                self.backward(loss)
                self.model.reset(epoch_seed)

                print(
                    f"Epoch [{epochs + 1}/{self.max_epochs}],"
                    f"Iteration {iterations + 1}/{max_iterations},"
                    f"Epoch Loss: {running_loss / max_iterations}"
                )

    def evaluate(self):
        """
        Evaluate the model.

        This method is a placeholder. Implement this method in your subclass.

        You can do in-place operations on the model in this method.

        E.g., `model.wind_velocity = torch.rand_like(model.wind_velocity)`, or `model.a.data = torch.rand(())`
        """

        print('Modify the evaluate method to evaluate your model')

        self.reset()
        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            for steps in range(self.max_steps):
                self.model.compute()
                print(f"Step [{steps + 1}/{self.max_steps}]")
