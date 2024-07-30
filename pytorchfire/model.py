import torch
from einops import repeat, reduce
from torch import nn

DEFAULT_SIZE = 500


# noinspection PyAttributeOutsideInit
class WildfireModel(nn.Module):
    """
    A model for predicting wildfire spread based on various input features.
    The model extends the PyTorch `nn.Module` and is tailored to perform both prediction and parameter calibration.

    The model is based on the paper
    [A Cellular Automata Model for Forest Fire Spread Prediction: The Case of the Wildfire That Swept Through Spetses
    Island in 1990](https://doi.org/10.1016/j.amc.2008.06.046).

    The model uses the following formula to calculate the probability of a cell propagating fire to its neighbors:

    $$ p_\\text{propagate} = p_h (1 + p_\\text{veg}) (1 + p_\\text{den}) p_w p_s $$

    in which,

    $$ p_w = \\exp(c_1 V_w) \\exp(c_2 V_w (\\cos(\\theta_w) - 1)) $$

    $$ p_s = \\exp(a \\theta_s) $$

    The model also uses the following formula to correct the probability into correct range:

    $$ f_p(x) = \\tanh(c \\cdot x) \\text{, where } c=1.1486328125$$

    To calculate the probability of a cell igniting by its neighbors, the model uses the following formula:

    $$ p_{\\text{ignite}, i} = 1 - \\prod_{j=1}^{8} (1 - p_{\\text{propagate}, i, j}) $$

    Attributes:
        keep_acc_mask (bool):
            If True, the accumulator mask will be kept in the model.

            - `false`, by default, to speed up the model.
            - `true` if `model.accumulator_mask` is needed to see how the rings of cells are accumulated.

        a (torch.Parameter):
            The scaling factor for ground elevation.

            - dtype: `torch.float`
            - shape: `[]`

        p_h (torch.Parameter):
            The probability that a fire propagate from a burning cell to an adjacent cell under normal conditions.

            - dtype: `torch.float`
            - shape: `[]`

        c_1 (torch.Parameter):
            The scaling factor for wind velocity.

            - dtype: `torch.float`
            - shape: `[]`

        c_2 (torch.Parameter):
            The scaling factor for wind direction.

            - dtype: `torch.float`
            - shape: `[]`

        p_continue (torch.Parameter):
            The probability that a burning cell continue to burn at next time step.

            This is NOT a learnable parameter.

            The parameter affects the shape of boundary according to
            [Li's experiment](https://github.com/XC-Li/Parallel_CellularAutomaton_Wildfire#interesting-findings).

            We confirmed this by our observation, and we found this parameter doesn't have a great impact on
            the affected area.

            - dtype: `torch.float`
            - shape: `[]`

        p_veg (torch.Tensor):
            The scaling factor for vegetation type.

            - dtype: `torch.float`
            - shape: `[Height, Width]`

        p_den (torch.Tensor):
            The scaling factor for vegetation density.

            - dtype: `torch.float`
            - shape: `[Height, Width]`

        wind_velocity (torch.Tensor):
            The wind velocity. Unit is m/s.

            - dtype: `torch.float`
            - shape: `[Height, Width]`

        wind_towards_direction (torch.Tensor):
            The wind direction. Starting from East and going counterclockwise in degrees.

            - dtype: `torch.float`
            - shape: `[Height, Width]`

        slope (torch.Tensor):
            The slope of the cell to its neighboring cells. Unit is degrees.

            - dtype: `torch.float`
            - shape: `[Height, Width, 3, 3]`

        initial_ignition (torch.Tensor):
            The initial ignition of the cells.

            - dtype: `torch.bool`
            - shape: `[Height, Width]`

        state (torch.Tensor):
            The state of the cells.

            The first dimension saves if the cell is burning, the second dimension saves if the cell has burnt out.

            | Description | Channel 0 | Channel 1 |
            |:---:|:---:|:---:|
            | burnable | 0 | 0 |
            | burning | 1 | 0 |
            | burned | 0 | 1 |

            - dtype: `torch.bool`
            - shape: `[2, Height, Width]`

        accumulator (torch.Tensor):
            The accumulator of the cells.

            The accumulator is used to accumulate the rings of cells for parameter calibration
            (used for back-propagation).

            - dtype: `torch.float`
            - shape: `[Height, Width]`

        accumulator_mask (torch.Tensor):
            The mask of the accumulator.
            This attribute will only be created if `model.keep_acc_mask` is `true`.
            And it is typically used for visualization only.

            - dtype: `torch.bool`
            - shape: `[Height, Width]`

        seed (int):
            The seed for random number generator.

            Manually change this attribute will not affect the random number generator.
            It has to be changed by `model._initialize_seed(seed)`, or `model.reset(seed)` method.

    Examples:
        ```python
        from pytorchfire import WildfireModel

        model = WildfireModel() # Create a model with default parameters and environment data
        model = model.cuda() # Move the model to GPU
        # model.reset(seed=seed) # Reset the model with a seed
        for _ in range(100): # Run the model for 100 steps
            model.compute() # Compute the next state
        ```

    """

    keep_acc_mask: bool
    a: nn.Parameter
    p_h: nn.Parameter
    c_1: nn.Parameter
    c_2: nn.Parameter
    p_continue: nn.Parameter
    p_veg: torch.Tensor
    p_den: torch.Tensor
    wind_velocity: torch.Tensor
    wind_towards_direction: torch.Tensor
    slope: torch.Tensor
    initial_ignition: torch.Tensor
    state: torch.Tensor
    accumulator: torch.Tensor
    accumulator_mask: torch.Tensor
    seed: int

    def __init__(self, env_data: dict = None, params: dict = None, keep_acc_mask: bool = False):
        """
        Initialize the WildfireModel.

        Parameters:
            env_data (dict):
                The environment data for the model.

                - `p_veg` (torch.Tensor): The scaling factor for vegetation type.
                - `p_den` (torch.Tensor): The scaling factor for vegetation density.
                - `wind_velocity` (torch.Tensor): The wind velocity. Unit is m/s.
                - `wind_towards_direction` (torch.Tensor): The wind direction. Starting from East and going
                    counterclockwise in degrees.
                - `slope` (torch.Tensor): The slope of the cell to its neighboring cells. Unit is degrees.
                - `initial_ignition` (torch.Tensor): The initial ignition of the cells.

            params (dict):
                The parameters for the model.

                - `a` (torch.float): The scaling factor for ground elevation.
                - `p_h` (torch.float): The probability that a fire propagate from a burning cell to an adjacent cell
                    under normal conditions.
                - `c_1` (torch.float): The scaling factor for wind velocity.
                - `c_2` (torch.float): The scaling factor for wind direction.
                - `p_continue` (torch.float): The probability that a burning cell continue to burn at next time step.

            keep_acc_mask (bool):
                If `true`, the accumulator mask will be kept in the model.

                - `false`, by default, to speed up the model.
                - `true` if `model.accumulator_mask` is needed to see how the rings of cells are accumulated.

        Examples:
            ```python
            SIZE = 500
            environment_data = {
                'p_veg': torch.rand(SIZE, SIZE),
                'p_den': torch.rand(SIZE, SIZE),
                'wind_velocity': torch.rand(SIZE, SIZE) * 10,
                'wind_towards_direction': torch.rand(SIZE, SIZE) * 365,
                'slope': torch.rand(SIZE, SIZE, 3, 3) * 90,
                'initial_ignition': torch.rand(SIZE, SIZE) > .9,
            }
            parameters = {
                'a': torch.tensor(.1),
                'p_h': torch.tensor(.3),
                'c_1': torch.tensor(.1),
                'c_2': torch.tensor(.1),
                'p_continue': torch.tensor(.3),
            }
            model = WildfireModel(env_data=environment_data, params=parameters) # Create a model with custom parameters
            ```
        """
        super(WildfireModel, self).__init__()

        env_data = {} if env_data is None else env_data
        params = {} if params is None else params
        self.keep_acc_mask = keep_acc_mask

        self.register_parameter('a', nn.Parameter(params.get('a', torch.tensor(.0))))
        self.register_parameter('p_h', nn.Parameter(params.get('p_h', torch.tensor(.3))))
        self.register_parameter('c_1', nn.Parameter(params.get('c_1', torch.tensor(.0))))
        self.register_parameter('c_2', nn.Parameter(params.get('c_2', torch.tensor(.0))))
        self.register_parameter('p_continue',
                                nn.Parameter(params.get('p_continue', torch.tensor(.3)), requires_grad=False))

        self.register_buffer('p_veg', env_data.get('p_veg', torch.zeros(DEFAULT_SIZE, DEFAULT_SIZE)))
        self.register_buffer('p_den', env_data.get('p_den', torch.zeros_like(self.p_veg)))
        self.register_buffer('wind_velocity', env_data.get('wind_velocity', torch.zeros_like(self.p_veg)))
        self.register_buffer('wind_towards_direction',
                             env_data.get('wind_towards_direction', torch.zeros_like(self.p_veg)))
        self.register_buffer('slope', env_data.get('slope', repeat(torch.zeros_like(self.p_veg), 'h w -> h w 3 3')))
        self.register_buffer('initial_ignition',
                             env_data.get('initial_ignition', torch.zeros_like(self.p_veg, dtype=torch.bool)))
        self.register_buffer('state', self._initialize_state(self.initial_ignition))
        if self.training:
            self.register_buffer('accumulator', self._initialize_accumulator(self.initial_ignition))
            if self.keep_acc_mask:
                self.register_buffer('accumulator_mask', self._initialize_accumulator_mask(self.accumulator))
        self.seed = self._initialize_seed()

        self.sanity_check()

    def sanity_check(self):
        """
        Perform sanity checks on the model.

        This method checks if all parameters and buffers have the correct shape and device.

        Raises:
            AssertionError: If the model is not properly initialized.

        Examples:
            ```python
            model = WildfireModel()
            model.sanity_check()
            ```
        """
        assert self.a.shape == self.p_h.shape == self.c_1.shape == self.c_2.shape == self.p_continue.shape == ()
        assert (
                self.p_veg.shape == self.p_den.shape == self.wind_velocity.shape == self.wind_towards_direction.shape == self.initial_ignition.shape)
        assert self.slope.shape == (*self.p_veg.shape, 3, 3)
        assert self.state.shape == (2, *self.p_veg.shape)
        assert (
                self.a.device == self.p_h.device == self.c_1.device == self.c_2.device == self.p_continue.device == self.p_veg.device == self.p_den.device == self.wind_velocity.device == self.wind_towards_direction.device == self.slope.device == self.initial_ignition.device == self.state.device)
        assert self.initial_ignition.dtype == self.state.dtype == torch.bool
        if self.training:
            assert self.accumulator.shape == self.initial_ignition.shape
            assert self.accumulator.device == self.state.device
            if self.keep_acc_mask:
                assert self.accumulator_mask.shape == self.accumulator.shape
                assert self.accumulator_mask.device == self.state.device
                assert self.accumulator_mask.dtype == torch.bool

    @staticmethod
    def _initialize_state(initial_ignition: torch.Tensor) -> torch.Tensor:
        return torch.stack((initial_ignition.clone(), torch.zeros_like(initial_ignition, dtype=torch.bool)), dim=0)

    @staticmethod
    def _initialize_accumulator(initial_ignition: torch.Tensor) -> torch.Tensor:
        return (initial_ignition.clone() * 1.0).requires_grad_(True)

    @staticmethod
    def _initialize_accumulator_mask(accumulator: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(accumulator, dtype=torch.bool)

    @staticmethod
    def _initialize_seed(seed: int = None) -> int:
        seed = seed if seed is not None else torch.Generator().seed()
        torch.manual_seed(seed)
        return seed

    def reset(self, seed: int = None):
        """
        Reset the model with a new seed.

        Parameters:
            seed (int):
                The seed for random number generator.
                If not provided, a new seed will be generated.

        Examples:
            ```python
            model = WildfireModel()
            model.reset(seed=123)
            ```
        """
        self.sanity_check()

        self.state = self._initialize_state(self.initial_ignition)
        if self.training:
            self.accumulator = self._initialize_accumulator(self.initial_ignition)
            if self.keep_acc_mask:
                self.accumulator_mask = self._initialize_accumulator_mask(self.accumulator)
        self.seed = self._initialize_seed(seed)

    def detach_accumulator(self):
        """
        Detach the accumulator from the computation graph.

        This method is used to detach the accumulator from the computation graph to avoid memory leak or out of memory.

        This method will not affect the model's behavior on simulation, and the accumulator can still be attached
        in the next step.

        Examples:
            ```python
            model = WildfireModel()
            model.compute(attach=True)
            model.detach_accumulator()
            model.compute(attach=True)
            ```
        """
        if self.training:
            self.accumulator = self.accumulator.detach().clone().requires_grad_(True)

    def p_ignite(self) -> torch.Tensor:
        """
        Calculate the probability of a cell igniting by its neighbors.

        The returned tensor is connected to the computation graph, and it can be used for back-propagation.

        Returns:
            The probability of a cell igniting by its neighbors.

                - dtype: `torch.float`
                - shape: `[Height, Width]`

        Examples:
            ```python
            model = WildfireModel()
            p_ignite = model.p_ignite()
            ```python
        """
        burning, _ = self.state

        p_s = torch.exp(self.a * self.slope)

        # to be used to calculate the angle between wind and fire direction
        wind_offset = torch.tensor([[3, 2, 1], [4, 0, 0], [5, 6, 7]], device=p_s.device) * 45

        wind_offset_tiled = repeat(wind_offset, 'c1 c2 -> 1 1 c1 c2', c1=3, c2=3)
        wind_towards_direction_expanded = repeat(self.wind_towards_direction, 'h w -> h w 1 1')
        wind_velocity_expanded = repeat(self.wind_velocity, 'h w -> h w 1 1')
        p_w = torch.exp(self.c_1 * wind_velocity_expanded) * torch.exp(self.c_2 * wind_velocity_expanded * (
                torch.cos(torch.deg2rad((wind_offset_tiled - wind_towards_direction_expanded) % 360)) - 1))

        p_propagate = repeat(self.p_h * (1 + self.p_veg) * (1 + self.p_den), 'h w -> h w 1 1') * p_s * p_w

        prob_like_act_c = 1.1486328125
        p_propagate = torch.tanh(prob_like_act_c * p_propagate)

        # out-of-bounds access in p_propagate is avoided by the slicing, and in state will result in 0
        p_out = torch.zeros_like(p_propagate)
        p_out[:-1, :-1, 0, 0] = torch.where(burning[1:, 1:], p_propagate[:-1, :-1, 0, 0], 0)
        p_out[:-1, :, 0, 1] = torch.where(burning[1:, :], p_propagate[:-1, :, 0, 1], 0)
        p_out[:-1, 1:, 0, 2] = torch.where(burning[1:, :-1], p_propagate[:-1, 1:, 0, 2], 0)
        p_out[:, :-1, 1, 0] = torch.where(burning[:, 1:], p_propagate[:, :-1, 1, 0], 0)
        p_out[:, 1:, 1, 2] = torch.where(burning[:, :-1], p_propagate[:, 1:, 1, 2], 0)
        p_out[1:, :-1, 2, 0] = torch.where(burning[:-1, 1:], p_propagate[1:, :-1, 2, 0], 0)
        p_out[1:, :, 2, 1] = torch.where(burning[:-1, :], p_propagate[1:, :, 2, 1], 0)
        p_out[1:, 1:, 2, 2] = torch.where(burning[:-1, :-1], p_propagate[1:, 1:, 2, 2], 0)
        p_out = 1 - reduce(1 - p_out, 'h w c1 c2 -> h w', 'prod', c1=3, c2=3)

        return p_out

    def compute(self, attach: bool = False):
        """
        Compute the next state of the cells.

        Parameters:
            attach (bool):
                If `true`, all newly ignited cells in current step will be attached to the accumulator.
        """

        burning, burned = self.state
        new_state = self.state.clone()
        new_burning, new_burned = new_state

        p_ignite = self.p_ignite()
        rand_propagate, rand_continue = torch.rand_like(p_ignite), torch.rand_like(p_ignite)

        # burnable cells have p_burn probability to become burning
        burnable = ~(burning | burned)
        new_burning_digits = torch.where(burnable, nn.functional.relu(p_ignite - rand_propagate), 0)
        new_burning_mask = new_burning_digits > 0

        if self.training:
            if attach:
                self.accumulator = self.accumulator + torch.where(new_burning_mask, p_ignite, 0)
                if self.keep_acc_mask:
                    self.accumulator_mask[new_burning_mask] = True
            else:
                self.accumulator = self.accumulator + torch.where(new_burning_mask, p_ignite.detach(), 0)
        new_burning[new_burning_mask] = True

        # burning cells have p_continue probability to continue burning
        will_burn_out_digits = torch.where(burning, nn.functional.relu(rand_continue - self.p_continue), 0)
        will_burn_out_mask = will_burn_out_digits > 0
        new_burning[will_burn_out_mask] = False
        new_burned[will_burn_out_mask] = True

        self.state = new_state
