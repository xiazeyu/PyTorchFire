import torch
from einops import repeat, reduce
from torch import nn

DEFAULT_SIZE = 500


# noinspection PyAttributeOutsideInit
class WildfireModel(nn.Module):
    """
    Wildfire model.

    Parameters:
        env_data:
            The environment data.
        params:
            The parameters.
        keep_acc_mask:
            Whether to keep the accumulator mask.

    Examples:
        >>> model = WildfireModel()

    Shape:
        - Input: :math:`(N, H, W)`.
        - Output: :math:`(N, H, W)`.
    """
    def __init__(self, env_data: dict = None, params: dict = None, keep_acc_mask: bool = False):
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
        assert self.a.shape == self.p_h.shape == self.c_1.shape == self.c_2.shape == self.p_continue.shape == ()
        assert (self.p_veg.shape == self.p_den.shape == self.wind_velocity.shape ==
                self.wind_towards_direction.shape == self.initial_ignition.shape)
        assert self.slope.shape == (*self.p_veg.shape, 3, 3)
        assert self.state.shape == (2, *self.p_veg.shape)
        assert (self.a.device == self.p_h.device == self.c_1.device == self.c_2.device == self.p_continue.device ==
                self.p_veg.device == self.p_den.device == self.wind_velocity.device ==
                self.wind_towards_direction.device == self.slope.device == self.initial_ignition.device ==
                self.state.device)
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
        self.sanity_check()

        self.state = self._initialize_state(self.initial_ignition)
        if self.training:
            self.accumulator = self._initialize_accumulator(self.initial_ignition)
            if self.keep_acc_mask:
                self.accumulator_mask = self._initialize_accumulator_mask(self.accumulator)
        self.seed = self._initialize_seed(seed)

    def detach_accumulator(self):
        if self.training:
            self.accumulator = self.accumulator.detach().clone().requires_grad_(True)

    def p_ignite(self) -> torch.Tensor:
        burning, _ = self.state

        p_s = torch.exp(self.a * self.slope)

        # to be used to calculate the angle between wind and fire direction
        wind_offset = torch.tensor([[3, 2, 1], [4, 0, 0], [5, 6, 7]], device=p_s.device) * 45

        wind_offset_tiled = repeat(wind_offset, 'c1 c2 -> 1 1 c1 c2', c1=3, c2=3)
        wind_towards_direction_expanded = repeat(self.wind_towards_direction, 'h w -> h w 1 1')
        wind_velocity_expanded = repeat(self.wind_velocity, 'h w -> h w 1 1')
        p_w = torch.exp(self.c_1 * wind_velocity_expanded) * torch.exp(
            self.c_2 * wind_velocity_expanded * (
                    torch.cos(torch.deg2rad((wind_offset_tiled - wind_towards_direction_expanded) % 360)) - 1))

        p_propagate = repeat(self.p_h * (1 + self.p_veg) * (1 + self.p_den),
                             'h w -> h w 1 1') * p_s * p_w

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
        will_burn_out_digits = torch.where(burning, nn.functional.relu(rand_continue - self.p_continue),
                                           0)
        will_burn_out_mask = will_burn_out_digits > 0
        new_burning[will_burn_out_mask] = False
        new_burned[will_burn_out_mask] = True

        self.state = new_state

    def forward(self, attach: bool = False) -> torch.Tensor:
        self.compute(attach)
        if self.training:
            return self.accumulator
        else:
            return self.state
