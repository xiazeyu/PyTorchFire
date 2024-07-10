import torch
from einops import repeat, reduce
from torch import nn

KEEP_ACC_MASK = False
DEFAULT_DEVICE = torch.device('cpu')
DEFAULT_DTYPE = torch.float32
DEFAULT_SIZE = 500


class WildfireModel(nn.Module):
    def __init__(self, data: dict = None, params: dict = None):
        super(WildfireModel, self).__init__()
        if data is None:
            data = {}
        if params is None:
            params = {}

        self.p_veg = data.get('p_veg',
                              torch.zeros(DEFAULT_SIZE, DEFAULT_SIZE, dtype=torch.float32, device=DEFAULT_DEVICE))
        self.p_den = data.get('p_den', torch.zeros_like(self.p_veg))
        self.wind_V = data.get('wind_V', torch.zeros_like(self.p_veg))
        self.wind_towards_direction = data.get('wind_towards_direction', torch.zeros_like(self.p_veg))
        self.slope = data.get('slope', torch.zeros_like(self.p_veg))
        self.initial_ignition = data.get('initial_ignition', torch.zeros_like(self.p_veg, dtype=torch.bool))

        self.parameter_dict = nn.ParameterDict({'a': nn.Parameter(params.get('a', torch.tensor(.0))),
                                                'p_h': nn.Parameter(params.get('p_h', torch.tensor(.3))),
                                                'c_1': nn.Parameter(params.get('c_1', torch.tensor(.0))),
                                                'c_2': nn.Parameter(params.get('c_2', torch.tensor(.0))),
                                                'p_continue': nn.Parameter(params.get('p_continue', torch.tensor(.3)),
                                                                           requires_grad=False), })

        self.state = self._initialize_state(self.initial_ignition)
        self.accumulator = self._initialize_accumulator(self.initial_ignition)
        if KEEP_ACC_MASK:
            self.accumulator_mask = self._initialize_accumulator_mask(self.accumulator)
        self.seed = self._initialize_seed()

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
        self.state = self._initialize_state(self.initial_ignition)
        self.accumulator = self._initialize_accumulator(self.initial_ignition)
        if KEEP_ACC_MASK:
            self.accumulator_mask = self._initialize_accumulator_mask(self.accumulator)
        self.seed = self._initialize_seed(seed)

    def detach_accumulator(self):
        self.accumulator = self.accumulator.detach().clone().requires_grad_(True)

    def p_ignite(self) -> torch.Tensor:
        burning, _ = self.state

        p_s = torch.exp(self.parameter_dict.a * torch.deg2rad(self.slope))

        # to be used to calculate the angle between wind and fire direction
        wind_offset = torch.tensor([[3, 2, 1], [4, 0, 0], [5, 6, 7]], device=p_s.device) * 45

        wind_offset_tiled = repeat(wind_offset, 'c1 c2 -> 1 1 c1 c2')
        wind_towards_direction_expanded = repeat(self.wind_towards_direction, 'h w -> h w 1 1')
        wind_V_expanded = repeat(self.wind_V, 'h w -> h w 1 1')
        p_w = torch.exp(self.parameter_dict.c_1 * wind_V_expanded) * torch.exp(
            self.parameter_dict.c_2 * wind_V_expanded * (
                    torch.cos(torch.deg2rad((wind_offset_tiled - wind_towards_direction_expanded) % 360)) - 1))

        p_propagate = repeat(self.parameter_dict.p_h * (1 + self.p_veg) * (1 + self.p_den) * p_s,
                             'h w -> h w 1 1') * p_w

        prob_act_c = 1.1486328125
        p_propagate = torch.tanh(prob_act_c * p_propagate)

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
        new_burning_digits = torch.where(burnable, nn.ReLU()(p_ignite - rand_propagate), 0)
        new_burning_mask = new_burning_digits > 0

        if attach:
            self.accumulator = self.accumulator + torch.where(new_burning_mask, p_ignite, 0)
            if KEEP_ACC_MASK:
                self.accumulator_mask[new_burning_mask] = True
        else:
            self.accumulator = self.accumulator + torch.where(new_burning_mask, p_ignite.detach(), 0)
        new_burning[new_burning_mask] = True

        # burning cells have p_continue probability to continue burning
        will_burn_out_digits = torch.where(burning, nn.ReLU()(rand_continue - self.parameter_dict.p_continue), 0)
        will_burn_out_mask = will_burn_out_digits > 0
        new_burning[will_burn_out_mask] = False
        new_burned[will_burn_out_mask] = True

        self.state = new_state
