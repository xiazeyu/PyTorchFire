import torch
from einops import repeat, reduce
from torch import nn


class WildfireModel(nn.Module):
    def __init__(self, model_config: dict, params: dict):
        super(WildfireModel, self).__init__()
        self.parameter_dict = nn.ParameterDict(params)

        self.canopy = model_config.get('canopy')
        self.wind_V = model_config.get('wind_V', torch.zeros_like(self.canopy))
        self.wind_towards_direction = model_config.get('wind_towards_direction', torch.zeros_like(self.canopy))
        self.slope = model_config.get('slope', torch.zeros_like(self.canopy))
        self.density = model_config.get('density', torch.zeros_like(self.canopy))
        self.initial_fire = model_config.get('initial_fire', torch.zeros_like(self.canopy, dtype=torch.bool))

        self.fire_state: tuple[torch.Tensor, torch.Tensor] | None = None
        self.accumulator: torch.Tensor | None = None
        self.accumulator_mask: torch.Tensor | None = None
        self.seed: int | None = None

        self.reset()

    def detach_all(self):
        self.accumulator = self.accumulator.detach().clone().requires_grad_(True)
        self.accumulator_mask = self.accumulator_mask.detach().clone()

    def reset(self, seed: int = None):
        self.fire_state = (self.initial_fire.clone(), torch.zeros_like(self.canopy, dtype=torch.bool))
        self.accumulator = (self.initial_fire.clone() * 1.0).requires_grad_(True)
        self.accumulator_mask = torch.zeros_like(self.accumulator, dtype=torch.bool)
        self.seed = seed if seed is not None else torch.Generator().seed()
        torch.manual_seed(self.seed)

    def p_burn(self) -> torch.Tensor:

        burning, _ = self.fire_state

        p_veg = (self.canopy / 100.0) - 0.4  # todo: resolve magic numbers, [0, 100] -> [-0.4, 0.6]
        p_den = (self.density / 34) - 0.5  # todo: resolve magic numbers, [0, 34(?)] -> [-0.5, 0.5]
        p_s = torch.exp(self.parameter_dict.a * torch.deg2rad(self.slope))  # a*[0, 1.5708]
        p_propagate_constant = self.parameter_dict.p_h * (1 + p_veg) * (1 + p_den) * p_s

        # to be used to calculate the angle between wind and fire direction
        wind_offset = torch.tensor([[3, 2, 1],
                                    [4, 0, 0],
                                    [5, 6, 7]],
                                   device=p_s.device) * 45

        wind_offset_tiled = repeat(wind_offset, 'c1 c2 -> 1 1 c1 c2')
        wind_towards_direction_expanded = repeat(self.wind_towards_direction, 'h w -> h w 1 1')
        wind_V_expanded = repeat(self.wind_V, 'h w -> h w 1 1')
        p_w = torch.exp(self.parameter_dict.c_1 * wind_V_expanded) * torch.exp(
            self.parameter_dict.c_2 * wind_V_expanded * (
                    torch.cos(torch.deg2rad((wind_offset_tiled - wind_towards_direction_expanded) % 360)) - 1))

        p_propagate = repeat(p_propagate_constant, 'h w -> h w 1 1') * p_w

        tanh_c = 1.1486328125
        p_propagate = torch.tanh(tanh_c * p_propagate)

        # out-of-bounds access in p_propagate is avoided by the slicing, and in fire_state will result in 0

        p_burn = torch.zeros_like(p_propagate)
        p_burn[:-1, :-1, 0, 0] = torch.where(burning[1:, 1:], p_propagate[:-1, :-1, 0, 0], 0)
        p_burn[:-1, :, 0, 1] = torch.where(burning[1:, :], p_propagate[:-1, :, 0, 1], 0)
        p_burn[:-1, 1:, 0, 2] = torch.where(burning[1:, :-1], p_propagate[:-1, 1:, 0, 2], 0)
        p_burn[:, :-1, 1, 0] = torch.where(burning[:, 1:], p_propagate[:, :-1, 1, 0], 0)
        p_burn[:, 1:, 1, 2] = torch.where(burning[:, :-1], p_propagate[:, 1:, 1, 2], 0)
        p_burn[1:, :-1, 2, 0] = torch.where(burning[:-1, 1:], p_propagate[1:, :-1, 2, 0], 0)
        p_burn[1:, :, 2, 1] = torch.where(burning[:-1, :], p_propagate[1:, :, 2, 1], 0)
        p_burn[1:, 1:, 2, 2] = torch.where(burning[:-1, :-1], p_propagate[1:, 1:, 2, 2], 0)
        p_burn = 1 - reduce(1 - p_burn, 'h w c1 c2 -> h w', 'prod', c1=3, c2=3)

        return p_burn

    def compute(self, attach: bool = False):  # -> tuple[torch.Tensor, torch.Tensor]:

        burning, burned = self.fire_state
        burning_out, burned_out = burning.clone(), burned.clone()
        p_burn = self.p_burn()
        rand_propagate, rand_continue = torch.rand_like(p_burn), torch.rand_like(p_burn)

        # burnable patches have p_burn probability to become burning
        burnable = ~(burning | burned)
        new_burning_digits = torch.where(burnable, nn.ReLU()(p_burn - rand_propagate), 0)
        new_burning = new_burning_digits > 0

        if attach:
            self.accumulator = self.accumulator + torch.where(new_burning, p_burn, 0)
            self.accumulator_mask[new_burning] = True
        else:
            self.accumulator = self.accumulator + torch.where(new_burning, p_burn.detach(), 0)
        burning_out[new_burning] = True

        # burning patches have p_continue_burn probability to continue burning
        will_burn_out_digits = torch.where(burning, nn.ReLU()(rand_continue - self.parameter_dict.p_continue_burn), 0)
        will_burn_out = will_burn_out_digits > 0
        burning_out[will_burn_out] = False
        burned_out[will_burn_out] = True

        self.fire_state = (burning_out, burned_out)

        # return new_burning_digits, will_burn_out_digits
