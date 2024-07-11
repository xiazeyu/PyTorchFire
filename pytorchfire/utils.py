import torch
from einops import repeat, rearrange
from torch import nn


def convert_wind_components_to_velocity_and_direction(wind_u: torch.Tensor,
                                                      wind_v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert wind_u.shape == wind_v.shape

    wind_u = wind_u.clone()
    wind_v = wind_v.clone()
    # Wind data directs to where air moving towards.
    # Velocity in m/s
    wind_u[torch.isnan(wind_u)] = 0
    wind_v[torch.isnan(wind_v)] = 0

    wind_velocity = torch.sqrt(wind_u ** 2 + wind_v ** 2)
    wind_towards_direction = (torch.rad2deg(
        torch.arctan2(wind_v, wind_u)) + 360) % 360  # starting from East and going counterclockwise in degrees

    return wind_velocity, wind_towards_direction


def calculate_slope(altitude: torch.Tensor, cell_size: torch.Tensor) -> torch.Tensor:
    assert cell_size.shape == ()

    altitude = repeat(altitude, 'h w -> 1 1 h w')
    altitude = nn.functional.pad(altitude, (1, 1, 1, 1), mode='replicate')
    kernels = repeat(torch.tensor([
        [[1., 0., 0.],
         [0., -1., 0.],
         [0., 0., 0.]],  # to top-left
        [[0., 1., 0.],
         [0., -1., 0.],
         [0., 0., 0.]],  # to top
        [[0., 0., 1.],
         [0., -1., 0.],
         [0., 0., 0.]],  # to top-right
        [[0., 0., 0.],
         [1., -1., 0.],
         [0., 0., 0.]],  # to left
        [[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]],  # to self
        [[0., 0., 0.],
         [0., -1., 1.],
         [0., 0., 0.]],  # to right
        [[0., 0., 0.],
         [0., -1., 0.],
         [1., 0., 0.]],  # to bottom-left
        [[0., 0., 0.],
         [0., -1., 0.],
         [0., 1., 0.]],  # to bottom
        [[0., 0., 0.],
         [0., -1., 0.],
         [0., 0., 1.]],  # to bottom-right
    ]), 'c h w -> c 1 h w')

    diffs = rearrange(nn.functional.conv2d(altitude, kernels), '1 c h w -> c h w')

    diffs[[1, 3, 5, 7]] /= cell_size
    diffs[[0, 2, 4, 8]] /= (cell_size * torch.sqrt(torch.tensor(2.)))

    slope = rearrange(torch.rad2deg(torch.arctan(diffs)), '(a b) h w -> h w a b', a=3, b=3)

    return slope
