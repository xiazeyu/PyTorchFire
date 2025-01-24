import torch
from einops import repeat, rearrange
from torch import nn


def convert_wind_components_to_velocity_and_direction(wind_u: torch.Tensor,
                                                      wind_v: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Convert wind components to velocity and direction.

    Parameters:
        wind_u (torch.Tensor):
            Wind component u. (Eastward wind component)

            - dtype: `torch.float`
            - shape: `[Height, Width]`

        wind_v (torch.Tensor):
            Wind component v. (Northward wind component)

            - dtype: `torch.float`
            - shape: `[Height, Width]`

    Returns:
        A dictionary containing the following

            - `wind_velocity` (`torch.Tensor`): Wind velocity. (m/s)
                - dtype: `torch.float`
                - shape: `[Height, Width]`
            - `wind_towards_direction` (`torch.Tensor`): Wind direction. (degrees, starting from East and going
                    counterclockwise)
                - dtype: `torch.float`
                - shape: `[Height, Width]`

    Examples:
        ```python
        wind_u = torch.rand(3, 3)
        wind_v = torch.rand(3, 3)
        convert_wind_components_to_velocity_and_direction(wind_u, wind_v)
        ```

    Raises:
        AssertionError: If the shapes of `wind_u` and `wind_v` are not the same.
    """
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

    return {
        'wind_velocity': wind_velocity,
        'wind_towards_direction': wind_towards_direction,
    }


def calculate_slope(altitude: torch.Tensor, cell_size: torch.Tensor) -> torch.Tensor:
    """
    Calculate the slope of the terrain.

    Parameters:
        altitude (torch.Tensor):
            Altitude of the terrain. (m)
            - dtype: `torch.float`
            - shape: `[Height, Width]`

        cell_size (torch.Tensor):
            Size of the cell. (m)
            - dtype: `torch.float`
            - shape: `[]`

    Returns:
        Slope of the terrain. (degrees)
            - dtype: `torch.float`
            - shape: `[Height, Width]`

    Examples:
        ```python
        altitude = torch.rand(3, 3)
        cell_size = torch.tensor(1.0)
        calculate_slope(altitude, cell_size)
        ```

    Raises:
        AssertionError: If the shape of `cell_size` is not `[]`.
    """
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

def jaccard_index(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    # Ensure the inputs are boolean
    assert y_true.dtype == y_pred.dtype == torch.bool

    # Calculate intersection and union
    intersection = torch.sum(y_true & y_pred).float()
    union = torch.sum(y_true | y_pred).float()

    # Compute Jaccard Index
    jaccard = intersection / union

    return jaccard.item()


def manhattan_distance(tensor1: torch.Tensor, tensor2: torch.Tensor):
    # Ensure the tensors are of the same shape
    assert tensor1.shape == tensor2.shape

    # Compute the absolute differences
    abs_diff = torch.abs(tensor1 - tensor2)

    # Sum the absolute differences
    manhattan_dist = torch.sum(abs_diff)

    return manhattan_dist.item()
