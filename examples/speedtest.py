import argparse
import datetime
import timeit

import torch

from pytorchfire.model import WildfireModel


def run(map_size, device):
    seed = torch.Generator().seed()

    model = WildfireModel({
        'p_veg': torch.zeros(map_size, map_size),
        'wind_towards_direction': torch.ones(map_size, map_size) * 135,
        'wind_velocity': torch.ones(map_size, map_size) * 10,
        # 'slope': calculate_slope(alt_matrix, torch.tensor(1.)),
    }, {
        'a': torch.tensor(0.042),
        'p_h': torch.tensor(0.35),
        'p_continue': torch.tensor(.1),
        'c_1': torch.tensor(0.045),
        'c_2': torch.tensor(0.131),
    }).to(device)

    model.eval()

    center_x, center_y = model.initial_ignition.shape[0] // 2, model.initial_ignition.shape[1] // 2
    start_x, end_x = center_x - 1, center_x + 2
    start_y, end_y = center_y - 1, center_y + 2
    model.initial_ignition[start_x:end_x, start_y:end_y] = True

    model.reset(seed=seed)

    def code_to_test():
        for i in range(300):
            model.compute()

    execution_time = timeit.timeit(code_to_test, number=1)
    print(execution_time)

    current_time = datetime.datetime.now().strftime('%H%M%S')

    with open(f'{map_size}_{device}_{current_time}.txt', 'w') as file:
        file.write(str(execution_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiment.')
    parser.add_argument('--device', type=str, required=False, default='cuda:0')
    parser.add_argument('--size', type=int, required=True, default=500)

    args = parser.parse_args()
    run(args.size, args.device)
