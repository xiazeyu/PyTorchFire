# Wild Fire Simulation Using Cellular Automaton and Parallel Computing
# Designed By Xiaochi (George) Li
# Final Project for High Performance Computing and Parallel Computing
# Data Science @ George Washington University
# May. 2018

# Modified to maximum running time

# Algorithm based on this paper:A cellular automata model for forest fire spread prediction: The case
# of the wildfire that swept through Spetses Island in 1990
# Author: A. Alexandridis a, D. Vakalis b, C.I. Siettos c,*, G.V. Bafas a

import argparse
import copy
import math
import random
import timeit

from mpi4py import MPI


def run(size):
    global sub_forest

    n_row_total = size
    n_col = size

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    stat = MPI.Status()

    # ------------ Change the code below here for different initial environment ---------

    # -------------Quick Change Begin-------------------------
    # number of rows and columns of full grid and the generations
    generation = 300

    # the possibility a cell will continue to burn in next time step
    # change the value to change the boundary of fire
    p_continue_burn = 0.1

    # Quick switch for factors in the model, turn on: True, turn off: False
    wind = True
    vegetation = False
    density = False
    altitude = False

    # ------------- Quick Change End---------------------------------------

    n_row = n_row_total // size + 2

    thetas = [[180, 135, 90],
              [225, 0, 45],
              [270, 315, 0]]

    def init_vegetation():
        veg_matrix = [[0 for col in range(n_col)] for row in range(n_row)]
        if vegetation == False:  # turn off vegetation
            for i in range(n_row):
                for j in range(n_col):
                    veg_matrix[i][j] = 2
        else:
            for i in range(n_row):
                for j in range(n_col):
                    if j <= n_col // 3:
                        veg_matrix[i][j] = 1
                    elif j <= n_col * 2 // 3:
                        veg_matrix[i][j] = 2
                    else:
                        veg_matrix[i][j] = 3
        return veg_matrix

    def init_density():
        den_matrix = [[0 for col in range(n_col)] for row in range(n_row)]
        if density == False:  # turn off density
            for i in range(n_row):
                for j in range(n_col):
                    den_matrix[i][j] = 2
        else:
            for i in range(n_row):
                for j in range(n_col):
                    if j <= n_col // 3:
                        den_matrix[i][j] = 1
                    elif j <= n_col * 2 // 3:
                        den_matrix[i][j] = 2
                    else:
                        den_matrix[i][j] = 3
        return den_matrix

    def init_altitude():
        alt_matrix = [[0 for col in range(n_col)] for row in range(n_row)]
        if altitude == False:  # turn off altitude
            for i in range(n_row):
                for j in range(n_col):
                    alt_matrix[i][j] = 1
        else:
            for i in range(n_row):
                for j in range(n_col):
                    alt_matrix[i][j] = j
        return alt_matrix

    def init_forest():
        forest = [[1 for col in range(n_col)] for row in range(n_row)]

        for i in range(1, n_row - 1):
            for j in range(1, n_col - 1):
                forest[i][j] = 2

        ignite_col = n_col // 2
        ignite_row = n_row // 2
        if rank == size // 2:
            for row in range(ignite_row - 1, ignite_row + 1):
                for col in range(ignite_col - 1, ignite_col + 1):
                    forest[row][col] = 3

        return forest

    # ------------------ Do not change anything below this line ----------------

    # ------------------ Parallel Function ----------------
    def msg_up(sub_grid):
        # Sends and Receives rows with Rank+1
        comm.send(sub_grid[n_row - 2], dest=rank + 1)
        sub_grid[n_row - 1] = comm.recv(source=rank + 1)
        return 0

    def msg_down(sub_grid):
        # Sends and Receives rows with Rank-1
        comm.send(sub_grid[1], dest=rank - 1)
        sub_grid[0] = comm.recv(source=rank - 1)
        return 0

    # ------------------ Parallel Function End ----------------

    def tg(x):
        return math.degrees(math.atan(x))

    def get_slope(altitude_matrix):
        slope_matrix = [[0 for col in range(n_col)] for row in range(n_row)]
        for row in range(n_row):
            for col in range(n_col):
                sub_slope_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                if row == 0 or row == n_row - 1 or col == 0 or col == n_col - 1:  # margin is flat
                    slope_matrix[row][col] = sub_slope_matrix
                    continue
                current_altitude = altitude_matrix[row][col]
                sub_slope_matrix[0][0] = tg((current_altitude - altitude_matrix[row - 1][col - 1]) / 1.414)
                sub_slope_matrix[0][1] = tg(current_altitude - altitude_matrix[row - 1][col])
                sub_slope_matrix[0][2] = tg((current_altitude - altitude_matrix[row - 1][col + 1]) / 1.414)
                sub_slope_matrix[1][0] = tg(current_altitude - altitude_matrix[row][col - 1])
                sub_slope_matrix[1][1] = 0
                sub_slope_matrix[1][2] = tg(current_altitude - altitude_matrix[row][col + 1])
                sub_slope_matrix[2][0] = tg((current_altitude - altitude_matrix[row + 1][col - 1]) / 1.414)
                sub_slope_matrix[2][1] = tg(current_altitude - altitude_matrix[row + 1][col])
                sub_slope_matrix[2][2] = tg((current_altitude - altitude_matrix[row + 1][col + 1]) / 1.414)
                slope_matrix[row][col] = sub_slope_matrix
        return slope_matrix

    def calc_pw(theta):
        c_1 = 0.045
        c_2 = 0.131
        V = 10
        t = math.radians(theta)
        ft = math.exp(V * c_2 * (math.cos(t) - 1))
        return math.exp(c_1 * V) * ft

    def get_wind():

        wind_matrix = [[0 for col in [0, 1, 2]] for row in [0, 1, 2]]

        for row in [0, 1, 2]:
            for col in [0, 1, 2]:
                wind_matrix[row][col] = calc_pw(thetas[row][col])
        wind_matrix[1][1] = 0

        if wind == False:  # turn off wind
            wind_matrix = [[1 for col in [0, 1, 2]] for row in [0, 1, 2]]
        return wind_matrix

    def burn_or_not_burn(abs_row, abs_col, neighbour_matrix):
        p_veg = {1: -0.3, 2: 0, 3: 0.4}[vegetation_matrix[abs_row][abs_col]]
        p_den = {1: -0.4, 2: 0, 3: 0.3}[density_matrix[abs_row][abs_col]]
        p_h = 0.35
        a = 0.042

        for row in [0, 1, 2]:
            for col in [0, 1, 2]:
                if neighbour_matrix[row][col] == 3:  # we only care there is a neighbour that is burning
                    # print(row,col)
                    slope = slope_matrix[abs_row][abs_col][row][col]
                    p_slope = math.exp(a * slope)
                    p_wind = wind_matrix[row][col]
                    p_burn = p_h * (1 + p_veg) * (1 + p_den) * p_wind * p_slope
                    if p_burn > random.random():
                        return 3  # start burning

        return 2  # not burning

    def update_forest(old_forest):
        result_forest = [[1 for i in range(n_col)] for j in range(n_row)]
        for row in range(1, n_row - 1):
            for col in range(1, n_col - 1):

                if old_forest[row][col] == 1 or old_forest[row][col] == 4:
                    result_forest[row][col] = old_forest[row][col]  # no fuel or burnt down
                if old_forest[row][col] == 3:
                    if random.random() < p_continue_burn:
                        result_forest[row][col] = 3  # We can change here to control the burning time
                    else:
                        result_forest[row][col] = 4
                if old_forest[row][col] == 2:
                    neighbours = [[row_vec[col_vec] for col_vec in range(col - 1, col + 2)]
                                  for row_vec in old_forest[row - 1:row + 2]]
                    # print(neighbours)
                    result_forest[row][col] = burn_or_not_burn(row, col, neighbours)
        return result_forest

    vegetation_matrix = init_vegetation()
    density_matrix = init_density()
    altitude_matrix = init_altitude()
    wind_matrix = get_wind()
    slope_matrix = get_slope(altitude_matrix)
    sub_forest = init_forest()  # [parallel] each worker has their own sub grid

    # start simulation

    def code_to_test():
        global sub_forest
        for i in range(generation):
            sub_forest = copy.deepcopy(update_forest(sub_forest))
            # [parallel] message passing function
            if rank == 0:
                msg_up(sub_forest)
            elif rank == size - 1:
                msg_down(sub_forest)
            else:
                msg_up(sub_forest)
                msg_down(sub_forest)
            temp_grid = comm.gather(sub_forest[1:n_row - 1], root=0)

    execution_time = timeit.timeit(code_to_test, number=1)
    print(execution_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiment.')
    parser.add_argument('--size', type=int, required=True, default=500)

    args = parser.parse_args()
    run(args.size)
