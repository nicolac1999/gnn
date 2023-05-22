import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from lib.diag_common.numpy_helpers import load_numpy_dict
from subsystems.heat_flux.utils.plotting import plot_timber_T_vs_simulated_T

# %% CONFIGURATION BLOCK
# -----> values read from Timber

MAGNETS_TEMPERATURES_0W = [1.948848367, 1.950351834, 1.946064234, 1.947808027, 1.947463989, 1.947883844, 1.946063399,
                           1.94485116]
MAGNETS_AVG_0W = np.mean(MAGNETS_TEMPERATURES_0W)
MAGNETS_TEMPERATURES_650W = [1.949618526, 1.95105207, 1.945541739, 1.944453597, 1.936847329, 1.934995896, 1.910495024,
                             1.892742991]
MAGNETS_AVG_650W = np.mean(MAGNETS_TEMPERATURES_650W)
MAGNETS_TEMPERATURES_700w = [1.949453831, 1.950561047, 1.945541739, 1.944090605, 1.934944987, 1.93243885, 1.901815791,
                             1.887179732]
MAGNETS_AVG_700W = np.mean(MAGNETS_TEMPERATURES_700w)

ALL_MAGNETS = [MAGNETS_TEMPERATURES_0W, MAGNETS_TEMPERATURES_650W, MAGNETS_TEMPERATURES_700w]
ALL_MAGNETS_AVG = [MAGNETS_AVG_0W, MAGNETS_AVG_650W, MAGNETS_AVG_700W]

HEATERS_POWER_0 = [0., 0., 0., 0., 0., 0., 0., 0.]
HEATERS_POWER_650 = [4.113879681, 4.128593445, 0, 4.151591301, 4.254763126, 4.15112114, 0, 4.297861197]
HEATERS_POWER_700 = [4.702946186, 4.720263481, 0, 4.737853036, 4.856591104, 4.757156372, 0, 4.908631802]

ALL_HEATERS_POWER = [0, 4.113879681, 4.702946186]
ALL_HEATERS_POWER_per_m = [np.sum(HEATERS_POWER_0) / 107,
                           np.sum(HEATERS_POWER_650) / 107,
                           np.sum(HEATERS_POWER_700) / 107]

# %%
# ----->  plot setup
SENSORS_LOCATION = [1, 4, 9, 14, 19, 22, 27, 32]
SENSOR_ERROR = 0.005

# %%
# ------> Predictions no heat incoming from side 60 cm2
file_path_0W = 'results/heat/simulations/0W/Standard_Cell_Feb_0W_wl=5.1.npz'
file_path_650W = 'results/heat/simulations/650W/Standard_Cell_Feb_650W_wl=15.3.npz'
file_path_700W = 'results/heat/simulations/700W/Standard_Cell_Feb_700W_wl=20.1.npz'

sim_0 = load_numpy_dict(file_path_0W)  # < ------ this value stays the same for both simulations with and without
                                       # heat incoming from adjacent cell, because we assume the valve of the adjacent
                                       # cell is opened such that the static heat is extracted in each cell independently

sim_650 = load_numpy_dict(file_path_650W)
sim_700 = load_numpy_dict(file_path_700W)

T_pred_0W = sim_0['cells__T'][-1][SENSORS_LOCATION]
T_pred_650W = sim_650['cells__T'][-1][SENSORS_LOCATION]
T_pred_7000W = sim_700['cells__T'][-1][SENSORS_LOCATION]

ALL_MAGNETS_PREDICTIONS = [T_pred_0W, T_pred_650W, T_pred_7000W]

title = 'Comparison sensors reading data and model simulation ( NO HEAT FROM SIDE ) '
plot_timber_T_vs_simulated_T(ALL_HEATERS_POWER, ALL_MAGNETS, ALL_MAGNETS_PREDICTIONS, title=title)

# %%
# ------>  Prediction with heat incoming from adj cell

file_path_650W_plus_adj = 'results/heat/simulations/650W+adj/Standard_Cell_Feb_650W_+adj_cell_wl=20.1.npz'
file_path_700W_plus_adj = 'results/heat/simulations/700W+adj/Standard_Cell_Feb_700W_+adj_cell_wl=24.9.npz'

sim_650_plus_adj = load_numpy_dict(file_path_650W_plus_adj)
sim_700_plus_adj = load_numpy_dict(file_path_700W_plus_adj)

T_pred_650W_plus_adj = sim_650_plus_adj['cells__T'][-1][SENSORS_LOCATION]
T_pred_7000W_plus_adj = sim_700_plus_adj['cells__T'][-1][SENSORS_LOCATION]

ALL_MAGNETS_PREDICTIONS_plus_adj = [T_pred_0W, T_pred_650W_plus_adj, T_pred_7000W_plus_adj]

title = 'Comparison sensors reading data and model simulation ( + HEAT FROM SIDE ) '
plot_timber_T_vs_simulated_T(ALL_HEATERS_POWER, ALL_MAGNETS, ALL_MAGNETS_PREDICTIONS_plus_adj, title=title)


# %%
# -------> Predictions no heat incoming from side, 66 cm2

file_path_0W_66cm2 = 'results/heat/simulations/0W_66cm2/Standard_Cell_Feb_0W_66cm2_wl=5.1.npz'
file_path_650W_66cm2 = 'results/heat/simulations/650W_66cm2/Standard_Cell_Feb_650W_66cm2_wl=15.3.npz'
file_path_700W_66cm2 = 'results/heat/simulations/700W_66cm2/Standard_Cell_Feb_700W_66cm2_wl=20.1.npz'


sim_0_66cm2 = load_numpy_dict(file_path_0W_66cm2)
sim_650_66cm2 = load_numpy_dict(file_path_650W_66cm2)
sim_700_66cm2 = load_numpy_dict(file_path_700W_66cm2)


T_pred_0W_66cm2 = sim_0_66cm2['cells__T'][-1][SENSORS_LOCATION]
T_pred_650W_66cm2 = sim_650_66cm2['cells__T'][-1][SENSORS_LOCATION]
T_pred_7000W_66cm2 = sim_700_66cm2['cells__T'][-1][SENSORS_LOCATION]

ALL_MAGNETS_PREDICTIONS_66cm2 = [T_pred_0W_66cm2, T_pred_650W_66cm2, T_pred_7000W_66cm2]

title = 'Comparison sensors reading data and model simulation ( NO HEAT FROM SIDE, 66cm2 interconnections ) '
plot_timber_T_vs_simulated_T(ALL_HEATERS_POWER, ALL_MAGNETS, ALL_MAGNETS_PREDICTIONS_66cm2, title=title)


# %%
# -------> Predictions with heat from side, 66 cm2

file_path_650W_66cm2_plus_adj = 'results/heat/simulations/650W+adj_66cm2/Standard_Cell_Feb_650W+adj_66cm2_wl=20.1.npz'
file_path_700W_66cm2_plus_adj = 'results/heat/simulations/700W+adj_66cm2/Standard_Cell_Feb_700W+adj_66cm2_wl=24.9.npz'


sim_650_66cm2_plus_adj = load_numpy_dict(file_path_650W_66cm2_plus_adj)
sim_700_66cm2_plus_adj = load_numpy_dict(file_path_700W_66cm2_plus_adj)


T_pred_650W_66cm2_plus_adj = sim_650_66cm2_plus_adj['cells__T'][-1][SENSORS_LOCATION]
T_pred_7000W_66cm2_plus_adj = sim_700_66cm2_plus_adj['cells__T'][-1][SENSORS_LOCATION]


ALL_MAGNETS_PREDICTIONS_66cm2_plus_adj = [T_pred_0W_66cm2, T_pred_650W_66cm2_plus_adj, T_pred_7000W_66cm2_plus_adj]

title = 'Comparison sensors reading data and model simulation ( + HEAT FROM SIDE, 66cm2 interconnections ) '
plot_timber_T_vs_simulated_T(ALL_HEATERS_POWER, ALL_MAGNETS, ALL_MAGNETS_PREDICTIONS_66cm2_plus_adj, title=title)



# %% ----> plot wetted length temperatures

# ============> 650 exp <==========
file_path_650W_15_3 = 'results/heat/simulations/650W_66cm2/Standard_Cell_Feb_650W_66cm2_wl=15.3.npz'
file_path_650W_20_1 = 'results/heat/simulations/650W_66cm2/Standard_Cell_Feb_650W_66cm2_wl=20.1.npz'


sim_650_15_3 = load_numpy_dict(file_path_650W_15_3)
sim_650_20_1 = load_numpy_dict(file_path_650W_20_1)


T_pred_650W_15_3 = sim_650_15_3['cells__T'][-1][SENSORS_LOCATION]
T_pred_650W_20_1 = sim_650_20_1['cells__T'][-1][SENSORS_LOCATION]

temperatures = [T_pred_650W_15_3, T_pred_650W_20_1]
plt.figure(figsize=(10, 10))
wetted_lengths = [[15.3], [20.1]]

for i in range(len(wetted_lengths)):
    plt.plot(wetted_lengths[i] * len(SENSORS_LOCATION), temperatures[i], 'x')

x = [[15.3, 20.1]] * len(SENSORS_LOCATION)
y = [[T_pred_650W_15_3[i], T_pred_650W_20_1[i]] for i in range(len(SENSORS_LOCATION))]
for i in range(len(SENSORS_LOCATION)):
    plt.plot(x[i], y[i], marker='o')

plt.grid()
plt.show()
colors = plt.cm.tab20

new_x_650 = []
x_single = np.array([15.3, 20.1])
for i in range(len(SENSORS_LOCATION)):
    y_single = np.array([T_pred_650W_15_3[i], T_pred_650W_20_1[i]])
    A = np.vstack([x_single, np.ones(len(x_single))]).T
    m, c = np.linalg.lstsq(A, y_single, rcond=None)[0]
    new_x = (MAGNETS_TEMPERATURES_650W[i] - c) / m

    new_x_650.append(new_x)


legend_elements = [Line2D([0], [0], color='black', marker='x', label='Timber data', linestyle='None'),
                   Line2D([0], [0], color='black', marker='s', label='Simulated', linestyle='None')]

# %% 650 experiment
plt.figure(figsize=(10, 6))
wl = np.mean(new_x_650)
error_bar = 0.005
for i in range(len(SENSORS_LOCATION)):
    plt.plot(x[i], y[i], marker='x', color=colors(i))
    plt.plot(wl, MAGNETS_TEMPERATURES_650W[i], 's', color=colors(i))
    if error_bar:
        plt.errorbar(wl, MAGNETS_TEMPERATURES_650W[i], yerr=error_bar, capsize=2, ecolor=colors(i))

plt.xticks(np.arange(x_single[0], x_single[1], 0.5))
plt.xlabel('Wetted length [ m ]')
plt.ylabel('Temperature [ mK ]')
plt.title('Wetted length for 650 W experiment')
plt.legend(handles=legend_elements, loc='lower left')
plt.grid()
plt.show()

total_error = 0
# COMPUTE TOTAL ERROR HERE !!
# %% ============== >  700 W < =============================

file_path_700W_20_1 = 'results/heat/simulations/700W_66cm2/Standard_Cell_Feb_700W_66cm2_wl=20.1.npz'
file_path_700W_24_9 = 'results/heat/simulations/700W_66cm2/Standard_Cell_Feb_700W_66cm2_wl=24.9.npz'


sim_700_20_1 = load_numpy_dict(file_path_700W_20_1)
sim_700_24_9 = load_numpy_dict(file_path_700W_24_9)


T_pred_700_20_1 = sim_700_20_1['cells__T'][-1][SENSORS_LOCATION]
T_pred_700W_24_9 = sim_700_24_9['cells__T'][-1][SENSORS_LOCATION]

temperatures = [T_pred_700_20_1, T_pred_700W_24_9]
plt.figure(figsize=(10, 10))
wetted_lengths = [[20.1], [24.9]]

for i in range(len(wetted_lengths)):
    plt.plot(wetted_lengths[i] * len(SENSORS_LOCATION), temperatures[i], 'x')

x = [[20.1, 24.9]] * len(SENSORS_LOCATION)
y = [[T_pred_700_20_1[i], T_pred_700W_24_9[i]] for i in range(len(SENSORS_LOCATION))]
for i in range(len(SENSORS_LOCATION)):
    plt.plot(x[i], y[i], marker='o')

plt.grid()
plt.show()
colors = plt.cm.tab20

new_x_700 = []
x_single = np.array([20.1, 24.9])
for i in range(len(SENSORS_LOCATION)):
    y_single = np.array([T_pred_700_20_1[i], T_pred_700W_24_9[i]])
    A = np.vstack([x_single, np.ones(len(x_single))]).T
    m, c = np.linalg.lstsq(A, y_single, rcond=None)[0]
    new_x = (MAGNETS_TEMPERATURES_700w[i] - c) / m

    new_x_700.append(new_x)


legend_elements = [Line2D([0], [0], color='black', marker='x', label='Timber data', linestyle='None'),
                   Line2D([0], [0], color='black', marker='s', label='Simulated', linestyle='None')]

# %% ============== >  700 W < =============================
plt.figure(figsize=(10, 6))
wl = np.mean(new_x_700)
error_bar = 0.005
for i in range(len(SENSORS_LOCATION)):
    plt.plot(x[i], y[i], marker='x', color=colors(i))
    plt.plot(wl, MAGNETS_TEMPERATURES_700w[i], 's', color=colors(i))
    if error_bar:
        plt.errorbar(wl, MAGNETS_TEMPERATURES_700w[i], yerr=error_bar, capsize=2, ecolor=colors(i))

plt.xticks(np.arange(x_single[0], x_single[1], 0.5))
plt.xlabel('Wetted length [ m ]')
plt.ylabel('Temperature [ mK ]')
plt.title('Wetted length for 700 W experiment')
plt.legend(handles=legend_elements, loc='lower left')
plt.grid()
plt.show()
# %% # ============> 650 exp  + adj <==========

file_path_650W_plus_adj_20_1 = 'results/heat/simulations/650W+adj_66cm2/Standard_Cell_Feb_650W+adj_66cm2_wl=15.3.npz'
file_path_650W_plus_adj_24_9 = 'results/heat/simulations/650W+adj_66cm2/Standard_Cell_Feb_650W+adj_66cm2_wl=20.1.npz'

sim_650_plus_adj_20_1 = load_numpy_dict(file_path_650W_plus_adj_20_1)
sim_650_plus_adj_24_9 = load_numpy_dict(file_path_650W_plus_adj_24_9)


T_pred_650W_adj_20_1 = sim_650_plus_adj_20_1['cells__T'][-1][SENSORS_LOCATION]
T_pred_650W_adj_24_9 = sim_650_plus_adj_24_9['cells__T'][-1][SENSORS_LOCATION]

x = [[15.3, 20.1]] * len(SENSORS_LOCATION)
y = [[T_pred_650W_adj_20_1[i], T_pred_650W_adj_24_9[i]] for i in range(len(SENSORS_LOCATION))]

new_x_650_adj = []


x_single_adj = np.array([15.3, 20.1])
for i in range(len(SENSORS_LOCATION)):
    y_single = np.array([T_pred_650W_adj_20_1[i], T_pred_650W_adj_24_9[i]])
    A = np.vstack([x_single_adj, np.ones(len(x_single_adj))]).T
    m, c = np.linalg.lstsq(A, y_single, rcond=None)[0]
    new_x = (MAGNETS_TEMPERATURES_650W[i] - c) / m

    new_x_650_adj.append(new_x)


plt.figure(figsize=(10, 6))
wl_adj = np.mean(new_x_650_adj)
error_bar = 0.005
for i in range(len(SENSORS_LOCATION)):
    plt.plot(x[i], y[i], marker='x', color=colors(i))
    plt.plot(wl_adj, MAGNETS_TEMPERATURES_650W[i], 's', color=colors(i))
    if error_bar:
        plt.errorbar(wl_adj, MAGNETS_TEMPERATURES_650W[i], yerr=error_bar, capsize=2, ecolor=colors(i))

plt.xticks(np.arange(x_single_adj[0], x_single_adj[1], 0.5))
plt.xlabel('Wetted length [ m ]')
plt.ylabel('Temperature [ mK ]')
plt.title('Wetted length for 650 W experiment + heat from adj cell')
plt.legend(handles=legend_elements, loc='lower left')
plt.grid()
plt.show()


# %% ==============> 700 W + adj cell <===================

file_path_700W_plus_adj_20_1 = 'results/heat/simulations/700W+adj_66cm2/Standard_Cell_Feb_700W+adj_66cm2_wl=20.1.npz'
file_path_700W_plus_adj_24_9 = 'results/heat/simulations/700W+adj_66cm2/Standard_Cell_Feb_700W+adj_66cm2_wl=24.9.npz'

sim_700_plus_adj_20_1 = load_numpy_dict(file_path_700W_plus_adj_20_1)
sim_700_plus_adj_24_9 = load_numpy_dict(file_path_700W_plus_adj_24_9)


T_pred_700W_adj_20_1 = sim_700_plus_adj_20_1['cells__T'][-1][SENSORS_LOCATION]
T_pred_700W_adj_24_9 = sim_700_plus_adj_24_9['cells__T'][-1][SENSORS_LOCATION]

x = [[20.1, 24.9]] * len(SENSORS_LOCATION)
y = [[T_pred_700W_adj_20_1[i], T_pred_700W_adj_24_9[i]] for i in range(len(SENSORS_LOCATION))]

new_x_700_adj = []


x_single_adj = np.array([ 20.1, 24.9])
for i in range(len(SENSORS_LOCATION)):
    y_single = np.array([T_pred_700W_adj_20_1[i], T_pred_700W_adj_24_9[i]])
    A = np.vstack([x_single_adj, np.ones(len(x_single_adj))]).T
    m, c = np.linalg.lstsq(A, y_single, rcond=None)[0]
    new_x = (MAGNETS_TEMPERATURES_700w[i] - c) / m

    new_x_700_adj.append(new_x)


plt.figure(figsize=(10, 6))
wl_adj = np.mean(new_x_700_adj)
error_bar = 0.005
for i in range(len(SENSORS_LOCATION)):
    plt.plot(x[i], y[i], marker='x', color=colors(i))
    plt.plot(wl_adj, MAGNETS_TEMPERATURES_700w[i], 's', color=colors(i))
    if error_bar:
        plt.errorbar(wl_adj, MAGNETS_TEMPERATURES_700w[i], yerr=error_bar, capsize=2, ecolor=colors(i))

plt.xticks(np.arange(x_single_adj[0], x_single_adj[1], 0.5))
plt.xlabel('Wetted length [ m ]')
plt.ylabel('Temperature [ mK ]')
plt.title('Wetted length for 700 W experiment + heat from adj cell')
plt.legend(handles=legend_elements, loc='lower left')
plt.grid()
plt.show()


