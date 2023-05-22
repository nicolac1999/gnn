import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from tensorflow import keras
from keras import optimizers
from functools import partial

from common.graph_manipulation_helpers import set_graph_features
from common.math.bayonet_geometry import BayonetGeometryEstimator
from subsystems.heat_flux.graphs.linear_system import make_graph_tensor_from_tensors
from subsystems.heat_flux.models.heat_simplified_model import HeatSimplifiedModel
from subsystems.heat_flux.models.model_v1 import *
from subsystems.heat_flux.utils.data_helpers import sampling_heat_data, sample_generator
from subsystems.heat_flux.utils.plotting import plot_preview, plot_graph_tensor, plot_concatenate_predictions
from lib.diag_common.tf_helpers import set_all_gpu_to_incremental_memory
from subsystems.heat_flux.utils.simulation import simulate_time_period

set_all_gpu_to_incremental_memory()

# %%
######################################################################################
################### Proper initialization ############################################
######################################################################################


fraction_wetted_perimeter = np.array([0.30106554, 0.29528652, 0.28934577, 0.28322702, 0.27691142,
                                      0.27037692, 0.26359744, 0.25654187, 0.2491725, 0.24144304,
                                      0.23329555, 0.2246561, 0.2154279, 0.20548029, 0.19462992,
                                      0.18260624, 0.16898168, 0.1530102, 0.13316729, 0.1052322,
                                      0.])

mean_fractions = []
for i in range(0, len(fraction_wetted_perimeter) - 1):
    mean_fractions.append((fraction_wetted_perimeter[i] + fraction_wetted_perimeter[i + 1]) / 2)

evapor_mass_flow = np.array([5.519425, 5.519425, 5.519425, 5.519425, 5.519425, 5.519425,
                           5.519425, 5.519425, 5.519425, 5.519425, 5.519425, 5.519425,
                           5.519425, 5.519425, 5.519425, 5.519425, 5.519425, 5.519425,
                           5.519425, 5.519425], dtype=np.float32)

evapor_mass_flow_new_static_heat = np.array([8.1,
                                           6.8, 6.8, 6.8,
                                           6.8, 6.8, 6.8,
                                           6.8, 6.8, 6.8,
                                           8.1,
                                           6.8, 6.8, 6.8,
                                           6.8, 6.8, 6.8,
                                           6.8, 6.8, 6.8], dtype=np.float32) + 1

# %%


model = HeatSimplifiedModel(1000)

# %%
######################################################################################


graph_ref = tfgnn.GraphTensor.from_pieces(
    context=tfgnn.Context.from_fields(
        features={'time': [0.],
                  'time_step': [1.],
                  'specific_heat_capacity': [4000.],
                  'static_heat': [.01],
                  'latent_heat_vaporization': [23.],
                  'liquid_density': [0.145],
                  'total_static_heat': [0.],
                  'total_dynamic_heat': [1.],
                  'total_vaporization_heat': [0.]
                  }),

    node_sets={"cells": tfgnn.NodeSet.from_fields(sizes=[20],
                                                  features={
                                                      "T": np.ones(20, dtype=np.float32) * 1.85,

                                                      "mass": [36.,
                                                               21.75, 14.5, 21.75,
                                                               21.75, 14.5, 21.75,
                                                               21.75, 14.5, 21.75,
                                                               36,
                                                               21.75, 14.5, 21.75,
                                                               21.75, 14.5, 21.75,
                                                               21.75, 14.5, 21.75],
                                                      'L': np.array([6.,
                                                                     5., 5., 5.,
                                                                     5., 5., 5.,
                                                                     5., 5., 5.,
                                                                     6.,
                                                                     5., 5., 5.,
                                                                     5., 5., 5.,
                                                                     5., 5., 5.], dtype=np.float32)
                                                  }),
               "heater": tfgnn.NodeSet.from_fields(sizes=[8],
                                                   features={
                                                       "power": np.ones(8, dtype=np.float32) * 0.0,
                                                   }),
               "liquid": tfgnn.NodeSet.from_fields(sizes=[20],
                                                   features={
                                                       # "extraction capacity": np.pad(np.ones(WETTED_LENGTH, dtype=np.float32) * 1.53,
                                                       #                               (20 - WETTED_LENGTH, 0)),

                                                       # "extraction capacity": np.ones(20, dtype=np.float32) * 1.53,

                                                       "T": [1.85,
                                                             1.85, 1.85, 1.85,
                                                             1.85, 1.85, 1.85,
                                                             1.85, 1.85, 1.85,
                                                             1.85,
                                                             1.85, 1.85, 1.85,
                                                             1.85, 1.85, 1.85,
                                                             1.85, 1.85, 1.85],
                                                       "avg_f": mean_fractions,
                                                       'L': np.array([6.,
                                                                      5., 5., 5.,
                                                                      5., 5., 5.,
                                                                      5., 5., 5.,
                                                                      6.,
                                                                      5., 5., 5.,
                                                                      5., 5., 5.,
                                                                      5., 5., 5.], dtype=np.float32),
                                                       "evapor_mass_flow": evapor_mass_flow_new_static_heat,
                                                       "D": np.ones(20, dtype=np.float32) * 0.053
                                                   })},

    edge_sets={"conduction": tfgnn.EdgeSet.from_fields(sizes=[38],
                                                       features={
                                                           "conductivity": np.ones(38, dtype=np.float32) * 1500,

                                                           "L": [6.1, 5., 5., 5.6, 5., 5.,
                                                                 5.6, 5., 5., 6.1, 6.1, 5.,
                                                                 5., 5.6, 5., 5., 5.6, 5., 5.,

                                                                 6.1, 5., 5., 5.6, 5., 5.,
                                                                 5.6, 5., 5., 6.1, 6.1, 5.,
                                                                 5., 5.6, 5., 5., 5.6, 5., 5.
                                                                 ],
                                                           "A": [30.,
                                                                 150., 150., 30.,
                                                                 150., 150., 30.,
                                                                 150., 150., 30.,
                                                                 30.,
                                                                 150., 150., 30.,
                                                                 150., 150., 30.,
                                                                 150., 150.,

                                                                 30.,
                                                                 150., 150., 30.,
                                                                 150., 150., 30.,
                                                                 150., 150., 30.,
                                                                 30.,
                                                                 150., 150., 30.,
                                                                 150., 150., 30.,
                                                                 150., 150.,
                                                                 ],
                                                       },
                                                       adjacency=tfgnn.Adjacency.from_indices(
                                                           source=("cells",
                                                                   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                                                    15, 16, 17, 18,
                                                                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                                                    16, 17, 18, 19]),
                                                           target=("cells",
                                                                   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                                                    16, 17, 18, 19,
                                                                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                                                    15, 16, 17, 18]))
                                                       ),
               "heat supplied": tfgnn.EdgeSet.from_fields(sizes=[8],
                                                          features={},
                                                          adjacency=tfgnn.Adjacency.from_indices(
                                                              source=("heater", [0, 1, 2, 3, 4, 5, 6, 7]),
                                                              target=("cells", [0, 1, 4, 7, 10, 11, 14, 17])
                                                          )),
               "cell2liquid": tfgnn.EdgeSet.from_fields(sizes=[20],
                                                        features={
                                                            "conductivity": np.ones(20, dtype=np.float32) * 860,
                                                        },
                                                        adjacency=tfgnn.Adjacency.from_indices(
                                                            source=("cells",
                                                                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                                                     15, 16, 17, 18, 19, ]),
                                                            target=("liquid",
                                                                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                                                     15, 16, 17, 18, 19])
                                                        )),
               "liquid2cell": tfgnn.EdgeSet.from_fields(sizes=[20],
                                                        features={
                                                            "conductivity": np.ones(20, dtype=np.float32) * 860,
                                                        },
                                                        adjacency=tfgnn.Adjacency.from_indices(
                                                            source=("liquid",
                                                                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                                                     15, 16, 17, 18, 19, ]),
                                                            target=("cells",
                                                                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                                                     15, 16, 17, 18, 19])
                                                        ))}
)



# %%

###########
## first plot
########
from tqdm import tqdm

mean_delta_t = []
last_temperatures = []
static_heat_range = np.arange(0.342, 1.442, 0.2)
for stat_heat in tqdm(static_heat_range):
    dict_new_features = {'context.static_heat': [stat_heat]}

    graph_ref = graph_set_features(graph=graph_ref,
                                   dict_features=dict_new_features)

    forward_pass = model(graph_ref)

    last_temperatures.append(forward_pass.T[-1])
    delta_T = forward_pass.T[-1] - 1.85
    mean_delta_t.append(np.mean(delta_T))

# %%

fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 6))
mask = [0, 1, 4, 7, 10, 11, 14, 17]
ax.plot(static_heat_range, [m * 1000 for m in mean_delta_t], label='DT average')
last_temperatures = np.array(last_temperatures)
for t in range(len(last_temperatures[0])):
    if t in mask:
        ax.plot(static_heat_range, (last_temperatures[:, t] - 1.85) * 1000, 'o', label=f'DT magnet[{t}]')

# static_heat_range_shifted = static_heat_range-0.342
# static_heat_range_shifted_masked = [x if x >= 0 else 0 for x in static_heat_range_shifted]
# x_labels = ["{:.2f}".format(x) for x in static_heat_range]
# ax.set_xticklabels(x_labels)
ax.set_xticks(static_heat_range)
x_labels = ["{:.2f}".format(x) for x in static_heat_range - 0.342]
ax.set_xticklabels(x_labels)
ax.set_xlabel('Applied power [W/m]')
ax.set_ylabel('DeltaT (Tmagnets - Tsat) [mK]')
ax.set_ylim(0, 25)

ax.legend(bbox_to_anchor=(1, 1), loc='upper left')

secax = ax.secondary_xaxis('top')
secax.set_xlabel('Total heat load (static + dynamic) [W/m]')
x_labels_up = ["{:.2f}".format(x) for x in static_heat_range]
secax.set_xticks(static_heat_range)
secax.set_xticklabels(x_labels_up)
plt.grid()

plt.show()

# %%
##########
##### second plot
##########
mean_delta_t_2 = []
last_temperatures_2 = []
static_heat_range_2 = np.arange(0.342, 1.442, 0.2)
for stat_heat in tqdm(static_heat_range_2):
    dict_new_features = {'context.static_heat': [stat_heat]}

    graph_ref = graph_set_features(graph=graph_ref,
                                   dict_features=dict_new_features)

    forward_pass = model(graph_ref)

    last_temperatures_2.append(forward_pass.T[-1])
    delta_T_2 = forward_pass.T[-1] - 1.85
    mean_delta_t_2.append(np.mean(delta_T_2))

# %%

fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 6))
mask = [0, 1, 4, 7, 10, 11, 14, 17]
ax.plot(static_heat_range_2, [m * 1000 for m in mean_delta_t_2], label='DT average')
last_temperatures_2 = np.array(last_temperatures_2)
for t in range(len(last_temperatures_2[0])):
    if t in mask:
        ax.plot(static_heat_range_2, (last_temperatures_2[:, t] - 1.85) * 1000, 'o', label=f'DT magnet[{t}]')

# static_heat_range_shifted = static_heat_range-0.342
# static_heat_range_shifted_masked = [x if x >= 0 else 0 for x in static_heat_range_shifted]
# x_labels = ["{:.2f}".format(x) for x in static_heat_range]
# ax.set_xticklabels(x_labels)
ax.set_xticks(static_heat_range_2)
x_labels = ["{:.2f}".format(x) for x in static_heat_range_2 - 0.342]
ax.set_xticklabels(x_labels)
ax.set_xlabel('Applied power [W/m]')
ax.set_ylabel('DeltaT (Tmagnets - Tsat) [mK]')
ax.set_ylim(0, 25)

ax.legend(bbox_to_anchor=(1, 1), loc='upper left')

secax = ax.secondary_xaxis('top')
secax.set_xlabel('Total heat load (static + dynamic) [W/m]')
x_labels_up = ["{:.2f}".format(x) for x in static_heat_range_2]
secax.set_xticks(static_heat_range_2)
secax.set_xticklabels(x_labels_up)
plt.grid()

plt.show()

# %%

##################
######## prova con static + dynamic
################

mean_delta_t = []
last_temperatures = []
dynamic_heat_range = np.arange(0., 1.1, 0.2)
#length_for_heater = np.array([6, 15, 15, 15, 6, 15, 15, 15])

for dynamic_heat in tqdm(dynamic_heat_range):

    total_heating_power = dynamic_heat * 102
    heater_power = np.ones(8, dtype=np.float32) * (total_heating_power / 8)
    dict_new_features = {'context.static_heat': [0.342],
                         'heater.power': heater_power.astype(np.float32)}

    graph_ref = graph_set_features(graph=graph_ref,
                                   dict_features=dict_new_features)

    forward_pass = model(graph_ref)

    last_temperatures.append(forward_pass.T[-1])
    delta_T = forward_pass.T[-1] - 1.85
    mean_delta_t.append(np.mean(delta_T))

# %%
fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 6))
mask = [0, 1, 4, 7, 10, 11, 14, 17]
ax.plot(static_heat_range, [m * 1000 for m in mean_delta_t], label='DT average')
last_temperatures = np.array(last_temperatures)
for t in range(len(last_temperatures[0])):
    if t in mask:
        ax.plot(static_heat_range, (last_temperatures[:, t] - 1.85) * 1000, 'o', label=f'DT magnet[{t}]')

# static_heat_range_shifted = static_heat_range-0.342
# static_heat_range_shifted_masked = [x if x >= 0 else 0 for x in static_heat_range_shifted]
# x_labels = ["{:.2f}".format(x) for x in static_heat_range]
# ax.set_xticklabels(x_labels)
ax.set_xticks(static_heat_range)
x_labels = ["{:.2f}".format(x) for x in static_heat_range - 0.342]
ax.set_xticklabels(x_labels)
ax.set_xlabel('Applied power [W/m]')
ax.set_ylabel('DeltaT (Tmagnets - Tsat) [mK]')
ax.set_ylim(0, 25)

ax.legend(bbox_to_anchor=(1, 1), loc='upper left')

secax = ax.secondary_xaxis('top')
secax.set_xlabel('Total heat load (static + dynamic) [W/m]')
x_labels_up = ["{:.2f}".format(x) for x in static_heat_range]
secax.set_xticks(static_heat_range)
secax.set_xticklabels(x_labels_up)
plt.grid()

plt.show()

