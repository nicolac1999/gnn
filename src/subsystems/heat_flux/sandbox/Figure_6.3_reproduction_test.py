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

#%%

# fraction_wetted_perimeter = np.array([0.30106554, 0.29528652, 0.28934577, 0.28322702, 0.27691142,
#                                       0.27037692, 0.26359744, 0.25654187, 0.2491725, 0.24144304,
#                                       0.23329555, 0.2246561, 0.2154279, 0.20548029, 0.19462992,
#                                       0.18260624, 0.16898168, 0.1530102, 0.13316729, 0.1052322,
#                                       0.])

fraction_wetted_perimeter = np.array([0.24999961, 0.24539063, 0.24063999, 0.23573396, 0.23065665,
       0.22538943, 0.21991031, 0.21419296, 0.20820555, 0.20190889,
       0.19525401, 0.18817835, 0.18060001, 0.17240843, 0.16344845,
       0.15349109, 0.14217454, 0.12886744, 0.11227977, 0.08884026,
       0.        ])


mean_fractions = []
for i in range(0, len(fraction_wetted_perimeter) - 1):
    mean_fractions.append((fraction_wetted_perimeter[i] + fraction_wetted_perimeter[i + 1]) / 2)

evapor_mass_flow = np.array([5.519425, 5.519425, 5.519425, 5.519425, 5.519425, 5.519425,
                           5.519425, 5.519425, 5.519425, 5.519425, 5.519425, 5.519425,
                           5.519425, 5.519425, 5.519425, 5.519425, 5.519425, 5.519425,
                           5.519425, 5.519425], dtype=np.float32)

# evapor_mass_flow_new_static_heat = np.array([8.1,
#                                            6.8, 6.8, 6.8,
#                                            6.8, 6.8, 6.8,
#                                            6.8, 6.8, 6.8,
#                                            8.1,
#                                            6.8, 6.8, 6.8,
#                                            6.8, 6.8, 6.8,
#                                            6.8, 6.8, 6.8], dtype=np.float32) + 1

evapor_mass_flow_new_static_heat = np.array([12.66,
                                           10.55, 10.55, 10.55,
                                           10.55, 10.55, 10.55,
                                           10.55, 10.55, 10.55,
                                           12.66,
                                           10.55, 10.55, 10.55,
                                           10.55, 10.55, 10.55,
                                           10.55, 10.55, 10.55], dtype=np.float32) + 1

# %%

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

#%%
model = HeatSimplifiedModel(1000)

# %%

###########
## first plot
########
from tqdm import tqdm

mean_delta_t = []
last_temperatures = []
#static_heat_range = np.arange(0.08, 0.09, 0.1)
#static_heat_range = [0.08, 0.09, 0.1, ]
#static_heat_range = [.2]
# dynamic_heat_range = np.concatenate((np.arange(0., 2.1, 0.2), np.array([0.5, 0.9, 1.3])))
dynamic_heat_range = np.concatenate((np.arange(0., 2.1, 0.2), np.array([1.3, 0.9, 0.5])))
#dynamic_heat_range = [0.]
list_temperatures = []

current_temperatures = np.ones(20, dtype=np.float32) * 1.85
for dynamic_heat in tqdm(dynamic_heat_range):

    total_heating_power = dynamic_heat * 102
    heater_power = np.ones(8, dtype=np.float32) * (total_heating_power / 8)
    dict_new_features = {'context.static_heat': [0.11],
                         'cells.T': current_temperatures,
                         'heater.power': heater_power.astype(np.float32)}

    graph_ref = set_graph_features(graph=graph_ref,
                                   dict_features=dict_new_features)

    forward_pass = model(graph_ref)

    list_temperatures.append(forward_pass.T)
    last_temperatures.append(forward_pass.T[-1])
    delta_T = forward_pass.T[-1] - 1.85
    mean_delta_t.append(np.mean(delta_T))

    current_temperatures = forward_pass.T[-1]


#%%


EXPERIMENTAL_POINTS = [(0, 1.5), (0.2, 2.5), (0.4,4.5), (0.5, 6.5), (0.6, 6),
                       (0.8, 7.5), (0.9, 11), (1, 13.5), (1.2, 15.5), (1.3, 16.5), (1.4,17.5),
                       (1.6, 20.5), (1.8, 24.2), (2, 28.5)]
x_experimental = [points[0] for points in EXPERIMENTAL_POINTS]
y_experimental = [points[1] for points in EXPERIMENTAL_POINTS]

plt.figure(figsize=(10,6))
plt.plot(dynamic_heat_range, [m * 1000 for m in mean_delta_t], 'x', label='predicted values')
plt.plot(x_experimental, y_experimental,'s', label='experimental values')
x_labels = ["{:.2f}".format(x) for x in dynamic_heat_range]
plt.xticks(dynamic_heat_range, labels=x_labels)
plt.xlabel('Applied power [W/m]')

plt.ylabel('DeltaT (Tmagnets - Tsat) [mK]')
plt.ylim(0, 30)
plt.grid()
plt.legend()
plt.show()
# %%

colors = plt.cm.tab20
mask = [0, 1, 4, 7, 10, 11, 14, 17]

temperature_array = np.array([]).reshape((0, 20))

for temperature in list_temperatures:
    temperature = temperature.numpy()
    temperature_array = np.vstack((temperature_array, temperature))

fig = plt.figure(figsize=(16, 6))

num_magnets = temperature_array.shape[-1]
for i in range(num_magnets):
    if i in mask:
        plt.plot(np.arange(0, 14014, 1), temperature_array[:, i], label=f'DT magnet [{i}]', color=colors(i), alpha=1)

plt.xlabel('Time stamp [ s ]')
plt.ylabel('Temperature [ K ]')

plt.legend()
plt.grid()
plt.show()