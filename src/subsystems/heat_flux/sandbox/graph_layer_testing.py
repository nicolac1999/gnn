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

fraction_wetted_perimeter = np.array([0.30106554, 0.29528652, 0.28934577, 0.28322702, 0.27691142,
                                      0.27037692, 0.26359744, 0.25654187, 0.2491725, 0.24144304,
                                      0.23329555, 0.2246561, 0.2154279, 0.20548029, 0.19462992,
                                      0.18260624, 0.16898168, 0.1530102, 0.13316729, 0.1052322,
                                      0.])

mean_fractions = []
for i in range(0, len(fraction_wetted_perimeter) - 1):
    mean_fractions.append((fraction_wetted_perimeter[i] + fraction_wetted_perimeter[i + 1]) / 2)

mean_fractions = np.array(mean_fractions, dtype=np.float32)

evapor_mass_flow = np.array([5.519425, 5.519425, 5.519425, 5.519425, 5.519425, 5.519425,
                           5.519425, 5.519425, 5.519425, 5.519425, 5.519425, 5.519425,
                           5.519425, 5.519425, 5.519425, 5.519425, 5.519425, 5.519425,
                           5.519425, 5.519425], dtype=np.float32)


#%%

graph_ref = tfgnn.GraphTensor.from_pieces(
    context=tfgnn.Context.from_fields(
        features={'time': [0.],
                  'time_step': [1.],
                  'specific_heat_capacity': [4000.],
                  'static_heat': [1.1],
                  'latent_heat_vaporization': [23.],
                  'liquid_density': [0.145],
                  'total_static_heat': [0.],
                  'total_dynamic_heat': [1.],
                  'total_vaporization_heat': [1200.]}),

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
                                                       "evapor_mass_flow": evapor_mass_flow,
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

model = HeatSimplifiedModel()
#%%

res = model(graph_ref)

