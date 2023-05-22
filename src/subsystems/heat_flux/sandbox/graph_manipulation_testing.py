from common.graph_manipulation_helpers import *


# %%
graph_ref = tfgnn.GraphTensor.from_pieces(
    context=tfgnn.Context.from_fields(
        features={'time': [867600.],
                  'time_step': [5.],
                  'specific heat capacity': [4000.],
                  'static_heat': [0.2]}),

    node_sets={"cells": tfgnn.NodeSet.from_fields(sizes=[1],
                                                  features={
                                                      "T": [20],
                                                      'F': [10]
                                                  }),

               "liquid": tfgnn.NodeSet.from_fields(sizes=[1],
                                                   features={
                                                       'level': [10],
                                                       'velocity': [50]

                                                   })},
    edge_sets={"conduction": tfgnn.EdgeSet.from_fields(sizes=[38],
                                                       features={
                                                           "conductivity": [100., 100., 100.],
                                                           "L": [6.1, 5., 5.],
                                                       },
                                                       adjacency=tfgnn.Adjacency.from_indices(
                                                           source=("cells", [0, 1, 2]),
                                                           target=("cells", [1, 2, 3]))
                                                       )}
)

# %%

dict_features = {'cells__T': [1],
                 'cells__F': [2],
                 'liquid__level': [20, 40, 60, 90],
                 'liquid__velocity': [40],
                 'heater__power': [50],
                 'context__time': [200],
                 'context__dynamic_heat': [3030303030],
                 'conduction__L': [10, 20, 30],
                 'conduction__test_feature': [10, 20]}
graph_ref = set_graph_features(graph_ref, dict_features)

