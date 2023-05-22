from subsystems.heat_flux.utils.plotting import plot_graph_tensor
import tensorflow_gnn as tfgnn

# %%

graph = tfgnn.GraphTensor.from_pieces(
    context=tfgnn.Context.from_fields(
        features={'time_step': [1.]}),
    node_sets={"liquid": tfgnn.NodeSet.from_fields(sizes=[8],
                                                   features={
                                                       "extraction capacity": [5., 5., 5., 5., 5., 0., 0., 0. ],
                                                       "T": [1.85, 1.85, 1.85, 1.85, 1.85, 1.85, 1.85, 1.85 ]
                                                   }),
               "heater": tfgnn.NodeSet.from_fields(sizes=[8],
                                                   features={
                                                       "power": [1., 1., 1., 1., 1., 1., 1., 1.],
                                                   }),
               "cells": tfgnn.NodeSet.from_fields(sizes=[8],
                                                  features={
                                                      "T": [1.89, 1.90, 1.91, 1.92, 1.93, 1.94, 1.95, 1.98],
                                                      "Q": [0., 0., 0., 0., 0., 0., 0., 0.],
                                                      "m": [400., 400., 400., 400., 400., 400., 400., 400.]
                                                  }), },
    edge_sets={"conduction": tfgnn.EdgeSet.from_fields(sizes=[14],
                                                       features={
                                                           "capacity": [100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100.],
                                                           'L': [1., 1., 1., 1., 1., 1.,1., 1., 1., 1., 1., 1., 1., 1.]
                                                       },
                                                       adjacency=tfgnn.Adjacency.from_indices(
                                                           source=("cells", [0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7]),
                                                           target=("cells", [1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6]))
                                                       ),
               "heat supplied": tfgnn.EdgeSet.from_fields(sizes=[8],
                                                          features={},
                                                          adjacency=tfgnn.Adjacency.from_indices(
                                                              source=("heater", [0, 1, 2, 3, 4, 5, 6, 7]),
                                                              target=("cells", [0, 1, 2, 3, 4, 5, 6, 7])
                                                          )),
               "cell2liquid": tfgnn.EdgeSet.from_fields(sizes=[8],
                                                           features={
                                                               "capacity": [1200., 1200., 0., 0., 5., 5., 0., 0.],
                                                               "extraction capacity": [5., 5., 0., 0.,5., 5., 0., 0.]
                                                           },
                                                           adjacency=tfgnn.Adjacency.from_indices(
                                                               source=("cells", [0, 1, 2, 3, 4, 5, 6, 7]),
                                                               target=("liquid", [0, 1, 2, 3, 4, 5, 6, 7])
                                                           ))}
)

# %%
plot_graph_tensor(graph, title='Heat Flux graph')

