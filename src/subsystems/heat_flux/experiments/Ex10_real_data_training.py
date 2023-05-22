import tensorflow as tf
import numpy as np

from tensorflow import keras
from keras import optimizers
from functools import partial

from subsystems.heat_flux.graphs.linear_system import make_graph_tensor_from_tensors
from subsystems.heat_flux.models.model_v1 import *
from subsystems.heat_flux.utils.data_helpers import sampling_heat_data, sample_generator
from subsystems.heat_flux.utils.plotting import plot_preview, plot_graph_tensor
from lib.diag_common.tf_helpers import set_all_gpu_to_incremental_memory

set_all_gpu_to_incremental_memory()

# %%

ds_filepath = r'data/heat/DS02_training_0W.xlsx'
frames = sample_generator(ds_filepath=ds_filepath,
                          variables=['T', 'H'],
                          starting_time=0, duration=None,
                          max_frames=5, samples_per_frame=31, time_step=30., stride=2000)



# %%

WETTED_LENGTH = 6

graph_ref = tfgnn.GraphTensor.from_pieces(
    context=tfgnn.Context.from_fields(
        features={'time': [867600.],
                  'time_step': [5.],
                  'specific heat capacity': [4000.],
                  'static_heat': [0.2]}),

    node_sets={"cells": tfgnn.NodeSet.from_fields(sizes=[20],
                                                  features={
                                                      "T": [1.90, 1.90, 1.90, 1.90, 1.90,
                                                            1.90, 1.90, 1.90, 1.90, 1.90,
                                                            1.90, 1.90, 1.90, 1.90, 1.90,
                                                            1.90, 1.90, 1.90, 1.90, 1.90],

                                                      # "Q": [0., 0., 0., 0., 0.,
                                                      #       0., 0., 0., 0., 0.,
                                                      #       0., 0., 0., 0., 0.,
                                                      #       0., 0., 0., 0., 0.],

                                                      "mass": [36.,
                                                               21.75, 14.5, 21.75,
                                                               21.75, 14.5, 21.75,
                                                               21.75, 14.5, 21.75,
                                                               36,
                                                               21.75, 14.5, 21.75,
                                                               21.75, 14.5, 21.75,
                                                               21.75, 14.5, 21.75],
                                                      'L': [6.,
                                                            5., 5., 5.,
                                                            5., 5., 5.,
                                                            5., 5., 5.,
                                                            6.,
                                                            5., 5., 5.,
                                                            5., 5., 5.,
                                                            5., 5., 5.]
                                                  }),
               "heater": tfgnn.NodeSet.from_fields(sizes=[8],
                                                   features={
                                                       "power": [0., 0., 0., 0., 0., 0., 0., 0.],
                                                   }),
               "liquid": tfgnn.NodeSet.from_fields(sizes=[20],
                                                   features={
                                                       "extraction capacity": np.pad(np.ones(WETTED_LENGTH, dtype=np.float32) * 1.53,
                                                                                     (20 - WETTED_LENGTH, 0)),
                                                       "T": [1.85,
                                                             1.85, 1.85, 1.85,
                                                             1.85, 1.85, 1.85,
                                                             1.85, 1.85, 1.85,
                                                             1.85,
                                                             1.85, 1.85, 1.85,
                                                             1.85, 1.85, 1.85,
                                                             1.85, 1.85, 1.85]
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
                                                           # "E_transf": [0., 0., 0., 0., 0., 0.],
                                                           # "Q_transf": [0., 0., 0., 0., 0., 0.]
                                                       },
                                                       adjacency=tfgnn.Adjacency.from_indices(
                                                           source=("cells", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                                                                             1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
                                                           target=("cells", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                                                             0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]))
                                                       ),
               "heat supplied": tfgnn.EdgeSet.from_fields(sizes=[8],
                                                          features={},
                                                          adjacency=tfgnn.Adjacency.from_indices(
                                                              source=("heater", [0, 1, 2, 3, 4, 5, 6, 7]),
                                                              target=("cells", [0, 1, 4, 7, 10, 11, 14, 17])
                                                          )),
               "cell2liquid": tfgnn.EdgeSet.from_fields(sizes=[20],
                                                           features={
                                                               "conductivity": np.ones(20, dtype=np.float32) * 1200,
                                                           },
                                                           adjacency=tfgnn.Adjacency.from_indices(
                                                               source=("cells", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,]),
                                                               target=("liquid", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
                                                           )),
               "liquid2cell": tfgnn.EdgeSet.from_fields(sizes=[20],
                                                           features={
                                                                "conductivity": np.ones(20, dtype=np.float32) * 1200,
                                                           },
                                                            adjacency=tfgnn.Adjacency.from_indices(
                                                               source=("liquid", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,]),
                                                               target=("cells", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
                                                           ))}
)

#plot_graph_tensor(graph_ref, title='Heat Flux graph')

# %% tf.data.Dataset preparation:
NUM_REPEAT = 16


# -- 1) Stack frames together and embed them into NamedTuple:

from subsystems.heat_flux.models.data_pipeline import from_samples_to_named_tuples, DataItemFromDS_v1

frames_as_nt = from_samples_to_named_tuples(frames, DataItemFromDS_v1)


# -- 2) Convert Numpy arrays into tf.Tensor and initialize a tf.data.Dataset:

frames_as_nt_tf = tf.nest.map_structure(tf.convert_to_tensor, frames_as_nt)
ds = tf.data.Dataset.from_tensor_slices(frames_as_nt_tf)

# -- 3) apply a partial function to each element of the df.data.Dataset ( creating clones from a reference graph )
from subsystems.heat_flux.models.data_pipeline import construct_graph_and_gt_from_NT_with_cloning


construct_graph_and_gt_from_NT_with_cloning_part = partial(construct_graph_and_gt_from_NT_with_cloning, graph_ref)

ds2 = ds.map(construct_graph_and_gt_from_NT_with_cloning_part)\
        .repeat(NUM_REPEAT)

graph_and_gt_item_0 = next(iter(ds2))

# %%

model = create_model_v1(30)
tr_param = model.trainable_variables[0]


#%% Set Optimizer & Learning-Rate:
# !!! BEWARE !!!  This should be reviewed, because we have multiple outputs from model!

LEARNING_RATE = 0.05

# optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
optimizer = optimizers.RMSprop(learning_rate=LEARNING_RATE)

model.compile(loss='mean_absolute_error', optimizer=optimizer)

#%% Demonstration of forward computation for specified graph:

res_5 = model(graph_and_gt_item_0[0])


#%% Plot before training
plot_preview(duration=4000,
             model_result_index=1,
             variables=['T'],
             starting_time=867600.,
             ds_filepath=r'data/heat/DS02_training_0W.xlsx',
             #frames=frames,
             model=model,
             graph=graph_ref,
             use_init_graph_state=False,
             y_range=(1.84, 2.))


#%% Run the training for one epoch:

print(f'Trainable param value PRE: {tr_param.numpy()} ')

model.fit(x=ds2)

print(f'Trainable param value POST: {tr_param.numpy()} ')

