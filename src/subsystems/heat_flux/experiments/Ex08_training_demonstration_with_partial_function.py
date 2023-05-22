import tensorflow as tf
import tensorflow_gnn as tfgnn
from functools import partial


from subsystems.heat_flux.models.model_v1 import create_model_v1
from subsystems.heat_flux.utils.data_helpers import sample_generator

# %%

graph_ref = tfgnn.GraphTensor.from_pieces(
    context=tfgnn.Context.from_fields(
        features={'time': [0.],
                  'time_step': [0.2],
                  'specific heat capacity': [1.]}),
    node_sets={"cells": tfgnn.NodeSet.from_fields(sizes=[4],
                                                  features={
                                                      "T": [1.90, 1.90, 1.90, 1.90],
                                                      "Q": [0., 0., 0., 0.],
                                                      "mass": [400., 400., 400., 400.],
                                                      'L': [1., 1., 1., 1.]
                                                  }),
               "heater": tfgnn.NodeSet.from_fields(sizes=[4],
                                                   features={
                                                       "power": [1., 1., 1., 1.],
                                                   }),
               "liquid": tfgnn.NodeSet.from_fields(sizes=[4],
                                                   features={
                                                       "extraction capacity": [5., 5., .0, 0.],
                                                       "T": [1.85, 1.85, 1.85, 1.85]
                                                   })},
    edge_sets={"conduction": tfgnn.EdgeSet.from_fields(sizes=[3],
                                                       features={
                                                           "conductivity": [100., 100., 100., 100., 100., 100.],
                                                           "L": [1., 1., 1., 1., 1., 1.],
                                                           "A": [1., 1., 1., 1., 1., 1.],
                                                           "E_transf": [0., 0., 0., 0., 0., 0.],
                                                           "Q_transf": [0., 0., 0., 0., 0., 0.]
                                                       },
                                                       adjacency=tfgnn.Adjacency.from_indices(
                                                           source=("cells", [0, 1, 2, 1, 2, 3]),
                                                           target=("cells", [1, 2, 3, 0, 1, 2]))
                                                       ),
               "heat supplied": tfgnn.EdgeSet.from_fields(sizes=[4],
                                                          features={},
                                                          adjacency=tfgnn.Adjacency.from_indices(
                                                              source=("heater", [0, 1, 2, 3]),
                                                              target=("cells", [0, 1, 2, 3])
                                                          )),
               "cell2liquid": tfgnn.EdgeSet.from_fields(sizes=[4],
                                                           features={
                                                               "conductivity": [1200., 1200., 1200., 1200.],
                                                           },
                                                           adjacency=tfgnn.Adjacency.from_indices(
                                                               source=("cells", [0, 1, 2, 3]),
                                                               target=("liquid", [0, 1, 2, 3])
                                                           ))}
)


# %% Load frames from dataset:

frames = sample_generator(ds_filepath=r'data/heat/heat_dataset wet=2,t=1.8,K=3.45.xlsx',
                          variables=['T', 'WettedLength'],
                          starting_time=0, duration=None,
                          max_frames=3, samples_per_frame=31, time_step=0.5, stride=20)


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


#%% Model creation:

# model = create_model_v1(30)
# tr_param = model.trainable_variables[0]

from subsystems.heat_flux.models.heat_simplified_model import HeatSimplifiedModel
model = HeatSimplifiedModel(num_steps=30)
tr_param = model.trainable_variables[0]


#%% Set Optimizer & Learning-Rate:
# !!! BEWARE !!!  This should be reviewed, because we have multiple outputs from model!
from keras import optimizers

LEARNING_RATE = 0.1

# optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
optimizer = optimizers.RMSprop(learning_rate=LEARNING_RATE)

model.compile(loss='mean_absolute_error', optimizer=optimizer)

#%% Run the training for one epoch:

print(f'Trainable param value PRE: {tr_param.numpy()} ')

model.fit(x=ds2)

print(f'Trainable param value POST: {tr_param.numpy()} ')