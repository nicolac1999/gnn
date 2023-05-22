import tensorflow as tf
from tensorflow import keras
from keras import optimizers

from subsystems.heat_flux.models.model_v1 import *
from subsystems.heat_flux.utils.data_helpers import sampling_heat_data, sample_generator
from subsystems.heat_flux.utils.plotting import plot_preview
from lib.diag_common.tf_helpers import set_all_gpu_to_incremental_memory

set_all_gpu_to_incremental_memory()


#%% Load frames from dataset:

ds_filepath = r'data/heat/heat_dataset wet=2,t=1.8,K=3.45.xlsx'
frames = sample_generator(ds_filepath=ds_filepath,
                          variables=['T', 'WettedLength'],
                          starting_time=0, duration=None,
                          max_frames=2, samples_per_frame=31, time_step=0.75, stride=5)

#%% tf.data.Dataset preparation:

NUM_REPEATE = 16

# -- 1) Stack frames together and embed them into NamedTuple:
from subsystems.heat_flux.models.data_pipeline import from_samples_to_named_tuples, DataItemFromDS_v1, construct_graph_and_gt_from_NT
frames_as_nt = from_samples_to_named_tuples(frames, DataItemFromDS_v1)


# -- 2) Convert Numpy arrays into tf.Tensor and initialize a tf.data.Dataset:

frames_as_nt_tf = tf.nest.map_structure(tf.convert_to_tensor, frames_as_nt)
ds = tf.data.Dataset.from_tensor_slices(frames_as_nt_tf)
item0_NT = next(iter(ds))

graph_item0 = construct_graph_and_gt_from_NT(item0_NT)

ds2 = ds.map(construct_graph_and_gt_from_NT)\
        .repeat(NUM_REPEATE)

graph_and_gt_item_0 = next(iter(ds2))


#%% Model creation:

model = create_model_v1(30)
tr_param = model.trainable_variables[0]

#from subsystems.heat_flux.models.heat_simplified_model import HeatSimplifiedModel
#model = HeatSimplifiedModel(num_steps=30)
#tr_param = model.trainable_variables[0]

#%% Set Optimizer & Learning-Rate:
# !!! BEWARE !!!  This should be reviewed, because we have multiple outputs from model!

LEARNING_RATE = 0.05

# optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
optimizer = optimizers.RMSprop(learning_rate=LEARNING_RATE)

model.compile(loss='mean_absolute_error', optimizer=optimizer)


#%% Demonstration of forward computation for specified graph:

res_5 = model(graph_and_gt_item_0[0])

#%% Plot before training
plot_preview(duration=100,
             model_result_index=1,
             variables=['T'],
             starting_time=0.,
             ds_filepath=r'data/heat/heat_dataset wet=2,t=1.8,K=3.45.xlsx',
             frames=frames,
             model=model,
             graph=graph_and_gt_item_0[0],
             use_init_graph_state=True)


#%% Run the training for one epoch:

print(f'Trainable param value PRE: {tr_param.numpy()} ')

model.fit(x=ds2)

print(f'Trainable param value POST: {tr_param.numpy()} ')

#%% Plot after training
plot_preview(duration=100,
             model_result_index=1,
             variables=['T'],
             starting_time=0.,
             ds_filepath=r'data/heat/heat_dataset wet=2,t=1.8,K=3.45.xlsx',
             frames=frames,
             model=model,
             graph=graph_and_gt_item_0[0],
             use_init_graph_state=True)



