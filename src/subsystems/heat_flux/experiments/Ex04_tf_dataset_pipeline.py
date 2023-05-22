import tensorflow as tf
import numpy as np

from subsystems.heat_flux.graphs.linear_system import make_graph_tensor_from_tensors
from subsystems.heat_flux.models.model_v1 import create_model_v1
from keras import optimizers
import tensorflow_gnn as tfgnn
from subsystems.heat_flux.graphs.linear_system import make_graph_tensor, make_graph_tensor_from_tensors
from subsystems.heat_flux.models.model_v1 import create_model_v1
from subsystems.heat_flux.utils.data_helpers import sample_generator
from subsystems.heat_flux.utils.plotting import plot_preview
from subsystems.heat_flux.utils.simulation import simulate_time_period
import matplotlib.pyplot as plt

# %% Load frames from dataset:

frames = sample_generator(ds_filepath=r'data/heat/heat_dataset wet=2,t=1.8,K=3.45.xlsx',
                          variables=['T', 'WettedLength'],
                          starting_time=0, duration=None,
                          max_frames=3, samples_per_frame=31, time_step=0.5, stride=20)


# %% Debug :: evaluation on hand-created graph and model:

frame = frames[0]

g0 = make_graph_tensor_from_tensors(initial_temperatures=tf.constant(frame['T'][0, :], dtype=tf.float32),
                                    num_wetted_cells=tf.cast(tf.squeeze(frame['WettedLength'][0][0]), tf.int32),
                                    time=tf.constant(frame['time'][0][0], dtype=tf.float32),
                                    time_step=tf.constant(frame['time_step'], dtype=tf.float32))

model = create_model_v1(30)
res = model(g0)

# plot_preview(data_source=ds_filepath, starting_time_point=0, duration=100, time_step=0.5, model=model, graph=g0, title='Single run')


# %%

# %% Stack frames together and embed them into NamedTuple:

from subsystems.heat_flux.models.data_pipeline import from_samples_to_named_tuples, DataItemFromDS_v1, \
    construct_graph_and_gt_from_NT
frames_as_nt = from_samples_to_named_tuples(frames, DataItemFromDS_v1)

# %% Convert Numpy arrays into tf.Tensor and initialize a tf.data.Dataset:

frames_as_nt_tf = tf.nest.map_structure(tf.convert_to_tensor, frames_as_nt)
ds = tf.data.Dataset.from_tensor_slices(frames_as_nt_tf)
item0 = next(iter(ds))

graph_item0 = construct_graph_and_gt_from_NT(item0)

# %%

ds2 = ds.map(construct_graph_and_gt_from_NT)


