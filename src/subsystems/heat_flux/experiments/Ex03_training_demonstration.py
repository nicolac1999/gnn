from keras import optimizers
import numpy as np
from subsystems.heat_flux.models.model_v1 import *
from subsystems.heat_flux.models.data_pipeline import construct_graph_and_gt_from_NT
from subsystems.heat_flux.utils.data_helpers import sampling_heat_data, sample_generator
from subsystems.heat_flux.utils.plotting import plot_prediction_and_true_values
from subsystems.heat_flux.graphs.linear_system import make_graph_tensor_from_tensors
from lib.diag_common.tf_helpers import set_all_gpu_to_incremental_memory

set_all_gpu_to_incremental_memory()

# %%
m = create_model_v1(30)

# %%

x, y_true = sampling_heat_data('data/heat/heat_dataset_two_wetted_cells_slow.xlsx', 0, 30)

my_graph = make_graph_tensor_from_tensors(x, num_wetted_cells=2)

run_results = m(my_graph)

# %%

plot_prediction_and_true_values(x, y_true, run_results,
                                title='Temperature evolution where truth is "slow" dataset, setting thermal capacity factor=2')

# #%%
# graph_schema = tfgnn.read_schema(r'src/subsystems/heat_flux/graph_schemas/heat_flux_graph_basic.pbtxt')
# graph_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)
# train_dataset = train_dataset.map(lambda serialized: tfgnn.parse_single_example(serialized=serialized, spec=graph_spec))
# # %%
# #graph_tensor = next(iter(train_dataset))
# batch_size = 1
# batched_train_dataset = train_dataset.batch(batch_size)
# graph_tensor_batch = next(iter(batched_train_dataset))
# print(graph_tensor_batch.rank)
# scalar_graph_tensor = graph_tensor_batch.merge_batch_to_components()
# #merge_batch_to_components to solve the following error
# '''
#     GraphUpdate requires a scalar GraphTensor, that is, with `GraphTensor.rank=0`, but got `rank=1`.
#      Use GraphTensor.merge_batch_to_components() to merge all contained graphs into one contiguously
#       indexed graph of the scalar GraphTensor.
# '''


# %%

# dataset creation
x_1, y_true_1 = sampling_heat_data('data/heat/heat_dataset_two_wetted_cells_slow.xlsx', 0, 30)
x_2, y_true_2 = sampling_heat_data('data/heat/heat_dataset_two_wetted_cells_slow.xlsx', 0, 30)
x_3, y_true_3 = sampling_heat_data('data/heat/heat_dataset_two_wetted_cells_slow.xlsx', 0, 30)


matrix_1 = np.vstack([[x_1], y_true_1])
matrix_2 = np.vstack([[x_2], y_true_2])
matrix_3 = np.vstack([[x_3], y_true_3])

# %%

#stack_sices = np.stack((matrix_1, matrix_2, matrix_3))
stack_sices = np.stack((matrix_1,))
ds1 = tf.data.Dataset.from_tensor_slices(stack_sices)

next(iter(ds1))

ds2 = ds1.map(construct_graph_and_gt_from_NT) \
    .repeat(256)
# .map(lambda batch_GT, y_true: (batch_GT.merge_batch_to_components(), y_true))

ds_val = ds1.map(construct_graph_and_gt_from_NT)

it2 = iter(ds2)
item1 = next(it2)
# item2 = next(it2)
# item3 = next(it2)

# %%

m = create_model_v1(30)

# %%

m.compile(loss='mean_absolute_error',
          optimizer=optimizers.Adam(learning_rate=0.1)
          )

res_5 = m(item1[0])

# %%
m.evaluate(x=ds_val)

# %%
the_layer = m.layers[2]
trainable_param = the_layer.trainable_variables[0]
print(f'Trainable param value PRE: {trainable_param.numpy()} ')

m.fit(x=ds2)

print(f'Trainable param value POST: {trainable_param.numpy()} ')

# %%
plot_prediction_and_true_values(x_1, y_true_1, prediction_tensor=m(make_graph_tensor(x_1)),
                                title='Temperature evolution where truth is "slow" dataset, setting thermal capacity factor=2')

# %%

samples = sample_generator(r'data/heat/heat_dataset_two_wetted_cells_normal_2_time_step=1.8_thermal_fac=3.45.xlsx',
                           starting_time=0.,
                           duration=100,
                           max_frames=3,
                           samples_per_frame=30,
                           time_step=1.8,
                           stride=10.)
# %%
stack_slices = []
for k in samples.keys():
    print(k)
    matrix = tf.convert_to_tensor(samples[k], dtype=tf.float32)
    if len(stack_slices) == 0:
        stack_slices = matrix
    else:
        stack_slices = np.concatenate((stack_slices, matrix), axis=0)

# %%
ds1 = tf.data.Dataset.from_tensor_slices(stack_slices)

next(iter(ds1))

ds2 = ds1.map(construct_graph_and_gt_from_NT) \
    .repeat(256)