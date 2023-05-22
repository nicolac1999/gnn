import tensorflow_gnn as tfgnn
from subsystems.heat_flux.models.model_v1 import create_model_v1
from subsystems.heat_flux.utils.simulation import \
    simulate_time_period
from subsystems.heat_flux.utils.obsolete import repeat_model, make_interpolation_tensors
from subsystems.heat_flux.utils.data_helpers import save_to_excel, sample_generator
from subsystems.heat_flux.utils.plotting import plot_temperatures_time_series, plot_preview

# %%
my_graph = tfgnn.GraphTensor.from_pieces(
    context=tfgnn.Context.from_fields(
        features={'time': [0.],
                  'time_step': [0.2],
                  'specific heat capacity': [1.]}),
    node_sets={"cells": tfgnn.NodeSet.from_fields(sizes=[4],
                                                  features={
                                                      "T": [1.90, 1.90, 1.90, 1.90],
                                                      "Q": [0., 0., 0., 0.],
                                                      "mass": [400., 400., 400., 400.]
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


# graph_schema = tfgnn.read_schema(r'src/subsystems/heat_flux/graph_schemas/heat_flux_graph_basic.pbtxt')
# graph_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)
# graph_spec.is_compatible_with(my_graph)
# %%
#creation of a model, run the model on a defined graph, and plot the evolution
model = create_model_v1(30, include_initial_step=True)

model_outputs = model(my_graph)

plot_temperatures_time_series(model_outputs)

# %%
# creation of the model, the model is repeated 3 times and the results are plotted
model_2 = create_model_v1(30, include_initial_step=True)
model_2_outputs = repeat_model(my_graph, model_2, 3, include_initial_step=True)
plot_temperatures_time_series(model_2_outputs)

#%%
#list_tensors = outputs.values()
#concat_outputs = tf.concat([t for t in list_tensors], axis=0)
#concat_outputs_with_initial_temp = tf.concat([[my_graph.node_sets['cells']['T']], concat_outputs], axis=0)
#plot_temperatures_time_series(my_graph.node_sets['cells']['T'], concat_outputs)
#plot_temperatures_time_series(concat_outputs_with_initial_temp)


#concat_outputs_second_rows = concat_outputs_with_initial_temp[::2]
#plot_temperatures_time_series(my_graph.node_sets['cells']['T'], concat_outputs_second_rows)
#plot_temperatures_time_series(concat_outputs_second_rows)

#df_1 = pd.DataFrame(concat_outputs_with_initial_temp)
#df_1.to_excel('data/heat/heat_dataset_two_wetted_cells_normal.xlsx')

#df_2 = pd.DataFrame(concat_outputs_second_rows)
#df_2.to_excel('data/heat/heat_dataset_two_wetted_cells_fast.xlsx')


# interpolated_values = make_interpolation_tensors(concat_outputs_with_initial_temp)
# plot_temperatures_time_series(interpolated_values)
# df_3 = pd.DataFrame(interpolated_values)
# df_3.to_excel('data/heat/heat_dataset_two_wetted_cells_slow.xlsx')


save_to_excel(model_2_outputs, 'data/heat/heat_dataset_two_wetted_cells_normal_2.xlsx')
# %%

data_to_save_fast = model_2_outputs[::2]
save_to_excel(data_to_save_fast, 'data/heat/heat_dataset_two_wetted_cells_fast_2.xlsx')

# %%

interpolated_values = make_interpolation_tensors(model_2_outputs)

plot_temperatures_time_series(interpolated_values)

# %%
d = simulate_time_period(model, my_graph, 0., 10., time_step=0.3, clip_simulation=False)

# %%

#df = load_row_from_dataset(r'data/heat/heat_dataset_two_wetted_cells_normal_2.xlsx', 12.3, 40.5)
plot_preview(r'data/heat/heat_dataset_two_wetted_cells_normal_2.xlsx',
             #model=model,
             graph=my_graph,
             starting_time=0.,
             duration=30.,
             time_step=1.5,
             x_range=(0, 60),
             y_range=(1.84, 1.95))

# %%
m7 = create_model_v1(30)
m7.trainable_variables[0].assign(3.45)
# plot_preview(r'data/heat/heat_dataset_two_wetted_cells_normal_2.xlsx',
#              model=m7,
#              graph=my_graph,
#              starting_time_point=0.,
#              duration=120.,
#              time_step=.4,
#              x_range=(0, 60),
#              y_range=(1.84, 1.95))

dict_to_save = simulate_time_period(m7, my_graph, 0., 120., 1.8)
save_to_excel(dict_to_save, r'data/heat/heat_dataset wet=2,t=1.8,K=3.45.xlsx')
# %%

df_2 = sample_generator(r'data/heat/heat_dataset_two_wetted_cells_normal_2.xlsx', 10, 40.5, 0, 1, 60, 1.)

samples = sample_generator(r'data/heat/heat_dataset_two_wetted_cells_normal_2_time_step=1.8_thermal_fac=3.45.xlsx', 0, 100, 3, 30, 0.4)
