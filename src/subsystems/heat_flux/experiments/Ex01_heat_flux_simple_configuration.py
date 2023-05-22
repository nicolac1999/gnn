import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow import keras

import pandas as pd
import seaborn as sns

'''
The script is going to perform heat transfer between nodes , the purpose is to show a gradient on the temperatures
'''

# %%

graph = tfgnn.GraphTensor.from_pieces(
    context=tfgnn.Context.from_fields(
        features={'time_step': [1.]}),
    node_sets={"cells": tfgnn.NodeSet.from_fields(sizes=[4],
                                                  features={
                                                      "T": [1.90, 1.92, 1.94, 1.90],
                                                      "Q": [0., 0., 0., 0.],
                                                      "m": [400., 400., 400., 400.]
                                                  }),
               "heater": tfgnn.NodeSet.from_fields(sizes=[4],
                                                   features={
                                                       "power": [1., 1., 1., 1.],
                                                   }),
               "liquid": tfgnn.NodeSet.from_fields(sizes=[4],
                                                   features={
                                                       "extraction capacity": [5., 5., 0., 0.],
                                                       "T": [1.85, 1.85, 1.85, 1.85]
                                                   })},
    edge_sets={"conduction": tfgnn.EdgeSet.from_fields(sizes=[6],
                                                       features={
                                                           "capacity": [100., 100., 100., 100., 100., 100.]
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
                                                               "capacity": [1200., 1200., 0., 0.],
                                                               "extraction capacity": [5., 5., 0., 0.]
                                                           },
                                                           adjacency=tfgnn.Adjacency.from_indices(
                                                               source=("cells", [0, 1, 2, 3]),
                                                               target=("liquid", [0, 1, 2, 3])
                                                           ))}
)

# %%

cell_features = graph.node_sets['cells'].get_features_dict()
energy_supplied = graph.node_sets["heater"]["power"] * graph.context["time_step"]

# magnet_temperatures = graph.node_sets["cells"]["T"]
# magnet_temperatures_shift = tf.concat([[0.], magnet_temperatures[0:-1]], axis=0)
# delta_T_magnets = tf.math.subtract(magnet_temperatures, magnet_temperatures_shift)

edge_capacity = graph.edge_sets["conduction"]["capacity"]

liquid_temperatures = graph.node_sets["liquid"]["T"]
# delta_T_magnet_liquid = tf.math.subtract(magnet_temperatures, liquid_temperatures)


edge_liquid_capacity = graph.edge_sets["cell2liquid"]["capacity"]

# energy_extracted = edge_liquid_capacity * delta_T_magnet_liquid

max_amount_extraction = graph.edge_sets["cell2liquid"]["extraction capacity"]
# energy_extracted = tf.minimum(energy_extracted, max_amount_extraction)

# %%

'''
message passing steps
1) broadcasting
'''


energy_supplied_broadcast = tfgnn.broadcast_node_to_edges(graph,
                                                          "heat supplied",
                                                          tfgnn.SOURCE,
                                                          feature_value=energy_supplied)

T_broadcast_source = tfgnn.broadcast_node_to_edges(graph,
                                                  "conduction",
                                                  tfgnn.SOURCE,
                                                  feature_name='T')
T_broadcast_target = tfgnn.broadcast_node_to_edges(graph,
                                                  "conduction",
                                                  tfgnn.TARGET,
                                                  feature_name='T')

delta_T_magnets = T_broadcast_source - T_broadcast_target
delta_T_magnets = keras.activations.relu(delta_T_magnets)

energy_conduction = delta_T_magnets * edge_capacity


T_broadcast_magnets_to_liquid = tfgnn.broadcast_node_to_edges(graph,
                                                  "cell2liquid",
                                                  tfgnn.SOURCE,
                                                  feature_name='T')
T_broadcast_liquid = tfgnn.broadcast_node_to_edges(graph,
                                                  "cell2liquid",
                                                  tfgnn.TARGET,
                                                  feature_name='T')

delta_T_magnet_liquid = T_broadcast_magnets_to_liquid - T_broadcast_liquid
energy_extracted = edge_liquid_capacity * delta_T_magnet_liquid

max_amount_extraction = graph.edge_sets["cell2liquid"]["extraction capacity"]
energy_extracted_liquid = tf.minimum(energy_extracted, max_amount_extraction)

# %%

'''
2) aggregation
'''


heat_supplied = tfgnn.pool_edges_to_node(graph,
                                         "heat supplied",
                                         tfgnn.TARGET,
                                         feature_value=energy_supplied_broadcast)

heat_sent_conduction = tfgnn.pool_edges_to_node(graph,
                                                "conduction",
                                                tfgnn.SOURCE,
                                                feature_value=energy_conduction)

heat_received_conduction = tfgnn.pool_edges_to_node(graph,
                                                    "conduction",
                                                    tfgnn.TARGET,
                                                    feature_value=energy_conduction)

heat_extracted_sent = tfgnn.pool_edges_to_node(graph,
                                               "cell2liquid",
                                               tfgnn.SOURCE,
                                               feature_value=energy_extracted_liquid)
heat_extracted_received = tfgnn.pool_edges_to_node(graph,
                                                   "cell2liquid",
                                                   tfgnn.TARGET,
                                                   feature_value=energy_extracted_liquid)

# %%

'''
3) node updates
'''

heat_sum_magnets = heat_supplied \
                   - heat_sent_conduction \
                   + heat_received_conduction \
                   - heat_extracted_sent

print(heat_sum_magnets)

temp_variation = heat_sum_magnets / cell_features['m']

print(temp_variation)

new_temp = cell_features['T'] + temp_variation
cell_features['T'] = new_temp

graph = graph.replace_features(node_sets={"cells": cell_features})
print(cell_features["T"])


# %%
########################################################################################################
########################################################################################################

########################################  SIMULATION  ##################################################

########################################################################################################
########################################################################################################




graph = tfgnn.GraphTensor.from_pieces(
    context=tfgnn.Context.from_fields(
        features={'time_step': [1.]}),
    node_sets={"cells": tfgnn.NodeSet.from_fields(sizes=[4],
                                                  features={
                                                      "T": [1.90, 1.90, 1.90, 1.90],
                                                      "Q": [0., 0., 0., 0.],
                                                      "m": [400., 400., 400., 400.]
                                                  }),
               "heater": tfgnn.NodeSet.from_fields(sizes=[4],
                                                   features={
                                                       "power": [1., 1., 1., 1.],
                                                   }),
               "liquid": tfgnn.NodeSet.from_fields(sizes=[4],
                                                   features={
                                                       "extraction capacity": [5., 5., 5., 5.],
                                                       "T": [1.85, 1.85, 1.85, 1.85]
                                                   })},
    edge_sets={"conduction": tfgnn.EdgeSet.from_fields(sizes=[3],
                                                       features={
                                                           "capacity": [100., 100., 100.]
                                                       },
                                                       adjacency=tfgnn.Adjacency.from_indices(
                                                           source=("cells", [1, 2, 3]),
                                                           target=("cells", [0, 1, 2]))
                                                       ),
               "heat supplied": tfgnn.EdgeSet.from_fields(sizes=[4],
                                                          features={},
                                                          adjacency=tfgnn.Adjacency.from_indices(
                                                              source=("heater", [0, 1, 2, 3]),
                                                              target=("cells", [0, 1, 2, 3])
                                                          )),
               "cell2liquid": tfgnn.EdgeSet.from_fields(sizes=[4],
                                                           features={
                                                               "capacity": [1200., 1200., 1200., 1200.],
                                                               "extraction capacity": [5., 5., 0., 0.]
                                                           },
                                                           adjacency=tfgnn.Adjacency.from_indices(
                                                               source=("cells", [0, 1, 2, 3]),
                                                               target=("liquid", [0, 1, 2, 3])
                                                           ))}
)

list_temp = []
list_temp.append(graph.node_sets["cells"]['T'].numpy())


for i in range(30):

    cell_features = graph.node_sets["cells"].get_features_dict()

    energy_supplied = graph.node_sets["heater"]["power"] * graph.context["time_step"]

    magnet_temperatures = graph.node_sets["cells"]["T"]
    magnet_temperatures_shift = tf.concat([[0.], magnet_temperatures[0:-1]], axis=0)
    delta_T_magnets = tf.math.subtract(magnet_temperatures, magnet_temperatures_shift)

    edge_capacity = graph.edge_sets["conduction"]["capacity"]

    liquid_temperatures = graph.node_sets["liquid"]["T"]
    delta_T_magnet_liquid = tf.math.subtract(magnet_temperatures, liquid_temperatures)


    edge_liquid_capacity = graph.edge_sets["cell2liquid"]["capacity"]

    energy_extracted = edge_liquid_capacity * delta_T_magnet_liquid

    max_amount_extraction = graph.edge_sets["cell2liquid"]["max amount"]
    energy_extracted = tf.minimum(energy_extracted, max_amount_extraction)


    '''
    message passing steps
    1) broadcasting
    '''


    energy_supplied_broadcast = tfgnn.broadcast_node_to_edges(graph,
                                                              "heat supplied",
                                                              tfgnn.SOURCE,
                                                              feature_value=energy_supplied)

    delta_T_broadcast = tfgnn.broadcast_node_to_edges(graph,
                                                      "conduction",
                                                      tfgnn.SOURCE,
                                                      feature_value=delta_T_magnets)

    energy_conduction = delta_T_broadcast * edge_capacity

    energy_extracted_liquid = tfgnn.broadcast_node_to_edges(graph,
                                                            "cell2liquid",
                                                            tfgnn.SOURCE,
                                                            feature_value=energy_extracted)

    '''
    2) aggregation
    '''


    heat_supplied = tfgnn.pool_edges_to_node(graph,
                                             "heat supplied",
                                             tfgnn.TARGET,
                                             feature_value=energy_supplied_broadcast)

    heat_sent_conduction = tfgnn.pool_edges_to_node(graph,
                                                    "conduction",
                                                    tfgnn.SOURCE,
                                                    feature_value=energy_conduction)

    heat_received_conduction = tfgnn.pool_edges_to_node(graph,
                                                        "conduction",
                                                        tfgnn.TARGET,
                                                        feature_value=energy_conduction)

    heat_extracted_sent = tfgnn.pool_edges_to_node(graph,
                                                   "cell2liquid",
                                                   tfgnn.SOURCE,
                                                   feature_value=energy_extracted_liquid)
    heat_extracted_received = tfgnn.pool_edges_to_node(graph,
                                                       "cell2liquid",
                                                       tfgnn.TARGET,
                                                       feature_value=energy_extracted_liquid)


    '''
    3) node updates
    '''

    heat_sum_magnets = heat_supplied \
                       - heat_sent_conduction \
                       + heat_received_conduction \
                       - heat_extracted_sent

    #print(heat_sum_magnets)

    temp_variation = heat_sum_magnets / cell_features['m']

    #print(temp_variation)

    new_temp = tf.maximum(cell_features['T'] + temp_variation, liquid_temperatures)
    cell_features['T'] = new_temp

    graph = graph.replace_features(node_sets={"cells": cell_features})
    #print(cell_features["T"])
    list_temp.append(new_temp.numpy())


# %%

fig, axs = plt.subplots(30, 1, figsize=(5, 70))
plt.subplots_adjust(hspace=0.8, top=0.97)
k = 0
for i in range(30):
    axs[k].bar([1, 2, 3, 4], list_temp[i])
    axs[k].set_ylim(1.85, 1.92)
    axs[k].set_xlabel('magnets')
    axs[k].set_ylabel('temperature [ K ]')
    axs[k].set_yticks(np.arange(1.85, 1.92, 0.01))
    axs[k].set_xticks([1, 2, 3, 4])
    axs[k].text(0.5, 1.91, f'step {i}')
    k = k + 1

fig.suptitle('Magnets temperatures with constant incoming heat and fluid covering first magnet')
fig.savefig('heat_flux_temperatures_2nodes_wetted.png', dpi=fig.dpi,  bbox_inches='tight', pad_inches=0.5)

# %%

plt.figure()
df = pd.DataFrame(list_temp)
sns.lineplot(df, markers=True)
plt.xlabel("time step")
plt.ylabel("temperature [ K ]")
sns.set_theme()
plt.title('Temperatures evolution 4 magnets "wetted" ')
plt.show()


