import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn import keras as keras_gnn
from subsystems.heat_flux.layers.message_passing_layers import MessageFactoryConductionHeliumBath, MessageFactoryHelium2Bayonet,\
    MessagePassingHeatSupplied
from subsystems.heat_flux.layers.updates_layers import IncreaseTemperature
from common.layers.time_management import IncreaseTimeLayer
from subsystems.heat_flux.layers.pre_processing_layers import ThermalCapacityComputation
from subsystems.heat_flux.models.heat_simplified_model import ResHeatSimplifiedModel
from tensorflow import keras

def create_model_v1(num_steps, include_initial_step=True):
    results = []
    times = []
    #messages_magnet_to_magnet = []

    graph_schema = tfgnn.read_schema(r"src/subsystems/heat_flux/graph_schemas/heat_flux_graph_basic.pbtxt")
    graph_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)


    input = keras.layers.Input(type_spec=graph_spec)

    preprocessing_step = keras_gnn.layers.GraphUpdate(
        node_sets={
            'cells': keras_gnn.layers.NodeSetUpdate(
                edge_set_inputs={},
                next_state=ThermalCapacityComputation(),
                node_input_feature='mass',
                context_input_feature="specific_heat_capacity"
            )
        }
    )

    graph = preprocessing_step(input)

    if include_initial_step:
        results.append(keras_gnn.layers.Readout(node_set_name="cells", feature_name='T')(graph))
        times.append(keras_gnn.layers.Readout(from_context=True, feature_name='time')(graph))
        #messages_magnet_to_magnet.append(keras_gnn.layers.Readout(edge_set_name='conduction', feature_name='E_transf')(graph))

    one_step_processing = keras_gnn.layers.GraphUpdate(
        node_sets={
            "cells": keras_gnn.layers.NodeSetUpdate(
                edge_set_inputs={"conduction": MessageFactoryConductionHeliumBath(),
                                 "heat supplied": MessagePassingHeatSupplied(),
                                 "cell2liquid": MessageFactoryHelium2Bayonet()},
                next_state=IncreaseTemperature(),
                node_input_feature=['T', 'thermal capacity', 'L'],
                context_input_feature=['static_heat', 'time_step']
            )
        },
        context=keras_gnn.layers.ContextUpdate(
            node_set_inputs={},
            next_state=IncreaseTimeLayer(),
            context_input_feature=['time', 'time_step']
        ),
    )

    for i in range(num_steps):
        graph = one_step_processing(graph)
        output = keras_gnn.layers.Readout(node_set_name="cells", feature_name='T')(graph)
        results.append(output)
        new_time_stamp = keras_gnn.layers.Readout(from_context=True, feature_name='time')(graph)
        times.append(new_time_stamp)
        #E_transf = keras_gnn.layers.Readout(edge_set_name='conduction', feature_name='E_transf')(graph)
        #messages_magnet_to_magnet.append(E_transf)

    if include_initial_step:
        stack_of_steps = tf.reshape(keras.layers.Concatenate()(results), [num_steps + 1, -1])
        stack_of_time = tf.reshape(keras.layers.Concatenate()(times), [num_steps + 1, -1])
        #stack_of_messages_magnet_to_magnet = tf.reshape(keras.layers.Concatenate()(messages_magnet_to_magnet), [num_steps + 1, -1])

    else:
        stack_of_steps = tf.reshape(keras.layers.Concatenate()(results), [num_steps, -1], name='T')
        stack_of_time = tf.reshape(keras.layers.Concatenate()(times), [num_steps, -1], name='time')
        #stack_of_messages_magnet_to_magnet = tf.reshape(keras.layers.Concatenate()(messages_magnet_to_magnet), [num_steps, -1])

    # output = keras_gnn.layers.Readout(node_set_name="cells", feature_name='T')(graph)

    model = keras.Model(input, outputs=ResHeatSimplifiedModel(stack_of_time, stack_of_steps)) #, stack_of_messages_magnet_to_magnet])

    return model

