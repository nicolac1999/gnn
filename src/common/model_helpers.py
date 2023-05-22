"""
This file is intended to contain various helper functions that are useful for GNN Model-building, configuring, running,
etc.
"""
from collections import namedtuple

import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn import GraphTensor, keras as keras_gnn
from tensorflow_gnn import keras as keras_gnn


def readout_node_values_as_slice(graph: GraphTensor, node_set_name: str, feature_name: str):
    """
    This method will create a Readout layer for a given node_set_name and feature_name, and will
    perform 'reshaping' so that the returned tensor contains a "slicing" dimension as first one.

    We can call returned value as "slice", because it has first dime set to 1. One can stack these "slices"
    on top of each other to obtain, e.g., time-evolution of that particular feature values.

    It calls tf.expand_dims(..., axis=0) to add first "slicing" dimension

    :param graph:
    :param node_set_name:
    :param feature_name:
    :return:
    """
    result = tf.expand_dims(
        keras_gnn.layers.Readout(node_set_name=node_set_name, feature_name=feature_name)(graph),
        axis=0
    )
    return result

def readout_edge_values_as_slice(graph: GraphTensor, edge_set_name: str, feature_name: str):
    """
    This method will create a Readout layer for a given edge_set_name and feature_name, and will
    perform 'reshaping' so that the returned tensor contains a "slicing" dimension as first one.

    We can call returned value as "slice", because it has first dime set to 1. One can stack these "slices"
    on top of each other to obtain, e.g., time-evolution of that particular feature values.

    It calls tf.expand_dims(..., axis=0) to add first "slicing" dimension

    :param graph:
    :param edge_set_name:
    :param feature_name:
    :return:
    """
    result = tf.expand_dims(
        keras_gnn.layers.Readout(edge_set_name=edge_set_name, feature_name=feature_name)(graph),
        axis=0
    )
    return result

def readout_context_values_as_slice(graph: GraphTensor, feature_name: str):
    """
    This method will create a Readout layer for given feature_name from Context, and will
    perform 'reshaping' so that the returned tensor contains a "slicing" dimension as first one.

    We can call returned value as "slice", because it has first dime set to 1. One can stack these "slices"
    on top of each other to obtain, e.g., time-evolution of that particular feature values.

    It calls tf.expand_dims(..., axis=0) to add first "slicing" dimension

    :param graph:
    :param node_set_name:
    :param feature_name:
    :return:
    """
    result = tf.expand_dims(
        keras_gnn.layers.Readout(from_context=True, feature_name=feature_name)(graph),
        axis=0
    )
    return result


def create_readout_layers_for_all_node_features(graph: tfgnn.GraphTensor):
    """
    This function creates a NamedTuple, that contains all features from all node-sets.
    The naming convention is: `node-set-name`__`feature-name`

    This function is very useful when you would like to return 'debug-output' -- for inspection all the node features.
    :param graph:
    :return:
    """
    all_output_layers = []
    named_tuple_fields = []
    for node_set_name, ns in graph.node_sets.items():
        for feature_name in ns.features.keys():
            out_layer = keras_gnn.layers.Readout(node_set_name=node_set_name, feature_name=feature_name)(graph)
            # out_layer = readout_node_values_as_slice(graph, node_set_name, feature_name)
            all_output_layers.append(out_layer)
            named_tuple_fields.append(f'{node_set_name}__{feature_name}')
        pass

    ResultNTType = namedtuple("ResultNTType", named_tuple_fields)

    return ResultNTType(*all_output_layers)
