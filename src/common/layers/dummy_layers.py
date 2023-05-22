from typing import Tuple

import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow import keras
from tensorflow_gnn import keras as keras_gnn, GraphTensor, FieldOrFields
from tensorflow_gnn.graph.graph_constants import FieldsNest


class DummyEdgesToNodePoolingLayer(tf.keras.layers.Layer):
    """
    This is just a 'dummy' Layer to be placed as a 'next_state' argument in the tfgnn.keras.layers.NodeSetUpdate(...)
    call. It prints out the input arguments, so you can verify if they are compatible with your Layer implementation.
    """

    def __init__(self, debug_tag: str):
        super().__init__()
        self.debug_tag = debug_tag

    def call(self, graph: GraphTensor, edge_set_name) -> FieldOrFields:

        tf.print(f'{self.debug_tag} @ Graph, edge_set_name:\nG--> ', graph, "\nE-->", edge_set_name)
        return {}



class DummyNodeNewStateProcessingLayer(tf.keras.layers.Layer):
    """
    This is just a 'dummy' Layer to be placed as a 'next_state' argument in the tfgnn.keras.layers.NodeSetUpdate(...)
    call. It prints out the input arguments, so you can verify if they are compatible with your Layer implementation.
    """

    def __init__(self, debug_tag: str):
        super().__init__()
        self.debug_tag = debug_tag


    def call(self, inputs: Tuple[FieldsNest, FieldsNest, FieldsNest]):
        node_inputs, edge_inputs, context_inputs = inputs

        tf.print(f'{self.debug_tag} @ Node, Edge, Context inputs:\nN-->', node_inputs, "\nE-->", edge_inputs, "\nC-->", context_inputs)

        return {}


class DummyNodeNewState_SaveMessagesAsNodeFeatures_Layer(tf.keras.layers.Layer):
    """
    This 'dummy' layer enables you to 'inject' received messages from edges as new node features for further examination.
    """

    def __init__(self, node_set_name, debug_tag: str = None):
        super().__init__()
        self.node_set_name = node_set_name
        self.debug_tag = debug_tag


    def call(self, inputs: Tuple[FieldsNest, FieldsNest, FieldsNest]):
        node_inputs, edge_inputs, context_inputs = inputs

        result = {}
        for edgeset_name, edgeset_features in edge_inputs.items():
            for feat_name, feat_value in edgeset_features.items():
                result[f'{edgeset_name}_{feat_name}'] = feat_value

        return result


