import tensorflow as tf
from typing import Tuple
from tensorflow_gnn.graph.graph_constants import FieldsNest
from tensorflow import keras

class ThermalCapacityComputation(keras.layers.Layer):
    '''
    Preprocessing layer, it computes the thermal capacity of each magnet, where
    thermal capacity = specific heat capacity * mass
    When this feature is computed than it is made available as magnets feature
    '''
    def __init__(self):
        super().__init__()

    def call(self, inputs: Tuple[FieldsNest, FieldsNest, FieldsNest]):
        node_input_feature, edge_inputs, context_input_features = inputs

        thermal_capacity = node_input_feature * context_input_features

        return {'thermal capacity': thermal_capacity}

