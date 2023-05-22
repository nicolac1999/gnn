from typing import Tuple

import tensorflow as tf
from tensorflow_gnn.graph.graph_constants import FieldsNest


class IncreaseTimeLayer(tf.keras.layers.Layer):
    '''
    This keras layer is devoted to increase the time stamp of the graph at each iteration.
    '''

    def __init__(self):
        super().__init__()

    def call(self, inputs: Tuple[FieldsNest, FieldsNest, FieldsNest]):
        context_input_features, node_input_feature, edge_inputs = inputs

        new_time_stamp = context_input_features['time'] + context_input_features['time_step']

        return {'time': new_time_stamp}
