from typing import NamedTuple

import numpy as np
import tensorflow as tf

from subsystems.heat_flux.graphs.linear_system import make_graph_tensor_from_tensors
from subsystems.heat_flux.models.heat_simplified_model import ResHeatSimplifiedModel


class DataItemFromDS_v1(NamedTuple):
    time_step: np.ndarray
    time: np.ndarray
    T: np.ndarray
    #WettedLength: np.ndarray
    H: np.ndarray


def construct_graph_and_gt_from_NT(frame: DataItemFromDS_v1):
    """
    Mapping function , given a tf.data.Dataset , this function is applied to each element, takes the first row od the matrix
    and create the graph, the rest of the matrix is used for the training and testing
    The function returns a tuple ( GraphTensor, ground truth )
    :param matrix: matrix of temperatures
    :return: tuple ( GraphTensor, ground truth )
    """
    graph = make_graph_tensor_from_tensors(initial_temperatures=frame.T[0, :],
                                           time=frame.time[0][0],
                                           num_wetted_cells=tf.cast(tf.squeeze(frame.WettedLength[0][0]), tf.int32),
                                           # num_wetted_cells=frame.WettedLength[0][0],
                                           time_step=tf.cast(frame.time_step, tf.float32))
    y = ResHeatSimplifiedModel(time=frame.time, T=frame.T)
    # y = frame.T
    return (graph, y)


def construct_graph_and_gt_from_NT_with_cloning(graph_template, frame: DataItemFromDS_v1):

    context_features = graph_template.context.get_features_dict()
    node_set_features_cell = graph_template.node_sets['cells'].get_features_dict()
    node_set_features_heater = graph_template.node_sets['heater'].get_features_dict()
    edge_sets_features = {}

    context_features['time'] = [frame.time[0][0]]
    context_features['time_step'] = [tf.cast(frame.time_step, tf.float32)]

    node_set_features_cell['T'] = frame.T[0, :]

    node_set_features_heater['power'] = frame.H[0, :]


    graph_clone = graph_template.replace_features(context=context_features,
                                                  node_sets={'cells': node_set_features_cell,
                                                             'heater': node_set_features_heater},
                                                  edge_sets=edge_sets_features)

    y = ResHeatSimplifiedModel(time=frame.time, T=frame.T)

    return (graph_clone, y)





def from_samples_to_named_tuples(frames: list[dict], dst_named_tuple: NamedTuple):
    fields = dst_named_tuple._fields
    stacks = {n: [] for n in fields}

    for frame in frames:
        for field in fields:
            stacks[field].append(frame[field])

    values_for_nt = {}
    for field in fields:
        values_for_nt[field] = np.stack(stacks[field])

    result = dst_named_tuple(**values_for_nt)

    return result

    # time_steps = []
    # times = []
    # temps = []
    #
    # for frame in frames:
    #     time_step = frame['time_step']
    #     time = frame['time']
    #     T = frame['T']
    #     WettedLength = frame['WettedLength']
    #
    #     if type(time_steps) == list:
    #         time_steps = [time_step]
    #         times = [time]
    #         temps = [T]
    #     else:
    #         time_steps = np.vstack((time_steps, [time_step]))
    #         times = np.vstack((times, [time]))
    #         temps = np.vstack((temps, [T]))
    #
    # samples_nt = DataItemFromDS_v1(time_step=time_steps,
    #                                time=times,
    #                                temperatures=temps)
    # return samples_nt
