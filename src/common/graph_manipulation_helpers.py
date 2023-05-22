from typing import NamedTuple

import numpy as np
import tensorflow_gnn as tfgnn
from tensorflow_gnn import GraphTensor
from collections import defaultdict
from mergedeep import merge


# %%

def set_graph_features(graph: tfgnn.GraphTensor, dict_features: dict) -> tfgnn.GraphTensor:
    """
    :param graph: graph to edit
    :param dict_features: dictionary of the features to update where
                          key = 'node_set_name.feature_name'
                          value = tf.Tensor or tf.RaggedTensor

    :return: graph with updated features
    """
    FEATURE_NAME_SEPARATOR = '__'


    list_node_sets_name = [str(node_name) for node_name in graph.node_sets.keys()]
    list_edge_sets_name = [str(edge_name) for edge_name in graph.edge_sets.keys()]

    FEATURES_TO_EDIT = list(dict_features.keys())

    new_nodesets_features = defaultdict(dict)
    new_edgesets_features = defaultdict(dict)
    new_context_features = dict()

    for keyword, feat_value in dict_features.items():

        prefix_suffix = keyword.split('__')

        if len(prefix_suffix) == 1:
            print(f'(WW) The feature "{keyword}" is going to be ignored')
            continue
        try:
            prefix, suffix = prefix_suffix
        except:
            print(f"(EE) You are not following the right convention for '{keyword}' -- separator '{FEATURE_NAME_SEPARATOR}' should be used to separate node-set/edge-set/'context' name from feature name.")
            continue

        if prefix in list_node_sets_name:
            new_nodesets_features[prefix][suffix] = feat_value

        elif prefix in list_edge_sets_name:
            new_edgesets_features[prefix][suffix] = feat_value

        elif prefix == 'context':
            new_context_features[suffix] = feat_value

        else:
            print(f'The "{keyword}" is being ignored, it does not belong neither to the '
                  f'node sets, nor to the edge sett, nor to contex')

    nodesets_features = {ns_name: graph.node_sets[ns_name].get_features_dict() for ns_name in new_nodesets_features.keys()}
    merge(nodesets_features, new_nodesets_features)

    edgesets_features = {es_name: graph.edge_sets[es_name].get_features_dict() for es_name in new_edgesets_features.keys()}
    merge(edgesets_features, new_edgesets_features)

    context_features = graph.context.get_features_dict()
    merge(context_features, new_context_features)

    graph = graph.replace_features(node_sets=nodesets_features,
                                   edge_sets=edgesets_features,
                                   context=context_features)

    return graph



def set_features_from_result_tuple(graph: GraphTensor, prev_result: NamedTuple, which_step=-1, include_unknown=False):
    """
    This function will 'inject' features found in the `prev_result` and set all found features in the graph.
    It is also capable to set new features (e.g., derived features from the results) if `include_unknown` is set to True.

    By default, the function takes the last time-point (which_step == -1), but you can specify your own desired step.
    If `which_step` is None, this function will assume that the results are not in form of time-series ('slices' along first dimension),
    but will take the featues "as is" -- this allows to inject also a 'debug_output', that contains raw feature tensors.

    :param graph:
    :param prev_result:
    :param which_step:
    :param include_unknown:
    :return:
    """


    node_sets_feature_names = [f'{ns_name}__{feat_name}' for ns_name, node_set in graph.node_sets.items() for feat_name in node_set.get_features_dict().keys()]
    edge_sets_feature_names = [f'{es_name}__{feat_name}' for es_name, edge_set in graph.edge_sets.items() for feat_name in edge_set.get_features_dict().keys()]
    context_feature_names = [f'context__{feat_name}' for feat_name in graph.context.features.keys()]

    known_keys = node_sets_feature_names + edge_sets_feature_names + context_feature_names

    if which_step is not None:
        features_from_res = {f'{x[0]}': x[1][which_step, ...] for x in zip(prev_result._fields, prev_result)}
    else:
        features_from_res = {f'{x[0]}': x[1] for x in zip(prev_result._fields, prev_result)}

    if include_unknown:
        features_to_apply = features_from_res
    else:
        features_to_apply = {k: features_from_res[k] for k in known_keys if k in features_from_res.keys()}

    new_graph = set_graph_features(graph, features_to_apply)

    return new_graph


def concatenate_results(all_results: list[NamedTuple]):

    if len(all_results) <= 0:
        return None

    first_res = all_results[0]
    nt_type = type(first_res)
    fields = first_res._fields
    buffer = {name: [] for name in fields}

    for res in all_results:
        res: NamedTuple
        d = res._asdict()

        for k, value in d.items():
            buffer[k].append(value)

    concatenated_dict = {}
    for k in buffer.keys():
        concatenated_dict[k] = np.concatenate(buffer[k], axis=0)


    result_tuple = nt_type(**concatenated_dict)
    return result_tuple


def clip_results(result_namedtuple: NamedTuple, max_time: float = None, min_time: float = None):
    """
    This function will perform the "clipping" of NamedTuple results so that the results "ends" at specified `max_time` time.
    """

    times = result_namedtuple.context__time
    if max_time is None:
        max_time = np.max(times)

    if min_time is None:
        min_time = np.min(times)

    min_mask = np.squeeze(np.asarray(times >= min_time))
    max_mask = np.squeeze(np.asarray(times <= max_time))
    valid_mask = min_mask & max_mask

    if np.count_nonzero(valid_mask) <= 0:   # <-- if nothing to clip, return the original namedtuple
        return result_namedtuple

    # <--> so HERE we know, we should discard some values:
    clipped_values = [raw_value[valid_mask, ...] for raw_value in result_namedtuple]
    nt_type = type(result_namedtuple)
    clipped_result = nt_type(*clipped_values)

    return clipped_result



def untuple_results(result_namedtuple: NamedTuple):
    """
    This function makes "untupling" of the values in the NamedTuple model result.
    For some reason, when the model returns the debug_output, the 'automatically added features' are returned as {tuple: 1} type.
    So this function transforms them back to expected form, where each item of result is a Tensor.

    :param result_namedtuple:
    :return:
    """

    res_type = type(result_namedtuple)
    transformed_values = []
    for value in result_namedtuple:
        if isinstance(value, tuple):
            transformed_values.append(value[0])
        else:
            transformed_values.append(value)

    new_result = res_type(*transformed_values)
    return new_result
