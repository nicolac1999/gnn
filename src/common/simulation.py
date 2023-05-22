from typing import NamedTuple, Union, Any

from tensorflow_gnn import GraphTensor

from common.graph_manipulation_helpers import set_features_from_result_tuple, concatenate_results, clip_results


def simulate_time_period(model, graph: GraphTensor, duration: float = None, num_steps=None,
                         starting_time=None, time_step=None,
                         input_features=None,
                         clip_simulation=False) -> Union[NamedTuple, Any]:
    """
    Generic function, that can run the specified model on the initial graph, potentially starting from specified starting_time
    and running the simulation until it simulates requested duration (or, requested number of steps).

    !!! WARNING !!! The resulting NamedTuple may (and probably WILL) contain duplicated time-points -- because the last
                    time-point of previous model run and the first time-point of the following model run are typically
                    the same and are both part of the model results.
    """

    if starting_time is None:
        starting_time = graph.context['time']

    g_to_use = graph
    if time_step is not None:
        g_to_use.context['time_step'] = time_step
    else:
        time_step = float(g_to_use.context['time_step'])

    if duration is None:
        if num_steps is not None:
            duration = num_steps * time_step
        else:
            raise ValueError('You must specify either `duration` or `num_steps`. ')

    ending_time = starting_time + duration

    all_results = []
    current_time = starting_time

    while current_time < ending_time:
        # TODO: If an time-series (evolution) of INPUT feature values is required, then set it here for each model run call:
        #
        sim_res: NamedTuple = model(g_to_use)
        all_results.append(sim_res)
        current_time = sim_res.context__time[-1]
        g_to_use = set_features_from_result_tuple(g_to_use, sim_res, which_step=-1)

    merged_res = concatenate_results(all_results)

    if clip_simulation:
        merged_res = clip_results(merged_res, ending_time)

    return merged_res



