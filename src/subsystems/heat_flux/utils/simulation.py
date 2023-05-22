import numpy as np
import tensorflow as tf
from typing import NamedTuple

from common.graph_manipulation_helpers import set_features_from_result_tuple, concatenate_results, clip_results, \
    set_graph_features
from common.math.bayonet_geometry import BayonetGeometryEstimator
from subsystems.heat_flux.graphs.half_cell import half_cell
from subsystems.heat_flux.graphs.standard_cell_dynamic_BHX_virtual_pot import \
    create_standard_cell_with_dynamic_BHX_virtual_pot
from subsystems.heat_flux.graphs.string2_phase1 import string2_phase1
from subsystems.heat_flux.utils.system_properties import fractions_wetted_perimeter_from_mf_and_wetted_length
from tensorflow_gnn.graph import graph_tensor as gt


def simulate_time_period_till_convergence(model, graph, feature_to_converge,
                                          max_duration=40000,
                                          starting_time=None,
                                          time_step=None,
                                          gradient_convergence_threshold=5e-6,
                                          steady_state_duration=1000.,
                                          clip_simulation=False,
                                          time_track_step=1500.) -> NamedTuple:
    """
    Generic function, that can run the specified model on the initial graph, potentially starting from specified starting_time
    and running the simulation until the gradient of the means of a selected feature don't change more than a certain threshold.
    If the convergence is never reached the simulation is stopped when the max duration is reached.

    The feature_to_converge can be for example temperatures for magnets or velocity for liquid.
    IMPORTANT: feature_to_converge is a string which follows the usual convention of '__' for separating node set, edge set name
    or context from the name of the feature ------> EX : 'cells__T'

    The default value for the gradient_convergence_threshold is 5e-6, that can be interpreted in the case of temperatures
    like 5 mK over 1000 s

    :param model:
    :param graph:
    :param feature_to_converge:
    :param max_duration:
    :param starting_time:
    :param time_step:
    :param gradient_convergence_threshold:
    :param clip_simulation:
    :return:
    """

    if starting_time is None:
        starting_time = graph.context['time']

    ending_time = starting_time + max_duration

    g_to_use = graph
    if time_step is not None:
        g_to_use.context['time_step'] = time_step

    time_span = (model.num_steps * g_to_use.context['time_step']).numpy()
    # print(f'The time span is {time_span}')

    previous_state_feature_to_converge = None

    all_results = []
    current_time = starting_time
    # steady_state_duration = steady_state_duration + current_time

    while current_time < ending_time:

        if current_time.numpy() % time_track_step == 0:
            print(f'current time: {current_time}')
        # print(current_time, ending_time)

        sim_res: NamedTuple = model(g_to_use)
        all_results.append(sim_res)
        merged_res_current = concatenate_results(all_results)

        if feature_to_converge not in sim_res._fields:
            raise ValueError(
                'EE -- The feature you choose for the converging criteria is not in the model simulation output')

        current_state_feature_to_converge = sim_res.__getattribute__(feature_to_converge).numpy().mean(axis=0)

        if current_time >= steady_state_duration:  # METTERE > E VEDERE SE IL PROB E' DELLA EMPTY SLICE E" QUELLO
            mask = (merged_res_current.context__time > current_time - steady_state_duration) & \
                   (merged_res_current.context__time <= current_time - steady_state_duration + model.num_steps)
            mask = mask.numpy().flatten()
            # print(mask)
            merged_res_current_feature_to_converge = merged_res_current.__getattribute__(
                feature_to_converge)  # already numpy array

            previous_state_feature_to_converge = merged_res_current_feature_to_converge[mask].mean(axis=0)

            # if previous_state_feature_to_converge is not None:
            gradient = np.abs((previous_state_feature_to_converge - current_state_feature_to_converge) / time_span)
            # print(gradient)
            # print(gradient)

            # if (np.abs(previous_state_feature_to_converge - current_state_feature_to_converge) <= gradient_convergence_threshold).all():
            #     break
            if (gradient <= gradient_convergence_threshold).all():
                # print((gradient <= gradient_convergence_threshold).all(),)
                print(f'convergence {current_time}')
                # all_results.append(sim_res)
                break

        if (current_state_feature_to_converge >= 2.15).any():
            print('Liquid temperature over 2.15 K, not more superfluid conditions, simulation stopped')
            all_results.append(sim_res)
            break

        current_time = sim_res.context__time[-1]
        g_to_use = set_features_from_result_tuple(g_to_use, sim_res, which_step=-1, include_unknown=True)
        # previous_state_feature_to_converge = current_state_feature_to_converge

        if current_time > ending_time:
            print(f'max duration of simulation reached, {current_time}')

    merged_res = concatenate_results(all_results)

    if clip_simulation:
        merged_res = clip_results(merged_res, ending_time)

    return merged_res


def simulate_wetted_lengths_variation(configuration, model, graph: gt.GraphTensor, mass_flow: np.ndarray,
                                      wetted_lengths: np.ndarray,
                                      num_bhx: list,
                                      bhx_geometry: BayonetGeometryEstimator,
                                      feature_to_converge: str,
                                      steady_state_duration: float,
                                      time_track_step: float = 1500,
                                      merge_results=False,
                                      mass_flow_from_right=True,
                                      max_duration=40000,
                                      heat_coming_from_adj_cell=False,
                                      interconnection_cross_section=60.):
    """
    The function performs temperature evolution for different wetted lengths, given a graph, a defined incoming mass flow
    and the geometry of the bayonet heat exchanger.



    :param model: Keras model
    :param graph: GraphTensor
    :param mass_flow: incoming mass flow
    :param wetted_lengths: numpy array of wetted lengths, indicating which nodes can extract heat, the remaining nodes are
                           dry meaning heat can't pass in that part of the BHX
    :param bhx_geometry: BayonetGeometryEstimator instance
    :param feature_to_converge: feature for which the convergence criteria is evaluated
                                EX 'cells_T'
                                IMPORTANT: follow the convention node_set_name + '__' + feature name
    :param steady_state_duration: time for which it is possible to claim the feature_to_converge are stable
    :param time_track_step: "verbose" feature for the simulation, it is used for keeping track of the simulation time
    :param merge_results: True --> the result is returned as a NamedTuple
                          False --> the result is returned as a list of NamedTuple
                                    num_elem_list = num_wetted_lengths passed in input
    :param mass_flow_from_right: True --> the liquid is injected from the right, meaning from the end of the BHX,
                                          the magnets are met in this order : D-D-D-Q-D-D-D-Q
                                 False --> the liquid is injected from the left,
                                         the magnets are met in this order : Q-D-D-D-Q-D-D-D
    :param max_duration: max duration of the simulation
    :return:
    """

    all_results = []
    # g_to_use = graph

    INITIAL_TEMPERATURES = graph.node_sets['cells']['T']
    power_supplied = graph.node_sets['heater']['power']
    static_heat = float(graph.context['static_heat'])
    HEAT_COMING_FROM_ADJACENT_CELL = heat_coming_from_adj_cell
    print(len(wetted_lengths))

    for i in range(len(num_bhx)):

        if configuration == "standard_cell":

            g_to_use = create_standard_cell_with_dynamic_BHX_virtual_pot(temperatures=INITIAL_TEMPERATURES,
                                                                         num_nodes_BHX=num_bhx[i],
                                                                         power_supplied=power_supplied,
                                                                         static_heat=static_heat,
                                                                         liquid_flow_direction=1,
                                                                         num_liquid_nodes_per_cell=1,
                                                                         heat_coming_from_adjacent_cell=HEAT_COMING_FROM_ADJACENT_CELL,
                                                                         interconnection_cross_section=interconnection_cross_section)
        elif configuration == "string2":

            g_to_use = string2_phase1(temperatures=INITIAL_TEMPERATURES,
                                      num_nodes_BHX=num_bhx[i],
                                      power_supplied=power_supplied,
                                      static_heat=static_heat,
                                      liquid_flow_direction=1,
                                      num_liquid_nodes_per_cell=1,
                                      heat_coming_from_adjacent_cell=HEAT_COMING_FROM_ADJACENT_CELL,
                                      interconnection_cross_section=interconnection_cross_section)

        elif configuration == 'half_cell':

            g_to_use = half_cell(temperatures=INITIAL_TEMPERATURES,
                                 num_nodes_BHX=num_bhx[i],
                                 static_heat=static_heat,
                                 power_supplied=power_supplied,
                                 liquid_flow_direction=1,
                                 num_liquid_nodes_per_cell=1,
                                 heat_coming_from_adjacent_cell=HEAT_COMING_FROM_ADJACENT_CELL,
                                 interconnection_cross_section=66.)

        if i != 0:
            g_to_use = set_features_from_result_tuple(g_to_use, sim_res, which_step=-1, include_unknown=True)
        # ---> graph initialization of average_fractions and evaporation_mass_flow
        # ---> linear evaporation rate
        print(f'current wetted length: {wetted_lengths[i]}')
        mean_fractions_wp_nodes, evap_mass_flows_per_node, incoming_mass_flow = fractions_wetted_perimeter_from_mf_and_wetted_length(
            mass_flow=mass_flow,
            wetted_length=wetted_lengths[i],
            bhx_geometry=bhx_geometry,
            graph=g_to_use,
            mass_flow_from_right=mass_flow_from_right)

        # incoming_mass_flow = np.sum(evap_mass_flows_per_node)
        # incoming_mass_flow = np.concatenate([[incoming_mass_flow], np.zeros(len(mean_fractions_wp_nodes)-1)], axis=0, dtype=np.float32)
        evapor_mass_flow_per_m = evap_mass_flows_per_node / g_to_use.node_sets['liquid']['L']

        dict_new_features = {'liquid__avg_f': np.array(mean_fractions_wp_nodes, dtype=np.float32),
                             'liquid__evapor_mass_flow': np.array(evap_mass_flows_per_node, dtype=np.float32),
                             'liquid__incoming_mass_flow': incoming_mass_flow,
                             'liquid__evapor_mass_flow_per_m': np.array(evapor_mass_flow_per_m, dtype=np.float32)}

        g_to_use = set_graph_features(graph=g_to_use,
                                      dict_features=dict_new_features)

        sim_res = simulate_time_period_till_convergence(model=model, graph=g_to_use,
                                                        feature_to_converge=feature_to_converge,
                                                        steady_state_duration=steady_state_duration,
                                                        time_track_step=time_track_step,
                                                        max_duration=max_duration)

        all_results.append(sim_res)

    if merge_results:
        merged_res = concatenate_results(all_results)
        return merged_res
    else:
        return all_results
