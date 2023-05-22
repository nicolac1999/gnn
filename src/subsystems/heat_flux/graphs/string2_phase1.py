from common.constants import *
import tensorflow_gnn as tfgnn
import numpy as np
import math

from subsystems.heat_flux.utils.plotting import plot_graph_tensor


def string2_phase1(temperatures,
                    num_nodes_BHX,
                    avg_fractions_wetted_perimeter=None,
                    evapor_mass_flow=None,
                    masses=None,
                    length_cells=None,
                    power_supplied=None,
                    common_power_supplied=0.,
                    liquid_temperature=1.85,
                    D=5.4,   # in cm
                    time: float = 0.,
                    time_step: float = 1.,
                    specific_heat_capacity=get_specific_heat_capacity(),
                    static_heat: float = 0.1,
                    latent_heat_vaporization=get_latent_heat_vaporization(),
                    liquid_density=get_superfluid_liquid_density(),
                    liquid_thermal_conductivity=get_helium_conductivity(),
                    kapitza_conductance=get_kapitza_conductance_cm(),  # in cm,
                    sensor_location=None,
                    liquid_flow_direction=1,
                    num_liquid_nodes_per_cell=2,
                    heat_coming_from_adjacent_cell=True,
                    shrinking_factor=1.,
                    interconnection_cross_section=60.):

    num_nodes_BHX_plus_pot = num_nodes_BHX + 1
    NUM_CELLS = 21

    if heat_coming_from_adjacent_cell:
        NUM_HEATERS = 6
        HEAT_SUPPLIED_TARGET = [0, 3, 8, 13, 18, 20]
        HEAT_SUPPLIED_SOURCE = [0, 1, 2, 3, 4, 5]

    else:
        NUM_HEATERS = 5
        HEAT_SUPPLIED_TARGET = [0, 3, 8, 13, 18]
        HEAT_SUPPLIED_SOURCE = [0, 1, 2, 3, 4]

    # clean here in the way to create before the adjacency matrix and then to have the number of edges

    cells2cells_adjacency_source = np.concatenate([np.arange(0, NUM_CELLS-1, 1), np.arange(1, NUM_CELLS, 1)])
    cells2cells_adjacency_target = np.concatenate([np.arange(1, NUM_CELLS, 1), np.arange(0, NUM_CELLS-1, 1)])


    NUM_CELLS2CELLS = len(cells2cells_adjacency_source)
    NUM_CELL2LIQUID = num_nodes_BHX
    NUM_LIQUID2CELL = num_nodes_BHX
    LENGTH_CELLS = np.array([30., 540., 30.,
                             30., 480., 480., 480., 30.,
                             30., 480., 480., 480., 30.,
                             30., 480., 480., 480., 30.,
                             30., 540., 30.], dtype=np.float32) * shrinking_factor

    LENGHT_POT = np.array([100.], dtype=np.float32)

    # = = = = = = = = = = = = = = >  INPUTS CHECK < = = = = = = = = == = = = = = = = =

    if num_nodes_BHX % num_liquid_nodes_per_cell != 0:
        raise ValueError('!! EE !! The number of nodes for the BHX that you specified is not an integer multiple'
                         'of the number of BHX per node, set correctly these numbers')

    if sensor_location is None:
        SENSORS_LOCATION = [1, 4, 9, 14, 19]
        SENSORS_LOCATION_DISTRIBUTION = np.zeros(NUM_CELLS, dtype=np.int32)
        SENSORS_LOCATION_DISTRIBUTION[SENSORS_LOCATION] = 1
    else:
        if min(sensor_location) < 0:
            raise ValueError(" !! EE !! : you are introducing negative numbers for the sensors' locations")
        elif max(sensor_location) >= NUM_CELLS:
            raise ValueError(" !! EE !! : one or more of the sensors' positions you are indicating are outside the "
                             "number of nodes representing the helium bath ")

    if avg_fractions_wetted_perimeter:
        if len(avg_fractions_wetted_perimeter) != num_nodes_BHX:
            raise ValueError('Number of the fractions supplied not compatible with the'
                             'number of nodes for the BHX')
    else:
        avg_fractions_wetted_perimeter = np.zeros(num_nodes_BHX_plus_pot, dtype=np.float32)

    if evapor_mass_flow:
        if len(evapor_mass_flow) != num_nodes_BHX:
            raise ValueError('Number of the fractions supplied not compatible with the'
                             'number of nodes for the BHX')
    else:
        evapor_mass_flow = np.zeros(num_nodes_BHX_plus_pot, dtype=np.float32)

    if masses:
        if len(masses) != NUM_CELLS:
            raise ValueError('Number of the masses supplied not compatible with the'
                             'number of cells in this standard cell discretization')
    else:
        masses = np.array([7.7, 20.6, 7.7,
                           7.7, 14.2, 14.2, 14.2, 7.7,
                           7.7, 14.2, 14.2, 14.2, 7.7,
                           7.7, 14.2, 14.2, 14.2, 7.7,
                           7.7, 20.6, 7.7], dtype=np.float32) * shrinking_factor

    if length_cells:
        if len(length_cells) != NUM_CELLS:
            raise ValueError('Number of the lengths per cell supplied not compatible with the'
                             'number of cells in this standard cell discretization')
    else:
        length_cells = LENGTH_CELLS

    if power_supplied is not None:
        if len(power_supplied) != NUM_HEATERS:
            raise ValueError('Number of the heaters power supplied not compatible with the'
                             'number of heaters in the string2 = 5 ,+ 1 heater which '
                             'is reproducing the effect coming from the adjacent cell if '
                             'the parameter heat_coming_from_adjacent_cell is set to True')
    else:
        power_supplied = np.ones(NUM_HEATERS, dtype=np.float32) * common_power_supplied

    power_to_bhx = np.zeros(NUM_CELLS, dtype=np.float32)

    liquid_temperatures = np.ones(num_nodes_BHX_plus_pot, dtype=np.float32) * liquid_temperature
    HBX_diameter = np.ones(num_nodes_BHX_plus_pot, dtype=np.float32) * D
    edges_cells2cells_heat_flux_conduction = np.zeros(NUM_CELLS2CELLS, dtype=np.float32)
    edges_cells2cells_delta_t = np.zeros(NUM_CELLS2CELLS, dtype=np.float32)
    edges_cells2cells_thermal_conductivity = np.ones(NUM_CELLS2CELLS, dtype=np.float32) * liquid_thermal_conductivity
    edges_cells2liquid_kapitza_conductance = np.ones(NUM_CELL2LIQUID, dtype=np.float32) * kapitza_conductance
    edges_liquid2cells_kapitza_conductance = np.ones(NUM_CELL2LIQUID, dtype=np.float32) * kapitza_conductance

    LIQUID_NODE_ADJ_MATRIX = [x for x in range(0, num_nodes_BHX, 1)]

    if liquid_flow_direction == 0:
        cell_node_order_adj_matrix = [x for x in range(0, int(num_nodes_BHX / num_liquid_nodes_per_cell), 1)]
        cell_node_order_adj_matrix = np.repeat(cell_node_order_adj_matrix, num_liquid_nodes_per_cell)
        lengths = LENGTH_CELLS[:int(num_nodes_BHX / num_liquid_nodes_per_cell)] / num_liquid_nodes_per_cell
        lengths = np.repeat(lengths, num_liquid_nodes_per_cell)
        lengths = np.concatenate([lengths, LENGHT_POT])

    elif liquid_flow_direction == 1:
        cell_node_order_adj_matrix = [x for x in range(NUM_CELLS - 1,
                                                       NUM_CELLS - int(num_nodes_BHX / num_liquid_nodes_per_cell) - 1,
                                                       -1)]
        cell_node_order_adj_matrix = np.repeat(cell_node_order_adj_matrix, num_liquid_nodes_per_cell)
        lengths = LENGTH_CELLS[-int(num_nodes_BHX / num_liquid_nodes_per_cell):] / num_liquid_nodes_per_cell
        lengths = np.repeat(lengths, num_liquid_nodes_per_cell)
        lengths = np.concatenate([lengths, LENGHT_POT])

    else:
        raise ValueError('You can insert 1 or 0, 1 if the flow is coming from the right, 0 from the left')

    # = = = = = = = = = = = = = = >  TENSOR GRAPH CREATION < = = = = = = = = == = = = = = = = =

    graph_ref = tfgnn.GraphTensor.from_pieces(
        context=tfgnn.Context.from_fields(
            features={'time': [time],
                      'time_step': [time_step],
                      'specific_heat_capacity': [specific_heat_capacity],
                      'static_heat': [static_heat],
                      'latent_heat_vaporization': [latent_heat_vaporization],
                      'liquid_density': [liquid_density],
                      'total_static_heat': [0.],
                      'total_dynamic_heat': [0.],
                      'total_vaporization_heat': [0.]
                      }),

        node_sets={"cells": tfgnn.NodeSet.from_fields(sizes=[NUM_CELLS],
                                                      features={
                                                          "T": temperatures,
                                                          "mass": masses,
                                                          'L': length_cells,
                                                          'power_to_bhx': power_to_bhx,
                                                          "has_sensor": SENSORS_LOCATION_DISTRIBUTION,
                                                      }),
                   "heater": tfgnn.NodeSet.from_fields(sizes=[NUM_HEATERS],
                                                       features={
                                                           "power": power_supplied,
                                                       }),
                   "liquid": tfgnn.NodeSet.from_fields(sizes=[num_nodes_BHX_plus_pot],
                                                       features={
                                                           "T": liquid_temperatures,
                                                           "avg_f": avg_fractions_wetted_perimeter,
                                                           'L': lengths,
                                                           "evapor_mass_flow": evapor_mass_flow,
                                                           "D": HBX_diameter,
                                                       })},

        edge_sets={"conduction": tfgnn.EdgeSet.from_fields(sizes=[NUM_CELLS2CELLS],
                                                           features={
                                                               "conductivity": edges_cells2cells_thermal_conductivity,

                                                               "L": [300., 300.,
                                                                     60.,
                                                                     270., 480., 480., 270,
                                                                     60.,
                                                                     270., 480., 480., 270,
                                                                     60.,
                                                                     270., 480., 480., 270,
                                                                     60.,
                                                                     300., 300.,

                                                                     300., 300.,
                                                                     60.,
                                                                     270., 480., 480., 270,
                                                                     60.,
                                                                     270., 480., 480., 270,
                                                                     60.,
                                                                     270., 480., 480., 270,
                                                                     60.,
                                                                     300., 300.,

                                                                     ],
                                                               "A": [150., 150.,
                                                                     interconnection_cross_section,
                                                                     150., 150., 150., 150.,
                                                                     interconnection_cross_section,
                                                                     150., 150., 150., 150.,
                                                                     interconnection_cross_section,
                                                                     150., 150., 150., 150.,
                                                                     interconnection_cross_section,
                                                                     150., 150.,

                                                                     150., 150.,
                                                                     interconnection_cross_section,
                                                                     150., 150., 150., 150.,
                                                                     interconnection_cross_section,
                                                                     150., 150., 150., 150.,
                                                                     interconnection_cross_section,
                                                                     150., 150., 150., 150.,
                                                                     interconnection_cross_section,
                                                                     150., 150.,
                                                                     ],
                                                               "heat_flux_conduction": edges_cells2cells_heat_flux_conduction,
                                                               "delta_t_conduction": edges_cells2cells_delta_t,
                                                           },
                                                           adjacency=tfgnn.Adjacency.from_indices(
                                                               source=("cells", cells2cells_adjacency_source),
                                                               target=("cells", cells2cells_adjacency_target))
                                                           ),
                   "heat supplied": tfgnn.EdgeSet.from_fields(sizes=[NUM_HEATERS],
                                                              features={},
                                                              adjacency=tfgnn.Adjacency.from_indices(
                                                                  source=("heater", HEAT_SUPPLIED_SOURCE),
                                                                  target=("cells", HEAT_SUPPLIED_TARGET)
                                                              )),
                   "cell2liquid": tfgnn.EdgeSet.from_fields(sizes=[NUM_CELL2LIQUID],
                                                            features={
                                                                "conductivity": edges_cells2liquid_kapitza_conductance,
                                                            },
                                                            adjacency=tfgnn.Adjacency.from_indices(
                                                                source=("cells", cell_node_order_adj_matrix),
                                                                target=("liquid", LIQUID_NODE_ADJ_MATRIX)
                                                            )),
                   "liquid2cell": tfgnn.EdgeSet.from_fields(sizes=[NUM_LIQUID2CELL],
                                                            features={
                                                                "conductivity": edges_liquid2cells_kapitza_conductance,
                                                            },
                                                            adjacency=tfgnn.Adjacency.from_indices(
                                                                source=("liquid",
                                                                        LIQUID_NODE_ADJ_MATRIX),
                                                                target=("cells",
                                                                        cell_node_order_adj_matrix)
                                                            ))}
    )

    return graph_ref

# # %% Testing
# INITIAL_TEMPERATURES = np.ones(21, dtype=np.float32) * 1.888
#
# string2_graph = string2_phase1(temperatures=INITIAL_TEMPERATURES,
#                                num_nodes_BHX=5,
#                                num_liquid_nodes_per_cell=1,
#                                heat_coming_from_adjacent_cell=False)
#
#
# # %%
# plot_graph_tensor(string2_graph)