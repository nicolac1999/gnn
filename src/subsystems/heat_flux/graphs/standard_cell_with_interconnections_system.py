from common.constants import *
import tensorflow_gnn as tfgnn
import numpy as np


# %%

def create_standard_cell_with_interconnections(temperatures,
                         avg_fractions_wetted_perimeter=None,
                         evapor_mass_flow=None,
                         masses=None,
                         length_cells=None,
                         power_supplied=None,
                         common_power_supplied=0.,
                         liquid_temperature=1.85,
                         D=5.3,   # in cm
                         time: float = 0.,
                         time_step: float = 1.,
                         specific_heat_capacity=get_specific_heat_capacity(),
                         static_heat: float = 0.1,
                         latent_heat_vaporization=get_latent_heat_vaporization(),
                         liquid_density=get_superfluid_liquid_density(),
                         liquid_thermal_conductivity=get_helium_conductivity(),
                         kapitza_conductance=get_kapitza_conductance_cm(),  # in cm
                         liquid_flow_from_right=True):
    NUM_CELLS = 36
    NUM_LIQUID = 36
    NUM_HEATERS = 9
    NUM_CELLS2CELLS = 70
    NUM_CELL2LIQUID = 36
    NUM_LIQUID2CELL = 36


    if avg_fractions_wetted_perimeter:
        if len(masses) != NUM_CELLS:
            raise ValueError('Number of the fractions supplied not compatible with the'
                             'number of cells in this standard cell discretization = 20')
    else:
        avg_fractions_wetted_perimeter = np.zeros(NUM_CELLS, dtype=np.float32)

    if evapor_mass_flow:
        if len(evapor_mass_flow) != NUM_CELLS:
            raise ValueError('Number of the values for evaporation mass flows supplied not compatible with the'
                             'number of cells in this standard cell discretization = 20')
    else:
        evapor_mass_flow = np.zeros(NUM_CELLS, dtype=np.float32)

    if masses:
        if len(masses) != NUM_CELLS:
            raise ValueError('Number of the masses supplied not compatible with the'
                             'number of cells in this standard cell discretization = 20')
    else:
        masses = np.array([7.7, 20.6, 7.7,
                           7.7, 14.2, 14.2, 14.2, 7.7,
                           7.7, 14.2, 14.2, 14.2, 7.7,
                           7.7, 14.2, 14.2, 14.2, 7.7,
                           7.7, 20.6, 7.7,
                           7.7, 14.2, 14.2, 14.2, 7.7,
                           7.7, 14.2, 14.2, 14.2, 7.7,
                           7.7, 14.2, 14.2, 14.2, 7.7], dtype=np.float32)


    if length_cells:
        length_liquid = length_cells
        if len(length_cells) != NUM_CELLS:
            raise ValueError('Number of the lengths per cell supplied not compatible with the'
                             'number of cells in this standard cell discretization = 20')
    else:
        length_cells = np.array([30., 540., 30.,
                                 30., 480., 480., 480., 30.,
                                 30., 480., 480., 480., 30.,
                                 30., 480., 480., 480., 30.,
                                 30., 540., 30.,
                                 30., 480., 480., 480., 30.,
                                 30., 480., 480., 480., 30.,
                                 30., 480., 480., 480., 30.], dtype=np.float32)
        length_liquid = length_cells


    if power_supplied is not None:
        if len(power_supplied) != NUM_HEATERS:
            raise ValueError('Number of the heaters power supplied not compatible with the'
                             'number of heaters in the standard cell = 8')
    else:
        power_supplied = np.ones(NUM_HEATERS, dtype=np.float32) * common_power_supplied

    power_to_bhx = np.zeros(NUM_CELLS, dtype=np.float32)


    liquid_temperatures = np.ones(NUM_LIQUID, dtype=np.float32) * liquid_temperature
    HBX_diameter = np.ones(NUM_LIQUID, dtype=np.float32) * D
    edges_cells2cells_heat_flux_conduction = np.zeros(NUM_CELLS2CELLS, dtype=np.float32)
    edges_cells2cells_delta_t = np.zeros(NUM_CELLS2CELLS, dtype=np.float32)
    edges_cells2cells_thermal_conductivity = np.ones(NUM_CELLS2CELLS, dtype=np.float32) * liquid_thermal_conductivity
    edges_cells2liquid_kapitza_conductance = np.ones(NUM_CELL2LIQUID, dtype=np.float32) * kapitza_conductance
    edges_liquid2cells_kapitza_conductance = np.ones(NUM_CELL2LIQUID, dtype=np.float32) * kapitza_conductance

    if liquid_flow_from_right:
        cell_and_liquid_node_order_adj_matrix = [x for x in range(NUM_LIQUID - 1, -1, -1)]
    else:
        cell_and_liquid_node_order_adj_matrix = [x for x in range(0, NUM_LIQUID, 1)]

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
                                                      }),
                   "heater": tfgnn.NodeSet.from_fields(sizes=[NUM_HEATERS],
                                                       features={
                                                           "power": power_supplied,
                                                       }),
                   "liquid": tfgnn.NodeSet.from_fields(sizes=[NUM_LIQUID],
                                                       features={
                                                           "T": liquid_temperatures,
                                                           "avg_f": avg_fractions_wetted_perimeter,
                                                           'L': length_liquid,
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
                                                                     60.,
                                                                     270., 480., 480., 270,
                                                                     60.,
                                                                     270., 480., 480., 270,
                                                                     60.,
                                                                     270., 480., 480., 270,

                                                                     300., 300.,
                                                                     60.,
                                                                     270., 480., 480., 270,
                                                                     60.,
                                                                     270., 480., 480., 270,
                                                                     60.,
                                                                     270., 480., 480., 270,
                                                                     60.,
                                                                     300., 300.,
                                                                     60.,
                                                                     270., 480., 480., 270,
                                                                     60.,
                                                                     270., 480., 480., 270,
                                                                     60.,
                                                                     270., 480., 480., 270,

                                                                     ],
                                                               "A": [150., 150.,
                                                                     60.,
                                                                     150., 150., 150., 150.,
                                                                     60.,
                                                                     150., 150., 150., 150.,
                                                                     60.,
                                                                     150., 150., 150., 150.,
                                                                     60.,
                                                                     150., 150.,
                                                                     60.,
                                                                     150., 150., 150., 150.,
                                                                     60.,
                                                                     150., 150., 150., 150.,
                                                                     60.,
                                                                     150., 150., 150., 150.,

                                                                     150., 150.,
                                                                     60.,
                                                                     150., 150., 150., 150.,
                                                                     60.,
                                                                     150., 150., 150., 150.,
                                                                     60.,
                                                                     150., 150., 150., 150.,
                                                                     60.,
                                                                     150., 150.,
                                                                     60.,
                                                                     150., 150., 150., 150.,
                                                                     60.,
                                                                     150., 150., 150., 150.,
                                                                     60.,
                                                                     150., 150., 150., 150.,
                                                                     ],
                                                               "heat_flux_conduction": edges_cells2cells_heat_flux_conduction,
                                                               "delta_t_conduction": edges_cells2cells_delta_t,
                                                           },
                                                           adjacency=tfgnn.Adjacency.from_indices(
                                                               source=("cells",
                                                                       [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                                                        14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                                                                        27, 28, 29, 30, 31, 32, 33, 34,
                                                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                                                        15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                                                                        27, 28, 29, 30, 31, 32, 33, 34, 35]),
                                                               target=("cells",
                                                                       [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                                                        15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                                                                        27, 28, 29, 30, 31, 32, 33, 34, 35,
                                                                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                                                        14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                                                                        27, 28, 29, 30, 31, 32, 33, 34]))
                                                           ),
                   "heat supplied": tfgnn.EdgeSet.from_fields(sizes=[NUM_HEATERS],
                                                              features={},
                                                              adjacency=tfgnn.Adjacency.from_indices(
                                                                  source=("heater", [0, 1, 2, 3, 4, 5, 6, 7, 8]),
                                                                  target=("cells", [0, 3, 8, 13, 18, 21, 26, 31, 35])
                                                              )),
                   "cell2liquid": tfgnn.EdgeSet.from_fields(sizes=[NUM_CELL2LIQUID],
                                                            features={
                                                                "conductivity": edges_cells2liquid_kapitza_conductance,
                                                            },
                                                            adjacency=tfgnn.Adjacency.from_indices(
                                                                source=("cells", cell_and_liquid_node_order_adj_matrix),
                                                                target=("liquid", cell_and_liquid_node_order_adj_matrix)
                                                            )),
                   "liquid2cell": tfgnn.EdgeSet.from_fields(sizes=[NUM_LIQUID2CELL],
                                                            features={
                                                                "conductivity": edges_liquid2cells_kapitza_conductance,
                                                            },
                                                            adjacency=tfgnn.Adjacency.from_indices(
                                                                source=("liquid",
                                                                        cell_and_liquid_node_order_adj_matrix),
                                                                target=("cells",
                                                                        cell_and_liquid_node_order_adj_matrix)
                                                            ))}
    )

    return graph_ref
