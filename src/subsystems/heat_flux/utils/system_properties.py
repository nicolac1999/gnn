import numpy as np
import math
from tensorflow_gnn.graph import graph_tensor as gt

from common.constants import get_helium_conductivity, get_kapitza_conductance_cm
from common.math.bayonet_geometry import BayonetGeometryEstimator

def fractions_wetted_perimeter_from_mf_and_num_wet_nodes(mass_flow, wetted_length,
                                                         bayonet_geometry: BayonetGeometryEstimator):
    """
    #  ---- function for computing the fractions of wetted perimeter given the mass flow and the number of wetted nodes
    #  ---- (the function uses the length of the cell which should not be used, review the function id needed)
    # OLD FUNCTION, DOESN'T SUPPORT RIGHT FLOW

    :param mass_flow:
    :param wetted_length:
    :param bayonet_geometry:
    :return:
    """
    initial_fraction = bayonet_geometry.fraction_from_mass_flow(mass_flow)
    initial_height = bayonet_geometry.height_from_fraction(initial_fraction)
    volume = bayonet_geometry.volume_from_height(initial_height, np.array([500]))

    vs = np.linspace(volume, 0, wetted_length + 1).flatten()
    lengths = np.ones(wetted_length + 1) * 500
    liquid_levels = bayonet_geometry.height_from_volume(vs, lengths.flatten())
    fraction_wetted_perim = bayonet_geometry.fraction_from_height(liquid_height=liquid_levels)

    mean_fractions = []
    for i in range(0, len(fraction_wetted_perim) - 1):
        mean_fractions.append((fraction_wetted_perim[i] + fraction_wetted_perim[i + 1]) / 2)

    mean_fractions = np.pad(mean_fractions, (0, 20 - len(mean_fractions)))

    evaporation_mass_flows = np.pad(np.repeat(mass_flow / wetted_length, wetted_length), (0, 20 - wetted_length))

    return mean_fractions, evaporation_mass_flows


def fractions_wetted_perimeter_from_mf_and_wetted_length(mass_flow: np.ndarray,
                                                         wetted_length: np.ndarray,
                                                         bhx_geometry: BayonetGeometryEstimator,
                                                         graph: gt.GraphTensor,
                                                         mass_flow_from_right: bool = True
                                                         ):
    """
    # ---- function for computing the fractions of wetted perimeter given the mass flow and the wetted length in meters
    The function computes the average fraction of wetted perimeter for each node of the graph,
    where the average is between the fraction of wetted perimeter at the beginning of the node and the end of the node.

    ----- more details -----

    :param mass_flow: mass flow in g/s
    :param wetted_length: portion of the bayonet heat exchanger which is wet ( in meter )
    :param bhx_geometry: BayonetGeometryEstimator instance
    :param graph: GraphTensor instance
    :return:
    """

    evap_mass_flow_per_m = mass_flow / wetted_length

    nodes_lengths = np.array(graph.node_sets['liquid']['L']) / 100.  # this should go in m

    # if mass_flow_from_right:
    #     nodes_lengths = np.flip(nodes_lengths)

    cum_nodes_lengths = np.cumsum(nodes_lengths)

    wetted_nodes_lengths = np.where(np.round(cum_nodes_lengths, 2) <= np.round(wetted_length, 2), nodes_lengths, 0)

    evap_mass_flows_per_node = evap_mass_flow_per_m * wetted_nodes_lengths

    cum_evap_mass_flows = np.cumsum(evap_mass_flows_per_node)

    initial_mass_flows = np.repeat(mass_flow, len(nodes_lengths))

    remaining_mass_flows = initial_mass_flows - cum_evap_mass_flows

    mass_flows_start_end_nodes = np.concatenate((mass_flow, remaining_mass_flows))

    mass_flows_start_end_nodes = np.where(mass_flows_start_end_nodes > 0., mass_flows_start_end_nodes, 0.)

    cross_sections = mass_flows_start_end_nodes / (10. * 0.145)

    fractions_wp_start_end_nodes = bhx_geometry.fraction_from_cross_area(cross_sections)

    mean_fractions_wp_nodes = []
    for i in range(0, len(fractions_wp_start_end_nodes) - 1):
        mean_fractions_wp_nodes.append((fractions_wp_start_end_nodes[i] + fractions_wp_start_end_nodes[i + 1]) / 2)


    # if mass_flow_from_right:
    #     mean_fractions_wp_nodes = np.flip(mean_fractions_wp_nodes)
    #     evap_mass_flows_per_node = np.flip(evap_mass_flows_per_node)

    return mean_fractions_wp_nodes, evap_mass_flows_per_node, mass_flows_start_end_nodes[:-1]


def delta_T_from_heat_flux(heat_flux: np.ndarray,
                           cross_section: np.ndarray,
                           gradient_length: np.ndarray = np.array([60.]),
                           helium_thermal_conductivity=get_helium_conductivity(),
                           return_mK=False):
    """
    The function computes the delta T for having a specific heat flux given a cross-section where
    the fluz
    :param heat_flux:
    :param cross_section:
    :param gradient_length:
    :param helium_thermal_conductivuty:
    :param return_mK:
    :return:
    """

    heat_flux_density = heat_flux / cross_section

    delta_t = np.power(heat_flux_density, 3) / (helium_thermal_conductivity / gradient_length),

    if return_mK:
        return delta_t * 1000
    else:
        return delta_t


def delta_T_kapitza_from_heat_power(heat_power: np.ndarray,
                                    fraction_wetted_perim,
                                    T_cold_source=np.array([1.85]),
                                    surface=np.array([7992.211710732434]),
                                    kapitza_conductance=get_kapitza_conductance_cm(),
                                    return_mK=True):
    """
    The function computes the delta T between helium bath needed for extract a given amount of heat through a
    specified surface.
    The amount of heat is regulated by the Kapitza resistance/conductance
    IMPORTANT : the default surface value is the total external surface in cm2 of a cylinder of:
                - length 480 cm
                - diameter 5.3 cm

    """

    delta_t = heat_power / (kapitza_conductance / 2 * T_cold_source ** 3 * surface * fraction_wetted_perim)

    if return_mK:
        return delta_t * 1000
    else:
        return delta_t


def distance_point_line(point_1, point_2, point_3,):
    """
    Function for computing the distance between a point and a line.
    The function computes the perpendicular distance of point_3 to the line passing by point_1 and point_2

    :param point_1: tuple or list of the components of the first point
    :param point_2: tuple or list of the components of the second point
    :param point_3: tuple or list of the components of the third point
    :return:
    """

    d = np.abs(np.cross(point_2 - point_1, point_3 - point_1) / np.linalg.norm(point_2 - point_1))

    return d

def distance_point_line_given_coeffs(point, coeffs):
    """
    Function for computing the distance between a point and a line.
    The function computes the distance given the coefficients of the line ==> slope and intercept

    :param point: tuple or list of the components of the point
    :param coeffs: tuple or list of the line coefficients
    :return:
    """
    d = abs((coeffs[0] * point[0]) - point[1] + coeffs[1]) / math.sqrt((coeffs[0] * coeffs[0]) + 1)

    return d

