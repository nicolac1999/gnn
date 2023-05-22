from tensorflow_gnn.graph import graph_tensor as gt

from subsystems.heat_flux.graphs.standard_cell_with_interconnections_system import *
from common.math.bayonet_geometry import *


# %% Report of the script
'''
---- task 6 (14/11/2022) --> find the incoming mass flow given the total static heat and the total supplied power
'''

# %%

def mass_flow_given_total_heat(graph: gt.GraphTensor, incremental_percentage_vaporization_power=0.1):

    """
    The function takes in input a graph where the heaters power and the static heat should be correctly
    initialized, and returns the evaporation mass flow equivalent to the total heat supplied to the
    standard cell.

    In particular the mass flow can be incremented by a certain amount, this is done to avoid that the
    evaporation power is not enough for extracting the power supplied causing the raise of the temperatures.

    By default, the mass flow is computed in order to be able to extract 10% more power that the one supplied.

    :param graph: GraphTensor
    :param incremental_percentage_vaporization_power: input parameter, involved in the computation of the
                                                      mass flow, for being "conservative" in the simulation,
                                                      having a gap between the power that can be extracted and
                                                      the power supplied guarantees that the model converge to
                                                      a steady state.

    :return:
    """

    # ---> Dynamic heat
    total_dynamic_heat = graph.node_sets['heater']['power'].numpy().sum()

    # ---> Static heat
    static_heat = graph.context['static_heat']
    standard_cell_length = graph.node_sets['cells']['L'].numpy().sum()
    total_static_heat = static_heat * standard_cell_length
    total_static_heat = total_static_heat.numpy()

    # ---> Dynamic + Static
    total_heat = total_static_heat + total_dynamic_heat

    # ---> Vaporization Power ---> mass flow
    total_vaporization_power = total_heat + incremental_percentage_vaporization_power * total_heat
    latent_heat_vaporization = graph.context['latent_heat_vaporization']

    evapor_mass_flow = total_vaporization_power / latent_heat_vaporization
    incoming_mass_flow = evapor_mass_flow.numpy()

    return incoming_mass_flow


# %%
INITIAL_TEMPERATURES_g1 = np.ones(36, dtype=np.float32) * 1.90
g1 = create_standard_cell_with_interconnections(temperatures=INITIAL_TEMPERATURES_g1,
                                                common_power_supplied=10.,
                                                static_heat=0.002)

# %%
mass_flow_given_total_heat(g1)

