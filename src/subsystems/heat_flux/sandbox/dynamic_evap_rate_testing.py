import matplotlib.pyplot as plt

from common.graph_manipulation_helpers import set_graph_features, set_features_from_result_tuple, clip_results
from common.math.bayonet_geometry import *
from subsystems.heat_flux.models.heat_simplified_model import HeatSimplifiedModel
from subsystems.heat_flux.graphs.standard_cell_with_interconnections_system import *
from subsystems.heat_flux.graphs.standard_cell_dynamic_BHX_virtual_pot import *
from subsystems.heat_flux.sandbox.mass_flow_given_supplied_heat import mass_flow_given_total_heat
from subsystems.heat_flux.utils.plotting import plot_model_wetted_lengths_simulation, plot_graph_tensor, \
    plot_liquid_BHX_behavior
from subsystems.heat_flux.utils.simulation import simulate_wetted_lengths_variation, \
    simulate_time_period_till_convergence
from subsystems.heat_flux.utils.system_properties import fractions_wetted_perimeter_from_mf_and_wetted_length

# %%
CONFIGURATIONS_DIRECTORY = 'data/fluid/fraction_for_mass_flow_model_configurations/'
MODEL_CONFIGURATION = 'M_fraction_for_massflow-200-plus-slope-correction.npz'

bg_53_12 = BayonetGeometryEstimator(5.3, 1.2)

model_feb = HeatSimplifiedModel(configuration_file=CONFIGURATIONS_DIRECTORY + MODEL_CONFIGURATION, num_steps=30)

# %%
##          == == == > TEST without mask for the nodes that can be wet, potentially the
#                      flow can jump instantaneously till the end of the BHX wetting all the BHX < == == ==

#
# # ---- linear initialization of fractions and evaporation mass flow
# temperatures_mask = [1, 4, 9, 14, 19, 22, 27, 32]
#
# INITIAL_TEMPERATURES_g1 = np.ones(36, dtype=np.float32) * 1.888
#
# power_supplied_650_plus_side_cell = np.array([4.1138, 4.1286, 0., 4.1516, 4.2547, 4.1511, 0., 4.2978, 25.], dtype=np.float32)
#
# g1 = create_standard_cell_with_interconnections(temperatures=INITIAL_TEMPERATURES_g1,
#                                                 liquid_temperature=1.85,
#                                                 static_heat=0.002,
#                                                 power_supplied=power_supplied_650_plus_side_cell)
#
# mass_flow = mass_flow_given_total_heat(g1, incremental_percentage_vaporization_power=0.1)
#
# wetted_lengths_from_right = np.array([15.])
#
# # %% Model simulation
# res = simulate_wetted_lengths_variation(model=model,
#                                         graph=g1,
#                                         mass_flow=mass_flow,
#                                         wetted_lengths=wetted_lengths_from_right,
#                                         bhx_geometry=bg,
#                                         feature_to_converge='cells__T',
#                                         time_track_step=1200.,
#                                         steady_state_duration=1800.,
#                                         mass_flow_from_right=True)
#
# # %% Plot simulation
# plot_model_wetted_lengths_simulation(model_simulations=res,
#                                      temperatures_mask=temperatures_mask,
#                                      wetted_lengths=wetted_lengths_from_right,
#                                      title='')
#

# %%
##          == == == > TEST with short BHX and virtual pot  < == == ==

INITIAL_TEMPERATURES_g2 = np.ones(36, dtype=np.float32) * 1.888

power_supplied_650_plus_side_cell = np.array([4.1138, 4.1286, 0., 4.1516, 4.2547, 4.1511, 0., 4.2978, 25.], dtype=np.float32)

g2 = create_standard_cell_with_dynamic_BHX_virtual_pot(temperatures=INITIAL_TEMPERATURES_g2,
                                                       num_nodes_BHX=5,
                                                       static_heat=0.002,
                                                       power_supplied=power_supplied_650_plus_side_cell,
                                                       liquid_flow_direction=1,
                                                       num_liquid_nodes_per_cell=1)

mass_flow = mass_flow_given_total_heat(g2, incremental_percentage_vaporization_power=0.)


wetted_lengths_from_right = np.array([15.])

temperatures_mask = np.asarray(g2.node_sets['cells']['has_sensor']).nonzero()[0].tolist()

# plot_graph_tensor(g2)
# %%
res_4 = simulate_wetted_lengths_variation(model=model_feb,
                                        graph=g2,
                                        mass_flow=mass_flow,
                                        wetted_lengths=wetted_lengths_from_right,
                                        bhx_geometry=bg_53_12,
                                        feature_to_converge='cells__T',
                                        time_track_step=1200.,
                                        steady_state_duration=1800.,
                                        mass_flow_from_right=True,
                                        max_duration=60000)

plot_model_wetted_lengths_simulation(model_simulations=res_4,
                                     temperatures_mask=temperatures_mask,
                                     wetted_lengths=wetted_lengths_from_right,
                                     title='')


# %%
# ---- > plotting liquid node analysis
sim = res_4[0]
plot_liquid_BHX_behavior(simulation=sim,
                         temperatures_mask=temperatures_mask,
                         title_temperatures_subplt='')

# %% DEBUGGING

sim_clipped = clip_results(sim, max_time=20000., min_time=19800.)

plot_liquid_BHX_behavior(simulation=sim_clipped,
                         temperatures_mask=temperatures_mask,
                         title_temperatures_subplt='')

