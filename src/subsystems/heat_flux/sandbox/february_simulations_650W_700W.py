from common.graph_manipulation_helpers import set_graph_features, set_features_from_result_tuple
from common.math.bayonet_geometry import *
from subsystems.heat_flux.models.heat_simplified_model import HeatSimplifiedModel
from subsystems.heat_flux.graphs.standard_cell_with_interconnections_system import *
from subsystems.heat_flux.sandbox.mass_flow_given_supplied_heat import mass_flow_given_total_heat
from subsystems.heat_flux.utils.plotting import plot_model_wetted_lengths_simulation
from subsystems.heat_flux.utils.simulation import simulate_wetted_lengths_variation, \
    simulate_time_period_till_convergence
from subsystems.heat_flux.utils.system_properties import fractions_wetted_perimeter_from_mf_and_wetted_length

# %%
'''
--- task 7 (14/11/2022) ---> try to reproduce the experiment of february, fixing the incoming mass flow and
                            the heaters power derived from timber, run the model for 8 different 
                            configurations of wetted length and find for which one the final temperatures 
                            distribution is much similar to the one derived from timber 
                            ( look the picture on mattermost or the one on the paper )
                            
--- task 7 (16/11/2022) ---> new convergence criteria, it is possible to set the duration of the steady state,
                             the time for which we can claim the temperatures are stable. If the differences of 
                             temperatures don't change more than a certain threshold than the simulation stops. 
                            
'''

# %%  Initialization of the BayonetGeometry Estimator, model, supplied heaters power


bg = BayonetGeometryEstimator(5.3, 0)
model = HeatSimplifiedModel(30)
temperatures_mask = [1, 4, 9, 14, 19, 22, 27, 32]

power_supplied_650 = np.array([4.1138, 4.1286, 0., 4.1516, 4.2547, 4.1511, 0., 4.2978], dtype=np.float32)
power_supplied_700 = np.array([4.7029, 4.7202, 0., 4.7378, 4.8566, 4.7571, 0., 4.9086], dtype=np.float32)
power_supplied_0 = np.zeros(8, dtype=np.float32)

# %% A standard graph of 36 nodes is initialized

INITIAL_TEMPERATURES_g1 = np.ones(36, dtype=np.float32) * 1.888
g1 = create_standard_cell_with_interconnections(temperatures=INITIAL_TEMPERATURES_g1,
                                                liquid_temperature=1.85,
                                                static_heat=0.002,
                                                power_supplied=power_supplied_650)

# %% The mass flow is computed given the graph, from the static + dynamic heat + some additive percentage of the total


mass_flow = mass_flow_given_total_heat(g1, incremental_percentage_vaporization_power=0.2)

# wetted_lengths_right = np.array([102., 87., 72., 57., 51., 36., 21., 6.])
# wetted_lengths_from_left = np.array([15., 30., 45., 51., 66., 81., 96., 102.])
wetted_lengths_from_right = np.array([102., 96., 81., 66., 51., 45., 30., 15.])
wetted_lengths_fine_grained = np.array([15., 15.31, 20.11, 24.91, 29.71, 30.])

# %% Model simulation
res = simulate_wetted_lengths_variation(model=model,
                                        graph=g1,
                                        mass_flow=mass_flow,
                                        wetted_lengths=wetted_lengths_fine_grained,
                                        bhx_geometry=bg,
                                        feature_to_converge='cells__T',
                                        time_track_step=1200.,
                                        steady_state_duration=1800.,
                                        mass_flow_from_right=True)

# %%  Plot 650 W experiment simulation

title = f'Temperatures evolution for different wetted lengths, February experiment 650 W'

plot_model_wetted_lengths_simulation(model_simulations=res,
                                     temperatures_mask=temperatures_mask,
                                     wetted_lengths=wetted_lengths_fine_grained,
                                     title=title,
                                     plot_average_temp=True)


# %%
# ------- > 700 W experiment simulation + plot

INITIAL_TEMPERATURES_g2 = np.ones(36, dtype=np.float32) * 1.865
g2 = create_standard_cell_with_interconnections(temperatures=INITIAL_TEMPERATURES_g2,
                                                liquid_temperature=1.85,
                                                static_heat=0.002,
                                                power_supplied=power_supplied_700)

mass_flow_2 = mass_flow_given_total_heat(g2, incremental_percentage_vaporization_power=0.2)

res_2 = simulate_wetted_lengths_variation(model=model,
                                          graph=g2,
                                          mass_flow=mass_flow_2,
                                          wetted_lengths=wetted_lengths_fine_grained,
                                          bhx_geometry=bg,
                                          feature_to_converge='cells__T',
                                          time_track_step=1200.,
                                          steady_state_duration=1800.,
                                          mass_flow_from_right=True)

title_2 = f'Temperatures evolution for different wetted lengths, February experiment 700 W'

plot_model_wetted_lengths_simulation(model_simulations=res_2,
                                     temperatures_mask=temperatures_mask,
                                     wetted_lengths=wetted_lengths_fine_grained,
                                     title=title_2)


# %%
# -------> No heat load applied + plot

INITIAL_TEMPERATURES_g3 = np.ones(36, dtype=np.float32) * 1.855
g3 = create_standard_cell_with_interconnections(temperatures=INITIAL_TEMPERATURES_g3,
                                                liquid_temperature=1.85,
                                                static_heat=0.002,
                                                power_supplied=power_supplied_0)

mass_flow_3 = mass_flow_given_total_heat(g3, incremental_percentage_vaporization_power=0.1)

res_3 = simulate_wetted_lengths_variation(model=model,
                                          graph=g3,
                                          mass_flow=mass_flow_3,
                                          wetted_lengths=wetted_lengths_from_right,
                                          bhx_geometry=bg,
                                          feature_to_converge='cells__T',
                                          time_track_step=1200.,
                                          steady_state_duration=1800.,
                                          mass_flow_from_right=True)

title_3 = f'Temperatures evolution for different wetted lengths, February experiment 0 W (only static heat)'

plot_model_wetted_lengths_simulation(model_simulations=res_3,
                                     temperatures_mask=temperatures_mask,
                                     wetted_lengths=wetted_lengths_from_right,
                                     title=title_3,
                                     n_columns=3,
                                     fig_size=(30, 20))

#wetted_lengths_fine_grained_first_dipole = np.array([0.31, 5.101, 9.901, 14.701])
wetted_lengths_fine_grained_first_dipole = np.array([14.701, 9.901, 5.101, 0.31])

INITIAL_TEMPERATURES_g4 = np.ones(36, dtype=np.float32) * 1.88
g4 = create_standard_cell_with_interconnections(temperatures=INITIAL_TEMPERATURES_g4,
                                                liquid_temperature=1.85,
                                                static_heat=0.002,
                                                power_supplied=power_supplied_0)
mass_flow_4 = mass_flow_given_total_heat(g4, incremental_percentage_vaporization_power=0.1)

res_4 = simulate_wetted_lengths_variation(model=model,
                                          graph=g4,
                                          mass_flow=mass_flow_4,
                                          wetted_lengths=wetted_lengths_fine_grained_first_dipole,
                                          bhx_geometry=bg,
                                          feature_to_converge='cells__T',
                                          time_track_step=1200.,
                                          steady_state_duration=1800.,
                                          mass_flow_from_right=True)


title_4 = f'Temperatures evolution for different wetted lengths, February experiment 0 W (only static heat)'

plot_model_wetted_lengths_simulation(model_simulations=res_4,
                                     temperatures_mask=temperatures_mask,
                                     wetted_lengths=wetted_lengths_fine_grained_first_dipole,
                                     title=title_4,
                                     n_columns=2,
                                     fig_size=(20, 10))


# %%
power_supplied_650_plus_side_cell = np.array([4.1138, 4.1286, 0., 4.1516, 4.2547, 4.1511, 0., 4.2978, 25.], dtype=np.float32)


INITIAL_TEMPERATURES_g5 = np.ones(36, dtype=np.float32) * 1.888

g5 = create_standard_cell_with_interconnections(temperatures=INITIAL_TEMPERATURES_g5,
                                                liquid_temperature=1.85,
                                                static_heat=0.002,
                                                power_supplied=power_supplied_650_plus_side_cell)


mass_flow5 = mass_flow_given_total_heat(g5, incremental_percentage_vaporization_power=0.1)

evap_mass_flows_absolute_rate = np.array([0.        , 0.        , 0.        , 0.        , 0.        ,
                                    0.        , 0.        , 0.        , 0.        , 0.        ,
                                    0.        , 0.        , 0.        , 0.        , 0.        ,
                                    0.        , 0.        , 0.        , 0.        , 0.        ,
                                    0.        , 0.        , 0.        , 0.        , 0.        ,
                                    0.        , 0.        , 0.        , 0.        , 0.        ,
                                    0.        , 0.036, 0.578, 0.014, 0.352, 0.022])

evap_mass_flows_per_node = evap_mass_flows_absolute_rate * mass_flow5
evap_mass_flows_per_node_flipped = np.flip(evap_mass_flows_per_node)
cum_evap_mass_flows = np.cumsum(evap_mass_flows_per_node_flipped)
initial_mass_flows = np.repeat(mass_flow5, 36)
remaining_mass_flows = initial_mass_flows - cum_evap_mass_flows
mass_flows_start_end_nodes = np.concatenate((mass_flow5, remaining_mass_flows))
mass_flows_start_end_nodes = np.where(mass_flows_start_end_nodes > 0., mass_flows_start_end_nodes, 0.)
fractions_wp_start_end_nodes = bg.fraction_from_mass_flow(mass_flow=mass_flows_start_end_nodes)
mean_fractions_wp_nodes = []
for i in range(0, len(fractions_wp_start_end_nodes) - 1):
    mean_fractions_wp_nodes.append((fractions_wp_start_end_nodes[i] + fractions_wp_start_end_nodes[i + 1])/ 2)
mean_fractions_wp_nodes = np.flip(mean_fractions_wp_nodes)



dict_new_features = {'liquid__avg_f': np.array(mean_fractions_wp_nodes, dtype=np.float32),
                     'liquid__evapor_mass_flow': np.array(evap_mass_flows_per_node, dtype=np.float32)}

g5 = set_graph_features(graph=g5,
                            dict_features=dict_new_features)

sim_res = simulate_time_period_till_convergence(model=model, graph=g5,
                                                        feature_to_converge='cells__T',
                                                        steady_state_duration=1800.,
                                                        time_track_step=1200.,
                                                        max_duration=60000)

#g5 = set_features_from_result_tuple(g5, sim_res, which_step=-1)

sim_res_list = [sim_res]
title_5 = 'Temperatures evolution for wetted length 15 m, February experiment 650 W + heat incoming from ' \
          'adjacent cell'

plot_model_wetted_lengths_simulation(sim_res_list,
                                     temperatures_mask=temperatures_mask,
                                     wetted_lengths=[15.],
                                     title=title_5,
                                     n_columns=1)


# %%
power_supplied_650_plus_side_cell = np.array([4.1138, 4.1286, 0., 4.1516, 4.2547, 4.1511, 0., 4.2978, 25.], dtype=np.float32)


INITIAL_TEMPERATURES_g6 = np.ones(36, dtype=np.float32) * 1.888

g6 = create_standard_cell_with_interconnections(temperatures=INITIAL_TEMPERATURES_g6,
                                                liquid_temperature=1.85,
                                                static_heat=0.002,
                                                power_supplied=power_supplied_650_plus_side_cell)


mass_flow6 = mass_flow_given_total_heat(g6, incremental_percentage_vaporization_power=0.1)

mean_fractions_wp_nodes_2, evap_mass_flows_per_node_2 = fractions_wetted_perimeter_from_mf_and_wetted_length(
            mass_flow=mass_flow6,
            wetted_length=np.array([15.]),
            bhx_geometry=bg,
            graph=g6,
            mass_flow_from_right=True)

dict_new_features = {'liquid__avg_f': np.array(mean_fractions_wp_nodes, dtype=np.float32),
                             'liquid__evapor_mass_flow': np.array(evap_mass_flows_per_node, dtype=np.float32)}

g_to_use = set_graph_features(graph=g6,
                            dict_features=dict_new_features)

sim_res_2 = simulate_time_period_till_convergence(model=model, graph=g_to_use,
                                                feature_to_converge='cells__T',
                                                steady_state_duration=1800.,
                                                time_track_step=1200.,
                                                max_duration=60000)
sim_res_list_2 = [sim_res_2]
title_6 = 'Temperatures evolution for wetted length 15 m, February experiment 650 W + heat incoming from ' \
          'adjacent cell'

plot_model_wetted_lengths_simulation(sim_res_list_2,
                                     temperatures_mask=temperatures_mask,
                                     wetted_lengths=[15.],
                                     title=title_6,
                                     n_columns=1)

# %% simulation linear fraction wetted perimeter decreasing
#wetted_lengths_fine_grained = np.array([15., 15.31, 20.11, 24.91, 29.71, 30.])
wetted_lengths_fine_grained = np.array([30.  , 29.71, 24.91, 20.11, 15.31, 15.  ])

power_supplied_650_plus_side_cell = np.array([4.1138, 4.1286, 0., 4.1516, 4.2547, 4.1511, 0., 4.2978, 25.], dtype=np.float32)


INITIAL_TEMPERATURES_g7 = np.ones(36, dtype=np.float32) * 1.888

g7 = create_standard_cell_with_interconnections(temperatures=INITIAL_TEMPERATURES_g7,
                                                liquid_temperature=1.85,
                                                static_heat=0.002,
                                                power_supplied=power_supplied_650_plus_side_cell)

mass_flow_7 = mass_flow_given_total_heat(g7, incremental_percentage_vaporization_power=0.1)

res_7 = simulate_wetted_lengths_variation(model=model,
                                          graph=g7,
                                          mass_flow=mass_flow_7,
                                          wetted_lengths=wetted_lengths_fine_grained,
                                          bhx_geometry=bg,
                                          feature_to_converge='cells__T',
                                          time_track_step=1200.,
                                          steady_state_duration=1800.,
                                          mass_flow_from_right=True,
                                          max_duration=60000)
title_7 = f'Temperatures evolution for different wetted lengths, experiment of 650 W taking into account the heat coming' \
          f'from the adjacent cell'

plot_model_wetted_lengths_simulation(model_simulations=res_7,
                                     temperatures_mask=temperatures_mask,
                                     wetted_lengths=wetted_lengths_fine_grained,
                                     title=title_7,
                                     n_columns=2,
                                     fig_size=(20, 10))



# %%

EXPERIMETAL_POINTS_FRACTION_MASS_FLOW_HE = np.array([[0., 0.],
                                                     [0.05, 0.5/74*2],
                                                     [0.1, 6/74*2],
                                                     [0.15, 22/74*2],
                                                     [0.18, 48/74*2],
                                                     [0.2, 80/74*2],
                                                     [0.23, 149/74*2],
                                                     [0.25, 223/74*2],
                                                     [0.27, 322/74*2],
                                                     [0.3, 530/74*2]])