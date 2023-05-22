import matplotlib.pyplot as plt
import numpy as np

from common.graph_manipulation_helpers import set_graph_features, set_features_from_result_tuple
from common.math.bayonet_geometry import *
from lib.diag_common.numpy_helpers import save_numpy_dict_to_file
from subsystems.heat_flux.models.heat_simplified_model import HeatSimplifiedModel
from subsystems.heat_flux.graphs.standard_cell_dynamic_BHX_virtual_pot import *
from subsystems.heat_flux.sandbox.mass_flow_given_supplied_heat import mass_flow_given_total_heat
from subsystems.heat_flux.utils.simulation import simulate_wetted_lengths_variation
from subsystems.heat_flux.utils.system_properties import fractions_wetted_perimeter_from_mf_and_wetted_length

"""
The purpose of the script is to re-generate experiment result with the new current state of the model
for the experiment on the String 2, where the heat was supplied by steps of 0.2 W/m from 0 to 1 W/m,
and we assumed that all the BHX was wet
"""

# %%    = = = = = = = = = = = = = > Experiment set up < = = = = = = = = = = = = = = =
CONFIGURATIONS_DIRECTORY = 'data/fluid/fraction_for_mass_flow_model_configurations/'
MODEL_CONFIGURATION = 'M_fraction_for_massflow-200_version2.npz'

HEAT_COMING_FROM_ADJACENT_CELL = False
STATIC_HEAT = 0.00342 # < -- because it should go in W/cm
INITIAL_TEMPERATURES_g2 = np.ones(36, dtype=np.float32) * 1.888
NUM_HEATERS = 8
WETTED_LENGTH = np.array(
    [102.])  # <-- assumption (the value here should go starting from the right, due to the direction
# of the flow we are indicating is from the right
LIQUID_SATURATION_TEMPERATURES = 1.85

bg = BayonetGeometryEstimator(5.3, 0)
model = HeatSimplifiedModel(configuration_file=CONFIGURATIONS_DIRECTORY + MODEL_CONFIGURATION, num_steps=30)

g0 = create_standard_cell_with_dynamic_BHX_virtual_pot(temperatures=INITIAL_TEMPERATURES_g2,
                                                       num_nodes_BHX=36,
                                                       static_heat=STATIC_HEAT,
                                                       liquid_flow_direction=1,
                                                       num_liquid_nodes_per_cell=1,
                                                       heat_coming_from_adjacent_cell=HEAT_COMING_FROM_ADJACENT_CELL)

graph_standard_cell_length = np.sum(g0.node_sets['cells']['L']) / 100
dynamic_heat_per_m_steps = np.arange(0., 1.1, 0.2)
temperatures_mask = np.asarray(g0.node_sets['cells']['has_sensor']).nonzero()[0].tolist()
# %% = = = = = = = = = = = = = = = = > Simulation block < = = = = = = = = = = = = = = = = =

sims_results = []
for dynamic_heat_per_m in dynamic_heat_per_m_steps:
    total_heating_power = dynamic_heat_per_m * graph_standard_cell_length
    heater_power = np.ones(NUM_HEATERS, dtype=np.float32) * (total_heating_power / NUM_HEATERS)

    new_features = {'heater__power': heater_power}

    graph_ref = set_graph_features(graph=g0, dict_features=new_features)

    mass_flow = mass_flow_given_total_heat(graph_ref, incremental_percentage_vaporization_power=0.1)

    sim = simulate_wetted_lengths_variation(model=model,
                                            graph=graph_ref,
                                            mass_flow=mass_flow,
                                            wetted_lengths=WETTED_LENGTH,
                                            bhx_geometry=bg,
                                            feature_to_converge='cells__T',
                                            time_track_step=1200.,
                                            steady_state_duration=1800.,
                                            mass_flow_from_right=True,
                                            max_duration=60000)

    sims_results.append(sim)
# %% = = = = = = = = = = = = = = = > Analysis block < = = = = = = = = = = = = = = = = =

static_heat_range = np.arange(0.342, 1.442, 0.2)
stable_Ts_simulations = [sim[0].cells__T[-1] for sim in sims_results]
delta_T_with_sat_T = [stable_Ts_sim - LIQUID_SATURATION_TEMPERATURES for stable_Ts_sim in stable_Ts_simulations]
mean_delta_Ts = np.mean(delta_T_with_sat_T, axis=1)

fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 6))
ax.plot(static_heat_range, [mean_dT * 1000 for mean_dT in mean_delta_Ts], label='DT average')

num_T_simulated = len(stable_Ts_simulations[0])

for i in range(num_T_simulated):
    if i in temperatures_mask:
        ax.plot(static_heat_range, np.asarray(delta_T_with_sat_T)[:, i] * 1000, 'o', label=f'DT magnet[{i}]')

ax.set_xticks(static_heat_range)
x_labels = ["{:.2f}".format(x) for x in static_heat_range - 0.342]
ax.set_xticklabels(x_labels)
ax.set_xlabel('Applied power [W/m]')
ax.set_ylabel('DeltaT (Tmagnets - Tsat) [mK]')
ax.set_ylim(0, 25)

ax.legend(bbox_to_anchor=(1, 1), loc='upper left')

secax = ax.secondary_xaxis('top')
secax.set_xlabel('Total heat load (static + dynamic) [W/m]')
x_labels_up = ["{:.2f}".format(x) for x in static_heat_range]
secax.set_xticks(static_heat_range)
secax.set_xticklabels(x_labels_up)
plt.grid()

plt.show()

# %% = = = = = = = = = = = = = = = = = > Saving Results < = = = = = = = = = = = = = = =

path_to_directory = 'results/heat/simulations/'
experiment = 'STRING2_'
suffixes = ['0', '0.2', '0.4', '0.6', '0.8', '1.']
for i in range(len(sims_results)):
    save_numpy_dict_to_file(sims_results[i][0]._asdict(), path_to_directory + experiment + suffixes[i] + 'W' + '.npz')

# %% ==> result with 0.2 percent more <==
# sims_results_0_2 = []
# for dynamic_heat_per_m in dynamic_heat_per_m_steps:
#     total_heating_power = dynamic_heat_per_m * graph_standard_cell_length
#     heater_power = np.ones(NUM_HEATERS, dtype=np.float32) * (total_heating_power / NUM_HEATERS)
#
#     new_features = {'heater__power': heater_power}
#
#     graph_ref = set_graph_features(graph=g0, dict_features=new_features)
#
#     mass_flow = mass_flow_given_total_heat(graph_ref, incremental_percentage_vaporization_power=0.2)
#
#     sim_0_2 = simulate_wetted_lengths_variation(model=model,
#                                             graph=graph_ref,
#                                             mass_flow=mass_flow,
#                                             wetted_lengths=WETTED_LENGTH,
#                                             bhx_geometry=bg,
#                                             feature_to_converge='cells__T',
#                                             time_track_step=1200.,
#                                             steady_state_duration=1800.,
#                                             mass_flow_from_right=True,
#                                             max_duration=60000)
#
#     sims_results_0_2.append(sim_0_2)
# # %%
#
# static_heat_range = np.arange(0.342, 1.442, 0.2)
# stable_Ts_simulations = [sim[0].cells__T[-1] for sim in sims_results_0_2]
# delta_T_with_sat_T = [stable_Ts_sim - LIQUID_SATURATION_TEMPERATURES for stable_Ts_sim in stable_Ts_simulations]
# mean_delta_Ts = np.mean(delta_T_with_sat_T, axis=1)
#
# fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 6))
# ax.plot(static_heat_range, [mean_dT * 1000 for mean_dT in mean_delta_Ts], label='DT average')
#
# num_T_simulated = len(stable_Ts_simulations[0])
#
# for i in range(num_T_simulated):
#     if i in temperatures_mask:
#         ax.plot(static_heat_range, np.asarray(delta_T_with_sat_T)[:, i] * 1000, 'o', label=f'DT magnet[{i}]')
#
# ax.set_xticks(static_heat_range)
# x_labels = ["{:.2f}".format(x) for x in static_heat_range - 0.342]
# ax.set_xticklabels(x_labels)
# ax.set_xlabel('Applied power [W/m]')
# ax.set_ylabel('DeltaT (Tmagnets - Tsat) [mK]')
# ax.set_ylim(0, 25)
#
# ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
#
# secax = ax.secondary_xaxis('top')
# secax.set_xlabel('Total heat load (static + dynamic) [W/m]')
# x_labels_up = ["{:.2f}".format(x) for x in static_heat_range]
# secax.set_xticks(static_heat_range)
# secax.set_xticklabels(x_labels_up)
# plt.grid()
# plt.suptitle('Simulation with 0.2 additional mass flow')
# plt.show()

# %%
CONFIGURATIONS_DIRECTORY = 'data/fluid/fraction_for_mass_flow_model_configurations/'
MODEL_CONFIGURATION = 'M_fraction_for_massflow-200-plus-slope-correction.npz'

HEAT_COMING_FROM_ADJACENT_CELL = False
STATIC_HEAT = 0.00342 # < -- because it should go in W/cm
INITIAL_TEMPERATURES_g2 = np.ones(36, dtype=np.float32) * 1.87
NUM_HEATERS = 8
WETTED_LENGTH = np.array(
    [102.])  # <-- assumption (the value here should go starting from the right, due to the direction
# of the flow we are indicating is from the right
LIQUID_SATURATION_TEMPERATURES = 1.85

bg_53_12 = BayonetGeometryEstimator(5.3, 1.2)
model_2 = HeatSimplifiedModel(configuration_file=CONFIGURATIONS_DIRECTORY + MODEL_CONFIGURATION, num_steps=30)

g2 = create_standard_cell_with_dynamic_BHX_virtual_pot(temperatures=INITIAL_TEMPERATURES_g2,
                                                       num_nodes_BHX=36,
                                                       static_heat=STATIC_HEAT,
                                                       liquid_flow_direction=1,
                                                       num_liquid_nodes_per_cell=1,
                                                       heat_coming_from_adjacent_cell=HEAT_COMING_FROM_ADJACENT_CELL)

graph_standard_cell_length = np.sum(g0.node_sets['cells']['L']) / 100
dynamic_heat_per_m_steps = np.arange(0., 1.1, 0.2)
temperatures_mask = np.asarray(g0.node_sets['cells']['has_sensor']).nonzero()[0].tolist()

# %%
g_to_use = g2
sims_results_string2 = []
for dynamic_heat_per_m in dynamic_heat_per_m_steps:
    print('dynamic_heat going on :', dynamic_heat_per_m)
    total_heating_power = dynamic_heat_per_m * graph_standard_cell_length
    heater_power = np.ones(NUM_HEATERS, dtype=np.float32) * (total_heating_power / NUM_HEATERS)

    new_features = {'heater__power': heater_power,
                    'context__time': [0.]}

    g_to_use = set_graph_features(graph=g_to_use, dict_features=new_features)

    mass_flow = mass_flow_given_total_heat(g_to_use, incremental_percentage_vaporization_power=0.10)

    sim = simulate_wetted_lengths_variation(model=model_2,
                                            graph=g_to_use,
                                            mass_flow=mass_flow,
                                            wetted_lengths=WETTED_LENGTH,
                                            bhx_geometry=bg_53_12,
                                            feature_to_converge='cells__T',
                                            time_track_step=1200.,
                                            steady_state_duration=1800.,
                                            mass_flow_from_right=True,
                                            max_duration=60000)

    g_to_use = set_features_from_result_tuple(g_to_use, sim[0], which_step=-1, include_unknown=True)

    sims_results_string2.append(sim)
# %%
static_heat_range = np.arange(0.342, 1.442, 0.2)
stable_Ts_simulations = [sim[0].cells__T[-1] for sim in sims_results_string2]
delta_T_with_sat_T = [stable_Ts_sim - LIQUID_SATURATION_TEMPERATURES for stable_Ts_sim in stable_Ts_simulations]
mean_delta_Ts = np.mean(delta_T_with_sat_T, axis=1)

fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 6))
ax.plot(static_heat_range, [mean_dT * 1000 for mean_dT in mean_delta_Ts], label='DT average')

num_T_simulated = len(stable_Ts_simulations[0])

for i in range(num_T_simulated):
    if i in temperatures_mask:
        ax.plot(static_heat_range, np.asarray(delta_T_with_sat_T)[:, i] * 1000, 'o', label=f'DT magnet[{i}]')

ax.set_xticks(static_heat_range)
x_labels = ["{:.2f}".format(x) for x in static_heat_range - 0.342]
ax.set_xticklabels(x_labels)
ax.set_xlabel('Applied power [W/m]')
ax.set_ylabel('DeltaT (Tmagnets - Tsat) [mK]')
ax.set_ylim(0, 25)

ax.legend(bbox_to_anchor=(1, 1), loc='upper left')

secax = ax.secondary_xaxis('top')
secax.set_xlabel('Total heat load (static + dynamic) [W/m]')
x_labels_up = ["{:.2f}".format(x) for x in static_heat_range]
secax.set_xticks(static_heat_range)
secax.set_xticklabels(x_labels_up)
plt.title('STRING2 experiment + 10 % mass_flow')
plt.grid()

plt.show()
# %%
path_to_directory = 'results/heat/simulations/'
experiment = 'STRING2_'
suffixes = ['0', '0.2', '0.4', '0.6', '0.8', '1.']
for i in range(len(sims_results_string2)):
    save_numpy_dict_to_file(sims_results_string2[i][0]._asdict(), path_to_directory + experiment
                            + suffixes[i] + 'W' + 'geometry_53+12_+10%mf' + '.npz')

