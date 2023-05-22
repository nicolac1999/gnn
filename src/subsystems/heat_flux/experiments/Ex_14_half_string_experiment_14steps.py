import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
from sklearn.linear_model import LinearRegression

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
for the experiment on the Half String, where the heat was supplied by steps of 0.2 W/m from 0 to 2 W/m and then 
back to 1.3, 0.9 and 0.5 W/m and we assumed that all the BHX was wet.
"""

# %%     = = = = = = = = = = = = = > Experiment set up < = = = = = = = = = = = = = = =
CONFIGURATIONS_DIRECTORY = 'data/fluid/fraction_for_mass_flow_model_configurations/'
MODEL_CONFIGURATION = 'M_fraction_for_massflow-200_version2.npz'

HEAT_COMING_FROM_ADJACENT_CELL = False
STATIC_HEAT = 0.0011 # < -- because it should go in W/cm
INITIAL_TEMPERATURES_g0 = np.ones(36, dtype=np.float32) * 1.86
NUM_HEATERS = 8
WETTED_LENGTH = np.array(
    [102.])  # <-- assumption (the value here should go starting from the right, due to the direction
# of the flow we are indicating is from the right
LIQUID_SATURATION_TEMPERATURES = 1.85

bg = BayonetGeometryEstimator(5.3, 0)
model = HeatSimplifiedModel(configuration_file=CONFIGURATIONS_DIRECTORY + MODEL_CONFIGURATION, num_steps=30)

g0 = create_standard_cell_with_dynamic_BHX_virtual_pot(temperatures=INITIAL_TEMPERATURES_g0,
                                                       num_nodes_BHX=36,
                                                       static_heat=STATIC_HEAT,
                                                       liquid_flow_direction=1,
                                                       num_liquid_nodes_per_cell=1,
                                                       heat_coming_from_adjacent_cell=HEAT_COMING_FROM_ADJACENT_CELL)

graph_standard_cell_length = np.sum(g0.node_sets['cells']['L']) / 100
dynamic_heat_per_m_steps = np.concatenate((np.arange(0., 2.1, 0.2), np.array([1.3, 0.9, 0.5])))
temperatures_mask = np.asarray(g0.node_sets['cells']['has_sensor']).nonzero()[0].tolist()

# %% = = = = = = = = = = = = = = = = > Simulation block + 10% < = = = = = = = = = = = = = = = = =

sims_results_half_cell = []
g_to_use = g0
for dynamic_heat_per_m in dynamic_heat_per_m_steps:
    print('dynamic_heat going on :', dynamic_heat_per_m)
    total_heating_power = dynamic_heat_per_m * graph_standard_cell_length
    heater_power = np.ones(NUM_HEATERS, dtype=np.float32) * (total_heating_power / NUM_HEATERS)

    new_features = {'heater__power': heater_power,
                    'context__time': [0.]}

    g_to_use = set_graph_features(graph=g_to_use, dict_features=new_features)

    mass_flow = mass_flow_given_total_heat(g_to_use, incremental_percentage_vaporization_power=0.1)

    sim = simulate_wetted_lengths_variation(model=model,
                                            graph=g_to_use,
                                            mass_flow=mass_flow,
                                            wetted_lengths=WETTED_LENGTH,
                                            bhx_geometry=bg,
                                            feature_to_converge='cells__T',
                                            time_track_step=1200.,
                                            steady_state_duration=1800.,
                                            mass_flow_from_right=True,
                                            max_duration=60000)

    g_to_use = set_features_from_result_tuple(g_to_use, sim[0], which_step=-1, include_unknown=True)

    sims_results_half_cell.append(sim)
# %%  = = = = = = = = = = = = = = = > Analysis block < = = = = = = = = = = = = = = = = =
stable_Ts_simulations = [sim[0].cells__T[-1] for sim in sims_results_half_cell]
delta_T_with_sat_T = [stable_Ts_sim - LIQUID_SATURATION_TEMPERATURES for stable_Ts_sim in stable_Ts_simulations]
mean_delta_Ts = np.mean(delta_T_with_sat_T, axis=1)

EXPERIMENTAL_POINTS = [(0, 1.5), (0.2, 2.5), (0.4,4.5), (0.5, 6.5), (0.6, 6),
                       (0.8, 7.5), (0.9, 11), (1, 13.5), (1.2, 15.5), (1.3, 16.5), (1.4,17.5),
                       (1.6, 20.5), (1.8, 24.2), (2, 28.5)]
x_experimental = [points[0] for points in EXPERIMENTAL_POINTS]
y_experimental = [points[1] for points in EXPERIMENTAL_POINTS]
interpolation = scipy.interpolate.interp1d(x_experimental, y_experimental)
x_new = np.arange(0., 2.01, 0.01)
#y_new = interpolation(x_new)

reg = LinearRegression().fit(np.array(x_experimental).reshape(-1, 1), np.array(y_experimental).reshape(-1, 1))
y_new_reg = reg.predict(x_new.reshape(-1, 1))

plt.figure(figsize=(15, 6))
plt.plot(dynamic_heat_per_m_steps, [mean_dT * 1000 for mean_dT in mean_delta_Ts], 'x', label='predicted values')
plt.plot(x_experimental, y_experimental, 's', label='experimental values')
#plt.plot(x_new, y_new)
plt.plot(x_new, y_new_reg)
x_labels = ["{:.2f}".format(x) for x in dynamic_heat_per_m_steps]
plt.xticks(dynamic_heat_per_m_steps, labels=x_labels)
plt.xlabel('Applied power [W/m]')

plt.ylabel('DeltaT (Tmagnets - Tsat) [mK]')
plt.ylim(-5, 30)
plt.title('Comparison experimental values with predicted values, including interpolation'
          ' line for experimental points ( + 10% mass_flow )')
plt.grid()
plt.legend()
plt.show()

# %% = = = = = = = = = = = = = = = = > Simulation block + 1% < = = = = = = = = = = = = = = = = =

sims_results_half_cell_01 = []
g_to_use = g0
for dynamic_heat_per_m in dynamic_heat_per_m_steps:
    print('dynamic_heat going on :', dynamic_heat_per_m)
    total_heating_power = dynamic_heat_per_m * graph_standard_cell_length
    heater_power = np.ones(NUM_HEATERS, dtype=np.float32) * (total_heating_power / NUM_HEATERS)

    new_features = {'heater__power': heater_power,
                    'context__time': [0.]}

    g_to_use = set_graph_features(graph=g_to_use, dict_features=new_features)

    mass_flow = mass_flow_given_total_heat(g_to_use, incremental_percentage_vaporization_power=0.01)

    sim = simulate_wetted_lengths_variation(model=model,
                                            graph=g_to_use,
                                            mass_flow=mass_flow,
                                            wetted_lengths=WETTED_LENGTH,
                                            bhx_geometry=bg,
                                            feature_to_converge='cells__T',
                                            time_track_step=1200.,
                                            steady_state_duration=1800.,
                                            mass_flow_from_right=True,
                                            max_duration=60000)

    g_to_use = set_features_from_result_tuple(g_to_use, sim[0], which_step=-1, include_unknown=True)

    sims_results_half_cell_01.append(sim)

# %% = = = = = = = = = = = = = = = > Analysis block < = = = = = = = = = = = = = = = = =

stable_Ts_simulations = [sim[0].cells__T[-1] for sim in sims_results_half_cell_01]
delta_T_with_sat_T = [stable_Ts_sim - LIQUID_SATURATION_TEMPERATURES for stable_Ts_sim in stable_Ts_simulations]
mean_delta_Ts = np.mean(delta_T_with_sat_T, axis=1)

EXPERIMENTAL_POINTS = [(0, 1.5), (0.2, 2.5), (0.4,4.5), (0.5, 6.5), (0.6, 6),
                       (0.8, 7.5), (0.9, 11), (1, 13.5), (1.2, 15.5), (1.3, 16.5), (1.4,17.5),
                       (1.6, 20.5), (1.8, 24.2), (2, 28.5)]
x_experimental = [points[0] for points in EXPERIMENTAL_POINTS]
y_experimental = [points[1] for points in EXPERIMENTAL_POINTS]
interpolation = scipy.interpolate.interp1d(x_experimental, y_experimental)
x_new = np.arange(0., 2.01, 0.01)
#y_new = interpolation(x_new)

reg = LinearRegression().fit(np.array(x_experimental).reshape(-1, 1), np.array(y_experimental).reshape(-1, 1))
y_new_reg = reg.predict(x_new.reshape(-1, 1))

plt.figure(figsize=(15, 6))
plt.plot(dynamic_heat_per_m_steps, [mean_dT * 1000 for mean_dT in mean_delta_Ts], 'x', label='predicted values')
plt.plot(x_experimental, y_experimental, 's', label='experimental values')
#plt.plot(x_new, y_new)
plt.plot(x_new, y_new_reg)
x_labels = ["{:.2f}".format(x) for x in dynamic_heat_per_m_steps]
plt.xticks(dynamic_heat_per_m_steps, labels=x_labels)
plt.xlabel('Applied power [W/m]')

plt.ylabel('DeltaT (Tmagnets - Tsat) [mK]')
plt.ylim(-5, 30)
plt.title('Comparison experimental values with predicted values, including interpolation'
          ' line for experimental points ( + 1%  mass_flow )')
plt.grid()
plt.legend()
plt.show()

# %% = = = = = = = = Experiment with different geometry and different model = = = = = = = = =

MODEL_CONFIGURATION_1 = 'M_fraction_for_massflow-200-plus-slope-correction.npz'

HEAT_COMING_FROM_ADJACENT_CELL = False
STATIC_HEAT = 0.0011 # < -- because it should go in W/cm
INITIAL_TEMPERATURES_g1 = np.ones(36, dtype=np.float32) * 1.86
NUM_HEATERS = 8
WETTED_LENGTH = np.array(
    [102.])  # <-- assumption (the value here should go starting from the right, due to the direction
# of the flow we are indicating is from the right
LIQUID_SATURATION_TEMPERATURES = 1.85


bg_53_12 = BayonetGeometryEstimator(5.3, 1.2)
model_1 = HeatSimplifiedModel(configuration_file=CONFIGURATIONS_DIRECTORY + MODEL_CONFIGURATION_1, num_steps=30)

g1 = create_standard_cell_with_dynamic_BHX_virtual_pot(temperatures=INITIAL_TEMPERATURES_g1,
                                                       num_nodes_BHX=36,
                                                       static_heat=STATIC_HEAT,
                                                       liquid_flow_direction=1,
                                                       num_liquid_nodes_per_cell=1,
                                                       heat_coming_from_adjacent_cell=HEAT_COMING_FROM_ADJACENT_CELL)

graph_standard_cell_length = np.sum(g1.node_sets['cells']['L']) / 100
dynamic_heat_per_m_steps = np.concatenate((np.arange(0., 2.1, 0.2), np.array([1.3, 0.9, 0.5])))
temperatures_mask = np.asarray(g1.node_sets['cells']['has_sensor']).nonzero()[0].tolist()

# %%
sims_results_half_cell_plus_slope = []
g_to_use = g1
for dynamic_heat_per_m in dynamic_heat_per_m_steps:
    print('dynamic_heat going on :', dynamic_heat_per_m)
    total_heating_power = dynamic_heat_per_m * graph_standard_cell_length
    heater_power = np.ones(NUM_HEATERS, dtype=np.float32) * (total_heating_power / NUM_HEATERS)

    new_features = {'heater__power': heater_power,
                    'context__time': [0.]}

    g_to_use = set_graph_features(graph=g_to_use, dict_features=new_features)

    mass_flow = mass_flow_given_total_heat(g_to_use, incremental_percentage_vaporization_power=0.05)

    sim = simulate_wetted_lengths_variation(model=model_1,
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

    sims_results_half_cell_plus_slope.append(sim)


# %%

stable_Ts_simulations = [sim[0].cells__T[-1] for sim in sims_results_half_cell_plus_slope]
delta_T_with_sat_T = [stable_Ts_sim - LIQUID_SATURATION_TEMPERATURES for stable_Ts_sim in stable_Ts_simulations]
mean_delta_Ts = np.mean(delta_T_with_sat_T, axis=1)

EXPERIMENTAL_POINTS = [(0, 1.5), (0.2, 2.5), (0.4,4.5), (0.5, 6.5), (0.6, 6),
                       (0.8, 7.5), (0.9, 11), (1, 13.5), (1.2, 15.5), (1.3, 16.5), (1.4,17.5),
                       (1.6, 20.5), (1.8, 24.2), (2, 28.5)]
x_experimental = [points[0] for points in EXPERIMENTAL_POINTS]
y_experimental = [points[1] for points in EXPERIMENTAL_POINTS]
interpolation = scipy.interpolate.interp1d(x_experimental, y_experimental)
x_new = np.arange(0., 2.01, 0.01)
#y_new = interpolation(x_new)

reg = LinearRegression().fit(np.array(x_experimental).reshape(-1, 1), np.array(y_experimental).reshape(-1, 1))
y_new_reg = reg.predict(x_new.reshape(-1, 1))

plt.figure(figsize=(15, 6))
plt.plot(dynamic_heat_per_m_steps, [mean_dT * 1000 for mean_dT in mean_delta_Ts], 'x', label='predicted values')
plt.plot(x_experimental, y_experimental, 's', label='experimental values')
#plt.plot(x_new, y_new)
plt.plot(x_new, y_new_reg)
x_labels = ["{:.2f}".format(x) for x in dynamic_heat_per_m_steps]
plt.xticks(dynamic_heat_per_m_steps, labels=x_labels)
plt.xlabel('Applied power [W/m]')

plt.ylabel('DeltaT (Tmagnets - Tsat) [mK]')
plt.ylim(-5, 30)
plt.title('Comparison experimental values with predicted values, including interpolation'
          ' line for experimental points ( + 5%  mass_flow ) + new geometry ')
plt.grid()
plt.legend()
plt.show()
# %%
path_to_directory = 'results/heat/simulations/'
experiment = 'HALF_CELL_'
suffixes = ['0', '0.2', '0.4', '0.6', '0.8', '1.', '1.2', '1.4', '1.6', '1.8', '2.', '1.3', '0.9', '0.5']
for i in range(len(sims_results_half_cell_plus_slope)):
    save_numpy_dict_to_file(sims_results_half_cell_plus_slope[i][0]._asdict(), path_to_directory + experiment
                            + suffixes[i] + 'W' + 'geometry_53+12_+5%mf' + '.npz')