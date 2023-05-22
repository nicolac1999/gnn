import matplotlib.pyplot as plt
from common.graph_manipulation_helpers import set_features_from_result_tuple, concatenate_results, set_graph_features
from subsystems.heat_flux.models.heat_simplified_model import HeatSimplifiedModel
from subsystems.heat_flux.graphs.standard_cell_system import *
from subsystems.heat_flux.graphs.standard_cell_with_interconnections_system import *
from subsystems.heat_flux.sandbox.fraction_wp_from_mf_and_wl_testing import \
    fractions_wetted_perimeter_from_mf_and_num_wet_nodes, fractions_wetted_perimeter_from_mf_and_wetted_length
from common.math.bayonet_geometry import *
from subsystems.heat_flux.utils.plotting import plot_graph_tensor

# %%  Report of the script
'''


--- task 3 (07/11/2022) ---> simulate temperatures behaviour for wetted length of 1/3 , find the steady state and 
                             check if the result are compatible with Figure 7 , page 8, LHC Project Report 144.
                             
                             We saw a strange behaviour of the temperatures, the last ones were diverging and increasing 
                             going more than 2.2 K ( HERE SET LIMIT TO MAX TEMPERATURE ), then we realized this
                             is caused from the fact that to be able to pass a huge heat flux in 30 cm^2, considering 
                             a temperature gradient over 560 cm , the delta T should be of about 4.72 K.
                             Cross-section of interconnection => bottleneck => redefinition of the graph structure
                             
                             IMPORTANT: Statement page 118 thesis 2000 about effective cross-section interconnections   
                             
--- task 3 (10/11/2022) ---> The graph architecture is now redefined, we have 36 nodes, quadrupoles are divided in
                             3 and dipoles are divided in 5. 
                             An important assumptions about the mass distribution was made. 
                             First of all ,from Benjamin, we learned that the proper magnet length is 14.4 over 15 m, 
                             meaning the starting and ending caps are 30 cm. Then we computed the mass of helium inside
                             these 30 cm, the we reduced it to 75%, due to the fact we know these volumes are not fully
                             empty but some space is occupied by BHX, the two beam tubes, and the symmetric BHX tube
                             at the bottom of the magnets.
                             Then the interconnections cross section was fixed to 60 cm^2, 8 heaters with power equal
                             to 8 W are applied, and from paper computations ( see paper and photo on Mattermost ) the 
                             delta T between the last wetted node and the first dry node, has to be 26 mK to transfer 
                             53.3 W through 60 cm^2. 
                             Simulating the model, it reaches the steady state and the delta T is the same of the one 
                             found on paper.
                                
COMMENTS ON THE SCRIPT: --> In the first part the fractions of wetted perimeter are computed taking in input the 
                        number of wetted nodes ( old version )
                        --> In the second part the fractions are computing specifying the wetted length in meters
                        ( new version ).
                         IMPORTANT: use a wetted length equal to the end an entire magnets. If the wetted length falls
                         in the middle of a magnet, the all the magnet is considered dry. 
                                                
'''
# %% ----> First part (07/11/2022)
# Initialization BayonetGeometryEstimator and model

bg = BayonetGeometryEstimator(5.3, 0)
model = HeatSimplifiedModel(30)

# %% Graph creation

frac_evaporate = fractions_wetted_perimeter_from_mf_and_num_wet_nodes(np.array([5.]), 7, bg)
fractions_wetted_perim = frac_evaporate[0]
evaporation_mass_flow = frac_evaporate[1]

# INITIAL_TEMPERATURES = [1.85, 1.85, 1.85, 1.85, 1.85, 1.85, 1.85, 1.85, 1.85,
#                         1.85, 1.85, 1.85, 1.85, 1.85, 1.85, 1.85, 1.85, 1.85, 1.85, 1.85]

INITIAL_TEMPERATURES = np.ones(20, dtype=np.float32) * 1.90   # for speeding convergence

g0 = create_standard_cell(INITIAL_TEMPERATURES,
                          avg_fractions_wetted_perimeter=tf.cast(fractions_wetted_perim, tf.float32),
                          evapor_mass_flow=tf.cast(evaporation_mass_flow, tf.float32),
                          liquid_thermal_conductivity=tf.constant(1500.),
                          static_heat=0.2,
                          common_power_supplied=8.)

# %%  Simulation
required_steps = 5000
obtained_steps = 0
all_results = []
g_to_use = g0

while obtained_steps < required_steps:
    sim_res = model(g_to_use)
    all_results.append(sim_res)
    obtained_steps += sim_res[0].shape[0]
    g_to_use = set_features_from_result_tuple(g_to_use, sim_res)

merged_res = concatenate_results(all_results)

# %% Plot
colors = plt.cm.tab20
mask = [0, 1, 4, 7, 10, 11, 14, 17]
fig = plt.figure(figsize=(16, 6))
num_magnets = merged_res.cells__T.shape[-1]
for i in range(num_magnets):
    if i in mask:
        plt.plot(merged_res.context__time, merged_res.cells__T[:, i], label=f'DT magnet [{i}]', color=colors(i),
                 alpha=1)

plt.xlabel('Time stamp [ s ]')
plt.ylabel('Temperature [ K ]')
plt.legend()
plt.grid()
plt.show()

# Variation for Simulation and Plot

#
# last_temperatures = []
# list_temperatures = []
# g_to_use = g0
# for i in range(30):
#     print(i)
#
#     forward_pass = model(g_to_use)
#
#     list_temperatures.append(forward_pass.cells__T)
#     last_temperatures.append(forward_pass.cells__T[-1])
#
#     current_temperatures = forward_pass.cells__T[-1]
#
#     dict_new_features = {'cells__T': current_temperatures}
#
#     g_to_use = set_graph_features(graph=g_to_use,
#                                   dict_features=dict_new_features)
#
# colors = plt.cm.tab20
# mask = [0, 1, 4, 7, 10, 11, 14, 17]
#
# temperature_array = np.array([]).reshape((0, 20))
#
# for temperature in list_temperatures:
#     temperature = temperature.numpy()
#     temperature_array = np.vstack((temperature_array, temperature))
#
# fig = plt.figure(figsize=(16, 6))
#
# num_magnets = temperature_array.shape[-1]
# for i in range(num_magnets):
#     if i in mask:
#         # plt.plot(np.arange(0, 2790, 1), temperature_array[:, i], label=f'DT magnet [{i}]', color=colors(i), alpha=1)
#         plt.plot(np.arange(0, 930, 1), temperature_array[:, i], label=f'DT magnet [{i}]', color=colors(i), alpha=1)
#
# plt.xlabel('Time stamp [ s ]')
# plt.ylabel('Temperature [ K ]')
#
# plt.legend()
# plt.grid()
# plt.show()


# %% Second part ---->  (10/11/2022)
# Graph initialization

INITIAL_TEMPERATURES_g1 = np.ones(36, dtype=np.float32) * 1.90
g1 = create_standard_cell_with_interconnections(temperatures=INITIAL_TEMPERATURES_g1)

# np.sum(g1.node_sets['cells']['L']) = 10200.0 cm = 102. m
# np.sum(g1.edge_sets['conduction']['L']) = 21240. cm => 21240.0/2 = 10620.0 cm = 106.2 m
# g1.edge_sets['conduction']['A']

# %% Initialization inputs : mass flow and wetted length (in m)
mass_flow = np.array([5.])
wetted_length = np.array([36.])


mean_fractions_wp_nodes, evap_mass_flows_per_node = fractions_wetted_perimeter_from_mf_and_wetted_length(mass_flow=mass_flow,
                                                                                                         wetted_length=wetted_length,
                                                                                                         bhx_geometry=bg,
                                                                                                         graph=g1)

# %% Set the new features for the graph
dict_new_features = {'liquid__avg_f': np.array(mean_fractions_wp_nodes, dtype=np.float32),
                     'liquid__evapor_mass_flow': np.array(evap_mass_flows_per_node, dtype=np.float32),
                     'context__static_heat': [0.002],
                     'heater__power': np.ones(8, dtype=np.float32) * 8.}

g1 = set_graph_features(graph=g1,
                        dict_features=dict_new_features)

# %% graph plotting
plot_graph_tensor(g1)

# %% Model simulation
required_steps = 2000
obtained_steps = 0
all_results = []
#g_to_use = g1

while obtained_steps < required_steps:
    sim_res = model(g_to_use)
    all_results.append(sim_res)
    obtained_steps += sim_res[0].shape[0]
    g_to_use = set_features_from_result_tuple(g_to_use, sim_res)

merged_res = concatenate_results(all_results)

# %% Plotting model results
colors = plt.cm.tab20
#mask = [0, 1, 4, 7, 10, 11, 14, 17]
mask_new_configuration = [1, 4, 9, 14, 19, 22, 27, 32]
#mask_new_configuration = [0, 3, 8, 13, 18, 21, 26, 31]
fig = plt.figure(figsize=(16, 6))
num_magnets = merged_res.cells__T.shape[-1]
num_lines_plot = 0
for i in range(num_magnets):
    if i in mask_new_configuration:
        plt.plot(merged_res.context__time, merged_res.cells__T[:, i], label=f'DT magnet [{i}]',
                 color=colors(num_lines_plot), alpha=1)
        num_lines_plot = num_lines_plot + 1

plt.xlabel('Time stamp [ s ]')
plt.ylabel('Temperature [ K ]')
plt.legend()
plt.grid()
plt.show()
