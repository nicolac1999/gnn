from common.graph_manipulation_helpers import set_graph_features, set_features_from_result_tuple

from common.math.bayonet_geometry import *
from lib.diag_common.numpy_helpers import save_numpy_dict_to_file, load_numpy_dict
from subsystems.heat_flux.graphs.string2_phase1 import string2_phase1
from subsystems.heat_flux.models.heat_simplified_model import HeatSimplifiedModel
from subsystems.heat_flux.graphs.standard_cell_with_interconnections_system import *
from subsystems.heat_flux.sandbox.mass_flow_given_supplied_heat import mass_flow_given_total_heat
from subsystems.heat_flux.utils.plotting import plot_string2
from subsystems.heat_flux.utils.simulation import simulate_wetted_lengths_variation

# %% ======================> Experiment description <================================
'''
The aim of the script is try to reproduce the laboratory experiment of 2003 on String2, all the values and more details 
can be found in the paper " LHC internal notes, Enrique Blanco".
A graph representing the half string is created, 5 magnets ( Q-D-D-D-Q ) and the setup is the following one:
* 1.4 slope
* BHX fully wet
* Static heat load 0.45 W/m
* Heat applied from 0 W/m to 1 W/m with steps of 0.2 W/m  
'''
# %% =======================> Experimental setup <====================================

#CONFIGURATIONS_DIRECTORY = 'data/fluid/fraction_for_mass_flow_model_configurations/'
#MODEL_CONFIGURATION = 'frac_for_mf-slope=0..npz'
#
#bg = BayonetGeometryEstimator(5.4, 1.2)
#model = HeatSimplifiedModel(configuration_file=CONFIGURATIONS_DIRECTORY + MODEL_CONFIGURATION, num_steps=30)
#
#HEAT_COMING_FROM_ADJACENT_CELL = False
## STATIC_HEAT = 0.0045  # < -- because it should go in W/cm
#STATIC_HEAT = 0.00342
#INITIAL_TEMPERATURES = np.ones(21, dtype=np.float32) * 1.86
#NUM_HEATERS = 5
#WETTED_LENGTH = np.array([57.01])
#NUM_BHX_NODES = [21]
#LIQUID_SATURATION_TEMPERATURES = 1.85
#
#g_string2 = string2_phase1(temperatures=INITIAL_TEMPERATURES,
#                           num_nodes_BHX=21,
#                           static_heat=STATIC_HEAT,
#                           liquid_flow_direction=1,
#                           num_liquid_nodes_per_cell=1,
#                           heat_coming_from_adjacent_cell=HEAT_COMING_FROM_ADJACENT_CELL,
#                           interconnection_cross_section=66.)
#
#graph_string2_length = np.sum(g_string2.node_sets['cells']['L']) / 100
#dynamic_heat_per_m_steps = np.arange(0., 1.1, 0.2)
#temperatures_mask = np.asarray(g_string2.node_sets['cells']['has_sensor']).nonzero()[0].tolist()
#
## %% ===========================> Simulation <=====================================
#
#g_to_use = g_string2
#sims_results_string2 = []
#for dynamic_heat_per_m in dynamic_heat_per_m_steps:
#    print('dynamic_heat going on :', dynamic_heat_per_m)
#
#    total_heating_power = dynamic_heat_per_m * graph_string2_length
#    heater_power = np.ones(NUM_HEATERS, dtype=np.float32) * (total_heating_power / NUM_HEATERS)
#
#    new_features = {'heater__power': heater_power,
#                    'context__time': [0.]}
#
#    g_to_use = set_graph_features(graph=g_to_use, dict_features=new_features)
#
#    mass_flow = mass_flow_given_total_heat(g_to_use, incremental_percentage_vaporization_power=0.10)
#    # print(f'For {dynamic_heat_per_m} the mass flow is :{mass_flow}')
#
#    sim = simulate_wetted_lengths_variation(configuration='string2',
#                                            model=model,
#                                            graph=g_to_use,
#                                            mass_flow=mass_flow,
#                                            wetted_lengths=WETTED_LENGTH,
#                                            num_bhx=NUM_BHX_NODES,
#                                            bhx_geometry=bg,
#                                            feature_to_converge='cells__T',
#                                            time_track_step=1200.,
#                                            steady_state_duration=1800.,
#                                            mass_flow_from_right=True,
#                                            max_duration=60000,
#                                            interconnection_cross_section=66.)
#
#    g_to_use = set_features_from_result_tuple(g_to_use, sim[0], which_step=-1, include_unknown=True)
#
#    sims_results_string2.append(sim)
#
## %% =============================> Plot results <======================================
#
#plot_string2(simulations=sims_results_string2, static_heat=STATIC_HEAT, temperatures_mask=temperatures_mask)
#
## %% ==================================> Save results <===============================
#
#path_to_directory = f'results/heat/simulations/STRING_2/STATIC_HEAT_{round(STATIC_HEAT * 100, 3)}/'
#experiment = 'STRING2_'
#suffixes = ['0', '0.2', '0.4', '0.6', '0.8', '1.']
#for i in range(len(sims_results_string2)):
#    save_numpy_dict_to_file(sims_results_string2[i][0]._asdict(), path_to_directory + experiment
#                            + suffixes[i] + 'W' + f'geometry_53+12_+10%mf_sh={round(STATIC_HEAT * 100, 3)}' + '.npz')
#
# %% ==================================> Import results <===============================

file_path_string2_0 = 'results/heat/simulations/STRING_2/STATIC_HEAT_0.342/STRING2_0Wgeometry_53+12_+10%mf_sh=0.342.npz'
file_path_string2_0_2 = 'results/heat/simulations/STRING_2/STATIC_HEAT_0.342/STRING2_0.2Wgeometry_53+12_+10%mf_sh=0.342.npz'
file_path_string2_0_4 = 'results/heat/simulations/STRING_2/STATIC_HEAT_0.342/STRING2_0.4Wgeometry_53+12_+10%mf_sh=0.342.npz'
file_path_string2_0_6 = 'results/heat/simulations/STRING_2/STATIC_HEAT_0.342/STRING2_0.6Wgeometry_53+12_+10%mf_sh=0.342.npz'
file_path_string2_0_8 = 'results/heat/simulations/STRING_2/STATIC_HEAT_0.342/STRING2_0.8Wgeometry_53+12_+10%mf_sh=0.342.npz'
file_path_string2_1 = 'results/heat/simulations/STRING_2/STATIC_HEAT_0.342/STRING2_1.Wgeometry_53+12_+10%mf_sh=0.342.npz'

sim_string2_0 = load_numpy_dict(file_path_string2_0)
sim_string2_0_2 = load_numpy_dict(file_path_string2_0_2)
sim_string2_0_4 = load_numpy_dict(file_path_string2_0_4)
sim_string2_0_6 = load_numpy_dict(file_path_string2_0_6)
sim_string2_0_8 = load_numpy_dict(file_path_string2_0_8)
sim_string2_1 = load_numpy_dict(file_path_string2_1)

sims_results_string2 = [sim_string2_0, sim_string2_0_2,
                        sim_string2_0_4, sim_string2_0_6,
                        sim_string2_0_8, sim_string2_1]

STATIC_HEAT = 0.34
TEMPERATURES_MASK = [1, 4, 8, 14, 19]

plot_string2(simulations=sims_results_string2, static_heat=STATIC_HEAT, temperatures_mask=TEMPERATURES_MASK)
