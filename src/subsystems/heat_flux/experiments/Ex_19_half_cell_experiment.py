import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from common.graph_manipulation_helpers import set_graph_features, set_features_from_result_tuple
from common.math.bayonet_geometry import *
from lib.diag_common.numpy_helpers import save_numpy_dict_to_file, load_numpy_dict
from subsystems.heat_flux.graphs.half_cell import half_cell
from subsystems.heat_flux.models.heat_simplified_model import HeatSimplifiedModel
from subsystems.heat_flux.graphs.standard_cell_dynamic_BHX_virtual_pot import *
from subsystems.heat_flux.sandbox.mass_flow_given_supplied_heat import mass_flow_given_total_heat
from subsystems.heat_flux.utils.plotting import plot_half_cell
from subsystems.heat_flux.utils.simulation import simulate_wetted_lengths_variation

# %% ======================> Experiment description <================================
'''
This script creates a half cell 4 magnets (Q-D-D-D), and perform the laboratory experiment made in 2000:
* No slope
* Static heat load of 1 W/m
* Fully wet BHX, with small overflow (valve opened 10% more) 
* Heat applied from 0 W/m to 2 W/m with steps of 0.2 W/m
'''

# %% =======================> Experimental setup <====================================

CONFIGURATIONS_DIRECTORY = 'data/fluid/fraction_for_mass_flow_model_configurations/'
MODEL_CONFIGURATION = 'frac_for_mf-slope=1.4.npz'

bg = BayonetGeometryEstimator(5.4, 1.2)
model = HeatSimplifiedModel(configuration_file=CONFIGURATIONS_DIRECTORY + MODEL_CONFIGURATION, num_steps=30)

HEAT_COMING_FROM_ADJACENT_CELL = False
STATIC_HEAT = 0.01
INITIAL_TEMPERATURES = np.ones(18, dtype=np.float32) * 1.88
NUM_HEATERS = 4
WETTED_LENGTH = np.array([34.71])
NUM_BHX_NODES = [18]
LIQUID_SATURATION_TEMPERATURES = 1.85

g_half_cell = half_cell(temperatures=INITIAL_TEMPERATURES,
                        num_nodes_BHX=18,
                        static_heat=STATIC_HEAT,
                        liquid_flow_direction=1,
                        num_liquid_nodes_per_cell=1,
                        heat_coming_from_adjacent_cell=HEAT_COMING_FROM_ADJACENT_CELL,
                        interconnection_cross_section=66.
                        )

g_half_cell_length = np.sum(g_half_cell.node_sets['cells']['L']) / 100
dynamic_heat_per_m_steps = np.concatenate((np.arange(0., 2.1, 0.2), np.array([1.3, 0.9, 0.5])))
temperatures_mask = np.asarray(g_half_cell.node_sets['cells']['has_sensor']).nonzero()[0].tolist()

# %% ===========================> Simulation <=====================================
# dynamic_heat_per_m_steps = np.array([2.])
g_to_use = g_half_cell
sims_results_half_cell = []
for dynamic_heat_per_m in dynamic_heat_per_m_steps:
    print('dynamic_heat going on :', dynamic_heat_per_m)

    total_heating_power = dynamic_heat_per_m * g_half_cell_length
    heater_power = np.ones(NUM_HEATERS, dtype=np.float32) * (total_heating_power / NUM_HEATERS)

    new_features = {'heater__power': heater_power,
                    'context__time': [0.]}

    g_to_use = set_graph_features(graph=g_to_use, dict_features=new_features)

    mass_flow = mass_flow_given_total_heat(g_to_use, incremental_percentage_vaporization_power=.02)
    # print(f'For {dynamic_heat_per_m} the mass flow is :{mass_flow}')

    sim = simulate_wetted_lengths_variation(configuration='half_cell',
                                            model=model,
                                            graph=g_to_use,
                                            mass_flow=mass_flow,
                                            wetted_lengths=WETTED_LENGTH,
                                            num_bhx=NUM_BHX_NODES,
                                            bhx_geometry=bg,
                                            feature_to_converge='cells__T',
                                            time_track_step=1200.,
                                            steady_state_duration=1800.,
                                            mass_flow_from_right=True,
                                            max_duration=60000,
                                            interconnection_cross_section=66.)

    g_to_use = set_features_from_result_tuple(g_to_use, sim[0], which_step=-1, include_unknown=True)

    sims_results_half_cell.append(sim)

# %% =============================> Plot results <======================================

plot_half_cell(sims_results_half_cell, temperatures_mask)

# %% ==================================> Save results <===============================

path_to_directory = 'results/heat/simulations/HALF_CELL/'
experiment = 'HALF_CELL'
suffixes = ['0', '0.2', '0.4', '0.6', '0.8', '1.', '1.2', '1.4', '1.6', '1.8', '2.', '1.3', '0.9', '0.5']

for i in range(len(sims_results_half_cell)):
    save_numpy_dict_to_file(sims_results_half_cell[i][0]._asdict(), path_to_directory + experiment
                            + suffixes[i] + 'W' + 'geometry_53+12_+300%mf' + '.npz')

# %% ==================================> Import results <===============================

directory = 'results/heat/simulations/HALF_CELL/'
mass_flow = 'MASS_FLOW_2%/'

file_path_half_cell_0 = directory + mass_flow + 'HALF_CELL0Wgeometry_53+12_+2%mf.npz'
file_path_half_cell_0_2 = directory + mass_flow + 'HALF_CELL0.2Wgeometry_53+12_+2%mf.npz'
file_path_half_cell_0_4 = directory + mass_flow + 'HALF_CELL0.4Wgeometry_53+12_+2%mf.npz'
file_path_half_cell_0_5 = directory + mass_flow + 'HALF_CELL0.5Wgeometry_53+12_+2%mf.npz'
file_path_half_cell_0_6 = directory + mass_flow + 'HALF_CELL0.6Wgeometry_53+12_+2%mf.npz'
file_path_half_cell_0_8 = directory + mass_flow + 'HALF_CELL0.8Wgeometry_53+12_+2%mf.npz'
file_path_half_cell_0_9 = directory + mass_flow + 'HALF_CELL0.9Wgeometry_53+12_+2%mf.npz'
file_path_half_cell_1 = directory + mass_flow + 'HALF_CELL1.Wgeometry_53+12_+2%mf.npz'
file_path_half_cell_1_2 = directory + mass_flow + 'HALF_CELL1.2Wgeometry_53+12_+2%mf.npz'
file_path_half_cell_1_3 = directory + mass_flow + 'HALF_CELL1.3Wgeometry_53+12_+2%mf.npz'
file_path_half_cell_1_4 = directory + mass_flow + 'HALF_CELL1.4Wgeometry_53+12_+2%mf.npz'
file_path_half_cell_1_6 = directory + mass_flow + 'HALF_CELL1.6Wgeometry_53+12_+2%mf.npz'
file_path_half_cell_1_8 = directory + mass_flow + 'HALF_CELL1.8Wgeometry_53+12_+2%mf.npz'
file_path_half_cell_2 = directory + mass_flow + 'HALF_CELL2.Wgeometry_53+12_+2%mf.npz'

sim_half_cell_0 = load_numpy_dict(file_path_half_cell_0)
sim_half_cell_0_2 = load_numpy_dict(file_path_half_cell_0_2)
sim_half_cell_0_4 = load_numpy_dict(file_path_half_cell_0_4)
sim_half_cell_0_5 = load_numpy_dict(file_path_half_cell_0_5)
sim_half_cell_0_6 = load_numpy_dict(file_path_half_cell_0_6)
sim_half_cell_0_8 = load_numpy_dict(file_path_half_cell_0_8)
sim_half_cell_0_9 = load_numpy_dict(file_path_half_cell_0_9)
sim_half_cell_1 = load_numpy_dict(file_path_half_cell_1)
sim_half_cell_1_2 = load_numpy_dict(file_path_half_cell_1_2)
sim_half_cell_1_3 = load_numpy_dict(file_path_half_cell_1_3)
sim_half_cell_1_4 = load_numpy_dict(file_path_half_cell_1_4)
sim_half_cell_1_6 = load_numpy_dict(file_path_half_cell_1_6)
sim_half_cell_1_8 = load_numpy_dict(file_path_half_cell_1_8)
sim_half_cell_2 = load_numpy_dict(file_path_half_cell_2)

sim_half_cell = [sim_half_cell_0, sim_half_cell_0_2,
                 sim_half_cell_0_4,
                 sim_half_cell_0_6, sim_half_cell_0_8,
                 sim_half_cell_1,
                 sim_half_cell_1_2,
                 sim_half_cell_1_4, sim_half_cell_1_6,
                 sim_half_cell_1_8, sim_half_cell_2,
                 sim_half_cell_1_3,
                 sim_half_cell_0_9,
                 sim_half_cell_0_5]

# Experimental points derived handmade from the plot (center based)
# EXPERIMENTAL_POINTS = [(0, 1.5), (0.2, 2.5), (0.4, 4.5), (0.5, 6.5), (0.6, 6),
#                        (0.8, 7.5), (0.9, 11), (1, 13.5), (1.2, 15.5), (1.3, 16.5), (1.4,17.5),
#                        (1.6, 20.5), (1.8, 24.2), (2, 28.5)]

# Experimental points form Table 5 "thesis-2000" (bottom based)
EXPERIMENTAL_POINTS = [(0, 0.17), (0.2, 1.67), (0.4, 3.17), (0.5, 5.33), (0.6, 5.33),
                       (0.8, 7.), (0.9, 9.83), (1, 11.83), (1.2, 14.33), (1.3, 15.5), (1.4, 16.83),
                       (1.6, 19.33), (1.8, 22.67), (2, 27.33)]

# TEMPERATURES_MASK = [1, 4, 9, 14]  # <==== CHANGE THIS ( TWO SENSORS PER DIPOLE AND NONE ON THE QUADRUPOLE)
TEMPERATURES_MASK = [3, 7, 8, 12, 13, 17]  # <==== CHANGE THIS ( TWO SENSORS PER DIPOLE AND NONE ON THE QUADRUPOLE)

plot_half_cell(sim_half_cell, TEMPERATURES_MASK, EXPERIMENTAL_POINTS)
