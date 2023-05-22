from common.math.bayonet_geometry import *
from lib.diag_common.numpy_helpers import save_numpy_dict_to_file
from subsystems.heat_flux.graphs.standard_cell_dynamic_BHX_virtual_pot import \
    create_standard_cell_with_dynamic_BHX_virtual_pot
from subsystems.heat_flux.models.heat_simplified_model import HeatSimplifiedModel
from subsystems.heat_flux.graphs.standard_cell_with_interconnections_system import *
from subsystems.heat_flux.sandbox.mass_flow_given_supplied_heat import mass_flow_given_total_heat
from subsystems.heat_flux.utils.plotting import plot_model_wetted_lengths_simulation
from subsystems.heat_flux.utils.simulation import simulate_wetted_lengths_variation

#%%

CONFIGURATIONS_DIRECTORY = 'data/fluid/fraction_for_mass_flow_model_configurations/'
MODEL_CONFIGURATION = 'frac_for_mf-slope=1.4.npz'

bg = BayonetGeometryEstimator(5.4, 1.2)
model = HeatSimplifiedModel(configuration_file=CONFIGURATIONS_DIRECTORY + MODEL_CONFIGURATION, num_steps=30)

power_supplied_650 = np.array([4.1138, 4.1286, 0., 4.1516, 4.2547, 4.1511, 0., 4.2978], dtype=np.float32)
power_supplied_700 = np.array([4.7029, 4.7202, 0., 4.7378, 4.8566, 4.7571, 0., 4.9086], dtype=np.float32)
power_supplied_0 = np.zeros(8, dtype=np.float32)

HEAT_COMING_FROM_ADJACENT_CELL = False
STATIC_HEAT = 0.002  # < -- because it should go in W/cm
INITIAL_TEMPERATURES = np.ones(36, dtype=np.float32) * 1.92
NUM_HEATERS = 8
WETTED_LENGTHS = np.array([102., 96., 81., 66., 51., 45., 30., 15.])
WETTED_LENGTHS_fine_grained = np.array([15., 15.31, 20.11, 24.91, 29.71, 30.])
WETTED_LENGTHS_fine_grained_first_dipole = np.array([14.701, 9.901, 5.101, 0.31])
NUM_BHX_NODES = [36, 33, 28, 23, 18, 15, 10, 5]
NUM_BHX_NODES_fine_grained = [5, 6, 7, 8, 9, 10]
NUM_BHX_NODES_fine_grained_first_dipole = [4, 3, 2, 1]
LIQUID_SATURATION_TEMPERATURES = 1.85

#  -----------> 650 W


g_to_use_650 = create_standard_cell_with_dynamic_BHX_virtual_pot(temperatures=INITIAL_TEMPERATURES,
                                                                 num_nodes_BHX=36,
                                                                 power_supplied=power_supplied_650,
                                                                 static_heat=STATIC_HEAT,
                                                                 liquid_flow_direction=1,
                                                                 num_liquid_nodes_per_cell=1,
                                                                 heat_coming_from_adjacent_cell=HEAT_COMING_FROM_ADJACENT_CELL,
                                                                 interconnection_cross_section=66.)

mass_flow = mass_flow_given_total_heat(g_to_use_650, incremental_percentage_vaporization_power=0.0)
temperatures_mask = np.asarray(g_to_use_650.node_sets['cells']['has_sensor']).nonzero()[0].tolist()

sim_fine_grained_650 = simulate_wetted_lengths_variation(model=model,
                                                         graph=g_to_use_650,
                                                         mass_flow=mass_flow,
                                                         wetted_lengths=WETTED_LENGTHS_fine_grained,
                                                         num_bhx=NUM_BHX_NODES_fine_grained,
                                                         bhx_geometry=bg,
                                                         feature_to_converge='cells__T',
                                                         time_track_step=1200.,
                                                         steady_state_duration=1800.,
                                                         mass_flow_from_right=True,
                                                         max_duration=60000,
                                                         interconnection_cross_section=66.)

plot_model_wetted_lengths_simulation(model_simulations=sim_fine_grained_650,
                                     temperatures_mask=temperatures_mask,
                                     wetted_lengths=WETTED_LENGTHS_fine_grained,
                                     title='',
                                     plot_average_temp=True)


path_to_directory = 'results/heat/simulations/650W_66cm2/'
experiment = 'Standard_Cell_Feb_650W_66cm2_wl='
suffixes_sim_fine_grained = ['15.', '15.3', '20.1', '24.9', '29.7', '30.']

for i in range(len(suffixes_sim_fine_grained)):
    save_numpy_dict_to_file(sim_fine_grained_650[i]._asdict(),
                            path_to_directory + experiment + suffixes_sim_fine_grained[i])

#  ----------------> 700 W

g_to_use_700 = create_standard_cell_with_dynamic_BHX_virtual_pot(temperatures=INITIAL_TEMPERATURES,
                                                                 num_nodes_BHX=36,
                                                                 power_supplied=power_supplied_700,
                                                                 static_heat=STATIC_HEAT,
                                                                 liquid_flow_direction=1,
                                                                 num_liquid_nodes_per_cell=1,
                                                                 heat_coming_from_adjacent_cell=HEAT_COMING_FROM_ADJACENT_CELL,
                                                                 interconnection_cross_section=66.)

mass_flow = mass_flow_given_total_heat(g_to_use_700, incremental_percentage_vaporization_power=0.0)
temperatures_mask = np.asarray(g_to_use_700.node_sets['cells']['has_sensor']).nonzero()[0].tolist()

sim_fine_grained_700 = simulate_wetted_lengths_variation(model=model,
                                                         graph=g_to_use_700,
                                                         mass_flow=mass_flow,
                                                         wetted_lengths=WETTED_LENGTHS_fine_grained,
                                                         num_bhx=NUM_BHX_NODES_fine_grained,
                                                         bhx_geometry=bg,
                                                         feature_to_converge='cells__T',
                                                         time_track_step=1200.,
                                                         steady_state_duration=1800.,
                                                         mass_flow_from_right=True,
                                                         max_duration=60000,
                                                         interconnection_cross_section=66.)

path_to_directory = 'results/heat/simulations/700W_66cm2/'
experiment = 'Standard_Cell_Feb_700W_66cm2_wl='
suffixes_sim_fine_grained = ['15.', '15.3', '20.1', '24.9', '29.7', '30.']

for i in range(len(suffixes_sim_fine_grained)):
    save_numpy_dict_to_file(sim_fine_grained_700[i]._asdict(),
                            path_to_directory + experiment + suffixes_sim_fine_grained[i])

#  ----------------> 0 W

INITIAL_TEMPERATURES = np.ones(36, dtype=np.float32) * 1.90

g_to_use_000 = create_standard_cell_with_dynamic_BHX_virtual_pot(temperatures=INITIAL_TEMPERATURES,
                                                                 num_nodes_BHX=36,
                                                                 power_supplied=power_supplied_0,
                                                                 static_heat=STATIC_HEAT,
                                                                 liquid_flow_direction=1,
                                                                 num_liquid_nodes_per_cell=1,
                                                                 heat_coming_from_adjacent_cell=HEAT_COMING_FROM_ADJACENT_CELL,
                                                                 interconnection_cross_section=66.)

mass_flow = mass_flow_given_total_heat(g_to_use_000, incremental_percentage_vaporization_power=0.)
temperatures_mask = np.asarray(g_to_use_000.node_sets['cells']['has_sensor']).nonzero()[0].tolist()

sim_fine_grained_000 = simulate_wetted_lengths_variation(model=model,
                                                         graph=g_to_use_000,
                                                         mass_flow=mass_flow,
                                                         wetted_lengths=WETTED_LENGTHS_fine_grained_first_dipole,
                                                         num_bhx=NUM_BHX_NODES_fine_grained_first_dipole,
                                                         bhx_geometry=bg,
                                                         feature_to_converge='cells__T',
                                                         time_track_step=1200.,
                                                         steady_state_duration=1800.,
                                                         mass_flow_from_right=True,
                                                         max_duration=60000,
                                                         interconnection_cross_section=66.)

path_to_directory = 'results/heat/simulations/0W_66cm2/'
experiment = 'Standard_Cell_Feb_0W_66cm2_wl='
suffixes_sim_fine_grained = ['14.7', '9.9', '5.1', '0.3']

for i in range(len(suffixes_sim_fine_grained)):
    save_numpy_dict_to_file(sim_fine_grained_000[i]._asdict(),
                            path_to_directory + experiment + suffixes_sim_fine_grained[i])

# ---------> 650 + adj
HEAT_COMING_FROM_ADJACENT_CELL = True

power_supplied_650_plus_side_cell = np.array([4.1138, 4.1286, 0., 4.1516, 4.2547, 4.1511, 0.,
                                              4.2978, 25.], dtype=np.float32)

INITIAL_TEMPERATUREs = np.ones(36, dtype=np.float32) * 1.89

g_to_use_650_plus_side = create_standard_cell_with_dynamic_BHX_virtual_pot(temperatures=INITIAL_TEMPERATURES,
                                                                           num_nodes_BHX=36,
                                                                           power_supplied=power_supplied_650_plus_side_cell,
                                                                           static_heat=STATIC_HEAT,
                                                                           liquid_flow_direction=1,
                                                                           num_liquid_nodes_per_cell=1,
                                                                           heat_coming_from_adjacent_cell=HEAT_COMING_FROM_ADJACENT_CELL,
                                                                           interconnection_cross_section=66.)

mass_flow = mass_flow_given_total_heat(g_to_use_650_plus_side, incremental_percentage_vaporization_power=0.0)
temperatures_mask = np.asarray(g_to_use_650_plus_side.node_sets['cells']['has_sensor']).nonzero()[0].tolist()

sim_650_plus_side = simulate_wetted_lengths_variation(model=model,
                                                      graph=g_to_use_650_plus_side,
                                                      mass_flow=mass_flow,
                                                      wetted_lengths=WETTED_LENGTHS_fine_grained,
                                                      num_bhx=NUM_BHX_NODES_fine_grained,
                                                      bhx_geometry=bg,
                                                      feature_to_converge='cells__T',
                                                      time_track_step=1200.,
                                                      steady_state_duration=1800.,
                                                      mass_flow_from_right=True,
                                                      max_duration=60000,
                                                      heat_coming_from_adj_cell=True,
                                                      interconnection_cross_section=66.)

path_to_directory = 'results/heat/simulations/650W+adj_66cm2/'
experiment = 'Standard_Cell_Feb_650W+adj_66cm2_wl='
suffixes_sim_fine_grained = ['15.', '15.3', '20.1', '24.9', '29.7', '30.']

for i in range(len(suffixes_sim_fine_grained)):
    save_numpy_dict_to_file(sim_650_plus_side[i]._asdict(),
                            path_to_directory + experiment + suffixes_sim_fine_grained[i])


# ----------------> 700W + adj

HEAT_COMING_FROM_ADJACENT_CELL = True
power_supplied_700_plus_side_cell = np.array([4.1138, 4.1286, 0., 4.1516, 4.2547, 4.1511, 0.,
                                              4.2978, 28.7], dtype=np.float32)


g_to_use_700_plus_side = create_standard_cell_with_dynamic_BHX_virtual_pot(temperatures=INITIAL_TEMPERATURES,
                                                                           num_nodes_BHX=36,
                                                                           power_supplied=power_supplied_700_plus_side_cell,
                                                                           static_heat=STATIC_HEAT,
                                                                           liquid_flow_direction=1,
                                                                           num_liquid_nodes_per_cell=1,
                                                                           heat_coming_from_adjacent_cell=HEAT_COMING_FROM_ADJACENT_CELL,
                                                                           interconnection_cross_section=66.)

mass_flow = mass_flow_given_total_heat(g_to_use_700_plus_side, incremental_percentage_vaporization_power=0.)
temperatures_mask = np.asarray(g_to_use_700_plus_side.node_sets['cells']['has_sensor']).nonzero()[0].tolist()

sim_700_plus_side = simulate_wetted_lengths_variation(model=model,
                                                      graph=g_to_use_700_plus_side,
                                                      mass_flow=mass_flow,
                                                      wetted_lengths=WETTED_LENGTHS_fine_grained,
                                                      num_bhx=NUM_BHX_NODES_fine_grained,
                                                      bhx_geometry=bg,
                                                      feature_to_converge='cells__T',
                                                      time_track_step=1200.,
                                                      steady_state_duration=1800.,
                                                      mass_flow_from_right=True,
                                                      max_duration=60000,
                                                      heat_coming_from_adj_cell=True,
                                                      interconnection_cross_section=66.)

path_to_directory = 'results/heat/simulations/700W+adj_66cm2/'
experiment = 'Standard_Cell_Feb_700W+adj_66cm2_wl='
suffixes_sim_fine_grained = ['15.', '15.3', '20.1', '24.9', '29.7', '30.']

for i in range(len(suffixes_sim_fine_grained)):
    save_numpy_dict_to_file(sim_700_plus_side[i]._asdict(),
                            path_to_directory + experiment + suffixes_sim_fine_grained[i])

