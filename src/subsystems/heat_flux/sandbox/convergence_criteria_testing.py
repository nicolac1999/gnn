from common.graph_manipulation_helpers import set_graph_features
from common.math.bayonet_geometry import *
from subsystems.heat_flux.models.heat_simplified_model import HeatSimplifiedModel
from subsystems.heat_flux.graphs.standard_cell_with_interconnections_system import *
from subsystems.heat_flux.sandbox.fraction_wp_from_mf_and_wl_testing import fractions_wetted_perimeter_from_mf_and_wetted_length
from subsystems.heat_flux.sandbox.mass_flow_given_supplied_heat import mass_flow_given_total_heat
from subsystems.heat_flux.utils.simulation import simulate_time_period_till_convergence

import matplotlib.pyplot as plt

'''
task 8 (16/11/2022) ---> improve convergence criteria, plot longer steady state for claming 
                          the temperatures are stable
'''


# %%
bg = BayonetGeometryEstimator(5.3, 0)
model = HeatSimplifiedModel(30)

power_supplied_650 = np.array([4.1138, 4.1286, 0., 4.1516, 4.2547, 4.1511, 0., 4.2978], dtype=np.float32)

INITIAL_TEMPERATURES_g1 = np.ones(36, dtype=np.float32) * 1.865

g3 = create_standard_cell_with_interconnections(temperatures=INITIAL_TEMPERATURES_g1,
                                                liquid_temperature=1.85,
                                                static_heat=0.002,
                                                power_supplied=power_supplied_650,
                                                time_step=1.)

# %%

mass_flow = mass_flow_given_total_heat(g3, incremental_percentage_vaporization_power=0.2)
wetted_length = np.array([102.])

mean_fractions_wp_nodes, evap_mass_flows_per_node = fractions_wetted_perimeter_from_mf_and_wetted_length(
            mass_flow=mass_flow,
            wetted_length=wetted_length,
            bhx_geometry=bg,
            graph=g3,
            mass_flow_from_right=True)

dict_new_features = {'liquid__avg_f': np.array(mean_fractions_wp_nodes, dtype=np.float32),
                     'liquid__evapor_mass_flow': np.array(evap_mass_flows_per_node, dtype=np.float32)}

g3 = set_graph_features(graph=g3, dict_features=dict_new_features)

# %%

sim_res = simulate_time_period_till_convergence(model=model, graph=g3,
                                                feature_to_converge='cells__T',
                                                time_track_step=300.,
                                                steady_state_duration=900.)

# %%
fig = plt.figure(figsize=(20, 20))
colors = plt.cm.tab20
mask_new_configuration = [1, 4, 9, 14, 19, 22, 27, 32]
num_magnets = sim_res.cells__T.shape[-1]
num_lines_plot = 0

for i in range(num_magnets):
    if i in mask_new_configuration:
        plt.plot(sim_res.context__time, sim_res.cells__T[:, i], label=f'DT magnet [{i}]',
                 color=colors(num_lines_plot), alpha=1)
        num_lines_plot = num_lines_plot + 1

plt.xlabel('Time stamp [ s ]')
plt.ylabel('Temperature [ K ]')
plt.title('Simulation for time step 10 s, variation in computations ')
plt.legend()
plt.grid()

plt.show()

