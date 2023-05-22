from common.math.bayonet_geometry import *
from subsystems.heat_flux.models.heat_simplified_model import HeatSimplifiedModel
from subsystems.heat_flux.graphs.standard_cell_with_interconnections_system import *
from subsystems.heat_flux.utils.simulation import simulate_time_period_till_convergence

import matplotlib.pyplot as plt


# %%
'''

--- task 9 (15/11/2022) ---> check if the model is able to predict the stable condition
                        ---> check if the model is able to reach the stability when all the temperatures
                             are equal , only one is higher than the others
                             PROBLEM: when the time step is high the temperatures starts to oscillate, 
                                      the constraint between a couple of cells about the maximum 
                                      energy that can be transferred is correct, but it is not valid 
                                      for a triplet of nodes, because the central node is distributing 
                                      to both of the neighbors, so its final temperature goes below the neighbors 


'''
# %%  Stable case, no heat load

bg = BayonetGeometryEstimator(5.3, 0)
model = HeatSimplifiedModel(30)

INITIAL_TEMPERATURES_g1 = np.ones(36, dtype=np.float32) * 1.90

g3 = create_standard_cell_with_interconnections(temperatures=INITIAL_TEMPERATURES_g1,
                                                liquid_temperature=1.85,
                                                static_heat=0.,
                                                time_step=10.)

# %%
sim = model(g3)

# %%
fig = plt.figure()
colors = plt.cm.tab20
mask_new_configuration = [1, 4, 9, 14, 19, 22, 27, 32]
num_magnets = sim.cells__T.shape[-1]
num_lines_plot = 0

for i in range(num_magnets):
    if i in mask_new_configuration:
        plt.plot(sim.context__time, sim.cells__T[:, i], label=f'DT magnet [{i}]',
                 color=colors(num_lines_plot), alpha=1)
        num_lines_plot = num_lines_plot + 1

plt.xlabel('Time stamp [ s ]')
plt.ylabel('Temperature [ K ]')
plt.title('Simulation for time step 10 s, variation in computations ')
plt.legend()
plt.grid()

plt.show()


# %% Case when one temperature is higher than the others

INITIAL_TEMPERATURES_g4 = np.array([1.88, 1.88, 1.88, 1.88, 1.88, 1.88,
                                    1.88, 1.88, 1.88, 1.88, 1.88, 1.88,
                                    1.88, 1.88, 1.90, 1.88, 1.88, 1.88,
                                    1.88, 1.88, 1.88, 1.88, 1.88, 1.88,
                                    1.88, 1.88, 1.88, 1.88, 1.88, 1.88,
                                    1.88, 1.88, 1.88, 1.88, 1.88, 1.88], dtype=np.float32)

g4 = create_standard_cell_with_interconnections(temperatures=INITIAL_TEMPERATURES_g4,
                                                liquid_temperature=1.85,
                                                static_heat=0.,
                                                time_step=10.) # change here the time step for visualizing the differences
# %%
sim_res = simulate_time_period_till_convergence(model=model, graph=g4,
                                                feature_to_converge='cells__T')

# %%
fig = plt.figure()
colors = plt.cm.tab20
mask_new_configuration = [1, 4, 9, 14, 19, 22, 27, 32]
mask_for_debug = [13, 14, 15]
num_magnets = sim_res.cells__T.shape[-1]
num_lines_plot = 0

for i in range(num_magnets):
    if i in mask_for_debug:
        plt.plot(sim_res.context__time, sim_res.cells__T[:, i], label=f'DT magnet [{i}]',
                 color=colors(num_lines_plot), alpha=1)
        num_lines_plot = num_lines_plot + 1

plt.xlabel('Time stamp [ s ]')
plt.ylabel('Temperature [ K ]')
plt.title('Simulation for time step 1 s, variation in computations ')
plt.legend()
plt.grid()

plt.show()
