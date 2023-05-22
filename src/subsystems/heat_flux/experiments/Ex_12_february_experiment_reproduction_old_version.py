import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# %% CONFIGURATION BLOCK
MAGNETS_TEMPERATURES_0W = [1.948848367, 1.950351834, 1.946064234, 1.947808027, 1.947463989, 1.947883844, 1.946063399, 1.94485116]
MAGNETS_AVG_0W = np.mean(MAGNETS_TEMPERATURES_0W)
MAGNETS_TEMPERATURES_650W = [1.949618526, 1.95105207, 1.945541739, 1.944453597, 1.936847329, 1.934995896, 1.910495024, 1.892742991]
MAGNETS_AVG_650W = np.mean(MAGNETS_TEMPERATURES_650W)
MAGNETS_TEMPERATURES_700w = [1.949453831, 1.950561047, 1.945541739, 1.944090605, 1.934944987, 1.93243885, 1.901815791, 1.887179732]
MAGNETS_AVG_700W = np.mean(MAGNETS_TEMPERATURES_700w)

ALL_MAGNETS = [MAGNETS_TEMPERATURES_0W, MAGNETS_TEMPERATURES_650W,  MAGNETS_TEMPERATURES_700w]
ALL_MAGNETS_AVG = [MAGNETS_AVG_0W, MAGNETS_AVG_650W, MAGNETS_AVG_700W]

HEATERS_POWER_0 = [0., 0., 0., 0., 0., 0., 0., 0.]
HEATERS_POWER_650 = [4.113879681, 4.128593445, 0, 4.151591301, 4.254763126, 4.15112114, 0, 4.297861197]
HEATERS_POWER_700 = [4.702946186, 4.720263481, 0, 4.737853036, 4.856591104, 4.757156372, 0, 4.908631802]

ALL_HEATERS_POWER = [0, 4.113879681, 4.702946186]
ALL_HEATERS_POWER_per_m = [np.sum(HEATERS_POWER_0) / 107,
                     np.sum(HEATERS_POWER_650) / 107,
                     np.sum(HEATERS_POWER_700) / 107]

# %%
colors = plt.cm.tab20
plt.figure(figsize=(10,6))
for i in range(len(ALL_MAGNETS)):
    for j in range(len(ALL_MAGNETS[i])):
        plt.plot(ALL_HEATERS_POWER[i], ALL_MAGNETS[i][j], 'x', color=colors(j))
    plt.plot(ALL_HEATERS_POWER[i], ALL_MAGNETS_AVG[i], 'o', label='avg magnet temp')

plt.ylabel('Magnet temperature [ K ]')
plt.xlabel('Heater power [ W ]')
plt.legend()
plt.grid()
plt.show()

# %%

colors = plt.cm.tab20
plt.figure(figsize=(10,6))
for i in range(len(ALL_MAGNETS)):
    for j in range(len(ALL_MAGNETS[i])):
        plt.plot(ALL_HEATERS_POWER_per_m[i], (ALL_MAGNETS[i][j] - 1.85) * 1000, 'x', color=colors(j))
    plt.plot(ALL_HEATERS_POWER_per_m[i], (ALL_MAGNETS_AVG[i] - 1.85) * 1000, 'o', label='avg magnet temp')

plt.ylabel(' DeltaT (Tmagnets - Tsat) [mK]')
plt.xlabel('Heater power [ W/m ]')
plt.legend()
plt.grid()
plt.show()

# %% plot with the result of the first approximation

PREDICTION_0_W = np.array([1.9640825, 1.9638925, 1.9630164, 1.9614385,
                           1.959215, 1.9574094, 1.9541497, 1.9501874], dtype=np.float32)

# PREDICTION_650_W = np.array([1.96052632, 1.95947368, 1.95631579, 1.95263158,
#                              1.94631579, 1.93947368, 1.91947368, 1.8926])

# predictions with heat coming from adjacent cell
PREDICTION_650_W = np.array([1.9864445, 1.9853913, 1.9825737, 1.9788465, 1.9726433, 1.9655257,
                             1.9456899, 1.9192649])


PREDICTION_700_W = np.array([1.95692308, 1.95538462, 1.95230769, 1.94769231,
                             1.94, 1.93153846, 1.90538462, 1.88230769])

ALL_MAGNETS_PREDICTIONS_FIRST_APPROX = [PREDICTION_0_W, PREDICTION_650_W, PREDICTION_700_W]

legend_elements = [Line2D([0], [0], color='black', marker='x', label='Timber data', linestyle='None'),
                   Line2D([0], [0], color='black', marker='s', label='Simulated', linestyle='None')]

colors = plt.cm.tab20
plt.figure(figsize=(10,6))
for i in range(len(ALL_MAGNETS)):
    for j in range(len(ALL_MAGNETS[i])):
        plt.plot(ALL_HEATERS_POWER[i], ALL_MAGNETS[i][j], 'x', color=colors(j))
        plt.plot(ALL_HEATERS_POWER[i], ALL_MAGNETS_PREDICTIONS_FIRST_APPROX[i][j], 's', color=colors(j))
    #plt.plot(ALL_HEATERS_POWER[i], ALL_MAGNETS_AVG[i], 'o', label='avg magnet temp')

plt.ylabel('Magnet temperature [ K ]')
plt.xlabel('Heater power [ W ]')
plt.title('Comparison sensors data and model_v1 simulation')
plt.legend(handles=legend_elements)
plt.grid()
plt.show()
