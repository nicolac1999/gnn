import tensorflow_gnn as tfgnn
import numpy as np
import matplotlib.pyplot as plt
from common.math.bayonet_geometry import BayonetGeometryEstimator

# %%

VELOCITY = 10  # cm/s
TIME = 30  # s
D = 5.3  # cm
DENSITY_HELIUM_SUPERFLUID = 0.145  # g/cm3
LATENT_HEAT_VAPORIZATION = 23  # W/g


def liquid_depth_from_wetted_fraction(fraction, D):
    '''
    :param fraction: perimeter wetted fraction
    :param D: diameter of the tube
    :return h: liquid depth
    '''
    alpha = fraction * np.pi
    h = D / 2 * (1 - np.cos(alpha))
    return h

def area_from_liquid_level_circle(h, D):
    alpha = np.arccos(1 - 2 * h / D)
    A = (np.power(D, 2) / 4) * (alpha - np.sin(2 * alpha) / 2)

    return A


def mean_fraction(fractions):
    result = (fractions[0] + fractions[1]) / 2
    return result


def direct_kapitza(fraction_wetted_surface):
    result = 430 * 1.85 ** 3 * 0.832 * 0.015 * fraction_wetted_surface
    return result

def direct_kapitza_from_f_and_t(fraction_wetted_surface, delta_t):
    result = 430 * 1.85 ** 3 * 0.832 * delta_t * fraction_wetted_surface
    return result

def delta_t_from_kapitza(kapitza, fraction_wetted_surface):
    result = kapitza / (430 * 1.85 ** 3 * 0.832 * fraction_wetted_surface)
    return result


# ------------------- wetted fraction

wetted_percentages = np.linspace(0, 0.3, 21)
aws_awf_list = []
for i in range(len(wetted_percentages) - 1):
    aws_awf = (wetted_percentages[i], wetted_percentages[i + 1])
    aws_awf_list.append(aws_awf)

plt.plot(wetted_percentages)
plt.show()

#%%
# ------------------ wetted fraction --> liquid level

liquid_levels_list = []
for fs in aws_awf_list:
    liquid_levels = (liquid_depth_from_wetted_fraction(fs[0], D=D), liquid_depth_from_wetted_fraction(fs[1], D=D))
    liquid_levels_list.append(liquid_levels)

plt.plot([ll[0] for ll in liquid_levels_list])
plt.show()

#%%
# ----------------- liquid level --> liquid area
areas_list = []
for hs in liquid_levels_list:
    areas = (area_from_liquid_level_circle(hs[0], D=D), area_from_liquid_level_circle(hs[1], D=D))
    areas_list.append(areas)

# ----------------- liquid area --> volume
volumes_list = []
for areas in areas_list:
    volumes = (areas[0] * VELOCITY, areas[1] * VELOCITY)
    volumes_list.append(volumes)

plt.plot([v[0] for v in volumes_list])
plt.show()
#%%

# ----------------- volume evaporated
Vs_evap_list = []
for vs in volumes_list:
    v_evap = vs[1] - vs[0]
    Vs_evap_list.append(v_evap)

# ------------------ mass evaporated
masses_evap_list = []
for vs_evap in Vs_evap_list:
    masses_evap = vs_evap * DENSITY_HELIUM_SUPERFLUID * 1e6
    masses_evap_list.append(masses_evap)

# %%


# ------------------ W evaporation
W_evaporation_list = []
for m_evap in masses_evap_list:
    W_evaporation = m_evap * LATENT_HEAT_VAPORIZATION
    W_evaporation_list.append(W_evaporation)

# ----------------- Energy evaporation
E_evaporation_list = []
for w_evap in W_evaporation_list:
    E_evap = w_evap * TIME
    E_evaporation_list.append(E_evap)

# ------------------- Average wetted

A_bar_list = []
for fs in aws_awf_list:
    a_bar = mean_fraction(fs)
    A_bar_list.append(a_bar)

# ------------------- Kapitza flux
Q_kapitza_list = []
for a_bar in A_bar_list:
    Q_kapitza = direct_kapitza(a_bar)
    Q_kapitza_list.append(Q_kapitza)

# %%
###################################################################
# second approach
MASS_TO_EVAPORATE = 2.9512378170980443e1 * DENSITY_HELIUM_SUPERFLUID  # mass to evaporate over 100 m g/s
MASS_OVER_5_m = MASS_TO_EVAPORATE/20  # mass to evaporate over 5 m g/s

VOLUME_TO_EVAPORATE_5_m = (MASS_OVER_5_m / (DENSITY_HELIUM_SUPERFLUID)) * 1e-6  # m3/s

bg = BayonetGeometryEstimator(0.053, 0)
INITIAL_LEVEL = bg.height_from_fraction(0.3)

# %%


liquid_levels = [INITIAL_LEVEL]
current_level = INITIAL_LEVEL
for i in range(19):

    volume_first_ll = bg.cross_area_from_height(np.array([current_level])) * VELOCITY
    volume_second_ll = volume_first_ll - VOLUME_TO_EVAPORATE_5_m

    area_second_ll = volume_second_ll / VELOCITY

    second_ll = bg.height_from_cross_area(area_second_ll)

    current_level = second_ll.flatten()

    liquid_levels.append(float(current_level))

liquid_levels.reverse()

# %%
import matplotlib.pyplot as plt
cell_number = np.arange(0, 21)
plt.plot(cell_number, np.array(liquid_levels) * 100)
plt.xticks(cell_number)
plt.title('Liquid level vs BHX cell')
plt.ylabel('Liquid level [cm]')
plt.xlabel('BHX cell')
plt.show()


# %%

bg = BayonetGeometryEstimator(5.3, 0)

bg.volume_from_height(np.array([0.01, 0.005, 0.003]), np.array([5]))

cells_length = np.ones(20) * 5.

liquid_levels = np.ones(20) * 0.011

bg.volume_from_height(liquid_levels, cells_length)

bg.cross_area_from_height(np.array([0.011]))

vs = np.linspace(1655, 0, 21)
lengths = np.ones(21) * 500


liquid_levels = bg.height_from_volume(vs, lengths)
fraction_wetted_perim = bg.fraction_from_height(liquid_height=liquid_levels)


plt.plot(liquid_levels, fraction_wetted_perim)
plt.xlabel('')
plt.show()

# %%

liquid_levels_testing = bg.height_from_fraction(fraction_wetted_perim)
plt.plot(liquid_levels_testing)
plt.xlabel('BHX cell number ')
plt.ylabel('liquid_level')
plt.show()


# %%%


mean_fractions_for_kapitza = []
for i in range(0, len(fraction_wetted_perim)-1):
    mean_fractions_for_kapitza.append((fraction_wetted_perim[i] + fraction_wetted_perim[i+1]) / 2)

kapitza_power = []
for f in mean_fractions_for_kapitza:
    kapitza_power.append(direct_kapitza(f))


plt.plot(kapitza_power)
plt.xlabel('BHX cell number')
plt.ylabel('Kapitza power [W]')
plt.show()

# %%

vs = np.linspace(1655, 0, 21)
vs_shifted = np.roll(vs, -1)

volumes_evaporated = (vs - vs_shifted)[:-1]

mass_evaporated = volumes_evaporated * DENSITY_HELIUM_SUPERFLUID / 50

energy_evaporated = mass_evaporated * LATENT_HEAT_VAPORIZATION

# %%
plt.figure(figsize=(10, 6))
plt.plot(kapitza_power, label='power Kapitza')
plt.plot(energy_evaporated, label='power evaporization')
plt.plot(np.minimum(kapitza_power, energy_evaporated), 'o' , label='min(kap, evap)')
plt.title('Comparison between "power evaporation" and "power kapitza" where the last one \n'
          ' is computed for a ΔT = 15mK')
plt.xlabel('BHX cell number')
plt.ylabel('Power (Energy * time) [W = J * s]')
plt.legend()
plt.grid()
plt.show()


# %%

delta_ts = delta_t_from_kapitza(kapitza_power, np.array(mean_fractions_for_kapitza))
plt.plot(delta_ts)
plt.title('ΔT helium bath and BHX helium')
plt.xlabel('BHX cell number')
plt.ylabel('T [K]')
plt.show()




# %%

direct_kapitza_from_f_and_t(np.array(mean_fractions_for_kapitza), delta_ts, )


# %%

delta_t_to_equalize_evap = delta_t_from_kapitza(energy_evaporated, np.array(mean_fractions_for_kapitza))
plt.plot(delta_t_to_equalize_evap)
plt.title('ΔT to equalize the evaporation power')
plt.xlabel('BHX cell number')
plt.ylabel('ΔT [K]')
plt.show()
# %%

direct_kapitza_from_f_and_t(np.array(mean_fractions_for_kapitza), delta_t_to_equalize_evap, )

plt.plot(direct_kapitza_from_f_and_t(np.array(mean_fractions_for_kapitza), delta_t_to_equalize_evap))
plt.xlabel('BHX cell number')
plt.ylabel('Power [W]')
plt.show()

# %%

mask_delta_t = np.array([0, 1, 4, 7, 10, 12, 14, 17])
delta_t_masked = delta_t_to_equalize_evap[mask_delta_t]

delta_t_masked = delta_t_masked.reshape((1, -1))

temperatures = np.ones(120) * 1.85
temperatures = temperatures.reshape((-1, 1))

matrix_temp = temperatures + delta_t_masked
for i in range(len(matrix_temp[0])):
    plt.plot(matrix_temp[:, i], label=f'T_{i}')

plt.ylim(1.85, 1.875)
plt.legend()
plt.show()



matrix_temp_2 = matrix_temp + 0.087

for i in range(len(matrix_temp_2[0])):
    plt.plot(matrix_temp_2[:, i], label=f'T_{i}')
plt.legend()
plt.show()




matrix_temp_3 = matrix_temp + 0.042

for i in range(len(matrix_temp_3[0])):
    plt.plot(matrix_temp_3[:, i], label=f'T_{i}')
plt.legend()
plt.show()