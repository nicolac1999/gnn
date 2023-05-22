import numpy as np

from common.math.bayonet_geometry import *
# %%

bg = BayonetGeometryEstimator(5.3, 0)

# %%

f = np.array([0.3, 0.2, 0.1, 0.5, 0.98])
liquid_height = np.array([1.09236908, 1.5])
liquid_height2 = np.array([1.09236908, 0.50610496, 0.12970023, 2.65, 5.29477083])

cell_length = np.array([500.])
cross_area = np.array([3.27915313,  1.07296969,  0.14232247, 11.0309172, 22.06067404])
volume = np.array([1639.576562, 2566.51356937])

mass_flow = np.array([4.7547719, 1.555806, 0.20636758, 15.99482949, 31.98797643])

bg.height_from_fraction(fraction=f)

bg.cross_area_from_fraction(fraction=f)

bg.volume_from_height(liquid_height=liquid_height, length=cell_length)

bg.volume_from_fraction(fraction=f, length=cell_length)

bg.cross_area_from_height(liquid_height=liquid_height)

bg.fraction_from_cross_area(cross_area=cross_area)

bg.fraction_from_height(liquid_height=liquid_height2)

bg.height_from_cross_area(cross_areas=cross_area)

bg.height_from_volume(volume=volume, length=cell_length)




