import numpy as np
from common.math.bayonet_geometry import *
from tensorflow_gnn.graph import graph_tensor as gt

from subsystems.heat_flux.graphs.standard_cell_with_interconnections_system import \
    create_standard_cell_with_interconnections
from subsystems.heat_flux.utils.system_properties import fractions_wetted_perimeter_from_mf_and_num_wet_nodes, \
    fractions_wetted_perimeter_from_mf_and_wetted_length

# %% Report of the script
'''
--- task 2 (07/11/2022) --> compute the average fractions of wetted perimeter given the number of wetted nodes 
--- task 2 variation (08/11/2022) --> compute the average fractions of wetted perimeter given the wetted 
                                      length (in meters)
--- Note: on 08/11/2022 we realized that the small cross section imposed by the interconnections forced to have an
          high delta T to be able to transfer the power from the dry region to the wet one
          
          Computation on the whiteboard:
          
          A1 = 30 cm^2, L1 = 60 cm => delta T = 0.506 K
          A2 = 30 cm^2, L2 = 560 cm => delta T = 4.72 K
          A3 = 60 cm^2, L3 = 60 cm => delta T = 0.062 K  (cross section double to the first case =>
                                                         heat flux density half =>
                                                         delta T 8 times less (2**3) 
           
'''

# %%  BHX geometry initialization

bg = BayonetGeometryEstimator(5.3, 0)


#### POSSIBILITY OF DELETING THE FOLLOWING CODE
# # %%
# # fractions_wetted_perimeter_from_mf_and_num_wet_nodes function testing
#
# fractions_wetted_perimeter_from_mf_and_num_wet_nodes(np.array([4.78]), 10, bayonet_geometry=bg)
# fractions_wetted_perimeter_from_mf_and_num_wet_nodes(np.array([5.]), 7, bayonet_geometry=bg)
# fractions_wetted_perimeter_from_mf_and_num_wet_nodes(np.array([4.8]), 20, bg)



# %%
# fractions_wetted_perimeter_from_mf_and_wetted_length testing

# ---> initialization of the inputs: mass flow, wetted length, bayonet geometry ( already initialized )
INITIAL_TEMPERATURES = np.ones(20, dtype=np.float32) * 1.90
#g0 = create_standard_cell(INITIAL_TEMPERATURES)
g1 = create_standard_cell_with_interconnections(INITIAL_TEMPERATURES)
mass_flow = np.array([5.])
wetted_length = np.array([36.])

# %%
fractions_wp, evap_mass_flows, _ = fractions_wetted_perimeter_from_mf_and_wetted_length(mass_flow=mass_flow,
                                                                                     wetted_length=wetted_length,
                                                                                     bhx_geometry=bg,
                                                                                     graph=g1,
                                                                                     mass_flow_from_right=False)
wetted_length = np.array([30.])
fractions_wp, evap_mass_flows, _ = fractions_wetted_perimeter_from_mf_and_wetted_length(mass_flow=mass_flow,
                                                                                     wetted_length=wetted_length,
                                                                                     bhx_geometry=bg,
                                                                                     graph=g1,
                                                                                     mass_flow_from_right=True)


# %% Testing the function in the middle of magnets


wetted_length = np.array([14.701])
fractions_wp, evap_mass_flows, _ = fractions_wetted_perimeter_from_mf_and_wetted_length(mass_flow=mass_flow,
                                                                                     wetted_length=wetted_length,
                                                                                     bhx_geometry=bg,
                                                                                     graph=g1,
                                                                                     mass_flow_from_right=True)

