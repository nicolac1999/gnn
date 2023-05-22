from common.math.bayonet_geometry import *
from lib.diag_common.numpy_helpers import save_numpy_dict_to_file
from lib.diag_common.io_helpers import ensure_dir_exists

from lib.diag_common.tf_helpers import set_all_gpu_to_incremental_memory
set_all_gpu_to_incremental_memory()

# -- safe imports: not causing full GPU occupancy:
from subsystems.heat_flux.utils.plotting import plot_model_wetted_lengths_simulation
from subsystems.heat_flux.models.heat_simplified_model import HeatSimplifiedModel

# -- Problematic imporst
from subsystems.heat_flux.graphs.standard_cell_with_interconnections_system import *
from subsystems.heat_flux.utils.simulation import simulate_wetted_lengths_variation
from subsystems.heat_flux.graphs.standard_cell_dynamic_BHX_virtual_pot import create_standard_cell_with_dynamic_BHX_virtual_pot
from subsystems.heat_flux.sandbox.mass_flow_given_supplied_heat import mass_flow_given_total_heat


# %% ======================> Experiment description <================================
"""
The aim of this script is to try to simulate the heating test made on the standard cell Q15-Q16 on February,
when the total power of the ARC of the sector 23 was fist set to 650W and then to 700W.
The test was performed for enough time to have the steady state, so we selected the region where the opening 
of the valve was stable and we took the sensors' temperatures. 

Then we identified an other stable condition when the power was set to 0 W

Looking at the date we realized that some heat is coming also from the adjacent cell Q17-Q18.
So we performed two simulations:
1) taking into account only the heat coming from the heaters of that standard cell
   (8 heaters , located at the beginning of each magnet) 
2) adding heat coming from the adjacent cell: a "virtual" heater is added to the last dipole, on the right end
   with a power equal to the sum of all the heaters in the adjacent cell
   
Assumption: The static heat is extracted independently in each standard cell 
            (this means the JT valve has a minimum opening, it supplies only the amount of helium 
            needed for extract the static heat) 
"""
# %% =======================> Common Experimental setup <====================================

CONFIGURATIONS_DIRECTORY = 'data/fluid/fraction_for_mass_flow_model_configurations/'
MODEL_CONFIGURATION = 'frac_for_mf-slope=1.4.npz'

bg = BayonetGeometryEstimator(5.4, 1.2)
model = HeatSimplifiedModel(configuration_file=CONFIGURATIONS_DIRECTORY + MODEL_CONFIGURATION, num_steps=30)
STATIC_HEAT = 0.002  # < -- because it should go in W/cm
LIQUID_SATURATION_TEMPERATURES = 1.85
INTERCONNECTION = 66.

# %% ==============> Experimental setup for no heat coming from adjacent cell <===============

# GRID_SEARCH_OPTIONS = {
#     'coarse': {
#         'WETTED_LENGTHS': np.array([102., 96., 81., 66., 51., 45., 30., 15.]),
#         'NUM_BHX_NODES': [36, 33, 28, 23, 18, 15, 10, 5]
#     },
#     'fine-01': {
#         'WETTED_LENGTHS': np.array([14.701, 9.901, 5.101, 0.31]),
#         'NUM_BHX_NODES': [4, 3, 2, 1]
#     },
#     'fine-12': {
#         'WETTED_LENGTHS': np.array([15., 15.31, 20.11, 24.91, 29.71, 30.]),
#         'NUM_BHX_NODES': [5, 6, 7, 8, 9, 10]
#     },
#
# }
#
# GRID_SEARCH_VERSION = 'fine'
# # GRID_SEARCH_VERSION = 'coarse'


experiments = {
    '650W-single': {
        'power_supplied': np.array([4.1138, 4.1286, 0., 4.1516, 4.2547, 4.1511, 0., 4.2978], dtype=np.float32),
        'WETTED_LENGTHS': np.array([15., 15.31, 20.11, 24.91, 29.71, 30.]),
        'NUM_BHX_NODES': [5, 6, 7, 8, 9, 10],
        'INITIAL_TEMPERATURES': np.ones(36, dtype=np.float32) * 1.92,
        'HEAT_COMING_FROM_ADJACENT_CELL': False
    },
    '650W-double': {
        'power_supplied': np.array([4.1138, 4.1286, 0., 4.1516, 4.2547, 4.1511, 0., 4.2978, 25.], dtype=np.float32),
        'WETTED_LENGTHS': np.array([15., 15.31, 20.11, 24.91, 29.71, 30.]),
        'NUM_BHX_NODES': [5, 6, 7, 8, 9, 10],
        'INITIAL_TEMPERATURES': np.ones(36, dtype=np.float32) * 1.89,
        'HEAT_COMING_FROM_ADJACENT_CELL': True
    },
    '700W-single': {
        'power_supplied': np.array([4.7029, 4.7202, 0., 4.7378, 4.8566, 4.7571, 0., 4.9086], dtype=np.float32),
        'WETTED_LENGTHS': np.array([15., 15.31, 20.11, 24.91, 29.71, 30.]),
        'NUM_BHX_NODES': [5, 6, 7, 8, 9, 10],
        'INITIAL_TEMPERATURES': np.ones(36, dtype=np.float32) * 1.92,
        'HEAT_COMING_FROM_ADJACENT_CELL': False
    },
    '700W-double': {
        'power_supplied': np.array([4.7029, 4.7202, 0., 4.7378, 4.8566, 4.7571, 0., 4.9086, 28.7], dtype=np.float32),
        'WETTED_LENGTHS': np.array([15., 15.31, 20.11, 24.91, 29.71, 30.]),
        'NUM_BHX_NODES': [5, 6, 7, 8, 9, 10],
        'INITIAL_TEMPERATURES': np.ones(36, dtype=np.float32) * 1.89,
        'HEAT_COMING_FROM_ADJACENT_CELL': True
    },

    '0W-single': {
        'power_supplied': np.zeros(8, dtype=np.float32),
        'WETTED_LENGTHS': np.array([14.701, 9.901, 5.101, 0.31]),
        'NUM_BHX_NODES': [4, 3, 2, 1],
        'INITIAL_TEMPERATURES': np.ones(36, dtype=np.float32) * 1.90,
        'HEAT_COMING_FROM_ADJACENT_CELL': False
    },
}

# %%

EXPERIMENTS_TO_RUN = ['650W-single', '650W-double', '700W-single', '700W-double', '0W-single']
#EXPERIMENTS_TO_RUN = ['650W-single']


# %%  RUN SIMULATION FOR ALL REQUIRED VARIANTS:

for experiment_tag in EXPERIMENTS_TO_RUN:
    print(f'Experiment {experiment_tag}')
    WETTED_LENGTHS = experiments[experiment_tag]['WETTED_LENGTHS']
    NUM_BHX_NODES = experiments[experiment_tag]['NUM_BHX_NODES']
    INITIAL_TEMPERATURES = experiments[experiment_tag]['INITIAL_TEMPERATURES']
    POWER_SUPPLIED = experiments[experiment_tag]['power_supplied']
    HEAT_COMING_FROM_ADJACENT_CELL = experiments[experiment_tag]['HEAT_COMING_FROM_ADJACENT_CELL']

    g_to_use = create_standard_cell_with_dynamic_BHX_virtual_pot(temperatures=INITIAL_TEMPERATURES,
                                                                 num_nodes_BHX=36,  # <==== check this
                                                                 power_supplied=POWER_SUPPLIED,
                                                                 static_heat=STATIC_HEAT,
                                                                 liquid_flow_direction=1,
                                                                 num_liquid_nodes_per_cell=1,
                                                                 heat_coming_from_adjacent_cell=HEAT_COMING_FROM_ADJACENT_CELL,
                                                                 interconnection_cross_section=INTERCONNECTION)

    mass_flow = mass_flow_given_total_heat(g_to_use, incremental_percentage_vaporization_power=0.1)
    temperatures_mask = np.asarray(g_to_use.node_sets['cells']['has_sensor']).nonzero()[0].tolist()

    simulations_results = simulate_wetted_lengths_variation(configuration='standard_cell',
                                                            model=model,
                                                            graph=g_to_use,
                                                            mass_flow=mass_flow,
                                                            wetted_lengths=WETTED_LENGTHS,
                                                            num_bhx=NUM_BHX_NODES,
                                                            bhx_geometry=bg,
                                                            feature_to_converge='cells__T',
                                                            time_track_step=1200.,
                                                            steady_state_duration=1800.,
                                                            mass_flow_from_right=True,
                                                            max_duration=60000,
                                                            interconnection_cross_section=INTERCONNECTION)

    #  ==================================> Saving to disk <===============================
    path_to_directory = f'results/heat/simulations/TimberData-Feb22'
    experiment_name = f'Standard_Cell_Feb_{experiment_tag}='

    # if not os.path.exists(f'{path_to_directory}/{experiment_name}'):
    #     os.makedirs(f'{path_to_directory}/{experiment_name}')

    ensure_dir_exists(f'{path_to_directory}/{experiment_name}')

    for i in range(len(WETTED_LENGTHS)):
        save_numpy_dict_to_file(simulations_results[i]._asdict(),
                                f'{path_to_directory}/{experiment_name}/Results_[wl={WETTED_LENGTHS[i]}].npz')

    # ==================================> Plot results <===============================
    title = f'Temperatures evolution for different wetted lengths, February experiment {experiment_tag}'

    plot_model_wetted_lengths_simulation(model_simulations=simulations_results,
                                         temperatures_mask=temperatures_mask,
                                         wetted_lengths=WETTED_LENGTHS,
                                         title=title,
                                         plot_average_temp=True,
                                         save=True,
                                         file_path_and_name_for_saving=f'{path_to_directory}/{experiment_name}/Temperatures_evolution_{experiment_name}.png')

