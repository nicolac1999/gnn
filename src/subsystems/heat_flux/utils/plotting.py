import PIL
import io
import numpy as np
import tensorflow as tf
import pygraphviz as pgv
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression
from tensorflow_gnn import GraphTensor
from typing import NamedTuple, Union

from lib.diag_common.matplotlib_helpers import colormap_to_css
# from subsystems.heat_flux.utils.simulation import simulate_time_period
from subsystems.heat_flux.utils.data_helpers import load_row_from_dataset, extract_col_from_var


def plot_temperatures_time_series(time_temperature: list,
                                  title='Temperature evolution'):
    '''
    This function is used for plotting magnets' temperature and the relative time stamp
    on the x-axis
    :param time_temperature: list of time stamps and temperature coming from the model output
    :param title: Optional parameter for setting the title of the plot
    :return:
    '''

    colors = plt.cm.tab10

    time, temperature = time_temperature
    num_magnets = temperature.shape[-1]
    time = time.numpy()
    temperature = temperature.numpy()

    plt.figure(figsize=(10, 10))
    for i in range(num_magnets):
        plt.plot(time, temperature[:, i], marker='x', label=f'magnet {i + 1}', color=colors(i), linestyle='--')

    plt.xlabel('Time stamp [ s ]')
    plt.ylabel('Temperature [ K ]')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


def plot_prediction_and_true_values(initial_state, true_tensor, prediction_tensor, title='Temperatures evolution'):
    colors = plt.cm.tab10

    initial_state = initial_state.numpy()
    true_array = true_tensor.numpy()
    true_array = np.vstack((initial_state, true_array))
    prediction_array = prediction_tensor.numpy()
    prediction_array = np.vstack((initial_state, prediction_array))

    # num_magnets = true_tensor.shape[1]
    # columns = [f'magnet {i}' for i in range(1, num_magnets + 1)]
    # df_true = pd.DataFrame(true_tensor)
    # df_prediction = pd.DataFrame(prediction_tensor)
    plt.figure(figsize=(10, 10))
    for i in range(true_array.shape[1]):
        plt.plot(true_array[:, i], marker='x', label=f'magnet {i + 1} true', color=colors(i), linestyle='--')
    for i in range(prediction_array.shape[1]):
        plt.plot(prediction_array[:, i], marker='o', label=f'magnet {i + 1} prediction', color=colors(i))
    # sns.set_style("darkgrid")
    # sns.lineplot(df_true, markers=True)
    # sns.lineplot(df_prediction, linestyle='--', markers=True)
    plt.xlabel('Processing step')
    plt.ylabel('Temperature [ K ]')
    plt.title(title)
    plt.legend(labels=['magnet_0_true', 'magnet_1_true', 'magnet_2_true', 'magnet_4_true',
                       'magnet_0_prediction', 'magnet_1_prediction', 'magnet_2_prediction', 'magnet_4_prediction'],
               loc="upper right")

    plt.show()


def plot_preview(duration,
                 variables: list[str],
                 model_result_index: int,
                 starting_time=None,
                 time_step=None,
                 ds_filepath=None,
                 frames=None,
                 model=None,
                 graph=None,
                 title=None,
                 x_range=None,
                 y_range=None,
                 use_init_graph_state=False):
    """
    :param y_range:
    :param title:
    :param x_range:
    :param graph:
    :param frames:
    :param model: GNN model
    :param ds_filepath: Excel file
    :param starting_time: starting time point in seconds
    :param duration: duration of the period of interest in seconds
    :param time_step: "delta t" of the model
    :return: plot pf the "ground truth" and the model predictions
    """
    colors = plt.cm.tab20
    plt.figure(figsize=(10, 10))

    TIME_COL_NAME = 'time'

    if model and not graph:
        raise ValueError('You passed the model without the graph, please include it, otherwise remove also the model')

    if use_init_graph_state == False and ds_filepath == None:
        raise ValueError(
            'You don\'t want to initialize the graph bit you didn\'t pass the dataset path, please include it, otherwise set use_init_graph_state==True')

    if not title and model:
        title = f'Temperature comparison ground truth and model prediction, for time_step={time_step}'
    if not title and not model:
        title = f'Temperature evolution for time_step={time_step}'
    mask = [0, 1, 4, 7, 10, 11, 14, 17]

    if frames:
        counter_frames = 0
        for var in variables:
            for frame in frames:
                for i in range(len(frame[var][0])):
                    plt.plot(frame[TIME_COL_NAME].squeeze(), frame[var][:, i], 'o', mfc='none',
                             label=(f'FR {var} [{i}]' if counter_frames == 0 else '_'), color=colors(i), alpha=0.3)
                counter_frames += 1

    if ds_filepath:
        ground_truth_dataframe = load_row_from_dataset(ds_filepath, starting_time, duration)
        ground_truth_dataframe.drop_duplicates(keep=False, inplace=True)

        time_ground_truth = ground_truth_dataframe[TIME_COL_NAME].values

        # ground_truth_array = ground_truth_dataframe.to_numpy()
        # time_ground_truth = ground_truth_array[:, 0]
        # temperature_ground_truth = ground_truth_array[:, 1:5]
        # num_magnets = temperature_ground_truth.shape[-1]

        col_names_for_variables, all_used_col_names = extract_col_from_var(df=ground_truth_dataframe,
                                                                           variables=variables)
        # col_names_for_variables = {'T': ['T[00]', 'T[01]', 'T[04]', 'T[07]', 'T[10]', 'T[11]', 'T[14]', 'T[17]']}
        mask = [0, 1, 4, 7, 10, 11, 14, 17]
        sub_dfs_variables = {}

        for var_name, col_names in col_names_for_variables.items():
            sub_dfs_variables[var_name] = ground_truth_dataframe[col_names].to_numpy()

        for var_name, sub_df_variable in sub_dfs_variables.items():

            num_variables = sub_df_variable.shape[-1]

            for i in range(num_variables):
                if i in mask:
                    plt.plot(time_ground_truth, sub_df_variable[:, i], 'x',
                             label=f'DS {col_names_for_variables[var_name][i]}', color=colors(i), alpha=0.3)

    if model and graph:

        if not use_init_graph_state:
            starting_temperatures = sub_dfs_variables['T'][0, :]
            cell_features = graph.node_sets['cells'].get_features_dict()
            cell_features['T'] = tf.cast(starting_temperatures, dtype=tf.float32)
            graph = graph.replace_features(node_sets={"cells": cell_features})

        predictions_dict = simulate_time_period(model=model, graph=graph, starting_time=starting_time,
                                                duration=duration, time_step=time_step)

        time, temperature = predictions_dict['time'], predictions_dict['T']
        num_magnets = temperature.shape[-1]
        time = time.numpy()
        temperature = temperature.numpy()

        for i in range(num_magnets):
            if i in mask:
                plt.plot(time, temperature[:, i], label=f'MD res[][{i}]', color=colors(i), alpha=1)

    plt.xlabel('Time stamp [ s ]')
    plt.ylabel('Temperature [ K ]')
    plt.title(title)
    if x_range is not None:
        plt.xlim(x_range[0], x_range[1])
    if y_range is not None:
        plt.ylim(y_range[0], y_range[1])
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()
    plt.grid()
    plt.show()


def plot_concatenate_predictions(list_predictions: list[dict],
                                 title=None,
                                 x_range=None,
                                 y_range=None):
    colors = plt.cm.tab20
    mask = [0, 1, 4, 7, 10, 11, 14, 17]

    time_array = np.array([]).reshape([0, 1])
    temperature_array = np.array([]).reshape((0, 20))
    static_heat_array = np.array([]).reshape([0, 1])
    dynamic_heat_array = np.array([]).reshape([0, 1])
    vaporization_heat_array = np.array([]).reshape([0, 1])

    for prediction in list_predictions:
        time, temperature = prediction['time'], prediction['T']
        static_heat, dynamic_heat, vaporization_heat = prediction['total_static_heat'], prediction[
            'total_dynamic_heat'], prediction['total_vaporization_heat']
        time = time.numpy()
        temperature = temperature.numpy()
        static_heat = static_heat.numpy()
        dynamic_heat = dynamic_heat.numpy()
        vaporization_heat = vaporization_heat.numpy()
        time_array = np.vstack((time_array, time))
        temperature_array = np.vstack((temperature_array, temperature))
        static_heat_array = np.vstack((static_heat_array, static_heat))
        dynamic_heat_array = np.vstack((dynamic_heat_array, dynamic_heat))
        vaporization_heat_array = np.vstack((vaporization_heat_array, vaporization_heat))

    fig, axs = plt.subplots(2, 1)
    num_magnets = temperature_array.shape[-1]
    for i in range(num_magnets):
        if i in mask:
            axs[0].plot(time_array, temperature_array[:, i], label=f'MD res[][{i}]', color=colors(i), alpha=1)

    axs[0].set_xlabel('Time stamp [ s ]')
    axs[0].set_ylabel('Temperature [ K ]')
    if not title:
        title = f'Temperature evolution, mean ΔT = {list_predictions[0]["mean_delta_T"] * 1000} mK'  # da cambiare
    axs[0].set_title(title)
    if x_range is not None:
        axs[0].set_xlim(x_range[0], x_range[1])
    if y_range is not None:
        axs[0].set_ylim(y_range[0], y_range[1])
    axs[0].legend(bbox_to_anchor=(1, 1), loc='upper left')
    # axs[0].tight_layout()
    axs[0].grid()

    axs[1].plot(time_array, static_heat_array, label='total static heat load')
    axs[1].plot(time_array, dynamic_heat_array, label='total dynamic heat load')
    axs[1].plot(time_array, vaporization_heat_array, label='total vaporization power')
    axs[1].set_xlabel('Time stamp [ s ]')
    axs[1].set_ylabel('Power [ W ]')
    axs[1].set_title('Comparison between powers inside the cell')
    axs[1].legend()
    axs[1].grid()

    fig.tight_layout()
    plt.show()


def plot_graph_tensor(graph: GraphTensor, layout='dot', **kwargs):
    agraph = from_gt_to_agraph(graph, **kwargs)
    img_png = agraph.draw(format='png', prog=layout)

    MATPLOTLIB_DPI = 100

    img = PIL.Image.open(io.BytesIO(img_png))
    # print(img.size)
    # new_size = 4000, 2000
    # img = img.resize(new_size)
    # .figure(figsize=())

    underlying_pixels = img.size
    required_inches = (np.asarray(
        underlying_pixels) / MATPLOTLIB_DPI) + 1  # <-- "+1" inch as a buffer for titles and/or other 'decorations' in Matplotlib Figure.

    plt.figure(figsize=tuple(required_inches))
    plt.imshow(img)

    req_title = kwargs.get('title', None)
    if req_title is not None:
        plt.title(req_title, pad=20.)

    plt.axis('off')

    plt.show()


def from_gt_to_agraph(graph, custom_y_position_lut=None,
                      custom_node_attribute_show=None,
                      custom_edge_attribute_show=None,
                      pin_positions=False,
                      **kwargs):
    EDGE_LEN_NORM_CONST = 5.0
    EDGE_LEN_DEFAULT = 1.5
    MIN_TEMPERATURE_RANGE = 1.80
    MAX_TEMPERATURE_RANGE = 2.0
    MIN_LEN_EDGE = 0.5

    pgvGraph = pgv.AGraph(strict=False, directed=True,
                          size='100,50')  # <- limiting size to 100x50 inch, which leads to 10000x5000 px images -- still very large, but hopefully feasible.

    color_map = plt.cm.bwr

    shapes_lut = ['circle', 'box', 'diamond', 'polygon', 'triangle', 'doublecircle', 'trapezium', 'parallelogram',
                  'house',
                  'star', 'Msquare']
    # shapes_iter = iter(shapes_lut)

    colors_lut = ['red', 'black', 'blue', 'orchid', 'lime', 'orangered', 'deepskyblue', 'sienna1', 'magenta',
                  'lawngreen',
                  'forestgreen']
    # colors_iter = iter(colors_lut)

    y_position_lut = {
        'cells': 2.0,
        'liquid': 3.0,
        'heater': 1.0
    }

    if custom_y_position_lut is not None:
        y_position_lut |= custom_y_position_lut

    node_attribute_show = {
        'cells': 'T',
        'heater': 'power',
        'liquid': 'evapor_mass_flow'
    }

    if custom_node_attribute_show is not None:
        node_attribute_show |= custom_node_attribute_show

    edge_attribute_show = {
        'conduction': 'L'
    }

    edge_length_feature = {
        'conduction': 'L'
    }

    if custom_edge_attribute_show is not None:
        edge_attribute_show |= custom_edge_attribute_show

    node_sets = graph.node_sets
    edge_sets = graph.edge_sets
    offset = {}

    shapes_dict = {}
    colors_dict = {}

    current_offset = 0
    node_set_counter = 0

    for node_set_name in node_sets.keys():
        num_nodes = graph.node_sets[node_set_name].sizes.numpy()

        offset[node_set_name] = int(current_offset)

        current_offset = current_offset + num_nodes

        # shapes_dict[node_set_name] = next(shapes_iter)
        # colors_dict[node_set_name] = next(colors_iter)

        shapes_dict[node_set_name] = shapes_lut[node_set_counter % len(shapes_lut)]
        colors_dict[node_set_name] = colors_lut[node_set_counter % len(colors_lut)]

        node_set_counter = node_set_counter + 1

    max_observed_pos_y = 0

    for node_set_name in node_sets.keys():
        pos_x = 0.
        pos_y = y_position_lut.get(node_set_name, max_observed_pos_y + 1)
        max_observed_pos_y = max(max_observed_pos_y, pos_y)

        num_nodes = int(graph.node_sets[node_set_name].sizes.numpy())

        for i in range(num_nodes):
            # node_attribute = node_attribute_show.get(node_set_name, '')

            if node_attribute_show.get(node_set_name) is None:
                node_attribute_str = ''
            else:
                node_attribute_value_float = node_sets[node_set_name][node_attribute_show.get(node_set_name)][i].numpy()
                node_attribute_str = f'{node_attribute_value_float:.2f}'

            if node_attribute_show.get(node_set_name) == 'T':

                normalized_T = ((node_sets[node_set_name][node_attribute_show.get(node_set_name)][
                                     i] - MIN_TEMPERATURE_RANGE) / (
                                            MAX_TEMPERATURE_RANGE - MIN_TEMPERATURE_RANGE)).numpy()
                node_attribute_color = colormap_to_css(color_map(normalized_T))
                print(normalized_T)
            else:
                node_attribute_color = 'white'

            pgvGraph.add_node(i + offset[node_set_name],
                              shape=shapes_dict[node_set_name],
                              color=colors_dict[node_set_name],
                              label=f'{node_set_name[0].upper()}{i}\n{node_attribute_str}',
                              pos=f'{pos_x},{pos_y}',
                              pin=pin_positions,
                              xlp=f'10.',
                              height=0.2,
                              width=0.3,
                              fixed_size=True,
                              style='filled',
                              fillcolor=node_attribute_color)
            pos_x = pos_x + 1.

    for edge_set_name in edge_sets:

        source_name = graph.edge_sets[edge_set_name].adjacency.source_name
        target_name = graph.edge_sets[edge_set_name].adjacency.target_name

        source_idxs = graph.edge_sets[edge_set_name].adjacency.source.numpy()
        target_idxs = graph.edge_sets[edge_set_name].adjacency.target.numpy()

        source_idxs = source_idxs + offset[source_name]
        target_idxs = target_idxs + offset[target_name]

        edge_attribute_show_name = edge_attribute_show.get(edge_set_name)
        edge_attribute_length_name = edge_length_feature.get(edge_set_name)

        edge_counter = 0
        for edge in zip(source_idxs, target_idxs):

            if edge_attribute_show_name is not None:
                edge_attribute_str = np.round(edge_sets[edge_set_name][edge_attribute_show_name][edge_counter].numpy(),
                                              2)
            else:
                edge_attribute_str = ""

            if edge_attribute_length_name is not None:
                edge_length_raw_value = edge_sets[edge_set_name][edge_attribute_length_name][edge_counter].numpy()
                edge_length = (edge_length_raw_value / EDGE_LEN_NORM_CONST) + MIN_LEN_EDGE
            else:
                edge_length = MIN_LEN_EDGE

            # label = edge_attribute if edge_attribute == "" else edge_attribute:.2f
            pgvGraph.add_edge(edge[0], edge[1],
                              arrowType='normal',
                              label=edge_attribute_str,
                              fontsize=10.0,
                              len=edge_length
                              )

            edge_counter += 1

    return pgvGraph


def plot_model_wetted_lengths_simulation(model_simulations: Union[list[NamedTuple], NamedTuple],
                                         temperatures_mask: list,
                                         wetted_lengths: Union[list, np.ndarray],
                                         title: str,
                                         n_columns=2,
                                         plot_average_temp=False,
                                         fig_size: tuple = (20, 20),
                                         save=True,
                                         file_path_and_name_for_saving=None):
    colors = plt.cm.tab20

    if type(model_simulations) == NamedTuple:
        model_simulations = list(model_simulations)

    num_simulations = len(model_simulations)

    n_rows = int(np.ceil(num_simulations / n_columns))

    if n_rows == 1:
        n_columns = 1

    fig, axs = plt.subplots(n_rows, n_columns, figsize=fig_size)  # <-- see Roman code for matplotlib figsize
    # <-- try to generalize in a way to support
    #      more simulation cases

    num_magnets = model_simulations[0].cells__T.shape[-1]

    current_row = 0
    current_column = 0

    if n_columns >= 2 and n_rows >= 2:

        for i in range(num_simulations):
            current_simulation = model_simulations[i]
            current_wl = wetted_lengths[i]
            num_lines_plot = 0

            for j in range(num_magnets):
                if j in temperatures_mask:
                    axs[current_row][current_column].plot(current_simulation.context__time,
                                                          current_simulation.cells__T[:, j],
                                                          label=f'magnet [{j}] T',
                                                          color=colors(num_lines_plot),
                                                          alpha=1)
                    num_lines_plot = num_lines_plot + 1

            if plot_average_temp:
                axs[current_row][current_column].plot(current_simulation.context__time,
                                                      current_simulation.cells__T[:, temperatures_mask].mean(axis=1),
                                                      label=f'avg magnets T',
                                                      color=colors(num_lines_plot))

            axs[current_row][current_column].set_xlabel('Time stamp [ s ]')
            axs[current_row][current_column].set_ylabel('Temperature [ K ]')
            axs[current_row][current_column].set_title(
                f'Temperature evolution for a wetted length of {current_wl:.1f}'
                f' meters ({round(current_wl / 102. * 100, 0)} %)')  # <--- hard coded numbers

            axs[current_row][current_column].legend(bbox_to_anchor=(1, 1), loc='upper left')
            axs[current_row][current_column].grid()

            current_column = current_column + 1
            if current_column == n_columns:
                current_column = 0
                current_row = current_row + 1

        for i in range(n_rows):
            for j in range(n_columns):
                if not axs[i, j].lines:
                    axs[i, j].set_visible(False)
    elif n_rows > 1:

        for i in range(num_simulations):
            current_simulation = model_simulations[i]
            current_wl = wetted_lengths[i]
            num_lines_plot = 0

            for j in range(num_magnets):
                if j in temperatures_mask:
                    axs[current_row].plot(current_simulation.context__time,
                                          current_simulation.cells__T[:, j],
                                          label=f'magnet [{j}] T',
                                          color=colors(num_lines_plot),
                                          alpha=1)
                    num_lines_plot = num_lines_plot + 1

            if plot_average_temp:
                axs[current_row].plot(current_simulation.context__time,
                                      current_simulation.cells__T[:, temperatures_mask].mean(axis=1),
                                      label=f'avg magnets T',
                                      color=colors(num_lines_plot))

            axs[current_row].set_xlabel('Time stamp [ s ]')
            axs[current_row].set_ylabel('Temperature [ K ]')
            axs[current_row].set_title(
                f'Temperature evolution for a wetted length of {current_wl:.1f}'
                f' meters ({round(current_wl / 102. * 100, 0)} %)')  # <--- hard coded numbers

            axs[current_row].legend(bbox_to_anchor=(1, 1), loc='upper left')
            axs[current_row].grid()

            current_row = current_row + 1

    else:
        for i in range(num_simulations):
            current_simulation = model_simulations[i]
            current_wl = wetted_lengths[i]
            num_lines_plot = 0

            for j in range(num_magnets):
                if j in temperatures_mask:
                    axs.plot(current_simulation.context__time,
                             current_simulation.cells__T[:, j],
                             label=f'magnet [{j}] T',
                             color=colors(num_lines_plot),
                             alpha=1)
                    num_lines_plot = num_lines_plot + 1

            if plot_average_temp:
                axs.plot(current_simulation.context__time,
                         current_simulation.cells__T[:, temperatures_mask].mean(axis=1),
                         label=f'avg magnets T',
                         color=colors(num_lines_plot))

            axs.set_xlabel('Time stamp [ s ]')
            axs.set_ylabel('Temperature [ K ]')
            axs.set_title(
                f'Temperature evolution for a wetted length of {current_wl:.1f}'
                f' meters ({round(current_wl / 102. * 100, 0)} %)')  # <--- hard coded numbers

            axs.legend(bbox_to_anchor=(1, 1), loc='upper left')
            axs.grid()

    if title:
        plt.suptitle(title)
    else:
        plt.suptitle('Temperatures evolution for different wetted length')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save:
        plt.savefig(file_path_and_name_for_saving)
    plt.show()



def plot_liquid_BHX_behavior(simulation: NamedTuple,
                             temperatures_mask: list,
                             title_temperatures_subplt: str):
    """
    Function for plotting temperatures evolution and liquid behavior:
     ** evaporation mass flow per meter
     ** absolute evaporation mass flow
     ** incoming mass flow
     ** fraction wetted perimeter
    """
    colors = plt.cm.tab20

    fig = plt.figure(figsize=(20, 20))

    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((3, 2), (1, 0), colspan=1)
    ax3 = plt.subplot2grid((3, 2), (1, 1), colspan=1)
    ax4 = plt.subplot2grid((3, 2), (2, 0), colspan=1)
    ax5 = plt.subplot2grid((3, 2), (2, 1), colspan=1)

    time = simulation.context__time
    num_liquid_node = simulation.liquid__incoming_mass_flow.shape[1]
    num_magnets = simulation.cells__T.shape[-1]

    num_lines_plot = 0

    for j in range(num_magnets):
        if j in temperatures_mask:
            ax1.plot(time, simulation.cells__T[:, j], label=f'magnet [{j}] T', color=colors(num_lines_plot), alpha=1)
            num_lines_plot = num_lines_plot + 1
    ax1.set_xlabel('Time stamp [ s ]')
    ax1.set_ylabel('Temperature [ K ]')
    ax1.set_title(title_temperatures_subplt)
    ax1.legend(bbox_to_anchor=(1, 1), loc='upper left')
    ax1.grid()

    for i in range(0, num_liquid_node):
        label = f'Liquid [{i}]'

        ax2.plot(time, np.round(simulation.liquid__evapor_mass_flow_per_m[:, i], 4), label=label)
        ax3.plot(time, np.round(simulation.liquid__evapor_mass_flow[:, i], 4), label=label)
        ax4.plot(time, np.round(simulation.liquid__incoming_mass_flow[:, i], 4), label=label)
        ax5.plot(time, np.round(simulation.liquid__avg_f[:, i], 4), label=label)

    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Evapor mass flow per meter [g/(s * m)]')
    ax2.legend(bbox_to_anchor=(1, 1), loc='upper left')
    ax2.set_title('Evaporation mass flow per meter')
    ax2.grid()

    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Evapor mass flow [g/s]')
    ax3.legend(bbox_to_anchor=(1, 1), loc='upper left')
    ax3.set_title('Evaporation mass flow')
    ax3.grid()

    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Incoming mass flow [g/s]')
    ax4.legend(bbox_to_anchor=(1, 1), loc='upper left')
    ax4.set_title('Incoming mass flow')
    ax4.grid()

    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('Fraction wetted perimeter')
    ax5.legend(bbox_to_anchor=(1, 1), loc='upper left')
    ax5.set_title('Fraction wetted perimeter')
    ax5.grid()

    plt.suptitle('Simulation insights')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_timber_T_vs_simulated_T(x, y_timber, y_sim, title='', error_bar=None):
    colors = plt.cm.tab20

    legend_elements = [Line2D([0], [0], color='black', marker='x', label='Timber data', linestyle='None'),
                       Line2D([0], [0], color='black', marker='s', label='Simulated', linestyle='None')]

    plt.figure(figsize=(10, 6))

    for i in range(len(y_timber)):
        for j in range(len(y_timber[i])):
            plt.plot(x[i], y_timber[i][j], 'x', color=colors(j))
            plt.plot(x[i], y_sim[i][j], 's', color=colors(j), fillstyle='none')

            if error_bar:
                plt.errorbar(x[i], y_timber[i][j], yerr=error_bar, capsize=2, ecolor=colors(j))
        # plt.plot(ALL_HEATERS_POWER[i], ALL_MAGNETS_AVG[i], 'o', label='avg magnet temp')

    plt.ylabel('Magnet temperature [ K ]')
    plt.xlabel('Heater power [ W ]')
    plt.title(title)
    plt.legend(handles=legend_elements, loc='lower left')
    plt.grid()
    plt.show()


def plot_string2(simulations: list[NamedTuple], static_heat: float, temperatures_mask, liquid_sat_temp=1.85):
    static_heat_range = np.arange(static_heat, static_heat + 1.1, 0.2)
    stable_Ts_simulations = [sim['cells__T'][-1] for sim in simulations]
    delta_T_with_sat_T = [stable_Ts_sim - liquid_sat_temp for stable_Ts_sim in stable_Ts_simulations]
    mean_delta_Ts = np.mean(np.array(delta_T_with_sat_T)[:, temperatures_mask], axis=1)

    reg_string2_sim = LinearRegression().fit((static_heat_range - static_heat).reshape(-1, 1),
                                             mean_delta_Ts.reshape(-1, 1), )
    thermal_conductivity_string2_sim = 1 / reg_string2_sim.coef_

    # x_new = np.arange(0., 1.01, 0.01)
    # y_new = reg_string2_sim.predict(x_new.reshape(-1, 1))

    fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 6))
    ax.plot(static_heat_range, [mean_dT * 1000 for mean_dT in mean_delta_Ts], label='DT average')

    num_T_simulated = len(stable_Ts_simulations[0])

    for i in range(num_T_simulated):
        if i in temperatures_mask:
            ax.plot(static_heat_range, np.asarray(delta_T_with_sat_T)[:, i] * 1000, 'o', label=f'DT magnet[{i}]')
    # ax.plot(x_new + static_heat, y_new * 1000, label='Linear reg')
    ax.set_xticks(static_heat_range)
    x_labels = ["{:.2f}".format(x) for x in static_heat_range - static_heat]
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('Applied power [W/m]')
    ax.set_ylabel('DeltaT (Tmagnets - Tsat) [mK]')
    ax.set_ylim(0, 25)

    ax.legend(bbox_to_anchor=(1, 1), loc='upper left')

    secax = ax.secondary_xaxis('top')
    secax.set_xlabel('Total heat load (static + dynamic) [W/m]')
    x_labels_up = ["{:.2f}".format(x) for x in static_heat_range]
    secax.set_xticks(static_heat_range)
    secax.set_xticklabels(x_labels_up)
    plt.title(
        f'STRING2 experiment, thermal conductivity : '
        f'{float(np.round(np.array(thermal_conductivity_string2_sim).flatten(), 2))} W/Km')
    plt.grid()

    plt.show()


def plot_half_cell(simulations: list[NamedTuple], temperatures_mask, experimental_points=None):

    dynamic_heat_per_m_steps = np.concatenate((np.arange(0., 2.1, 0.2), np.array([1.3, 0.9, 0.5])))
    # The temperatures are "normalized" on the "lowest" temperature, not on the T saturation like in the String2 experiment

    stable_Ts_simulations = [sim['cells__T'][-1] for sim in simulations]
    lowest_temperature = stable_Ts_simulations[0][-1]
    # delta_T_with_sat_T = [stable_Ts_sim - LIQUID_SATURATION_TEMPERATURES for stable_Ts_sim in stable_Ts_simulations]
    delta_T_with_low_T = [stable_Ts_sim - lowest_temperature for stable_Ts_sim in stable_Ts_simulations]
    mean_delta_Ts = np.mean(np.array(delta_T_with_low_T)[:, temperatures_mask], axis=1)
    higher_temps = np.array(delta_T_with_low_T)[:, 1]
    lower_temps = np.array(delta_T_with_low_T)[:, 14]

    x_new = np.arange(0., 2.01, 0.01)

    # Linear Regression fit for simulation points ===> thermal conductivity of "our" half cell model
    reg_sim = LinearRegression().fit(dynamic_heat_per_m_steps.reshape(-1, 1), mean_delta_Ts.reshape(-1, 1), )
    thermal_conductivity_sim = 1 / reg_sim.coef_
    plt.figure(figsize=(15, 6))
    plt.plot(dynamic_heat_per_m_steps, np.array([mean_dT * 1000 for mean_dT in mean_delta_Ts]), '^',
             label='avg T increase', markersize=8) # <==== add '-^' if you want the line

    # reg_sim_high = LinearRegression().fit(dynamic_heat_per_m_steps.reshape(-1, 1), higher_temps.reshape(-1, 1), )
    # thermal_conductivity_sim_high = 1 / reg_sim_high.coef_
    #
    # reg_sim_low = LinearRegression().fit(dynamic_heat_per_m_steps.reshape(-1, 1), lower_temps.reshape(-1, 1), )
    # thermal_conductivity_sim_low = 1 / reg_sim_low.coef_

    for i in range(len(stable_Ts_simulations[0])):
        if i in temperatures_mask:
            plt.plot(dynamic_heat_per_m_steps, np.asarray(delta_T_with_low_T)[:, i] * 1000, 'x',
                     label=f'T increase [{i}]', markersize=8)

    if experimental_points:
        x_experimental = [points[0] for points in experimental_points]
        y_experimental = [points[1] for points in experimental_points]

        # Linear Regression fit for experimental points ==> thermal conductivity of real half cell

        reg = LinearRegression().fit(np.array(x_experimental).reshape(-1, 1), np.array(y_experimental).reshape(-1, 1))
        reg_mK = LinearRegression().fit(np.array(x_experimental).reshape(-1, 1),
                                        (np.array(y_experimental) / 1000).reshape(-1, 1))
        thermal_conductivity = 1 / reg_mK.coef_
        y_new_reg = reg.predict(x_new.reshape(-1, 1))

        plt.plot(x_experimental, y_experimental, 's', label='experimental values', fillstyle='none', color='black',
                markersize=10)

        plt.plot(x_new, y_new_reg)

    x_labels = ["{:.2f}".format(x) for x in dynamic_heat_per_m_steps]
    plt.xticks(dynamic_heat_per_m_steps, labels=x_labels)
    plt.xlabel('Applied power [W/m]')

    plt.ylabel('T increase [mK]')
    plt.ylim(-5, 30)
    plt.title(f'HALF CELL experiment, thermal conductivity :'
              f' {float(np.round(np.array(thermal_conductivity_sim).flatten(), 2))} W/Km')
    plt.grid()
    plt.legend()
    plt.show()
