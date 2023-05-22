import re

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.axes import Axes
from matplotlib.ticker import AutoMinorLocator

from tensorflow_gnn import GraphTensor

from subsystems.fluid.graphs.pipe_1d_v1 import get_pipe_geometry_v1
from subsystems.fluid.models.model_1 import Result_HydroModel


def _resolve_plot_metadata(feature_spec: str, model_res: Result_HydroModel, graph: GraphTensor):

    is_v1 = 'boundaries' in graph.node_sets['wells'].features.keys()

    if '+' in feature_spec:
        feat_spec_chunks = feature_spec.split('+', maxsplit=1)
        primary_plot_data = _resolve_plot_metadata(feat_spec_chunks[0], model_res, graph)
        secondary_plot_data = _resolve_plot_metadata(feat_spec_chunks[1], model_res, graph)

        return {
            'primary': primary_plot_data['primary'],
            'secondary': secondary_plot_data['primary']
        }


    if '[' in feature_spec:
        # -- parse out dimension to plot
        chunks = re.split('[][]', feature_spec)
        feature_name = chunks[0]
        requested_dimension_str = chunks[1]
    else:
        feature_name = feature_spec
        requested_dimension_str = None

    model_res_dict = model_res._asdict()
    feature_values = np.asarray(model_res_dict[feature_name])

    # boundaries = np.asarray(graph.node_sets['wells']['boundaries'])
    # x_coords = np.concatenate((boundaries[:, 0, 0], np.reshape(boundaries[-1, 0, 1], (1,))))

    if is_v1:
        x_coords_extr_func = _get_xcoords_from_graph_v1
        values_extr_func = _pad_mirror_last_value
    else:
        x_coords_extr_func = _get_double_xcoords_from_graph_v2
        values_extr_func = _get_values_reshaped_for_double_xs_v2
        # x_coords_extr_func = _get_plain_xcoords_from_graph_v2
        # values_extr_func = lambda x: x


    x_coords = x_coords_extr_func(graph)

    resulting_lines = []

    if feature_values.ndim == 3:
        if requested_dimension_str is None:
            feature_values = np.linalg.norm(feature_values, axis=-1)
            line_def = {
                'xs': x_coords,
                'values': values_extr_func(feature_values),
                'label': f'norm( {feature_name} )'
            }
            resulting_lines.append(line_def)

        else:
            if requested_dimension_str.isnumeric():
                requested_dimensions = [int(requested_dimension_str)]
            else:
                # -- all dimensions as separate lines:
                requested_dimensions = range(feature_values.shape[2])

            for d in requested_dimensions:
                line_def = {
                    'xs': x_coords,
                    'values': values_extr_func(feature_values[..., d]),
                    'label': f'{feature_name}[{d}]'
                }
                resulting_lines.append(line_def)

    else:
        # -- so the feature is just "scalar":
        line_def = {
            'xs': x_coords,
            'values': values_extr_func(feature_values),
            'label': f'{feature_name}'
        }
        resulting_lines.append(line_def)


    result = {
        'primary': {
            'specifier': feature_spec,
            'lines': resulting_lines
        }
    }

    return result


def _get_xcoords_from_graph_v1(graph: GraphTensor):
    """
    Will extract LEFT positions of wells, and will add the RIGHT position of the last well.
    """
    if 'boundaries' in graph.node_sets['wells'].features.keys():
        boundaries = np.asarray(graph.node_sets['wells']['boundaries'])
        x_coords = np.concatenate((boundaries[:, 0, 0], np.reshape(boundaries[-1, 0, 1], (1,))))
    else:
        xz_bottom_points = graph.node_sets['wells']['xz_bottom_points']
        MoB_xs = np.sum(0.5 * xz_bottom_points[:, :, 0], axis=1)
        x_coords = MoB_xs

    return x_coords


def _get_plain_xcoords_from_graph_v2(graph: GraphTensor):
    """
    Will create x-coordinates of the middle of each well.
    """
    xz_bottom_points = graph.node_sets['wells']['xz_bottom_points']
    MoB_xs = np.sum(0.5 * xz_bottom_points[:, :, 0], axis=1)
    x_coords = MoB_xs

    return x_coords



def _get_double_xcoords_from_graph_v2(graph: GraphTensor):
    """
    Will extract LEFT and RIGHT positions of wells -- this will double the size of points to plot (R and L points of adjacent cells may overlay (of the pipe is continuous).
    """

    xz_bottom_points = np.asarray(graph.node_sets['wells']['xz_bottom_points'])
    xpoints = xz_bottom_points[:, :, 0]   # <-- x coordinates of nodes in form: [[L:, R:], [...], ...]
    x_coords = xpoints.flatten()

    # x_coords = np.concatenate((boundaries[:, 0, 0], np.reshape(boundaries[-1, 0, 1], (1,))))

    return x_coords


def _get_values_reshaped_for_double_xs_v2(ys: np.ndarray):

    if ys.ndim == 1:
        expanded_ys = np.broadcast_to(np.expand_dims(ys, 1), (ys.shape[0], 2))
        ys_flatten = expanded_ys.flatten()
    elif ys.ndim == 2:
        expanded_ys = np.broadcast_to(np.expand_dims(ys, -1), (ys.shape[0], ys.shape[1], 2))
        ys_flatten = np.reshape(expanded_ys, (ys.shape[0], -1))
        pass
    else:
        raise ValueError('Incorrect ndims of ys! ')
    return ys_flatten




def _pad_mirror_last_value(ys):
    """
    Will pad 1 value on the 'RIGHT' to the ys array, which is expected to be rank-2 or rank-3 matrix.
    :param ys:
    :return:
    """
    if ys.ndim == 1:
        res = np.pad(ys, [(0, 1)], mode='symmetric')
    elif ys.ndim == 2:
        res = np.pad(ys, [(0, 0), (0, 1)], mode='symmetric')
    elif ys.ndim == 3:
        res = np.pad(ys, [(0, 0), (0, 1), (0,0)], mode='symmetric')
    else:
        raise ValueError(f'Expected matrix of rank 1, 2 or 3. Received array of shape={ys.shape} | values: {ys}')

    return res

def _create_all_requested_lines(ax: Axes, req_lines_meta:list[dict], color_offset=0, is_secondary=False):

    if len(req_lines_meta) <= 0:
        return

    cm = plt.colormaps['tab10']
    is_secondary_char = '' if not is_secondary else '→'

    # -- resolve x-lims and y-lims:
    x_min, x_max, y_min, y_max = +np.Inf, -np.Inf, +np.Inf, -np.Inf
    for line_meta in req_lines_meta:
        xs = line_meta['xs']
        ys = line_meta['values']
        x_min = np.minimum(x_min, np.min(xs))
        x_max = np.maximum(x_max, np.max(xs))
        y_min = np.minimum(y_min, np.min(ys))
        y_max = np.maximum(y_max, np.max(ys))

    x_lims = (x_min, x_max)
    y_lims = (y_min-0.1*np.abs(y_min), y_max+0.1*np.abs(y_max))
    ax.set(xlim=x_lims, ylim=y_lims)
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    # -- create line-objects:
    line_objs = []
    y_label_chunks = []
    color_counter = color_offset
    for line_meta in req_lines_meta:
        xs = line_meta['xs']
        ys = line_meta['values']

        line, = ax.plot([], [], drawstyle='steps-post', label=f"{line_meta['label']} {is_secondary_char}", color=cm(color_counter))  # <-- empty line object
        line_objs.append({
            'line_obj': line,
            'xs': xs,
            'values': ys,
            'label': line_meta['label'],
        })
        y_label_chunks.append(line_meta['label'])
        color_counter += 1

    ax.set_xlabel('Position [cm]')
    ax.set_ylabel(', '.join(y_label_chunks) + ' [cm | cm/s ]')

    return line_objs


def plot_liquid_results_v1(graph: GraphTensor, model_res: Result_HydroModel, features_to_plot=['wells__liq_depth', 'wells__speed', 'wells__speed[0]', 'wells__liq_surf_alt'], steps_to_depict=None):

    DEFAULT_NUM_OF_STEPS_TO_DEPICT = 7

    if len(features_to_plot) <= 0:
        raise ValueError('I have nothing to plot -- please specify at least one feature in features_to_plot!')

    model_res_dict = model_res._asdict()
    times = np.asarray(model_res.context__time)

    if steps_to_depict is None:
        steps_to_depict = np.round(np.linspace(start=0, stop=times.shape[0]-1, num=DEFAULT_NUM_OF_STEPS_TO_DEPICT)).astype(np.int)


    # fig = plt.figure(figsize=(10, len(features_to_plot)*3))
    fig, axis = plt.subplots(len(features_to_plot), 1, figsize=(10, len(features_to_plot) * 4), squeeze=False)

    pipe_geometry = get_pipe_geometry_v1(graph)

    for i in range(len(features_to_plot)):
        feature_name_raw = features_to_plot[i]
        plot_metadata = _resolve_plot_metadata(feature_name_raw, model_res, graph)

        primary_plot_meta = plot_metadata.get('primary', {})
        primary_lines_meta = primary_plot_meta.get('lines', [])
        feature_spec = primary_plot_meta.get('specifier')

        ax: Axes = axis[i, 0]

        title = feature_spec
        ax.set_title(title)

        # -- print boundaries of the tube?
        if feature_spec == 'wells__liq_surf_alt':
            # -- for plot with liquid altitudes, plot also bottom and upper pipe boundaries:
            x_coords = _get_xcoords_from_graph_v1(graph)
            ax.plot(x_coords, _pad_mirror_last_value(pipe_geometry['wells_base_alt']) , '--', color='lightgrey', drawstyle='steps-post')

        # -- print the values:
        for step in steps_to_depict:
            for line_meta in primary_lines_meta:
                xs = line_meta['xs']
                feature_values = line_meta['values']
                if step > feature_values.shape[0]:
                    continue
                ys = feature_values[step, :]
                # print(f'Will plot: xs={xs} | ys={ys}')
                time_value = times[step, 0]
                ax.plot(xs, ys, label=f"@ T={time_value:.1f} s", drawstyle='steps-post')

            pass

        ax.set_xlabel('Position [cm]')
        ax.set_ylabel(feature_spec + ' [cm | cm/s ]')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.legend()
        ax.grid()


    plt.tight_layout()
    plt.show()

    pass


def animate_liquid_results_v1(graph: GraphTensor, model_res: Result_HydroModel, features_to_plot=None):
    """
    This function will create the Matplotlib animation object, that then can be either stored as video, or depicted in
    Python Notebook as interactive animation.

    :return: `anim` object
    """

    # if len(features_to_plot) <= 0:
    #     raise ValueError('I have nothing to plot -- please specify at least one feature in features_to_plot!')


    times = np.asarray(model_res.context__time)

    is_v1 = 'boundaries' in graph.node_sets['wells'].features.keys()

    if features_to_plot is None:
        if is_v1:
            features_to_plot = ['wells__liq_surf_alt', 'wells__speed']
        else:
            features_to_plot = ['wells__liq_surf_alt', 'wells__speed_x']

    num_features = len(features_to_plot)
    num_frames = times.shape[0]

    # %% Basic Frame preparation:
    fig, axes = plt.subplots(num_features, 1, figsize=(14, 4 * num_features), squeeze=False)    # unused: , sharex='col'

    subplot_specs = []
    all_lines_to_draw = []

    for i in range(num_features):
        plot_metadata = _resolve_plot_metadata(features_to_plot[i], model_res, graph)
        subplot_specs.append(plot_metadata)

        # -- 1) primary plot:
        primary_plot_meta = plot_metadata.get('primary', {})
        primary_lines_meta = primary_plot_meta.get('lines', [])
        feature_spec = primary_plot_meta.get('specifier')

        ax: Axes = axes[i, 0]
        lines_in_this_subplot = []
        if i == 0:  # <-- Text object - displayed only once:
            time_text = ax.text(0.05, 0.9, 'Time: unkn.', transform=ax.transAxes)

        if feature_spec == 'wells__liq_surf_alt':
            # -- for plot with liquid altitudes, plot also bottom and upper pipe boundaries:
            if is_v1:
                pipe_geometry = get_pipe_geometry_v1(graph)
                x_coords = _get_xcoords_from_graph_v1(graph)
                ax.plot(x_coords, _pad_mirror_last_value(pipe_geometry['wells_base_alt']) , '--', color='lightgrey', drawstyle='steps-post')
            else:
                _plot_pipe_geometry_into_ax(ax, graph)

        lines_in_this_subplot += _create_all_requested_lines(ax, primary_lines_meta)

        title = feature_spec
        ax.set_title(title)

        # -- 2) Secondary plot, if any:
        secondary_plot_meta = plot_metadata.get('secondary', {})
        secondary_lines_meta = secondary_plot_meta.get('lines', [])
        if len(secondary_lines_meta) > 0:
            ax2 = ax.twinx()
            lines_in_this_subplot += _create_all_requested_lines(ax2, secondary_lines_meta, color_offset=len(primary_lines_meta), is_secondary=True)

        line_artists = [x['line_obj'] for x in lines_in_this_subplot]
        ax.legend(loc='upper right', handles=line_artists)
        ax.grid()

        all_lines_to_draw += lines_in_this_subplot


    plt.tight_layout()


    def animate(i):
        this_time = float(times[i])
        text_to_set = f'Time: {this_time:.2f} s'

        objs_to_return = [time_text]

        for foo_line in all_lines_to_draw:
            line_obj = foo_line['line_obj']
            this_xs = foo_line['xs']
            this_ys = foo_line['values'][i, :]

            line_obj.set_data(this_xs, this_ys)
            objs_to_return.append(line_obj)

            if foo_line['label'] == 'wells__volume':
                total_volume_in_system = np.sum(this_ys)
                text_to_set += f' | Total Volume = {total_volume_in_system:.2f}'

        # for feature_name in features_to_plot:
        #     feature_values = model_res_dict[feature_name]
        #     n_dims = feature_values.shape[2] if feature_values.ndim == 3 else 1
        #
        #     for d in range(n_dims):
        #         line = lines_objs[feature_name][d]
        #
        #         this_xs = x_coords
        #         this_ys = feature_values[i, :] if feature_values.ndim == 2 else feature_values[i, :, d]
        #
        #         line.set_data(this_xs, this_ys)
        #         objs_to_return.append(line)


        time_text.set_text(text_to_set)

        return objs_to_return


    # %%
    anim = None
    time_step = graph.context['time_step']  # <-- in seconds
    time_step_ms = int(time_step * 1000)
    anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=time_step_ms, blit=True)

    return anim, fig, axes



def _plot_pipe_geometry_into_ax(ax: Axes, graph):

    x_coords = _get_double_xcoords_from_graph_v2(graph)
    bottom_zs_rank2 = graph.node_sets['wells']['xz_bottom_points'][:, :, 1].numpy()
    bottom_zs = bottom_zs_rank2.flatten()
    ceil_zs = bottom_zs + _get_values_reshaped_for_double_xs_v2(graph.node_sets['wells']['max_height'].numpy())

    ax.plot(x_coords, bottom_zs, '--', color='darkgrey', linewidth='3')
    ax.plot(x_coords, ceil_zs, ':', color='darkgrey', linewidth='3')

    return ax


def plot_pipe_geometry_v2(graph: GraphTensor):

    fig, axis = plt.subplots(1, 1, figsize=(10, 1 * 4), squeeze=False)

    ax: Axes = axis[0, 0]
    _plot_pipe_geometry_into_ax(ax, graph)
    ax.grid()

    plt.tight_layout()
    plt.show()
    return


def plot_liquid_results_v2(graph: GraphTensor, model_res: Result_HydroModel, features_to_plot=['wells__liq_surf_alt+wells__liq_depth'], steps_to_depict=None):

    DEFAULT_NUM_OF_STEPS_TO_DEPICT = 7

    if len(features_to_plot) <= 0:
        raise ValueError('I have nothing to plot -- please specify at least one feature in features_to_plot!')

    is_v1 = 'boundaries' in graph.node_sets['wells'].features.keys()

    model_res_dict = model_res._asdict()
    times = np.asarray(model_res.context__time)

    if steps_to_depict is None:
        steps_to_depict = np.round(np.linspace(start=0, stop=times.shape[0]-1, num=DEFAULT_NUM_OF_STEPS_TO_DEPICT)).astype(np.int)


    # fig = plt.figure(figsize=(10, len(features_to_plot)*3))
    fig, axis = plt.subplots(len(features_to_plot), 1, figsize=(10, len(features_to_plot) * 4), squeeze=False)

    # pipe_geometry = get_pipe_geometry_v2(graph)

    for i in range(len(features_to_plot)):
        feature_name_raw = features_to_plot[i]
        plot_metadata = _resolve_plot_metadata(feature_name_raw, model_res, graph)

        primary_plot_meta = plot_metadata.get('primary', {})
        primary_lines_meta = primary_plot_meta.get('lines', [])
        feature_spec = primary_plot_meta.get('specifier')

        ax: Axes = axis[i, 0]

        title = feature_spec
        ax.set_title(title)

        # -- print boundaries of the tube?
        if 'wells__liq_surf_alt' in feature_spec:
            # -- for plot with liquid altitudes, plot also bottom and upper pipe boundaries:
            _plot_pipe_geometry_into_ax(ax, graph)

        # -- print the values:
        for step in steps_to_depict:
            for line_meta in primary_lines_meta:
                xs = line_meta['xs']
                feature_values = line_meta['values']
                if step > feature_values.shape[0]:
                    continue

                time_value = times[step, 0]
                if is_v1:
                    ax.plot(xs, feature_values[step, :], label=f"@ T={time_value:.1f} s", drawstyle='steps-post')
                else:
                    ax.plot(xs, feature_values[step, :], '-', label=f"@ T={time_value:.1f} s", linewidth='1')

            pass

        ax.set_xlabel('Position [cm]')
        ax.set_ylabel(feature_spec + ' [cm | cm/s ]')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.legend()
        ax.set_title(title)
        ax.grid()

    plt.tight_layout()
    plt.show()

    pass


