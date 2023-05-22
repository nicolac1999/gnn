import numpy as np
from matplotlib import pyplot as plt


def plot_knots(model):
    layer_config = model.layers[0].get_config()
    knot_pos = layer_config['knot_pos']
    knot_vals = layer_config['knot_vals']

    plt.plot(knot_pos, knot_vals, '.', label="Knots Pos & vals")
    plt.title('Knot-points')
    plt.grid()
    plt.legend()
    plt.show()


def plot_analytic_vs_predicted(model, xs, ys_true, xy_labels=(None,None), val_sampling_points=None, title=None):

    plt.plot(xs, ys_true, label="Analytical values")

    if model is not None:
        if val_sampling_points is not None:
            min_x = np.min(xs)
            max_x = np.max(xs)
            xs_val = np.linspace(min_x, max_x, val_sampling_points)
        else:
            xs_val = xs

        ys_pred = model(xs_val)
        plt.plot(xs_val, ys_pred, label="Predicted values")

    if xy_labels is not None:
        plt.xlabel(xy_labels[0])
        plt.ylabel(xy_labels[1])

    if title is not None:
        plt.title(title)

    plt.grid()
    plt.legend()
    plt.show()

    pass
