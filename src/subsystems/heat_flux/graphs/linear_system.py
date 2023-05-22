import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn


def make_graph_tensor_from_tensors(initial_temperatures,
                      masses=None,
                      common_mass=400.,
                      length_center_cells=None,
                      common_length=1.,
                      cross_sections=None,
                      common_cross_section=1.,
                      power_supplied=None,
                      common_power_supplied=0.,
                      liquid_temperature=1.85,
                      num_wetted_cells=1,
                      wetted_extraction_capacity=5.,
                      thermal_conductivity_magnet_to_magnet=100.,
                      thermal_conductivity_magnet_to_liquid=1200.,
                      time_step: float = 1.,
                      time: float = 0.,
                      specific_heat_capacity: float = 1.,
                      static_heat: float = 1.):
    '''
    - This function creates a GraphTensor given all the features of the "nodes" of the graph
    if the features are not passed as input default values are used

    - This function is intended to be used with some file (e.g. json ) where the user can specify a
    proper configuration of the graph he has and creates the corresponding GraphTensor

    - By default all the magnets are connected between them, each heater is supposed to be connected to
    all the magnets --> future work is to give the possibility to specify the distribution of the heaters and create the
    adjacency matrix consequently

    - By default the number of nodes representing the liquid is equal to the one representing the magnets, this is in
    the real word a true assumption due to the fact the heat exchanger is going along all the standard LHC cell

    - Future work --> the extraction_capacity can be substituted by a mask or a single number which indicates where
    the liquid is, then the extraction capacity is computed consequently

    :param temperatures:
    :return:
    '''

    # magnets_temperatures = tf.nest.flatten(magnets_temperatures)
    # magnets_temperatures = tf.reshape(magnets_temperatures, [4])
    # print(magnets_temperatures)

    num_cells = tf.squeeze(tf.shape(initial_temperatures)[-1])
    # num_cells = initial_temperatures.shape[-1]

    # std_cells_shape = tf.constant(num_cells, dtype=tf.int32, shape=(1,), name="std_cells_shape")
    std_cells_shape = tf.reshape(num_cells, shape=(1,), name="std_cells_shape")

    if masses:
        if len(masses) != len(initial_temperatures):
            raise ValueError('Dimension of the masses supplied not compatible with magnets temperatures,'
                             'please use the common_mass or insert the correct number of masses')
    else:
        masses = tf.repeat(common_mass, num_cells)
        # masses = tf.repeat(common_mass, len(magnets_temperatures))

    if power_supplied:
        if len(power_supplied) != len(initial_temperatures):
            raise ValueError(
                'Dimension of the power supplied for the heaters is not compatible with magnets temperatures,'
                'please use the common_power supplied or insert the correct number of powers')
    else:
        # power_supplied = np.repeat(common_power_supplied, num_cells)
        power_supplied = tf.repeat(common_power_supplied, num_cells)

   #tf.print("num_wetted_cells: ", num_wetted_cells)
    # if tf.rank(num_wetted_cells) > 1:
    #     raise ValueError(f'ERROR - num_wetted_cells is in incorrect shape! Expected: scalar, or (1,); given: {num_wetted_cells} of shape: {tf.shape(num_wetted_cells)}. Rank: {tf.rank(num_wetted_cells)}')

    #liquid_temperatures = np.repeat(liquid_temperature, num_cells)
    liquid_temperatures = tf.repeat(liquid_temperature, num_cells)

    # liquid_extraction_capacity = np.pad(np.repeat(wetted_extraction_capacity, num_wetted_cells),
    #                                     [0, int(num_cells - num_wetted_cells)])
    # paddings = tf.constant([[0, num_cells - num_wetted_cells]])
    # paddings = tf.constant([[0, 2]])
    # liquid_extraction_capacity = tf.pad(tensor=tf.repeat(wetted_extraction_capacity, num_wetted_cells),
    #                                     paddings=paddings)
    # liquid_extraction_capacity = tf.concat(
    #     [liquid_extraction_capacity, tf.repeat(0., len(magnets_temperatures) - num_wetted_cells)], axis=0)

    #tf.print(tf.zeros(num_cells - num_wetted_cells))
    liquid_extraction_capacity = tf.concat(
        [tf.repeat(wetted_extraction_capacity, num_wetted_cells), tf.zeros(num_cells - num_wetted_cells)], axis=0)

    #accumulated_heat = np.repeat(0., num_cells)
    accumulated_heat = tf.repeat(0., num_cells)

    cells = tfgnn.NodeSet.from_fields(sizes=std_cells_shape,
                                      features={
                                          "T": tf.cast(initial_temperatures, tf.float32),
                                          "Q": tf.cast(accumulated_heat, tf.float32),
                                          "mass": tf.cast(masses, tf.float32),
                                          "L": tf.cast(tf.repeat(common_length, 4), tf.float32)
                                      })

    heater = tfgnn.NodeSet.from_fields(sizes=std_cells_shape,
                                       features={
                                           "power": tf.cast(power_supplied, tf.float32),
                                       })

    liquid = tfgnn.NodeSet.from_fields(sizes=std_cells_shape,
                                       features={
                                           "extraction capacity": tf.cast(liquid_extraction_capacity, tf.float32),
                                           "T": tf.cast(liquid_temperatures, tf.float32)
                                       })

    # cells_sources = tf.concat(
    #     [tf.range(0, tf.shape(magnets_temperatures) - 1), tf.range(1, tf.shape(magnets_temperatures))], axis=0)
    # cells_target = tf.concat(
    #     [tf.range(1, tf.shape(magnets_temperatures)), tf.range(0, tf.shape(magnets_temperatures) - 1)], axis=0)

    # edge_indices = [(i, i + 1) for i in range(num_cells - 1)]
    # src_idx = [s[0] for s in edge_indices] + [s[1] for s in edge_indices]
    # target_idx = [s[1] for s in edge_indices] + [s[0] for s in edge_indices]
    # num_conductivity_edges = len(src_idx)

    id_L = tf.constant(0)
    id_R = num_cells - 1
    src_R = tf.range(id_L, id_R)
    tgt_R = src_R + tf.constant(1)
    src = tf.concat((src_R, tgt_R), 0)
    tgt = tf.concat((tgt_R, src_R), 0)
    num_conductivity_edges = len(src)

    cell_adjacency = tfgnn.Adjacency.from_indices(source=('cells', src),
                                                  target=('cells', tgt))

    conduction = tfgnn.EdgeSet.from_fields(sizes=tf.constant([num_conductivity_edges]),
                                           features={
                                               "conductivity": tf.cast(np.repeat(thermal_conductivity_magnet_to_magnet,
                                                                                 num_conductivity_edges), tf.float32),
                                               "L": tf.cast(np.repeat(1., num_conductivity_edges), tf.float32),
                                               "A": tf.cast(np.repeat(1., num_conductivity_edges), tf.float32),
                                               "E_transf": tf.cast(np.repeat(0., num_conductivity_edges), tf.float32),
                                               "Q_transf": tf.cast(np.repeat(0., num_conductivity_edges), tf.float32),
                                           },
                                           adjacency=cell_adjacency)

    heater_sources = tf.range(0, num_cells)
    cells_target = tf.range(0, num_cells)

    heater_cells_adjacency = tfgnn.Adjacency.from_indices(source=('heater', heater_sources),
                                                          target=('cells', cells_target))

    heat_supplied = tfgnn.EdgeSet.from_fields(sizes=std_cells_shape,
                                              features={},
                                              adjacency=heater_cells_adjacency)

    cells_sources = tf.range(0, num_cells)
    liquid_target = tf.range(0, num_cells)

    cells_liquid_adjacency = tfgnn.Adjacency.from_indices(source=('cells', cells_sources),
                                                          target=('liquid', liquid_target))

    heat_extracted = tfgnn.EdgeSet.from_fields(sizes=std_cells_shape,
                                               features={
                                                   "conductivity": tf.cast(
                                                       tf.repeat(thermal_conductivity_magnet_to_liquid, num_cells),
                                                       tf.float32),
                                               },
                                               adjacency=cells_liquid_adjacency)

    context = tfgnn.Context.from_fields(
        features={'time': tf.cast([time], tf.float32),
                  'time_step': tf.cast([time_step], tf.float32),
                  'specific heat capacity': tf.cast([specific_heat_capacity], tf.float32),
                  'static_heat': tf.cast([static_heat], tf.float32)})

    result = tfgnn.GraphTensor.from_pieces(node_sets={'cells': cells,
                                                      'heater': heater,
                                                      'liquid': liquid
                                                      },
                                           edge_sets={'conduction': conduction,
                                                      'heat supplied': heat_supplied,
                                                      'cell2liquid': heat_extracted
                                                      },
                                           context=context)

    return result


