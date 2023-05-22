import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow import keras
import math as m
from tensorflow_gnn.graph.graph_constants import FieldOrFields


class MessageFactoryConductionHeliumBath(keras.layers.Layer):
    '''
    This keras layer is responsible to compute the amount of heat transferred due to conduction between magnets
    '''

    def __init__(self):
        super().__init__()

    def call(self, graph: gt.GraphTensor, edge_set_name) -> FieldOrFields:
        T_broadcast_source = tfgnn.broadcast_node_to_edges(graph,
                                                           edge_set_name,
                                                           tfgnn.SOURCE,
                                                           feature_name='T')

        T_broadcast_target = tfgnn.broadcast_node_to_edges(graph,
                                                           edge_set_name,
                                                           tfgnn.TARGET,
                                                           feature_name='T')

        delta_T_magnets = T_broadcast_target - T_broadcast_source

        delta_T_magnets = keras.activations.relu(-delta_T_magnets)

        gradient_temperatures = delta_T_magnets / (graph.edge_sets[edge_set_name]['L'])

        # gradient_temperatures = keras.activations.relu(-gradient_temperatures)

        conductivity = tf.cast(graph.edge_sets[edge_set_name]["conductivity"], tf.float32)

        heat_flux_density = tf.math.pow(gradient_temperatures * conductivity, 1 / 3)

        heat_flux = heat_flux_density * graph.edge_sets[edge_set_name]["A"]
        E_transf_conduction = heat_flux * graph.context['time_step']

        # for the conservation law we can't transfer the amount of energy such that the temperature of
        # the second node is less than the other
        ## we have to limit the amount of energy transfered by the energy for
        # which the two temperature will equalize after the energy transfer

        cp_SOURCE = tfgnn.broadcast_node_to_edges(graph,
                                                  edge_set_name,
                                                  tfgnn.SOURCE,
                                                  feature_name='thermal capacity')

        cp_TARGET = tfgnn.broadcast_node_to_edges(graph,
                                                  edge_set_name,
                                                  tfgnn.TARGET,
                                                  feature_name='thermal capacity')

        combined_cp_coefficient = (cp_TARGET * cp_SOURCE) / (cp_TARGET + cp_SOURCE)

        max_energy_transferable = delta_T_magnets * combined_cp_coefficient

        E_transf = tf.minimum(E_transf_conduction, max_energy_transferable)

        heat_sent_conduction = tfgnn.pool_edges_to_node(graph,
                                                        edge_set_name,
                                                        tfgnn.SOURCE,
                                                        feature_value=E_transf)

        heat_received_conduction = tfgnn.pool_edges_to_node(graph,
                                                            edge_set_name,
                                                            tfgnn.TARGET,
                                                            feature_value=E_transf)

        heat_sum_magnets = - heat_sent_conduction + heat_received_conduction

        return heat_sum_magnets


class MessagePassingHeatSupplied(keras.layers.Layer):
    '''
    This keras layer is responsible to compute the amount of heat transferred by the heater to the magnets
    '''

    def __init__(self):
        super().__init__()

    def call(self, graph: gt.GraphTensor, edge_set_name) -> FieldOrFields:
        energy_supplied = graph.node_sets["heater"]["power"] * graph.context["time_step"]

        energy_supplied_broadcast = tfgnn.broadcast_node_to_edges(graph,
                                                                  edge_set_name,
                                                                  tfgnn.SOURCE,
                                                                  feature_value=energy_supplied)

        heat_supplied = tfgnn.pool_edges_to_node(graph,
                                                 edge_set_name,
                                                 tfgnn.TARGET,
                                                 feature_value=energy_supplied_broadcast)

        return heat_supplied


class MessageFactoryHelium2Bayonet(keras.layers.Layer):
    '''
    This keras layer is responsible to compute the amount of heat extracted by the liquid, so the
    heat going from magnets to liquid
    '''

    def __init__(self, he_creeping_lo_hfd=0.0, he_creeping_middle_hfd=0.0016, he_creeping_hi_hfd=0.0040,
                 he_creeping_distance_lo=0., he_creeping_distance_middle=0.):
        self.he_creeping_lo_hfd = tf.Variable(he_creeping_lo_hfd, trainable=True)
        self.he_creeping_middle_hfd = tf.Variable(he_creeping_middle_hfd, trainable=True)
        self.he_creeping_hi_hfd = tf.Variable(he_creeping_hi_hfd, trainable=True)
        # self.he_creeping_max_height = tf.Variable(he_creeping_max_height, trainable=True)
        self.m1 = - (tf.ones(1) * he_creeping_distance_lo - tf.ones(1) * he_creeping_distance_middle) / \
                    (self.he_creeping_middle_hfd - self.he_creeping_lo_hfd)
        self.b1 = he_creeping_distance_lo - self.m1 * self.he_creeping_lo_hfd

        self.m2 = - (tf.ones(1) * he_creeping_distance_middle) / (self.he_creeping_hi_hfd - self.he_creeping_middle_hfd)
        self.b2 = - self.m2 * he_creeping_hi_hfd
        super().__init__()

    def call(self, graph: gt.GraphTensor, edge_set_name) -> FieldOrFields:
        pi = tf.constant(m.pi)
        latent_heat_vaporization = graph.context['latent_heat_vaporization']

        T_broadcast_magnets_to_liquid = tfgnn.broadcast_node_to_edges(graph,
                                                                      edge_set_name,
                                                                      tfgnn.SOURCE,
                                                                      feature_name='T')

        T_broadcast_liquid_to_magnets = tfgnn.broadcast_node_to_edges(graph,
                                                                      edge_set_name,
                                                                      tfgnn.TARGET,
                                                                      feature_name='T')

        L_broadcast = tfgnn.broadcast_node_to_edges(graph,
                                                    edge_set_name,
                                                    tfgnn.TARGET,
                                                    feature_name='L')

        D_broadcast = tfgnn.broadcast_node_to_edges(graph,
                                                    edge_set_name,
                                                    tfgnn.TARGET,
                                                    feature_name='D')

        avg_f_broadcast = tfgnn.broadcast_node_to_edges(graph,
                                                        edge_set_name,
                                                        tfgnn.TARGET,
                                                        feature_name='avg_f')

        kapitza_heat_flux_density = graph.edge_sets[edge_set_name]['kapitza_induced_heat_flux_density']
        effective_fract_broadcast = self._compute_effective_fraction_for_hfd(kapitza_heat_flux_density, avg_f_broadcast,
                                                                             D_broadcast)

        # calcoliamo IL DELTA TRA i magneti e il liquido, se il delta t e' negativo l energia non puo passare

        delta_T_magnet_liquid = T_broadcast_liquid_to_magnets - T_broadcast_magnets_to_liquid
        delta_T_magnet_liquid = keras.activations.relu(- delta_T_magnet_liquid)

        # prediamo la kapitza coefficient 860
        # edge_liquid_kapitza_coefficient = tf.cast(graph.edge_sets[edge_set_name]['conductivity'], tf.float32)

        wetted_area = tf.cast(L_broadcast, tf.float32) * \
                      tf.cast(D_broadcast, tf.float32) * \
                      pi * \
                      tf.cast(effective_fract_broadcast, tf.float32)

        # kapitza_power = (delta_T_magnet_liquid *
        #                  edge_liquid_kapitza_coefficient/2 *
        #                  wetted_area *
        #                  tf.pow(T_broadcast_liquid_to_magnets, 3))

        kapitza_power = kapitza_heat_flux_density * wetted_area

        evap_mass_flow_from_kapitza = kapitza_power / latent_heat_vaporization

        prev_evapor_mass_flow = tfgnn.broadcast_node_to_edges(graph,
                                                              edge_set_name,
                                                              tfgnn.TARGET,
                                                              feature_name='incoming_mass_flow')

        # total_mass_flow = tf.math.reduce_sum(prev_evapor_mass_flow)
        total_mass_flow = prev_evapor_mass_flow[:1]
        # cum_mass_flow = tf.math.cumsum(evap_mass_flow_from_kapitza, exclusive=True)
        cum_mass_flow = tf.math.cumsum(evap_mass_flow_from_kapitza)
        cum_mass_flow = tf.concat([tf.zeros(1), cum_mass_flow], axis=0)  # lo zero va messo alla fine dato che
        # il flusso di elio viene da destra

        incoming_mass_flow = total_mass_flow - cum_mass_flow
        incoming_mass_flow = keras.activations.relu(incoming_mass_flow)

        # perche dobbiamo prendere il minimo tra quello che arriva nel nodo e quello che volgiamo evaporare
        # incoming mass flow e' un tensore piu lungo quindi lo dobbiamo accorciare per
        # i comparisons, abbiamo bisogno di quello che entra nel nodo
        # quindi prendiamo tutto tranne il primo elemento che sarebbe quello che esce
        # dell ultimo nodo
        curr_evaporation_mass_flow = tf.minimum(incoming_mass_flow[:-1], evap_mass_flow_from_kapitza)

        # kapitza_energy = kapitza_power * graph.context['time_step']

        energy_extractable = curr_evaporation_mass_flow * latent_heat_vaporization * graph.context['time_step']
        # evaporation_power = tf.cast(curr_evaporation_mass_flow * latent_heat_vaporization, tf.float32)
        # evaporation_energy = tf.cast(curr_evaporation_mass_flow * latent_heat_vaporization, tf.float32) * graph.context['time_step']

        # l heat extracted e' il min tra quanto puo essere evaporato,
        # quanto e' possibile trasferire tramite kapitza
        # quanto e' possibile traferire in base alla energy conservation law,
        # la temperatura del helium bath non puo andare sotto quella dell elio nella bhx, quindi
        # l energia massima trasferibile e' quella per cui le due temperature diventano uguali

        cp_SOURCE = tfgnn.broadcast_node_to_edges(graph,
                                                  edge_set_name,
                                                  tfgnn.SOURCE,
                                                  feature_name='thermal capacity')

        max_energy_transferable = delta_T_magnet_liquid * cp_SOURCE

        # heat_extracted_kapitza_evap = tf.minimum(kapitza_energy, evaporation_energy)

        # heat_extracted = tf.minimum(heat_extracted_kapitza_evap, max_energy_transferable)
        heat_extracted = tf.minimum(energy_extractable, max_energy_transferable)
        # heat_extracted = tf.minimum(kapitza_energy, evaporation_energy)
        # the function returns the heat extracted and the temperature of the liquid
        # which we need because the temperature of one node can't go below the temperature of its neighboors
        # max evaporization power is returned for the heat analysis layer so it should be shaped like context features

        # max_power_sum_for_context = tfgnn.pool_edges_to_context(graph,
        #                                             edge_set_name,
        #                                             'sum',
        #                                             feature_value=evaporation_power)

        max_power_sum_for_context = total_mass_flow * latent_heat_vaporization

        # mass_flow_going_to_the_pot = incoming_mass_flow[-1:]

        curr_evaporation_mass_flow = tfgnn.pool_edges_to_node(graph,
                                                              edge_set_name,
                                                              tfgnn.TARGET,
                                                              feature_value=curr_evaporation_mass_flow)

        # incoming_mass_flow_to_liquid_nodes = tfgnn.pool_edges_to_node(graph,
        #                                               edge_set_name,
        #                                               tfgnn.TARGET,
        #                                               feature_value=incoming_mass_flow[:-1])  #<-- si puo togliere
        #
        # incoming_mass_flow_to_liquid_nodes = tf.concat([incoming_mass_flow_to_liquid_nodes[1:],
        #                                                 mass_flow_going_to_the_pot, ], axis=0)

        heat_extracted_pool = tfgnn.pool_edges_to_node(graph,
                                                       edge_set_name,
                                                       tfgnn.SOURCE,
                                                       feature_value=heat_extracted)
        power_to_BHX = tfgnn.pool_edges_to_node(graph,
                                                edge_set_name,
                                                tfgnn.SOURCE,
                                                feature_value=heat_extracted)  # < -- stessa cosa di sopra si potrebbe togliere

        avg_f_pool = tfgnn.pool_edges_to_node(graph,
                                              edge_set_name,
                                              tfgnn.TARGET,
                                              feature_value=avg_f_broadcast)

        return {'heat_extracted': heat_extracted_pool,
                'max_power_sum_for_context': max_power_sum_for_context,
                'power_to_bhx': power_to_BHX,
                'evapor_mass_flow': curr_evaporation_mass_flow,
                'incoming_mass_flow': incoming_mass_flow,
                'avg_f': avg_f_pool}
        # 'mass_flow_going_to_the_pot': mass_flow_going_to_the_pot}#T_broadcast_liquid_to_magnets

    def _compute_effective_fraction_for_hfd(self, heat_flux_density, original_frac_wet_perim, diameter):
        circumference = tf.constant(m.pi) * diameter

        perimeter = original_frac_wet_perim * circumference

        creeping_additive_factor_1 = self.m1 * heat_flux_density + self.b1
        creeping_additive_factor_2 = self.m2 * heat_flux_density + self.b2
        creeping_additive_factor = tf.maximum(creeping_additive_factor_1, creeping_additive_factor_2)
        creeping_additive_factor = tf.keras.activations.relu(creeping_additive_factor)

        new_perimeter = perimeter + 2 * creeping_additive_factor

        effective_fract_broadcast = new_perimeter / circumference
        effective_fract_broadcast = tf.minimum(effective_fract_broadcast, 1.)

        return effective_fract_broadcast


class DynamicHeatPoolLayer(keras.layers.Layer):

    def __init__(self):
        super().__init__()

    def call(self, graph: gt.GraphTensor, node_set_name) -> FieldOrFields:
        total_power = tfgnn.pool_nodes_to_context(graph,
                                                  node_set_name=node_set_name,
                                                  reduce_type='sum',
                                                  feature_name='power')
        return tf.cast(total_power, tf.float32)


class StaticHeatPoolLayer(keras.layers.Layer):

    def __init__(self):
        super().__init__()

    def call(self, graph: gt.GraphTensor, node_set_name) -> FieldOrFields:
        total_length = tfgnn.pool_nodes_to_context(graph,
                                                   node_set_name=node_set_name,
                                                   reduce_type='sum',
                                                   feature_name='L')
        return tf.cast(total_length, tf.float32)


class MessageFactoryBayonet2Helium(keras.layers.Layer):

    def __init__(self):
        super().__init__()

    def call(self, graph: gt.GraphTensor, edge_set_name) -> FieldOrFields:
        pi = tf.constant(m.pi)

        T_broadcast_liquid_to_magnets = tfgnn.broadcast_node_to_edges(graph,
                                                                      edge_set_name,
                                                                      tfgnn.SOURCE,
                                                                      feature_name='T')

        T_broadcast_magnets_to_liquid = tfgnn.broadcast_node_to_edges(graph,
                                                                      edge_set_name,
                                                                      tfgnn.TARGET,
                                                                      feature_name='T')

        L_broadcast = tfgnn.broadcast_node_to_edges(graph,
                                                    edge_set_name,
                                                    tfgnn.SOURCE,
                                                    feature_name='L')

        D_broadcast = tfgnn.broadcast_node_to_edges(graph,
                                                    edge_set_name,
                                                    tfgnn.SOURCE,
                                                    feature_name='D')

        fractions_wetted_perim_broadcast = tfgnn.broadcast_node_to_edges(graph,
                                                                         edge_set_name,
                                                                         tfgnn.SOURCE,
                                                                         feature_name='avg_f')

        # calcoliamo IL DELTA TRA il liquido i magneti e, se il delta t e' negativo l'energia non puo passare
        # consideriamo sempre finale - iniziale
        delta_T_liquid_magnet = T_broadcast_magnets_to_liquid - T_broadcast_liquid_to_magnets
        delta_T_liquid_magnet = keras.activations.relu(- delta_T_liquid_magnet)

        edge_liquid_kapitza_coefficient = tf.cast(graph.edge_sets[edge_set_name]['conductivity'], tf.float32)

        wetted_area = tf.cast(L_broadcast, tf.float32) * \
                      tf.cast(D_broadcast, tf.float32) * \
                      pi * \
                      tf.cast(fractions_wetted_perim_broadcast, tf.float32)

        kapitza_power = (delta_T_liquid_magnet *
                         edge_liquid_kapitza_coefficient / 2 *
                         wetted_area *
                         tf.pow(T_broadcast_magnets_to_liquid, 3))  # third power of the cold source

        kapitza_energy = kapitza_power * graph.context['time_step']

        kapitza_energy_pool = tfgnn.pool_edges_to_node(graph,
                                                       edge_set_name,
                                                       tfgnn.TARGET,
                                                       feature_value=kapitza_energy)

        return {'power_from_bhx': kapitza_energy_pool}
