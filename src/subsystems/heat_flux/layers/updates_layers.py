import tensorflow as tf
from typing import Tuple
from tensorflow_gnn.graph.graph_constants import FieldsNest
from tensorflow import keras
from lib.diag_common.tf.layers.PWL_Layer import PieceWiseLinear_1D_Layer



class IncreaseTemperature(keras.layers.Layer):

    """
    This keras layer is devoted to the increase the magnets temperatures based on the variation of enthalpy
    (sum of all the energy incoming and outgoing from that magnet)

    """

    def __init__(self):
        super().__init__()
        self.thermal_capacity_factor = self.add_weight(shape=(), initializer='ones', trainable=True,
                                                       name='thermal_capacity_factor')

    def call(self, inputs: Tuple[FieldsNest, FieldsNest, FieldsNest]):
        node_input_feature, edge_inputs, context_input_features = inputs

        static_heat = context_input_features['time_step'] * \
                      context_input_features['static_heat'] * \
                      node_input_feature['L']

        heat_sum_magnets = edge_inputs['conduction'] + \
                           edge_inputs['heat supplied'] - \
                           edge_inputs['cell2liquid']['heat_extracted'] + \
                           static_heat

        temp_variation = heat_sum_magnets / (self.thermal_capacity_factor * node_input_feature['thermal capacity'])
        new_temp = node_input_feature['T'] + temp_variation

        return {'T': new_temp}

# class DynamicHeatUpdateLayer(keras.layers.Layer):
#     def __init__(self):
#         super().__init__()
#
#     def call(self, inputs: Tuple[FieldsNest, FieldsNest, FieldsNest]):
#         context_input_features, node_set_inputs, edge_set_inputs = inputs
#         #print(context_input_features)
#         #print(node_set_inputs)
#         #print(edge_set_inputs)
#
#         total_dynamic_heat = node_set_inputs['heater']
#
#         return {'total_dynamic_heat': total_dynamic_heat}
#
#
# class StaticHeatUpdateLayer(keras.layers.Layer):
#     def __init__(self):
#         super().__init__()
#
#     def call(self, inputs: Tuple[FieldsNest, FieldsNest, FieldsNest]):
#         context_input_features, node_set_inputs, edge_set_inputs = inputs
#
#         total_static_heat = tf.cast(context_input_features['static_heat'], tf.float32) * node_set_inputs['cells']
#
#         return {'total_static_heat': total_static_heat}


class HeatAnalysisUpdateLayer(keras.layers.Layer):
    """
    Analysis Layer ==> computes the total static heat, dynamic heat and the max power extractable
    ( = helium evaporation flow * 23 J/g ) at each time step , and set them as graph context features

    """

    def __init__(self):
        super().__init__()

    def call(self, inputs: Tuple[FieldsNest, FieldsNest, FieldsNest]):
        context_input_features, node_set_inputs, edge_set_inputs = inputs

        total_dynamic_heat = node_set_inputs['heater']
        total_static_heat = tf.cast(context_input_features['static_heat'], tf.float32) * node_set_inputs['cells']
        total_vaporization_heat = edge_set_inputs['cell2liquid']['max_power_sum_for_context']

        return {'total_dynamic_heat': total_dynamic_heat,
                'total_static_heat': total_static_heat,
                'total_vaporization_heat': total_vaporization_heat}


class PowerBHXLayer(keras.layers.Layer):
    """
    Layer which computes the heat flux between helium bath and BHX. Bidirectional edges potentially gives the possibility
    to transfer heat from BHX to helium bath in the case the latter is at a lower temperature.

    This happens for example during the 'Inverse Response' (increasing of pressure => increasing of saturation temperatures)

    In our simplified model only the case when the helium in the BHX is at a lower temperature is tested.
    Another simplification is that the temperatures stay fixed along all the BHX, when in reality due to pressure drop
    the saturation temperature changes along the BHX

    'Inverse Response' and 'Saturation temperature variation' are left for future experiments.

    """

    def __init__(self):
        super().__init__()

    def call(self, inputs: Tuple[FieldsNest, FieldsNest, FieldsNest]):

        node_input_feature, edge_inputs, context_input_features = inputs

        power_to_bhx = edge_inputs['cell2liquid']['power_to_bhx']
        power_from_bhx = edge_inputs['liquid2cell']['power_from_bhx']


        results = tf.maximum(power_to_bhx, power_from_bhx)

        return {'power_to_bhx': results}


class HeatFluxConductionLayer(keras.layers.Layer):

    """
    Layer which computes the heat flux between magnets, caused by difference of temperatures
    (synonyms : horizontal heat flux, heat flux by conduction)

    The heat flux is computed as:

     (ΔT / L * K) ^ (1/3) * A

     where:
           *  ΔT/L is gradient temperatures [K/cm]
           *  K is the thermal conductivity [W^3 / cm^5 * K]
           *  A is the cross-section of the interface [cm^2]

    """


    def __init__(self):
        super().__init__()

    def call(self, inputs: Tuple[FieldsNest, FieldsNest, FieldsNest]):
        edge_inputs, node_inputs, context_inputs = inputs

        # edge_inputs is a dict with all the features,
        # node_inputs is a dict with keys 0 and 1
        # 0 is related to the features of the SOURCES of the edges
        # 1 is related to the features of the TARGETS of the edges


        T_source = node_inputs[0]
        T_target = node_inputs[1]
        delta_T = T_target - T_source

        gradient_temperatures = delta_T / (edge_inputs['L'])
        gradient_temperatures = keras.activations.relu(-gradient_temperatures)

        conductivity = tf.cast(edge_inputs['conductivity'], tf.float32)

        heat_flux_density = tf.math.pow(gradient_temperatures * conductivity, 1/3)
        heat_flux = heat_flux_density * edge_inputs['A']


        return {'heat_flux_conduction': heat_flux,
                'delta_t_conduction': delta_T}

class KapitzaInducedHeatDensityLayer(keras.layers.Layer):

    """
    This layer computes the amount of heat flux density induced by Kapitza effect which happens between the helium
    bath and BHX ('vertical heat flux density' due to the graph configuration and/or string design)

    The heat flux density is computes as :

     ΔT * C/2 * T_cold ^3

     where:
            **  ΔT is the difference of temperatures between bath and BHX [K]
            **  C is the kapitza conductance ( over 2 because there are 2 surfaces) [W/ cm^2 * K^4]
            **  T_cold is the temperature of the cold source [K]
    """

    def __init__(self):
        super().__init__()

    def call(self, inputs: Tuple[FieldsNest, FieldsNest, FieldsNest]):

        edge_inputs, node_inputs, context_inputs = inputs

        T_source = node_inputs[0]
        T_target = node_inputs[1]

        delta_T_magnet_liquid = T_target - T_source
        delta_T_magnet_liquid = keras.activations.relu(- delta_T_magnet_liquid)

        kapitza_induced_heat_flux_density = delta_T_magnet_liquid * \
                         edge_inputs/2 * tf.pow(T_target, 3)

        return {'kapitza_induced_heat_flux_density': kapitza_induced_heat_flux_density}


class FractionFromMassFlowLayer(keras.layers.Layer):

    """


    """

    def __init__(self, layer_configuration_file):
        self.layer_configuration_file = layer_configuration_file
        super().__init__()

    def call(self, inputs: Tuple[FieldsNest, FieldsNest, FieldsNest]):
        node_input_feature, edge_inputs, context_input_features = inputs

        evaporation_mass_flow = edge_inputs['cell2liquid']['evapor_mass_flow']
        incoming_mass_flow = edge_inputs['cell2liquid']['incoming_mass_flow']
        #mass_flow_going_to_pot = edge_inputs['cell2liquid']['mass_flow_going_to_the_pot']
        #incoming_mass_flow = tf.concat([tf.zeros(1), incoming_mass_flow], axis=0)

        #all_incoming_mass_flow = tf.concat([mass_flow_going_to_pot, incoming_mass_flow], axis=0)

        # fractions e' un tensore piu lungo quindi ha 37 elementi
        # perche sarebbero le fractions all inizio e alla fine del nodo

        fractions_from_mass_flow_layer = PieceWiseLinear_1D_Layer.load_from_config_file(self.layer_configuration_file)

        #fractions = fractions_from_mass_flow_layer(all_incoming_mass_flow)
        fractions = fractions_from_mass_flow_layer(incoming_mass_flow)
        fractions = keras.activations.relu(fractions)
        fractions = tf.concat([fractions, tf.zeros(1)], axis=0)

        # per le starting fractions prendiamo tutti gli elementi dalla posizione 1
        starting_fractions = fractions[:-1]
        #per le ending fractions prendiamo tutti gli elementi dalla posizione 0 fino al penultimo
        ending_fractions = fractions[1:]

        mean_fractions = (starting_fractions + ending_fractions) / 2
        #mean_fractions = tf.concat([tf.zeros(1), mean_fractions], axis=0)

        #mean_fractions = keras.activations.relu(mean_fractions)


        return {'evapor_mass_flow': evaporation_mass_flow,
                'avg_f': mean_fractions}


class LiquidVolumeLayer(keras.layers.Layer):

    def __init__(self, layer_configuration_file):
        self.layer_configuration_file = layer_configuration_file
        super().__init__()

    def call(self, inputs: Tuple[FieldsNest, FieldsNest, FieldsNest]):
        node_input_feature, edge_inputs, context_input_features = inputs

        lengths = node_input_feature['L']
        fractions = node_input_feature['avg_f']

        cross_area_from_fractions = PieceWiseLinear_1D_Layer.load_from_config_file(self.layer_configuration_file)

        cross_areas = cross_area_from_fractions(fractions)
        cross_areas = keras.activations.relu(cross_areas)

        volumes = cross_areas * lengths

        return {'volume': volumes}



class LiquidAnalysisLayer(keras.layers.Layer):
    def __init__(self):
        super().__init__()


    def call(self, inputs: Tuple[FieldsNest, FieldsNest, FieldsNest]):
        node_input_feature, edge_inputs, context_input_features = inputs

        lengths = node_input_feature['L'] / 100.

        incoming_mass_flow = edge_inputs['cell2liquid']['incoming_mass_flow']
        evapor_mass_flow = edge_inputs['cell2liquid']['evapor_mass_flow']
        avg_f = edge_inputs['cell2liquid']['avg_f']


        evapor_mass_flow_per_m = evapor_mass_flow / lengths

        return {'incoming_mass_flow': incoming_mass_flow,
                'evapor_mass_flow': evapor_mass_flow,
                'evapor_mass_flow_per_m': evapor_mass_flow_per_m,
                'avg_f': avg_f}



