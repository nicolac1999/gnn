from typing import NamedTuple
from tensorflow_gnn import keras as keras_gnn
from tensorflow import keras
import numpy as np

from common.layers.time_management import IncreaseTimeLayer
from subsystems.heat_flux.layers.message_passing_layers import *
from subsystems.heat_flux.layers.pre_processing_layers import *
from subsystems.heat_flux.layers.updates_layers import *
from lib.diag_common.tf.layers.PWL_Layer import PieceWiseLinear_1D_Layer

from common.model_helpers import readout_node_values_as_slice, readout_context_values_as_slice, readout_edge_values_as_slice

#%%

class ResHeatSimplifiedModel(NamedTuple):
    """
    NamedTuple class defining a 'tuple' of results, that are produced by the HeatSimplifiedModel.
    """
    context__time: np.ndarray
    cells__T: np.ndarray
    cells__power_to_bhx: np.ndarray
    context__total_static_heat: np.ndarray
    context__total_dynamic_heat: np.ndarray
    context__total_vaporization_heat: np.ndarray
    conduction__heat_flux_conduction: np.ndarray
    conduction__delta_t_conduction: np.ndarray
    liquid__incoming_mass_flow: np.ndarray
    liquid__evapor_mass_flow: np.ndarray
    liquid__evapor_mass_flow_per_m: np.ndarray
    liquid__avg_f: np.ndarray




#%%

class HeatSimplifiedModel(keras.Model):
    """
    Wrapper class for our GNN model, which simulates a simplified Heat-Flux physics.
    The purpose of this wrapper class is to demonstrate one possible way how the model can be defined as a subclass of
    keras.Model.

    """

    #GRAPH_SCHEMA_PATH = r"src/subsystems/heat_flux/graph_schemas/heat_flux_graph_basic.pbtxt"
    GRAPH_SCHEMA_PATH = r"src/subsystems/heat_flux/graph_schemas/heat_flux_graph_virtual_pot.pbtxt"
    OUTPUT_NT_TYPE = ResHeatSimplifiedModel

    def __init__(self, configuration_file, num_steps=30):
        self.num_steps = num_steps
        self.configuration_file = configuration_file

        in_out = self._create_model(num_steps=num_steps)


        # wrapped_outputs = self.OUTPUT_NT_TYPE(*in_out['outputs'])
        super(HeatSimplifiedModel, self).__init__(inputs=in_out['inputs'], outputs=in_out['outputs'])


    def _create_model(self, num_steps, include_initial_step=True):
        res_temperatures = []
        res_times = []
        res_static_heat = []
        res_dynamic_heat = []
        res_vaporiz_heat = []
        res_power_to_bhx = []
        res_heat_flux_conduction = []
        res_delta_t_conduction = []
        res_incoming_mass_flow = []
        res_evapor_mass_flow = []
        res_evapor_mass_flow_per_m = []
        res_fractions_wetted_perimeter = []

        graph_schema = tfgnn.read_schema(self.GRAPH_SCHEMA_PATH)
        graph_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)

        input_layer = keras.layers.Input(type_spec=graph_spec)

        preprocessing_step = keras_gnn.layers.GraphUpdate(
            node_sets={
                'cells': keras_gnn.layers.NodeSetUpdate(
                    edge_set_inputs={},
                    next_state=ThermalCapacityComputation(),
                    node_input_feature='mass',
                    context_input_feature="specific_heat_capacity"
                )
            }
        )
        graph = preprocessing_step(input_layer)

        if include_initial_step:

            res_temperatures.append(readout_node_values_as_slice(graph, 'cells', 'T'))
            res_times.append(readout_context_values_as_slice(graph, 'time'))
            res_static_heat.append((readout_context_values_as_slice(graph, 'total_static_heat')))
            res_dynamic_heat.append(readout_context_values_as_slice(graph, 'total_dynamic_heat'))
            res_vaporiz_heat.append(readout_context_values_as_slice(graph, 'total_vaporization_heat'))
            res_power_to_bhx.append(readout_node_values_as_slice(graph, 'cells', 'power_to_bhx'))
            res_heat_flux_conduction.append(readout_edge_values_as_slice(graph, 'conduction', 'heat_flux_conduction'))
            res_delta_t_conduction.append(readout_edge_values_as_slice(graph, 'conduction', 'delta_t_conduction'))
            res_incoming_mass_flow.append(readout_node_values_as_slice(graph, 'liquid', 'incoming_mass_flow'))
            res_evapor_mass_flow.append(readout_node_values_as_slice(graph, 'liquid', 'evapor_mass_flow'))
            res_evapor_mass_flow_per_m.append(readout_node_values_as_slice(graph, 'liquid', 'evapor_mass_flow_per_m'))
            res_fractions_wetted_perimeter.append(readout_node_values_as_slice(graph, 'liquid', 'avg_f'))

        kapitza_processing = keras_gnn.layers.GraphUpdate(
            edge_sets={
                'cell2liquid': keras_gnn.layers.EdgeSetUpdate(
                    next_state=KapitzaInducedHeatDensityLayer(),
                    edge_input_feature='conductivity',
                    node_input_feature='T',
                )
            }
        )

        layer_helium2bayonet_message = MessageFactoryHelium2Bayonet()

        temperature_step_processing = keras_gnn.layers.GraphUpdate(
            node_sets={
                "cells": keras_gnn.layers.NodeSetUpdate(
                    edge_set_inputs={"conduction": MessageFactoryConductionHeliumBath(),
                                     "heat supplied": MessagePassingHeatSupplied(),
                                     "cell2liquid": layer_helium2bayonet_message},
                    next_state=IncreaseTemperature(),
                    node_input_feature=['T', 'thermal capacity', 'L'],
                    context_input_feature=['static_heat', 'time_step']
                )
            },)

        fraction_wetted_perim_processing = keras_gnn.layers.GraphUpdate(
            node_sets={
                'liquid': keras_gnn.layers.NodeSetUpdate(
                    edge_set_inputs={"cell2liquid": layer_helium2bayonet_message},
                    next_state=FractionFromMassFlowLayer(layer_configuration_file=self.configuration_file),
                    node_input_feature={},
                    context_input_feature={}
                )
            }
        )

        liquid_analysis_processing = keras_gnn.layers.GraphUpdate(
            node_sets={
                'liquid': keras_gnn.layers.NodeSetUpdate(
                    edge_set_inputs={"cell2liquid": layer_helium2bayonet_message},
                    next_state=LiquidAnalysisLayer(),
                    node_input_feature=['L'],
                    context_input_feature={}
                )
            }
        )

        time_processing_step = keras_gnn.layers.GraphUpdate(
            context=keras_gnn.layers.ContextUpdate(
                node_set_inputs={},
                next_state=IncreaseTimeLayer(),
                context_input_feature=['time', 'time_step']
            ),
        )

        # static_heat_processing_step = keras_gnn.layers.GraphUpdate(
        #     context=keras_gnn.layers.ContextUpdate(
        #         node_set_inputs={'cells': StaticHeatPoolLayer()},
        #         next_state=StaticHeatUpdateLayer(),
        #         context_input_feature=['static_heat']
        #     )
        # )
        #
        # dynamic_heat_processing_step = keras_gnn.layers.GraphUpdate(
        #     context=keras_gnn.layers.ContextUpdate(
        #         node_set_inputs={'heater': DynamicHeatPoolLayer()},
        #         next_state=DynamicHeatUpdateLayer(),
        #         context_input_feature=None
        #
        #     )
        # )

        heat_flux_to_bhx_step = keras_gnn.layers.GraphUpdate(
            node_sets={
                'cells': keras_gnn.layers.NodeSetUpdate(
                    edge_set_inputs={'cell2liquid': MessageFactoryHelium2Bayonet(),
                                     'liquid2cell': MessageFactoryBayonet2Helium()},
                    next_state=PowerBHXLayer(),
                    node_input_feature=['power_to_bhx']
                )
            }
        )

        heat_flux_conduction_step = keras_gnn.layers.GraphUpdate(
            edge_sets={
                'conduction': keras_gnn.layers.EdgeSetUpdate(
                    next_state=HeatFluxConductionLayer(),
                    edge_input_feature=['L', 'A', 'conductivity'],
                    node_input_feature='T',
                )
            }
        )

        heat_analysis_step = keras_gnn.layers.GraphUpdate(
            context=keras_gnn.layers.ContextUpdate(
                node_set_inputs={'cells': StaticHeatPoolLayer(),
                                 'heater': DynamicHeatPoolLayer()},
                edge_set_inputs={"cell2liquid": MessageFactoryHelium2Bayonet()},
                next_state=HeatAnalysisUpdateLayer(),
                context_input_feature=['static_heat']

            )
        )


        for i in range(num_steps):
            graph = kapitza_processing(graph)

            graph = temperature_step_processing(graph)
            res_temperatures.append(readout_node_values_as_slice(graph, 'cells', 'T'))


            graph = heat_flux_to_bhx_step(graph)
            res_power_to_bhx.append(readout_node_values_as_slice(graph, 'cells', 'power_to_bhx'))

            graph = liquid_analysis_processing(graph)
            res_incoming_mass_flow.append(readout_node_values_as_slice(graph, 'liquid', 'incoming_mass_flow'))
            res_evapor_mass_flow.append(readout_node_values_as_slice(graph, 'liquid', 'evapor_mass_flow'))
            res_evapor_mass_flow_per_m.append(readout_node_values_as_slice(graph, 'liquid', 'evapor_mass_flow_per_m'))
            res_fractions_wetted_perimeter.append(readout_node_values_as_slice(graph, 'liquid', 'avg_f'))


            graph = time_processing_step(graph)
            res_times.append(readout_context_values_as_slice(graph, 'time'))

            graph = heat_analysis_step(graph)

            graph = fraction_wetted_perim_processing(graph)

            res_static_heat.append(readout_context_values_as_slice(graph, 'total_static_heat'))
            res_dynamic_heat.append(readout_context_values_as_slice(graph, 'total_dynamic_heat'))
            res_vaporiz_heat.append(readout_context_values_as_slice(graph, 'total_vaporization_heat'))

            graph = heat_flux_conduction_step(graph)
            res_heat_flux_conduction.append(readout_edge_values_as_slice(graph, 'conduction', 'heat_flux_conduction'))
            res_delta_t_conduction.append(readout_edge_values_as_slice(graph, 'conduction', 'delta_t_conduction'))

        stack_of_time = keras.layers.Concatenate(axis=0)(res_times)
        stack_of_temperatures = keras.layers.Concatenate(axis=0)(res_temperatures)
        stack_of_power_to_bhx = keras.layers.Concatenate(axis=0)(res_power_to_bhx)
        stack_of_static_heat = keras.layers.Concatenate(axis=0)(res_static_heat)
        stack_of_dynamic_heat = keras.layers.Concatenate(axis=0)(res_dynamic_heat)
        stack_of_vaporization_heat = keras.layers.Concatenate(axis=0)(res_vaporiz_heat)
        stack_of_heat_flux_conduction = keras.layers.Concatenate(axis=0)(res_heat_flux_conduction)
        stack_of_delta_t_conduction = keras.layers.Concatenate(axis=0)(res_delta_t_conduction)

        stack_of_incoming_mass_flow = keras.layers.Concatenate(axis=0)(res_incoming_mass_flow)
        stack_of_evapor_mass_flow = keras.layers.Concatenate(axis=0)(res_evapor_mass_flow)
        stack_of_evapor_mass_flow_per_m =keras.layers.Concatenate(axis=0)(res_evapor_mass_flow_per_m)
        stack_of_fractions_wetted_perimeter =keras.layers.Concatenate(axis=0)(res_fractions_wetted_perimeter)


        return {
            'inputs': input_layer,
            'outputs': self.OUTPUT_NT_TYPE(stack_of_time,
                                           stack_of_temperatures,
                                           stack_of_power_to_bhx,
                                           stack_of_static_heat,
                                           stack_of_dynamic_heat,
                                           stack_of_vaporization_heat,
                                           stack_of_heat_flux_conduction,
                                           stack_of_delta_t_conduction,
                                           stack_of_incoming_mass_flow,
                                           stack_of_evapor_mass_flow,
                                           stack_of_evapor_mass_flow_per_m,
                                           stack_of_fractions_wetted_perimeter),
        }


    def list_output_names(self):
        return self.OUTPUT_NT_TYPE._fields

