
import torch
from torch import nn
import torch.nn.functional as F

import os
from typing import Callable, List, Any
from collections import defaultdict
from munch import Munch

import numpy as np

from spikingjelly.activation_based import neuron

import snn_signgd.dictfs as dictfs

from ..neuronal_dynamical_system.spikingjelly.template import BaseNeuron, BaseCodec, BaseNeuronWrapper 
from ..neuronal_dynamical_system.spikingjelly.psychoactive_substance import stimulant, depressant, Psychoactive
from ..neuronal_dynamical_system.spikingjelly.dtype import TensorPair

            
class hook_context(object):
    def __init__(self, hook_fn: Callable, module:nn.Module = None) -> None:
        self.module = module
        self.hook_fn = hook_fn

    def __enter__(self) -> None:
        if self.module is None:
            hook = nn.modules.module.register_module_forward_hook(
                self.hook_fn
            )
        else:
            hook = self.module.register_forward_hook(self.hook_fn)
        self.hooks = [hook]
        return self

    def __exit__(self, exc_type:Any, exc_value:Any, traceback:Any) -> None:
        for hook in self.hooks:
            hook.remove()

def running_average(state, input, timestep):
    return timestep / (1.0 + timestep) * state + 1.0 / (1.0 + timestep) * input

def activation_stats_hook(module, input, output):
    
    if not hasattr(module, 'activation_stats'):
        module.activation_stats = defaultdict(lambda: None)
        module.activation_stats['timestep'] = 0

    #print("Collect Activation Stats:",module.activation_stats)
    
    time_dependent_stats = ['mean', 'square_moment']
    for name, stat_func, update_func in [
            ('max',lambda x: torch.max(x, dim = 0)[0], torch.maximum), 
            ('min',lambda x: torch.min(x, dim = 0)[0], torch.minimum),
            ('mean',lambda x: torch.mean(x, dim = 0), running_average),
            ('square_moment',lambda x: torch.mean(torch.square(x), dim = 0), running_average),
        
        ]:
        for indicator, x in zip(['input', 'output'], [input[0],output]):
            if not isinstance(x, torch.Tensor): continue
                
            x_np = x.detach()
            
            path = os.path.join(indicator,name)
            value = stat_func(x_np)
            
            if module.activation_stats[path] is None:
                module.activation_stats[path] = value
            else:
                if name in time_dependent_stats:
                    module.activation_stats[path] = update_func(
                        module.activation_stats[path], value,
                        timestep = module.activation_stats['timestep']
                    )
                else:
                    module.activation_stats[path] = update_func(
                        module.activation_stats[path], value
                    )
    module.activation_stats['timestep'] += 1
    return

class SNNActHook:
    def __init__(self, define_codec , timestamps, stats_for_input, stats_for_output):
        self.timestamps = timestamps
        self.indicators = ['input_0', 'output']
        self.define_codec = define_codec
        #print("Codec Hook Initialization:", self.indicators)

        self.stats_for_input = stats_for_input
        self.stats_for_output = stats_for_output

        self.measurements = defaultdict(lambda : 0)
        self.measurements['timestep'] = 1
        self.codecs = (
            self.define_codec( # TODO make this codec initialization universal
                statistics = self.stats_for_input
            ),
            self.define_codec(
                statistics = self.stats_for_output
            ),
        )
        self.reset()
            
    def __call__(self, module, input, output):
        #print("Codechook iteration")
        
        measurements = self.measurements
        timestep = measurements['timestep']

        for id, current, codec in zip(self.indicators, [input[0], output], self.codecs):
            
            state_path = os.path.join(id,'state')
            
            spikes = current.detach()
            value = codec.decode(
                state = measurements[state_path], 
                spikes = spikes, timestep = timestep
            )

            measurements[state_path] = value
            #print("Timestep:", timestep, "timestamps:", self.timestamps)

            if timestep in self.timestamps:
                path = os.path.join(id,str(timestep))
                measurements[path] = measurements[state_path].detach() + measurements[path]
                    
        measurements['timestep'] += 1
        
    def reset(self):
        for codec in self.codecs:
            if hasattr(codec, 'reset'):
                codec.reset()


class ANNActHook:
    def __init__(self):
        self.indicators = ['input_0', 'output']
        #print("ANNActHook Initialization:", self.indicators)
        self.reset()

    def reset(self):
        self.measurements = defaultdict(lambda : 0)
        
    def __call__(self, module, input, output):
        #print("ANN hook called")
        measurements = self.measurements
        for id, activation in zip(self.indicators, [input[0], output]):
            value = activation.detach()

            path = str(id)
            measurements[path] = value + measurements[path]

            
class ANNIOCollector:
    def __init__(self, model, target_module):
        self.model = model
        self.module = target_module

    @torch.no_grad()
    def __call__(self, input):
        # do not use train mode here (avoid bn update)
        self.model.eval()

        #print("Prepare codec for:", self.module)
        self.data_saver = ANNActHook()
        
        #print("Get Layer IO")
        with hook_context(hook_fn = self.data_saver, module = self.module) :
            _ = self.model(input)

        print("Measurements:", self.data_saver.measurements.keys())
        return self.data_saver.measurements

class SNNIOCollector:
    def __init__(self, model, target_module, simulation_length:int, define_codec):
        self.model = model
        self.module = target_module
        self.simulation_length = simulation_length
        self.define_codec = define_codec

    @torch.no_grad()
    def __call__(self, input):
        self.model.eval()
        
        net = self.model.model
        collector = ANNIOCollector(model = net, target_module = self.module)
        
        with stimulant(net) as S:
            excited = collector(S.excite(input[0:1]))
        
        collector = ANNIOCollector(model = net, target_module = self.module)
        
        with depressant(net) as D:
            depressed = collector(D.depress(input[0:1]))
        
        statistics_in = Munch(excited = excited['input_0'], depressed = depressed['input_0'])
        statistics_out = Munch(excited = excited['output'], depressed = depressed['output'])
        
        data_saver = SNNActHook(
            define_codec =  self.define_codec,
            timestamps = [self.simulation_length],
            stats_for_input = statistics_in ,
            stats_for_output = statistics_out
        )
        
        with hook_context( hook_fn = data_saver, module = self.module ) :
            _ = self.model(input)
        print("Measurements:", data_saver.measurements.keys())
        output = { 
            key: data_saver.measurements[os.path.join(key, str(self.simulation_length))] 
            for key in data_saver.indicators
        }
        return output

class MicroElectrodeArray:
    def __init__(self, module_name_map, codec_generator, timestamps):
        self.module_name_map = module_name_map
        
        
        snn_hooks = nn.modules.module.register_module_forward_hook(
            self.inject_measurement_tools_to_neuron(
                define_codec = codec_generator, 
                timestamps = timestamps,
            )
        )    
    
        ann_hooks = nn.modules.module.register_module_forward_hook(
            self.measure_ann_activations()
        )
        self.hooks = [snn_hooks, ann_hooks]
        self.reset()
        
    def reset(self):
        self.timestep = 1
        self.measurements = dictfs.DictFS()
        self.codecs = {}

    def __del__(self):
        for hook in self.hooks:
            hook.remove()
            
    def measure_ann_activations(device, label = 'ann'):
        def monitor_hook(module, input, output):
            if module in device.module_name_map and not (isinstance(module, BaseNeuron) or isinstance(module, BaseNeuronWrapper)): 
                
                module_name = device.module_name_map[module]

                #if module_name == "encoder.layers.encoder_layer_3.mlp.1":
                print( "Names:", label, module_name, "output:", output.shape, "input:", input[0].shape)
                for id, activation in zip( ["input_" + str(i) for i in range(1, len(input) + 1)] + ["output"], list(input) + [output]):
                    device.measurements[os.path.join(module_name, str(id), label)] = activation.detach().cpu().numpy()
                #exit()

                return
        return monitor_hook
        
    def inject_measurement_tools_to_neuron(device, define_codec, timestamps, label = 'snn'):
        def monitor_hook(module, input, output):
            #print("Module hook register:", module.__class__.__name__, isinstance(module, BaseNeuron), isinstance(module, BaseNeuronWrapper))
            if module in device.module_name_map and (isinstance(module, BaseNeuron) or isinstance(module, BaseNeuronWrapper)): 
                
                module_name = device.module_name_map[module]
                timestep_path = os.path.join(module_name, label, 'timestep')

                if not device.measurements.exists(timestep_path): 
                    
                    '''
                    device.codecs[module] = (
                        define_codec( # TODO make this codec initialization universal
                            statistics = (module.activations if hasattr(module, 'activations') else None)
                        ),
                        define_codec(),
                    )
                    '''
                    device.codecs[module_name] = (
                        define_codec(),
                    )
                    device.measurements[timestep_path] = 1
                
                timestep = device.measurements[timestep_path]
        
                for id, current, codec in zip(["output"], [output], device.codecs[module_name]):
                #for id, current, codec in zip(indicators, [output], device.codecs[module_name]):
                    state_path = os.path.join(module_name, id, label, 'state')

                    if device.measurements.exists(state_path):
                        state = device.measurements[state_path]
                    else:
                        state = 0.0

                    value = codec.decode(
                        state = state, 
                        spikes = current.detach(), 
                        timestep = timestep
                    )
    
                    device.measurements[state_path] = value
    
                    if timestep in timestamps:
                        device.measurements[os.path.join(module_name, id,label,str(timestep))] = \
                            device.measurements[state_path].detach().cpu().numpy()
                    
                        
                device.measurements[timestep_path] += 1
                        
        return monitor_hook