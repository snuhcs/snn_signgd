from .unary import Neuron as UnaryNeuron
from .binary import NeuronWrapper as BinaryNeuron
from .unary import Codec

import torch
from torch import nn
from ...psychoactive_substance import stimulant, depressant
from munch import Munch
from tqdm import tqdm
import time
from snn_signgd.pretty_printer import print
from snn_signgd.system_optimizations import reduce_duplicates

def correction(net:nn.Module, sample_data: torch.Tensor):
    '''
    with stimulant(net) as S:
        excited_output = net(S.excite(sample_data[0:1]))
    with depressant(net) as D:
        depressed_output = net(D.depress(sample_data[0:1]))
    
    for m in tqdm(net.modules(), desc = "Dynamics Correction", total = len(list(net.modules()))):
        if hasattr(m, 'activations'):
            del(m.activations)
            del(m.v)
    return None
    '''
    with stimulant(net) as S:
        excited_output = net(S.excite(sample_data[0:1]))
    with depressant(net) as D:
        depressed_output = net(D.depress(sample_data[0:1]))
    
    for m in tqdm(net.modules(), desc = "Dynamics Correction", total = len(list(net.modules()))):
        if hasattr(m, 'activations'):
            offset = m.activations["excited"] - m.activations["depressed"] + 2 * m.activations["depressed"]
            bias = m.activations["depressed"]

            offset = reduce_duplicates(offset[0])
            bias = reduce_duplicates(bias[0])

            del(m.activations)
            del(m.v)
            #m.offset = nn.Parameter(offset)
            #m.bias = nn.Parameter(bias)
            m.offset = offset  
            m.bias = bias

            #print("Offset:", m.offset)
            #print("Bias:", m.bias)
            #m.input_init =  m.activations["excited"] - m.activations["depressed"] # W \cdot 1
            # If initial function value p_0 is different for all input neurons, we need W \cdot p_0

    statistics = Munch(excited = excited_output.clone(), depressed = depressed_output.clone())
    offset = statistics["excited"] - statistics["depressed"] + 2 * statistics["depressed"]
    bias = statistics["depressed"]

    offset = reduce_duplicates(offset[0])
    bias = reduce_duplicates(bias[0])

    del(statistics)

    #print("Codec Offset:", str(offset).replace("<", ""))
    #print("Codec Bias:", str(bias).replace("<", ""))
    return Munch(offset = offset, bias = bias, input_init = None)
