import torch
from torch import nn, fx

from typing import Callable, Union, Tuple, Iterable, Dict, Any, Type

import importlib.util
import sys, os
import uuid, re
import types
from collections import defaultdict

def repeat_shape_base(x:torch.Tensor):
    dimension = len(x.shape)
    return [1 for _ in range(dimension)]

def import_module_from_path(path:Union[str, os.PathLike], module_name:str):
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(path, "__init__.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, module_name)

def save_and_reimport_nn_module(model: fx.GraphModule, temporary_directory:str = "tmp"):
    model_name = str(uuid.uuid4())
    model_name = re.sub("^\d+", "", model_name.replace("-",""))
    model.to_folder(temporary_directory, model_name)

    return import_module_from_path(path = temporary_directory, module_name = model_name)

def keyword_matching(dict1, dict2, str_to_keywords:Callable):
    mapping = defaultdict(list)
    for key1 in dict1:
        keywords = str_to_keywords(key1) 
        keywords = list(set(keywords)) # Unique keywords
        
        for key2 in dict2:
            keywords_dst = str_to_keywords(key2)
            
            if all([keyword in keywords_dst for keyword in keywords]):
                mapping[key1].append(key2)
                
    return mapping