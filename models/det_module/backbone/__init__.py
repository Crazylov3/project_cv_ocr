from .mobilenet_v3 import *


def build_backbone(config):
    module_name = config.pop("name")
    module_class = eval(module_name)(**config)
    return module_class
