from .db_fpn import DBFPN


def build_neck(config):
    module_name = config.pop("name")
    module_class = eval(module_name)(**config)
    return module_class
