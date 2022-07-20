from .db_post_process import DBPostProcess


def build_post_process(config):
    module_name = config.pop("name")
    module_class = eval(module_name)(**config)
    return module_class
