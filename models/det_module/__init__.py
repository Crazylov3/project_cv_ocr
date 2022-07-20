from .base_model import BaseModel


def build_model(config):
    model = BaseModel(config)
    return model
