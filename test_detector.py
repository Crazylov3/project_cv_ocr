import yaml
import torch
from models.detector.base_model import BaseModel

if __name__ == "__main__":
    path2config = "./configs/det/db.yaml"
    config = yaml.load(open(path2config, "rb"), Loader=yaml.FullLoader)
    model = BaseModel(config)
    inp = torch.randn(1, 3, 512, 512)
    out = model(inp)
    print(out.shape)
