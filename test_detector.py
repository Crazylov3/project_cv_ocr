import yaml
# import torch
import paddle
from models.det_module.base_model import BaseModel
from models.det_module.post_process import build_post_process

if __name__ == "__main__":
    path2config = "./configs/det/db.yaml"
    config = yaml.load(open(path2config, "rb"), Loader=yaml.Loader)
    model = BaseModel(config["Architecture"])
    post_processor = build_post_process(config["PostProcess"])
    inp = paddle.randn([1, 3, 224, 224])
    # inp = paddle.tensor(inp)
    out = model(inp)

    # print(type(model))
    print(out["maps"].shape)
    shape_list = [[224, 224, 1, 1]]
    out = post_processor(out, shape_list)
    print(out[0]["points"])
