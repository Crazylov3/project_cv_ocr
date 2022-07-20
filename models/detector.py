from .det_module.post_process import build_post_process
from .det_module import build_model as build_det_model
import PIL.Image as Image
import paddle
from utils.generals import crop_sub_image
from copy import deepcopy
from utils.generals import image_transform as default_image_transform


class Detector:
    def __init__(self, config, transform=None):
        self.config = config
        self.model = build_det_model(config["Architecture"])
        self.post_processor = build_post_process(config["PostProcess"])
        self.transform = transform if transform is not None else default_image_transform

    def __call__(self, inp):
        assert isinstance(inp, Image.Image)
        w, h = inp.width, inp.height
        # TODO: correct shape_list
        img = self.transform(inp) if self.transform is not None else inp
        img = img.unsqueeze(0)
        img = paddle.to_tensor(img.numpy())
        out = self.model(img)
        out = self.post_processor(out, [[w, h, 1, 1]])
        return out[0]["points"]

    @staticmethod
    def post_process(img, bboxes):
        sub_imgs = []
        for bbox in bboxes:
            sub_img = deepcopy(crop_sub_image(img, *bbox))
            sub_imgs.append(sub_img)
        return sub_imgs
