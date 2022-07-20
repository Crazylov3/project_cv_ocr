import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as tf
from copy import deepcopy


def crop_sub_image(image, xy1, xy2, xy3, xy4):
    """
    Crop by 4 corner of image
    """
    top_left_x = min([xy1[0], xy2[0], xy3[0], xy4[0]])
    top_left_y = min([xy1[1], xy2[1], xy3[1], xy4[1]])
    bot_right_x = max([xy1[0], xy2[0], xy3[0], xy4[0]])
    bot_right_y = max([xy1[1], xy2[1], xy3[1], xy4[1]])
    image = np.asarray(image)
    img = image[top_left_y:bot_right_y + 1, top_left_x:bot_right_x + 1]
    return img


def draw_bbox_with_text(image, bbox, text, color, font_size=None, font_path=None):
    """
    Draw a bounding box with text on an image.
    """
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox, outline=color)
    draw.text(bbox[:2], text, fill=color, font=font_path)
    return image


def image_transform(image):
    ops = tf.Compose([
        tf.Resize(size=(224, 224)),
        tf.ToTensor(),
        tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return ops(image)