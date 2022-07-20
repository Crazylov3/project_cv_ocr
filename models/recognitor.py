from .reg_module import build_model as build_reg_model
from collections import defaultdict
import torch
import numpy as np
import math
from PIL import Image


def resize(w, h, expected_height, image_min_width, image_max_width):
    new_w = int(expected_height * float(w) / float(h))
    round_to = 10
    new_w = math.ceil(new_w / round_to) * round_to
    new_w = max(new_w, image_min_width)
    new_w = min(new_w, image_max_width)

    return new_w, expected_height


def process_image(image, image_height, image_min_width, image_max_width):
    # img = image.convert('RGB')
    img = image

    w, h = img.size
    new_w, image_height = resize(w, h, image_height, image_min_width, image_max_width)

    img = img.resize((new_w, image_height), Image.ANTIALIAS)

    img = np.asarray(img).transpose(2, 0, 1)
    img = img / 255
    return img


def process_input(image, image_height, image_min_width, image_max_width):
    img = process_image(image, image_height, image_min_width, image_max_width)
    img = img[np.newaxis, ...]
    img = torch.FloatTensor(img)
    return img


def translate(img, model, max_seq_length=128, sos_token=1, eos_token=2):
    "data: BxCXHxW"
    model.eval()
    device = img.device

    with torch.no_grad():
        src = model.cnn(img)
        memory = model.transformer.forward_encoder(src)

        translated_sentence = [[sos_token] * len(img)]
        char_probs = [[1] * len(img)]

        max_length = 0

        while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T == eos_token, axis=1)):
            tgt_inp = torch.LongTensor(translated_sentence).to(device)

            #            output = model(img, tgt_inp, tgt_key_padding_mask=None)
            #            output = model.transformer(src, tgt_inp, tgt_key_padding_mask=None)
            output, memory = model.transformer.forward_decoder(tgt_inp, memory)
            output = torch.softmax(output, dim=-1)
            output = output.to('cpu')

            values, indices = torch.topk(output, 5)

            indices = indices[:, -1, 0]
            indices = indices.tolist()

            values = values[:, -1, 0]
            values = values.tolist()
            char_probs.append(values)

            translated_sentence.append(indices)
            max_length += 1

            del output

        translated_sentence = np.asarray(translated_sentence).T

        char_probs = np.asarray(char_probs).T
        char_probs = np.multiply(char_probs, translated_sentence > 3)
        char_probs = np.sum(char_probs, axis=-1) / (char_probs > 0).sum(-1)

    return translated_sentence, char_probs


class Predictor:
    def __init__(self, config):
        device = config['device']
        model, vocab = build_reg_model(config)
        weights = config['pretrained_path']
        # model.load_state_dict(torch.load(weights, map_location=torch.device(device)))
        self.config = config
        self.model = model
        self.vocab = vocab
        self.device = device

    def __call__(self, imgs, return_prob=False):
        return self.predict_batch(imgs, return_prob)

    def predict_batch(self, imgs, return_prob=False):
        bucket = defaultdict(list)
        bucket_idx = defaultdict(list)
        bucket_pred = {}
        sents, probs = [0] * len(imgs), [0] * len(imgs)
        for i, img in enumerate(imgs):
            img = process_input(img, self.config['dataset']['image_height'],
                                self.config['dataset']['image_min_width'], self.config['dataset']['image_max_width'])
            bucket[img.shape[-1]].append(img)
            bucket_idx[img.shape[-1]].append(i)

        for k, batch in bucket.items():
            batch = torch.cat(batch, 0).to(self.device)
            s, prob = translate(batch, self.model)
            prob = prob.tolist()

            s = s.tolist()
            s = self.vocab.batch_decode(s)

            bucket_pred[k] = (s, prob)

        for k in bucket_pred:
            idx = bucket_idx[k]
            sent, prob = bucket_pred[k]
            for i, j in enumerate(idx):
                sents[j] = sent[i]
                probs[j] = prob[i]

        if return_prob:
            return sents, probs
        else:
            return sents
