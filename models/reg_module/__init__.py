from .base_model import VietOCR
from .vocab.vocab import Vocab


def build_model(config):
    vocab = Vocab(config['vocab'])
    device = config['device']
    model = VietOCR(len(vocab), config['backbone'], config['cnn'], config['transformer'], config['seq_modeling'])
    model = model.to(device)
    return model, vocab
