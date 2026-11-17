from src.afrcnn import AFRCNN
from src.conv_tasnet import ConvTasNet
from src.earlier_speaker_split import EarlySplitAFRCNN


def build_model(config):
    options = dict(config)
    name = options.pop("name")
    if name == "conv_tasnet":
        return ConvTasNet(**options)
    if name == "afrcnn":
        return AFRCNN(**options)
    if name == "early_split_afrcnn":
        return EarlySplitAFRCNN(**options)
    raise ValueError("Unknown model name; supported: conv_tasnet, afrcnn, early_split_afrcnn")
