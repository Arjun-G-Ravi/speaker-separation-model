from src.afrcnn import AFRCNN
from src.conv_tasnet import ConvTasNet


def build_model(config):
    options = dict(config)
    name = options.pop("name")
    if name == "conv_tasnet":
        return ConvTasNet(**options)
    if name == "afrcnn":
        return AFRCNN(**options)
    raise ValueError("Unknown model name; supported: conv_tasnet, afrcnn")
