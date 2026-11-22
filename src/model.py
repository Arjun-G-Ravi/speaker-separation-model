from src.afrcnn import AFRCNN
from src.conv_tasnet import ConvTasNet
from src.earlier_speaker_split import EarlySplitAFRCNN
from src.pilot1_cross_speaker.model import CrossSpeakerAFRCNN
from src.pilot1_coarse_to_fine.model import CoarseToFineAFRCNN


def build_model(config):
    options = dict(config)
    name = options.pop("name")
    if name == "conv_tasnet":
        return ConvTasNet(**options)
    if name == "afrcnn":
        return AFRCNN(**options)
    if name == "early_split_afrcnn":
        return EarlySplitAFRCNN(**options)
    if name == "pilot1_cross_speaker":
        return CrossSpeakerAFRCNN(**options)
    if name == "pilot1_coarse_to_fine":
        return CoarseToFineAFRCNN(**options)
    raise ValueError("Unknown model name; supported: conv_tasnet, afrcnn, early_split_afrcnn, "
                     "pilot1_cross_speaker, pilot1_coarse_to_fine")
