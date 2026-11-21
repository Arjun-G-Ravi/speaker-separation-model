Not for any ai agents to touch.

# User notes

So I have to perform decently on libre2mix and libre3mix datasets. I have currently taken a small subset of 5000 train + 500 val + 500 test for both of the dataset. I Ideally have to use a CNN based model - so we'll try to make the best possible CNN based model for the task

# Best CNN based models
Strongest CNN-based models to study
Here are established published baselines on Libri2Mix.
Model	SI-SNRi (dB)	Parameters	CNN-only?
Conv-TasNet	12.2	~5.6M	Yes
SuDoRM-RF 1.0x	13.5	2.7M	Yes
SuDoRM-RF 2.5x	14.0	6.4M	Yes
A-FRCNN-16	16.7	6.1M	Yes
TDANet Large	17.4	Variant-dependent	No, includes attention

The CNN-only figures are reported in the NeurIPS 2021 A-FRCNN comparison. TDANet's ICLR 2023 architecture explicitly includes multi-head self-attention, so it should not be classified as purely convolutiona


So, we need to make a model better than A-FRCNN-16's 16.7 score. This have to be done with a subset of data used in the original thing(16hrs vs 56 hrs). We just have a 3090.

# First and Second Attempts

| Measurement | First model: Conv-TasNet | Current model: A-FRCNN-16 |
|---|---:|---:|
| Dataset | Libri3Mix — 3 speakers | Libri2Mix — 2 speakers |
| Best validation SI-SDR | 6.05 dB* | 10.40 dB |
| Test SI-SDR | 5.71 dB | 10.20 dB |
| Test SI-SDR improvement | 9.10 dB | 10.20 dB |
| Best checkpoint | Epoch 65 | Epoch 69 |
| Completed epochs | 74 logged | 79 |
| Recorded training time | 10 h 21 min | 8 h 37 min |

*6.05 dB was the first model’s final logged validation score, not its best-checkpoint score.

The first model struggled with separation: 149 of 500 test mixtures had at least one speaker below 0 dB SI-SDR. The current model’s validation and test scores are close, which is encouraging for generalization. However, separating two speakers is easier than separating three, and the current run also uses a different crop, mixing, and loss recipe.

Compared with the original A-FRCNN paper, your current result remains substantially lower: approximately 10.20 dB SI-SDR improvement versus 16.7 dB published SI-SNR improvement, a roughly 6.5 dB gap. These metrics are comparable, but the evaluation protocols are not identical. The paper used 58 hours of training mixtures and 200 epochs, whereas your prepared subset contains 15.58 hours and training stopped after 79 epochs; your loss, scheduler, dynamic mixing, and selected test subset also differ. More data and a closer reproduction of the training recipe are reasonable next steps, but we haven’t established which difference explains the gap. A-FRCNN paper, Sections 4–5 and Table 3.

Conv-TasNet, in three sentences: A learned convolutional encoder converts the waveform into audio features. Stacked dilated convolution blocks gather context over increasingly wide time ranges and predict a feature mask for each speaker. A learned decoder converts each masked representation back into a waveform.

A-FRCNN-16, in three sentences: A learned encoder converts the waveform into audio features at multiple temporal resolutions. A shared convolutional block repeatedly exchanges and combines information across those resolutions, refining the representation over 16 iterations. Speaker masks and a learned decoder then reconstruct the separated waveforms.

# next run
got to 13.03 or something. 
