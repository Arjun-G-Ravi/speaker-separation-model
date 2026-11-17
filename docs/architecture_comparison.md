# Speech separation architecture comparison

Recorded on 2026-11-15 for this project's offline, mono, three-speaker separation task.

## Recommendation

First evaluate Conv-TasNet with the newly implemented fixed crops, dynamic mixing,
and scale-sensitive loss. Then test **TDANet** as the next architecture if attention
is acceptable. If the separator must be entirely convolutional, test **improved
SuDoRM-RF**. Compare a pretrained three-speaker **SepFormer** when maximum quality
is the priority and greater compute cost is acceptable.

These are experiment priorities, not a verified ranking on this project's data.
No alternative architecture has been trained or evaluated here yet.

## Architecture choices

| Model | Structure | Reason to consider it | Main qualification |
| --- | --- | --- | --- |
| Conv-TasNet | Learned waveform encoder/decoder and dilated temporal convolutions | Existing implementation and a useful baseline | Current checkpoint performance does not establish the architecture's ceiling |
| Improved SuDoRM-RF | Multiscale convolutional separator with repeated downsampling, upsampling, and feature fusion | First candidate under a strict CNN-only requirement | Three-speaker gains must be measured; available two-speaker weights are not a direct replacement |
| TDANet | Multiscale convolutional encoder/decoder with global and local attention | Attractive quality/compute compromise | Its global module includes Transformer self-attention; it is not a pure CNN |
| SepFormer | Dual-path Transformer separator | Strong pretrained comparison and potential fine-tuning starting point | Higher compute requirements and possible dataset mismatch |

SuDoRM-RF processes speech at multiple temporal resolutions, combining broader
patterns with fine detail. Its improved reference implementation supports a
configurable source count. An initial three-output experiment could use its
reference settings of 16 U-convolution blocks, depth 4, and 512 encoder features.
This configuration is a starting point, not a tuned recipe for Libri3Mix.
[Paper](https://arxiv.org/abs/2103.02644),
[reference implementation](https://raw.githubusercontent.com/etzinis/sudo_rm_rf/master/sudo_rm_rf/dnn/models/improved_sudormrf.py).

## Published two-speaker comparison

The following results are reported in the TDANet paper's Table 2 for **Libri2Mix**.
Higher SI-SNR improvement indicates better separation relative to the mixture.

| Model | SI-SNR improvement | Parameters |
| --- | ---: | ---: |
| Conv-TasNet | 12.2 dB | 5.6M |
| SuDoRM-RF 1.0× | 13.5 dB | 2.7M |
| SuDoRM-RF 2.5× | 14.0 dB | 6.4M |
| TDANet | 16.9 dB | 2.3M |
| TDANet Large | 17.4 dB | 2.3M |
| SepFormer | 16.5 dB | 26M |

The comparison combines different training recipes, so it is not a controlled
architecture ablation. The SuDoRM-RF rows concern the original variants, not the
improved implementation recommended above. These scores are not predictions for
our three-speaker task. TDANet Large retains the parameter count while increasing
compute through different encoder/decoder kernel and stride settings.
[TDANet paper, Table 2](https://arxiv.org/html/2209.15200).

In its separate LRS2-2Mix timing experiment, the paper reports 4.7 versus 86.9
GMAC/s and CPU processing ratios of 0.79 versus 7.55 for TDANet and SepFormer,
respectively. That is approximately **18× fewer operations** and **9.6× faster CPU
inference** for TDANet in that setup. These are not measurements on this project's
hardware or three-speaker data. Parameter count alone does not determine speed.
[TDANet paper, Table 3 and evaluation protocol](https://arxiv.org/html/2209.15200).

## Comparison with our current checkpoint

Our November 14 Conv-TasNet checkpoint, selected at epoch 65, was evaluated on all
500 recordings in the selected three-speaker test subset:

| Measurement | Result |
| --- | ---: |
| Mean SI-SDR | 5.71 dB |
| Mean SI-SDR improvement over mixture | 9.10 dB |
| Recordings with at least one source below 0 dB SI-SDR | 149 / 500 (29.8%) |

These are measurements of the checkpoint trained **before** the new crop, mixing,
and scale-loss changes. Its training subset contains about 13.9 hours of fixed
mixtures from 251 speakers. The changes require fresh training before their effect
can be assessed. [Local evaluation results](critiques/separation_quality_metrics.json).

The original LibriMix paper reports **12.1 dB SI-SDR improvement** for Conv-TasNet
on clean three-speaker, 8 kHz/min data, trained using train-360. The broader training
data and full benchmark differ from our selected subsets. This suggests room to
improve the current model's training; it does not guarantee a 3 dB gain from our
changes. [LibriMix paper, Table 4](https://arxiv.org/html/2005.11262v1).

For a three-speaker Transformer reference, SpeechBrain's available SepFormer
checkpoint reports **19.8 dB SI-SNR improvement on WSJ0-3Mix**. This demonstrates
three-speaker capability but cannot be directly compared with our Libri3Mix score.
Its expected input is mono 8 kHz audio.
[SepFormer WSJ0-3Mix model card](https://huggingface.co/speechbrain/sepformer-wsj03mix).

## How to decide for this project

1. Train the updated Conv-TasNet recipe and establish its validation performance.
2. Compare three-output TDANet, or improved SuDoRM-RF if CNN-only is required, using
   the same data pipeline, loss, and a declared optimizer-step or GPU-time budget.
3. Evaluate a pretrained three-speaker SepFormer on the same recordings, recording
   that its pretraining data and compute budget differ.
4. Report mean SI-SDR/SI-SDRi, the worst-speaker distribution, failure rate, listening
   quality, inference time, and peak memory. Use consistent playback scaling.
5. Select configurations using validation data and reserve test evaluation for
   final reporting.

My expectation is that SuDoRM-RF offers an incremental CNN upgrade, while TDANet
offers a stronger quality/compute compromise. A well-trained Transformer is worth
testing for maximum quality. Neither greater size nor an architecture change alone
guarantees better output with the available data.
