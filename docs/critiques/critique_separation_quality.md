# Critique: Three-speaker separation quality

Reviewed 2026-11-15. This is a code, artifact, waveform, and literature review. No model, training configuration, dataset, or Git state was changed. No dependencies, weights, or datasets were downloaded. Diagnostic scripts ran with the existing `/home/arjun/ai_env/bin/python` on CPU. The exact audio that prompted the complaint has not been identified; both available real-data runs were inspected.

## 1. What the idea actually is

A roughly 5.12-million-parameter, noncausal Conv-TasNet takes mono 8 kHz audio and always emits three waveforms. Training uses utterance-level permutation-invariant negative SI-SDR on a fixed clean Libri3Mix subset. The UI also accepts arbitrary uploaded recordings.

That is a reasonable narrow benchmark baseline. The upload interface covers a much wider acoustic problem than the training distribution.

## 2. Problem being solved

Recover intelligible individual speakers without excessive interference, missing speech, distortion, or unstable identity. Clean three-speaker mixtures, noisy conversations with varying speaker counts, and extracting one known person are different tasks. They need different training coverage and sometimes different model interfaces.

Success should mean better held-out separation **and** better listening quality on representative recordings. A rising training SI-SDR is insufficient.

## 3. Initial verdict

**Decent baseline; inadequate training coverage and output calibration for dependable inference.** There are concrete pipeline issues worth addressing before attributing every failure to Conv-TasNet.

| Local evidence | November 11 run | November 14 run |
|---|---:|---:|
| Training mixtures | 500 | 5,000 |
| Actual stored mixture duration | 1.381 hours | 13.932 hours |
| Training speakers | 251 | 251 |
| Best validation SI-SDR | 0.400 dB | 6.055 dB |
| Selected epoch | 31 | 65 |
| Latest logged training SI-SDR | 6.674 dB | 10.474 dB |
| Latest logged validation SI-SDR | 0.325 dB | 6.048 dB |
| Logged epochs | 41 | 74 |

The validation sets differ in size, so this is not a controlled estimate of the effect of adding data. Nevertheless, the old checkpoint is demonstrably weak. Its existing 50-example test assessment reports 0.588 dB SI-SDR, 3.983 dB SI-SDR improvement, and 65 of 150 individual sources below 0 dB.

For the newer run, validation plateaus while training improves. This is consistent with overfitting and distribution mismatch. The train/validation gap is not a pure overfitting measurement: training uses random crops, padding, and AMP; validation uses unpadded full recordings and float32. The newer run has a recovery checkpoint and no completion result, so the available record shows an unfinished run, not a successfully completed 100-epoch run.

Fresh evaluation of the newer epoch-65 checkpoint on **all 500 held-out test recordings**, using CPU float32, complete recordings, and utterance PIT:

| Metric | Result |
|---|---:|
| Mean SI-SDR | **5.707 dB** |
| Mean SI-SDR improvement over mixture | **9.095 dB** |
| Median utterance SI-SDR | 6.619 dB |
| 10th-percentile utterance SI-SDR | 1.157 dB |
| 90th-percentile utterance SI-SDR | 9.380 dB |
| Recordings with at least one speaker below 0 dB SI-SDR | **149 / 500 (29.8%)** |
| Individual speakers below 0 dB | 202 / 1,500 (13.5%) |
| Recordings whose mean SI-SDR is negative | 31 / 500 |
| Recordings whose mean score is worse than the unseparated mixture | 4 / 500 |

The below-zero threshold is a diagnostic marker of substantial interference/distortion, not a calibrated human-listening boundary. The result explains why the average can improve while many individual tracks remain poor. These are results for the selected subset, not the full official Libri3Mix benchmark. The evaluation checked the checkpoint's test manifest hash and took 237 seconds. Per-example results and checkpoint SHA-256 are saved in [separation_quality_metrics.json](separation_quality_metrics.json).

## 4. What is good/decent

- Core optimization order is sensible: train/eval modes, gradient scaling, unscale before clipping, optimizer step, validation, scheduler, and checkpoint selection are present.
- PIT enumerates all six permutations for three speakers, and the loss correctly excludes padded samples. The inspection did not reveal an obvious sign or permutation bug.
- Data preparation checks alignment, sample rates, source energy, speaker split separation, and mixture/source summation. Three newly checked validation examples matched their source sum to within 0.0000611 amplitude, consistent with PCM16 quantization.
- Checkpoints retain configuration and manifest identities; inference restores the checkpoint's settings.
- Current WAV export applies a shared attenuation when necessary. Its existing regression check passed during this review.
- Conv-TasNet remains a useful baseline. Its dimensions and ReLU-mask option closely resemble established implementations. ReLU masks alone are not evidence of a faulty architecture. [Original paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6726126/), [Asteroid configuration](https://raw.githubusercontent.com/asteroid-team/asteroid/master/egs/librimix/ConvTasNet/local/conf.yml).

## 5. What is weak or risky

### A. The demo selects the old, substantially weaker checkpoint — confirmed

`config/demo.yaml:1` explicitly selects `out/2026-11-11_07-00-31_005338/best.pt`. Changing `config/train.yaml` or creating a newer run does not change that selection. The November 15 inference previews explicitly used the newer checkpoint, so this does not explain every available bad output.

**Action:** explicitly select the intended checkpoint and restart the demo, which loads the model once. Expose its run ID, dataset size, epoch, and validation score together. Do not silently choose by timestamp, because this repository also contains synthetic smoke-test checkpoints.

### B. Output amplitude is uncontrolled — confirmed; current export already mitigates clipping

`src/loss.py:6` optimizes a scale-invariant objective. It does not require the waveform to have the reference's gain or polarity. The model also has an unconstrained decoder and nonnegative, unbounded masks (`src/model.py:56`).

The three newer saved test previews have raw peaks of **562.34, 695.66, and 1,027.19**. Between about 79% and 94% of their source samples exceed the nominal playback range. Three independent validation probes reproduced peaks of 839–1,199. This is an actual calibration failure, not a hypothetical risk.

Float WAV can store these values. A playback stage that clips at ±1 can destroy the waveform. Existing saved diagnostics show one source falling from 11.25 dB SI-SDR to -5.08 dB under simulated clipping. They do not prove how a particular player behaved. The current `playback_audio` implementation safely attenuates all tracks; it does not repair interference or recover correct relative source gains. The newer attenuated preview still contains individual speakers with negative SI-SDR.

**Action:** retain safe export. Log raw peak/RMS, attenuation, and source-sum residual. In a separate training experiment, add a scale-sensitive waveform reconstruction term after consistent PIT alignment, or compare an appropriate scale-dependent SDR objective. A multi-resolution spectral term is another perceptual-quality experiment, not a guaranteed cure. Validate weights and use the same source assignment across losses. Do not bolt an uncalibrated mixture-consistency correction onto outputs hundreds of times too large: first constrain/calibrate gains. For noisy input, clean speech outputs should not be forced to sum to speech plus noise without a residual/noise output. [SI-SDR definition](https://arxiv.org/abs/1811.02508), [consistency constraints](https://arxiv.org/abs/1811.08521).

### C. Variable-length padding changes the model's normalization — reproduced

`src/data.py:48` pads each batch to its longest example. `src/model.py:12` computes global statistics across all channels and time, including this padding. Lengths reach the loss, but not the model. Biases and intermediate activations also mean waveform zero padding is not harmless throughout the separator.

Training has batch size four; validation has batch size one. In the larger training set, **43.74%** of examples are shorter than the 10-second crop cap. A short clip's training prediction can therefore depend on the duration of other examples in its batch through its own padded length; normalization does not mix different examples' actual values.

Fresh checkpoint probe, scoring only the original samples:

| Clip duration | Unpadded SI-SDR | SI-SDR after padding to 10 seconds | Difference |
|---|---:|---:|---:|
| 3.005 seconds | 10.376 | 9.453 | -0.923 dB |
| 4.335 seconds | 3.136 | 3.039 | -0.097 dB |
| 10.755 seconds | 10.200 | 10.200 | 0 dB; unchanged control |

This demonstrates padding sensitivity. Three probes do not establish its average contribution to poor quality or the improvement from retraining.

**Action:** start with a controlled **3-second crop experiment**, since all current training mixtures are at least three seconds. Ensure selected crops contain usable reference speech. This avoids variable batch padding and increases crop diversity while keeping batch size, optimizer, learning rate, and precision fixed for the comparison. Compare equal optimizer-step or GPU-time budgets and evaluate full utterances. Longer-term alternatives are duration bucketing (reduces, but does not eliminate padding) or length-aware normalization and intermediate masking (more invasive). Simply masking the final loss again will not solve it. The published LibriMix baseline uses 3-second segments. [LibriMix paper](https://arxiv.org/html/2005.11262v1), [sampling study](https://arxiv.org/abs/2304.07142).

### D. The model sees fixed mixtures with limited diversity — confirmed; likely high impact

`src/data.py:25` reads the same prepared mixture and source triplet each epoch. The only stochastic audio operation is a synchronized crop when a recording exceeds ten seconds. Thus 43.74% of larger-set examples have no crop variation at all. There is no dynamic selection of speakers, relative gain, temporal offset, speed, noise, room response, or channel coloration.

The 5,000 mixtures contain 15,000 unique source utterances but only 251 training speakers. Increasing epochs repeatedly exposes the same interference relationships. The current record already shows training improvement with little validation improvement.

**Action:** implement on-the-fly mixing from training-only sources: select three distinct speakers, choose independent utterances/crops, vary relative loudness, and apply shared final scaling to mixture and targets. Existing `s1/s2/s3` files permit an initial remix experiment without a download, although they are gain-scaled and truncated. The complete `train-clean-100.tar.gz` is already cached locally; extracting its original utterances offers better coverage and permits 16 kHz preparation. Do not claim remixing creates new voices. If speaker diversity remains limiting, consider train-clean-360 as a separately approved data expansion. [SpeechBrain's dynamic mixer](https://raw.githubusercontent.com/speechbrain/speechbrain/develop/recipes/LibriMix/separation/dynamic_mixing.py).

### E. Clean three-source training does not cover arbitrary uploads — confirmed scope mismatch

Preparation produces `8k/min/mix_clean`, truncating at the shortest source. Every example must contain three distinct speakers, and both the dataset and loss reject silent reference sources. There is no intentional coverage of one/two-speaker recordings, long inactive tracks, room reverberation, ambient noise, music, codecs, or real microphone channels.

**Action:** if real recordings are the goal, build a validation suite representing them, then simulate matching room responses, noise, loudness differences, offsets and partial overlap. Choose whether targets are dry speech or reverberant source images before adding augmentation. Mixtures must be constructed consistently with those targets. Supporting 1–3 active speakers requires a silence-aware objective and data changes; padding targets with zeros currently raises an error. An activity head or inactive-source energy penalty can be evaluated. [WHAMR](https://arxiv.org/abs/1910.10279), [SA-SDR](https://arxiv.org/abs/2110.15581), [LibriCSS](https://arxiv.org/abs/2001.11482).

### F. Architecture and bandwidth impose limits — structural facts, not diagnosed sole causes

The dilated convolution stack has nominal temporal support of `1 + 2 × 3 × (2^8 − 1) = 1,531` encoded frames, approximately **1.53 seconds** at a 1 ms encoder hop. Global normalization introduces whole-clip statistical dependence, so it would be incorrect to claim the entire model is strictly local to 1.53 seconds. It nevertheless lacks explicit long-range content interaction or a persistent speaker representation.

Dual-path recurrence/attention and TF-GridNet offer stronger ways to model longer context. This may help similar voices and identity continuity, but the cause of specific speaker swaps has not been measured here. Whole-clip inference avoids inter-chunk assignment changes; it does not guarantee that a learned output never switches speakers internally. [DPRNN](https://arxiv.org/abs/1910.06379), [SepFormer](https://arxiv.org/abs/2010.13154), [Wavesplit](https://arxiv.org/abs/2002.08933).

At 8 kHz, recoverable input bandwidth stops at 4 kHz. Better separation cannot restore the original high-frequency detail that resampling removed. For natural-sounding wideband output, use genuinely 16 kHz sources and a matching model/training recipe; merely saving or upsampling 8 kHz predictions at 16 kHz adds no information. Keeping a 16-sample kernel and 8-sample stride at the higher rate also changes their physical durations, so review those settings. `prepare_data.py` currently explicitly rejects non-8-kHz preparation. Stereo averaging additionally discards spatial cues when channels contain useful microphone differences.

### G. Mean SI-SDR hides individual failures — confirmed evaluation limitation

`src/evaluate.py` records utterance means and SI-SDR improvement, but does not summarize the worst speaker, failure rate, intelligibility, missing words, or listening preference. One saved newer preview averages 2.65 dB while its speakers score **-4.88, 2.33, and 10.50 dB**. A tolerable average does not mean all three outputs are usable.

**Action:** retain mean SI-SDR/SI-SDRi, add per-speaker distributions, median, 10th percentile, any-speaker-failure rate, source activity leakage, and export gain statistics. Add STOI/ESTOI where available and an unchanged ASR system's WER when intelligibility is relevant; keep resampling and metric protocols identical. Use fixed listening comparisons on validation examples with comparable loudness. No-reference quality scores alone cannot establish that the intended speaker was preserved. Synthetic tone overfitting tests establish numerical behavior, not speech quality. [LibriCSS evaluation motivation](https://arxiv.org/abs/2001.11482), [DNSMOS scope](https://arxiv.org/abs/2010.15258).

## 6. Evidence supporting the approach

The original Conv-TasNet demonstrates that this model family can separate three speakers, and the LibriMix paper reports **12.1 dB SI-SDR improvement** on clean three-speaker 8 kHz/min data. That is improvement over the mixture, not absolute SI-SDR. Its training scope and full test set differ from this repository's subsets, so the number is context rather than an apples-to-apples target. [Conv-TasNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC6726126/), [LibriMix Table 4](https://arxiv.org/html/2005.11262v1).

Established libraries support very similar blocks and ReLU masks. PIT is appropriate for unordered sources. These findings argue for retaining the existing model as a measured baseline while improving its training conditions. [Torchaudio reference](https://docs.pytorch.org/audio/2.7.0/_modules/torchaudio/models/conv_tasnet.html), [uPIT](https://arxiv.org/abs/1703.06284).

## 7. Evidence against expecting dependable output from the current setup

Published work identifies noise/reverberation, mismatched domains, silence, and continuous conversation as distinct challenges. Strong results on clean mixtures are not sufficient evidence for robust uploads. Domain adaptation studies also report improvement while holding the architecture fixed, supporting a data-focused experiment before a complete rewrite. [WHAM](https://arxiv.org/abs/1907.01160), [WHAMR](https://arxiv.org/abs/1910.10279), [meeting adaptation](https://arxiv.org/abs/2110.10739), [multi-channel adaptation](https://arxiv.org/abs/2305.11151).

## 8. Similar existing implementations

| Implementation | Use here | Important qualification |
|---|---|---|
| [Asteroid Conv-TasNet/LibriMix](https://github.com/asteroid-team/asteroid/blob/master/egs/librimix/ConvTasNet/local/conf.yml) | Closest reference recipe; isolate recipe vs architecture effects | Configure three sources; the displayed default is two |
| [SpeechBrain LibriMix dynamic mixer](https://github.com/speechbrain/speechbrain/blob/develop/recipes/LibriMix/separation/dynamic_mixing.py) | Reference for speaker selection, crops, loudness and source-consistent scaling | Adapt to this project's config and dependency constraints |
| [Pretrained three-speaker SepFormer](https://huggingface.co/speechbrain/sepformer-wsj03mix) | Fastest meaningful external three-output comparison; possible fine-tuning start | 8 kHz WSJ0-3Mix domain; reported 19.8 dB SI-SNRi is not a Libri3Mix result |
| [ESPnet TF-GridNet](https://github.com/espnet/espnet/blob/master/espnet2/enh/separator/tfgridnet_separator.py) | Strong next offline training candidate with complex spectral estimation | Set three outputs and matching sample-rate/STFT/data recipe; verify memory on the 3090 |
| [ClearerVoice/MossFormer2](https://github.com/modelscope/ClearerVoice-Studio) | Additional practical pretrained comparison for supported tasks | Check checkpoint source count; its 16 kHz model card alone does not establish three-source compatibility |
| [SpeakerBeam](https://github.com/BUTSpeechFIT/speakerbeam) | Better task formulation when one known person's voice is wanted | Requires enrollment audio and extraction training |
| [SVoice](https://github.com/facebookresearch/svoice) | Research reference for multiple/unknown speaker counts | Older dependency stack and multiple count-specific models; not the first integration choice |
| [Google sound-separation](https://github.com/google-research/sound-separation) | MixIT and meeting-domain adaptation references | Archived; adaptation research, not a turnkey replacement |
| [Torchaudio Conv-TasNet](https://docs.pytorch.org/audio/2.7.0/_modules/torchaudio/models/conv_tasnet.html) | Reference implementation for validating model choices | Framework/version compatibility must be checked before adoption |
| [Official LibriMix](https://github.com/JorisCos/LibriMix) | Reference data variants, construction and metadata | Preserve split identity; more data is not automatically matched real-world data |

The basic architecture is already well established. The useful contribution here would be a reproducible, well-tested system adapted to the intended recordings, not another unvalidated implementation of the same baseline.

## 9. How to improve it: ranked experiments

| Order | Experiment | Verification and decision |
|---|---|---|
| 0 | Pin the intended checkpoint and verify safe playback | Confirm displayed run/epoch; export regression passes; no overflow in delivered WAVs |
| 1 | Fixed valid 3-second crop baseline | Keep other hyperparameters fixed; compare full-utterance validation, failure rate, throughput and listening against the current recipe |
| 2 | Dynamic training-only remixing | Hold step/time budget fixed; seek validation and lower-tail improvement, not just higher training score |
| 3 | Separate amplitude-loss ablation | Check raw gain distribution, mixture residual, SI-SDR and listening; reject scale control that harms separation |
| 4 | Match actual acoustics and active-speaker counts | Evaluate each augmentation family on a corresponding held-out validation condition; use a silence-aware loss before empty targets |
| 5 | Pretrained three-speaker SepFormer comparison | Same recordings and scoring; compare domain transfer, latency and memory before deciding to fine-tune |
| 6 | Three-output TF-GridNet at 16 kHz | Prepare genuine wideband data; compare against a matched-rate baseline, not against only the old narrowband checkpoint |

Do one substantive change per experiment. Keep W&B logging, fixed validation IDs, representative audio snapshots, and a record of data/model/precision settings. Freeze choices using validation; reserve test results for final reporting. The test audit in this review diagnoses the existing frozen checkpoint and should not become a per-experiment tuning target.

Do not first increase model width, run many more epochs, change all optimizer settings, append a denoiser, or switch masks to softmax. None has evidence here of being the dominant cause; a denoiser cannot reconstruct a discarded speaker, and a softmax over latent masks does not guarantee waveform mixture consistency with an unconstrained encoder/decoder.

## 10. Better alternate plans

**Fastest route to a usable quality comparison:** keep the current model as baseline, compare the available three-speaker SepFormer, and fine-tune the better starting point on dynamically mixed, matched-domain data. No weights were downloaded during this review.

**Best next architecture experiment for offline quality:** TF-GridNet with three outputs, a genuine 16 kHz dataset, and a trusted training recipe. Its complex spectral mapping and time/frequency context are attractive; its reported 23.5 dB WSJ0-2mix SI-SDRi is a two-speaker result and must not be presented as a three-speaker prediction. [TF-GridNet](https://arxiv.org/abs/2211.12433).

**If training resources constrain the experiment:** DPRNN is a reasonable intermediate architecture before a large attention model. Actual memory and speed depend on the recipe and segment length; parameter count alone is insufficient. [DPRNN](https://arxiv.org/abs/1910.06379).

**If the user wants one person:** use an enrollment-conditioned target-speaker extractor such as SpeakerBeam/SpEx+. This makes target identity explicit, but changes the task. [SpeakerBeam explanation](https://www.ntt-review.jp/archive/ntttechnical.php?contents=ntr202109fa4.html), [SpEx+](https://arxiv.org/abs/2005.04686).

**If the user wants meetings:** use continuous separation with activity/silence support and explicit stream assignment. Graph-PIT addresses the distinction between total speakers and simultaneous speakers; it does not itself produce persistent identity tracks for an unlimited number of people. [Graph-PIT](https://arxiv.org/abs/2107.14446).

**If only unlabeled real recordings are available:** investigate MixIT adaptation after establishing a reliable supervised baseline. RemixIT is principally an enhancement/self-training reference, not evidence of a ready-made three-speaker solution. [MixIT](https://arxiv.org/abs/2006.12701), [RemixIT](https://arxiv.org/abs/2202.08862).

## 11. Questions that determine the next experiment

1. Did the complaint concern the older demo checkpoint, raw earlier exports, or the newer attenuated previews?
2. Must every recording contain exactly three people, or should one/two-speaker and inactive periods work?
3. Are target recordings clean synthetic mixtures, phone audio, interviews, meetings, or media with music?
4. Is the objective pleasant listening, accurate transcription, or extraction of one enrolled person?
5. Is offline processing acceptable, and what latency/memory budget should an alternative meet?

These affect priorities; they do not invalidate the confirmed amplitude, padding, and checkpoint findings.

## 12. Final recommendation

**Build a small, measured improvement first.** Correct checkpoint selection, retain safe export, run the crop and dynamic-mixing ablations, and compare a pretrained three-speaker SepFormer. For a larger quality investment, train or fine-tune a three-output TF-GridNet with matched wideband data. Improve the evaluation of individual speaker failures alongside the model.

## 13. Source log

Sources were searched and inspected as primary research, author repositories, model cards, or official documentation. Abstracts establish scope; claims about specific implementation details or benchmark numbers were checked in the corresponding paper text, code or model card. Separate URLs for the same paper or the raw and HTML form of the same file are counted once. The 25 evidence pages were reviewed before the separate implementation search. This is a targeted comparison, not an exhaustive current leaderboard claim.

### Evidence sources: 25 distinct works/pages

| ID | Title and URL | Domain | Classification | Relevance |
|---|---|---|---|---|
| E01 | [Conv-TasNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC6726126/) | pmc.ncbi.nlm.nih.gov | Supports; background | Validates baseline family; mask alternatives and three-speaker experiment |
| E02 | [LibriMix](https://arxiv.org/html/2005.11262v1) | arxiv.org | Mixed evidence | Same benchmark family, 3-second recipe, generalization and sparse overlap |
| E03 | [TF-GridNet](https://arxiv.org/abs/2211.12433) | arxiv.org | Opposes architecture ceiling; alternative | Complex spectral mapping with broader context; task-specific evidence |
| E04 | [Data sampling strategies](https://arxiv.org/abs/2304.07142) | arxiv.org | Mixed evidence | Crop length trades diversity, training efficiency and context |
| E05 | [SepFormer](https://arxiv.org/abs/2010.13154) | arxiv.org | Alternative | Dual-path attention with two/three-speaker benchmark results |
| E06 | [SDR—half-baked or well done?](https://arxiv.org/abs/1811.02508) | arxiv.org | Background | Clarifies metric definitions and scale invariance |
| E07 | [Dual-path RNN](https://arxiv.org/abs/1910.06379) | arxiv.org | Alternative | Long-sequence content modeling beyond convolutional receptive fields |
| E08 | [WHAM](https://arxiv.org/abs/1907.01160) | arxiv.org | Opposes clean-only scope | Noisy separation requires explicit evaluation |
| E09 | [WHAMR](https://arxiv.org/abs/1910.10279) | arxiv.org | Opposes clean-only scope | Reverberation creates an additional failure mode |
| E10 | [Differentiable consistency constraints](https://arxiv.org/abs/1811.08521) | arxiv.org | Supports improvement | Mixture consistency as an explicit constraint |
| E11 | [SA-SDR](https://arxiv.org/abs/2110.15581) | arxiv.org | Opposes current silence handling | Aggregate objective handles silent sources when another source is active |
| E12 | [Graph-PIT](https://arxiv.org/abs/2107.14446) | arxiv.org | Alternative | Continuous separation with varying total speaker count |
| E13 | [Wavesplit](https://arxiv.org/abs/2002.08933) | arxiv.org | Alternative | Sequence-wide speaker representations and clustering |
| E14 | [LibriCSS](https://arxiv.org/abs/2001.11482) | arxiv.org | Opposes narrow evaluation | Real continuous recordings and ASR-based evaluation |
| E15 | [DPTNet](https://arxiv.org/abs/2007.13975) | arxiv.org | Alternative | Direct contextual interaction through dual-path modeling |
| E16 | [MossFormer2](https://arxiv.org/abs/2312.11825) | arxiv.org | Alternative | Combines long-range attention and finer temporal modeling |
| E17 | [MixIT](https://arxiv.org/abs/2006.12701) | arxiv.org | Supports adaptation | Learning from unseparated mixtures |
| E18 | [Meeting adaptation with MixIT](https://arxiv.org/abs/2110.10739) | arxiv.org | Supports data-first work | Improvement with architecture held constant and real meeting adaptation |
| E19 | [Unsupervised multi-channel adaptation](https://arxiv.org/abs/2305.11151) | arxiv.org | Supports adaptation | Real-room and multi-channel evidence |
| E20 | [SMS-WSJ](https://arxiv.org/abs/1910.13934) | arxiv.org | Background | Spatial/reverberant evaluation and recognition metrics |
| E21 | [RemixIT](https://arxiv.org/abs/2202.08862) | arxiv.org | Mixed; alternative | Enhancement self-training; applicability requires care |
| E22 | [SpEx+](https://arxiv.org/abs/2005.04686) | arxiv.org | Alternative | Enrollment-conditioned extraction instead of blind separation |
| E23 | [NTT SpeakerBeam explanation](https://www.ntt-review.jp/archive/ntttechnical.php?contents=ntr202109fa4.html) | ntt-review.jp | Alternative | Practical task framing for a desired person's voice |
| E24 | [Utterance-level PIT](https://arxiv.org/abs/1703.06284) | arxiv.org | Supports | Correct permutation treatment for unordered outputs |
| E25 | [DNSMOS](https://arxiv.org/abs/2010.15258) | arxiv.org | Background | Perceptual quality proxy designed for noise suppression, not speaker identity |

### Similar implementation sources: 10 additional distinct pages

| ID | Page and URL | Domain | Classification | Why it matters |
|---|---|---|---|---|
| I01 | [Official LibriMix repository](https://github.com/JorisCos/LibriMix) | github.com | Existing implementation | Data construction and variants |
| I02 | [Asteroid recipe config](https://raw.githubusercontent.com/asteroid-team/asteroid/master/egs/librimix/ConvTasNet/local/conf.yml) | raw.githubusercontent.com | Existing implementation | Closest model/optimization/crop reference |
| I03 | [SpeechBrain dynamic mixer](https://raw.githubusercontent.com/speechbrain/speechbrain/develop/recipes/LibriMix/separation/dynamic_mixing.py) | raw.githubusercontent.com | Existing implementation | Concrete remixing and scaling recipe |
| I04 | [SepFormer WSJ0-3Mix model card](https://huggingface.co/speechbrain/sepformer-wsj03mix) | huggingface.co | Existing implementation | Available three-speaker checkpoint and exact benchmark provenance |
| I05 | [ESPnet TF-GridNet source](https://raw.githubusercontent.com/espnet/espnet/master/espnet2/enh/separator/tfgridnet_separator.py) | raw.githubusercontent.com | Existing implementation | Configurable source count, STFT and normalization behavior |
| I06 | [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) | github.com | Existing implementation | Practical pretrained alternatives; linked ClearVoice README/model card also inspected |
| I07 | [SpeakerBeam repository](https://github.com/BUTSpeechFIT/speakerbeam) | github.com | Existing implementation | Enrollment-aware LibriMix extraction recipe |
| I08 | [SVoice](https://github.com/facebookresearch/svoice) | github.com | Existing implementation | Multi-speaker/count-specific modeling; older stack |
| I09 | [Google sound-separation](https://github.com/google-research/sound-separation) | github.com | Existing implementation | Archived MixIT/meeting reference code |
| I10 | [Torchaudio Conv-TasNet source](https://docs.pytorch.org/audio/2.7.0/_modules/torchaudio/models/conv_tasnet.html) | docs.pytorch.org | Existing implementation | Independent block, norm, and mask implementation |

## Local evidence and reproducibility

- `config/demo.yaml`, `config/train.yaml`; `src/model.py`, `src/data.py`, `src/loss.py`, `src/train.py`, `src/evaluate.py`, `src/inference.py`, `src/audio.py`, `src/settings.py`; `scripts/prepare_data.py`, `scripts/serve_demo.py`.
- Real-run configs, environments and metrics under `logs/2026-11-11_07-00-31_005338/` and `logs/2026-11-14_19-52-54_688479/`.
- Older assessment: `logs/2026-11-11_07-00-31_005338/quality_assessment.json`.
- Newer previews and clipping simulation: `out/2026-11-14_19-52-54_688479/inference-preview-2026-11-15_07-32-49_500623/`.
- Newly generated `separation_quality_metrics.json` and `padding_probe.json` alongside this report. Metrics use raw predictions; amplitude clipping is not introduced into separation scoring.
- `/tmp/separation_audit.py`, `/tmp/separation_probe.py`, and `/tmp/separation_evaluate_audit.py` contain the diagnostic scripts used in this session. No training was run and no claimed improvement was inferred from retraining.
