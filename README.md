# Speech separation with A-FRCNN and Conv-TasNet

Noncausal PyTorch speech separation for mono audio at 8 kHz. The default experiment
uses A-FRCNN with an earlier speaker split; the original Conv-TasNet is preserved
in `src/conv_tasnet.py`, including
compatibility with existing checkpoints. Models map `[batch, samples]` to
`[batch, sources, samples]`, with two or three sources selected by the dataset.
Source identities are unordered; speaker 1 does not identify the same person
across different clips. These are offline models.

### Selecting the training dataset

In `config/train.yaml`, set `data.dataset: libri2mix` (the default) or `libri3mix`.
The selector chooses the corresponding `data.roots` path and sets `model.sources`
automatically. Both use A-FRCNN with `iterations: 16`. The resolved root and source
count are saved in each run's configuration and checkpoint.

Prepare the selected dataset first. For Libri2Mix, run:

```fish
/home/arjun/ai_env/bin/python -m scripts.download_libri2mix
```

Settings are in `config/download_libri2mix.yaml`: 5,000 training, 500 validation,
and 500 test mixtures, with two sources at 8 kHz, saved to `data/Libri2Mix-5000/`.
The downloader verifies archive checksums and resumes completed chunks when rerun.
It can download about 7 GB of LibriSpeech archives and enforces a 20 GB storage
ceiling under its dataset root. The original `scripts.download_dataset` command
uses the separate three-speaker configuration. Selecting a training dataset
never downloads it automatically.

Train using `/home/arjun/ai_env/bin/python -m src.train`. Later, set
`data.dataset: libri3mix`, verify its root contains prepared manifests, and keep
`checkpoint.resume: null` to start a separate three-speaker experiment. Changing
source counts changes the output head; this is not an automatic fine-tuning step.
Resume supports interruptions of the same model/dataset configuration.

The listening demo still selects the existing trained Conv-TasNet checkpoint.
After training A-FRCNN, set the new checkpoint in `config/demo.yaml`,
`config/evaluate.yaml`, `config/inference.yaml`, or `config/preview.yaml` as needed.
Those tools recover the architecture and source count from the checkpoint.

## Pilot 1: expand the LibriSpeech source pool

The three matched Libri2Mix pilots fine-tune the epoch 138 early-split model.
A uses the original source pool. B also calibrates the decoder's gain and polarity.
C uses the calibrated decoder and remixes full source utterances from cached
`train-clean-100`. All three retain 5,000 training draws per epoch and select
their best checkpoint on the existing 500 validation recordings.

Edit `config/pilot1_librispeech_expansion.yaml` to choose the arm. Prepare the
source pool once before arm C, using the cached archive. The preparation command
requires no network and uses up to 8 GB of additional storage.

```fish
/home/arjun/ai_env/bin/python -m src.pilot1_librispeech_expansion.prepare
/home/arjun/ai_env/bin/python -m src.pilot1_librispeech_expansion.train
```

Run training once per arm (A, B, C) on the RTX 3090. The command checks one full
batch before starting each arm, then prints its `out/<run_id>/best.pt` path.
For an interrupted arm, set `resume` to its `recovery.pt` and keep the same arm.
Clear `resume` before moving to the next arm. W&B settings come from the baseline
checkpoint's training configuration.

Fill in the three `evaluation.checkpoints` paths. Run the command below with
`selection_report: null` to compare the original checkpoint and the three pilots
on validation. It saves the selected arm and projection decision to
`logs/pilot1_<timestamp>/validation.json`. Once the choice is locked, set
`selection_report` to that file and run the same command to evaluate the original
checkpoint and the selected arm on the test set. If no arm clears the validation
gate, the test report covers only the original checkpoint.

```fish
/home/arjun/ai_env/bin/python -m src.pilot1_librispeech_expansion.evaluate
```

For listening previews of B or C, use `playback_gain_db: 0` in
`config/preview.yaml`. Their decoder already incorporates the calibration.

## Local listening room

Run the dataset browser and upload inference UI in your uv environment:

```fish
uv run -m scripts.serve_demo
```

Open **http://localhost:7860**. If your uv environment has not been prepared,
`uv run` will resolve/install the declared dependencies when you run it yourself.
To reuse the existing global environment without installing anything:

```fish
uv run --no-project --no-sync --python /home/arjun/ai_env/bin/python -m scripts.serve_demo
```

The viewer lists all training and validation examples with search and pagination.
Select a mixture to play its ground-truth sources; **Run separation** adds
the model outputs, matched to the reference order using PIT, with per-example
SI-SDR and improvement over the mixture. Predictions are generated on demand and
cached for the current server session. The training history and checkpoint score
are shown separately from the current example's score.

The **Your audio** tab accepts WAV, MP3, FLAC, OGG, M4A, and other formats supported
by the installed ffmpeg. Uploads default to at most 50 MB and 60 seconds. Uploaded
audio is converted to mono/8 kHz, with one playable/downloadable float WAV per source;
there are no reference tracks or quality scores for uploads. Only one
inference request runs at a time. The model loads once at startup.

Set the checkpoint, device, port, upload limits, and output directory in
`config/demo.yaml`. It selects the November 14 run's best checkpoint (epoch 65).
Generated tracks stay under `out/demo/<session>/`; source dataset files are untouched.
The service binds only to localhost. When running on your SSH machine, forward
the port from your laptop (`ssh -L 7860:127.0.0.1:7860 your-server`), then open the
same localhost URL in your laptop browser. No hosting service is required.

To generate a listening preview with three held-out test examples and two training
examples from that checkpoint:

```fish
/home/arjun/ai_env/bin/python -m scripts.preview_inference
```

`config/preview.yaml` controls the checkpoint and example counts. Each run writes
a fresh `out/<run_id>/inference-preview-<timestamp>/index.html` with playable mixtures,
references, predictions, and separate train/test metrics. Examples are the first
manifest entries, selected before inference; training examples contain speech used
during training. Open the generated `index.html` to listen.

## Environment

Use your existing environment; no environment creation or dependency installation
is performed by the project. `ffmpeg` must be on PATH for fallback FLAC decoding.
Installing the optional `soundfile` package enables faster native FLAC decoding
without starting a subprocess per source. WAV I/O uses SciPy. Python 3.11–3.14 is supported.

The inspected environment already contains PyTorch, NumPy, SciPy, and PyYAML.
W&B is missing. Run this yourself if using online logging:

```fish
/home/arjun/ai_env/bin/python -m pip install 'wandb>=0.19,<1'
/home/arjun/ai_env/bin/wandb login
```

For faster dataset preparation, run this yourself:

```fish
/home/arjun/ai_env/bin/python -m pip install 'soundfile>=0.13,<0.14'
```

`pyproject.toml` declares dependencies; `uv.lock` records a complete resolution.
On a separate machine, you may choose `uv sync --locked` yourself, then use
`uv run --no-sync python -m src.train` (and the other modules below).
Choose a PyTorch CUDA installation compatible with that machine's driver.
The project never invokes `uv sync`, `pip install`, or a dataset download at import.

## Configuration and commands

Run commands from this repository. All paths in YAML are relative to the repository
unless absolute. Edit YAML files rather than passing command-line arguments.
`config/train.yaml` contains every real-experiment training hyperparameter.

```fish
# Offline synthetic correctness and optimization checks; no dataset download
/home/arjun/ai_env/bin/python -m scripts.smoke_check

# A-FRCNN-16 two/three-speaker preparation, training, evaluation, and inference checks
/home/arjun/ai_env/bin/python -m scripts.check_afrcnn

# Explicitly starts approximately 7 GB of source archive transfer
/home/arjun/ai_env/bin/python -m scripts.prepare_data

# Uses config/train.yaml; defaults target the RTX 3090, with CPU fallback
/home/arjun/ai_env/bin/python -m src.train

# Set checkpoint in config/evaluate.yaml first
/home/arjun/ai_env/bin/python -m src.evaluate

# Set checkpoint, input, and output in config/inference.yaml first
/home/arjun/ai_env/bin/python -m src.inference
```

Use `training.device: cuda` to require a GPU, `auto` for CUDA when available,
or `cpu`. CUDA training uses float16 autocast and gradient scaling when `amp: true`.
SI-SDR and normalization reductions use float32. CPU uses float32.
The smoke script runs an additional full-size, 10-second CUDA AMP optimizer step
when CUDA is available; otherwise it reports that check as skipped.

For W&B, `logging.wandb_mode` can be `online`, `offline`, or `disabled`.
Online is the real-training default. Local metrics are always recorded.

## Libri3Mix subset

To prepare the larger **5,000 training / 500 validation / 500 test** subset:

```fish
/home/arjun/ai_env/bin/python -m scripts.download_dataset
```

Settings live in `config/download.yaml`; output goes to `data/Libri3Mix-5000/`.
This command downloads the official clean LibriSpeech archives (about 7 GB total),
then builds the selected three-speaker, 8 kHz mixtures and manifests. It samples
only the requested mixtures; it does not generate the full LibriMix dataset.
It probes three official mirrors concurrently, chooses the fastest sample,
uses 16 parallel connections, and retries interrupted connections. Terminal logs
show downloaded GB, percentage, MB/s, ETA, cache reuse, extracted sources, and
mixtures prepared. Probe speeds are estimates, not guaranteed sustained speeds.

Completed 8 MiB download chunks survive interruption: run the same command to
resume without fetching those chunks again. An incomplete chunk is fetched again.
Archives are checked against the official OpenSLR MD5 checksums before preparation
and retained in `data/Libri3Mix-5000/.archives/` for reuse. Unlike the streaming
small-subset preparer below, this trades disk space for reliable download resumption.
The separate large-subset config has a **20 GB storage ceiling**, including chunks,
archive assembly, cached archives, temporary sources, and generated WAVs; free space
is checked before each archive download. Mirror URLs, connection count, timeouts,
and storage limits can be changed in the YAML. No new Python packages are needed.
If preparation is interrupted after downloading, rerunning reuses verified archives
and completed mixtures. Run only one preparer against a dataset root at a time.

The `libri3mix` entry in `config/train.yaml` targets `data/Libri3Mix-5000`: 5,000
training, 500 validation, and 500 test examples. Select `data.dataset: libri3mix`
to use these prepared files. Training defaults to Libri2Mix in its separate root.
The download command preserves the small dataset and does not change training
settings or start training.

Preparation pins the [official LibriMix metadata](https://github.com/JorisCos/LibriMix)
to the revision in `config/prepare.yaml`. With seed 42 it selects exactly:

| Output split | Original split | Mixtures | Unique speech utterances |
| --- | --- | ---: | ---: |
| train | train-clean-100 | 500 | 1,500 |
| validation | dev-clean | 50 | 144 |
| test | test-clean | 50 | 146 |

This is the small trial configuration in `config/prepare.yaml`. The larger
5,000/500/500 experiment uses the separate `config/download.yaml` described above.
Existing larger-run files under
`data/Libri3Mix` are preserved; the small trial uses `data/Libri3Mix-small`.
Reducing the counts speeds training and reduces stored audio, but sequential
archive preparation can still read most of the original training archive.

It streams the [official LibriSpeech archives](https://www.openslr.org/12),
stores only selected sources, and creates clean mixtures using the published
gains. Source amplitudes are scaled before polyphase resampling to 8 kHz; each
mixture ends at its shortest speech source. Full recording lengths are preserved.
Only `mix_clean`, `s1`, `s2`, and `s3` PCM16 WAV files are generated.
No WHAM noise or train-clean-360 archive is required.

This avoids generating/downloading full LibriMix. Because the official speech
archives are sequential gzip files, preparation may still transfer nearly all
three source archives (roughly 7 GB); it never stores those archives. An interrupted
stream reconnects automatically at its compressed byte offset using HTTP Range.
If a mirror ignores Range, its already-read prefix is replayed without storing it.
`network` in `config/prepare.yaml` controls the read timeout (120 seconds), retries
(six per stalled read), and initial retry delay (three seconds, with capped backoff).
If retries are exhausted, rerun the same command: finished mixtures and available
temporary sources are reused, though a new process restarts its archive network pass.
Do not delete the dataset directory after a timeout.

The default source host is the official China mirror, which was faster than the
original host in short transfer probes from this workspace. Actual throughput varies.
Preparation uses 16 parallel HTTP range requests with 4 MiB chunks, feeds them to
the archive decoder in order, and limits the prefetch window to about 68 MiB of payload.
No full archive is written to disk. Transfer progress prints archive bytes, throughput,
and an estimated archive completion time, including local processing pauses. A new
process must still reread the archive prefix; cached sources and completed mixtures
are reused. Stop an already-running preparation process before starting the updated
one. Set `network.download_workers: 1` for mirrors that cannot serve parallel ranges.

The **10 GB ceiling is decimal bytes and covers files under the dataset root**,
including source cache, partial files, manifests, and outputs. It does not count
existing Python packages, filesystem allocation overhead, training logs, or models.
Preparation reserves 10 MB of headroom and checks space before writing sources,
mixtures, or metadata. It stops before exceeding the ceiling if the selected data
cannot fit. Exact full-subset storage has not been measured without downloading it.
Do not run multiple preparation processes against the same root.

```text
data/Libri3Mix-small/
  train/{mix_clean,s1,s2,s3}/<official_mixture_ID>.wav
  validation/{mix_clean,s1,s2,s3}/<official_mixture_ID>.wav
  test/{mix_clean,s1,s2,s3}/<official_mixture_ID>.wav
  train.json, validation.json, test.json
  train_selection.json, validation_selection.json, test_selection.json
  preparation.json
```

Selection files preserve the original gains, source paths, metadata digest, seed,
and revision. Manifests contain relative mixture/source paths, lengths, and selection
digests. Preparation validates alignment, sample rate, source energy, speaker split
disjointness, and mixture/source agreement within PCM16 quantization tolerance.
`preparation.json` reports actual preparation storage and manifest hashes.

Training uses fixed 3-second crops with no batch padding. With dynamic mixing enabled,
each sample selects the configured number of distinct training speakers and independent source crops
from the prepared training source tracks. Sources are normalized to RMS 0.05 and
receive independent gains sampled from -5 to +5 dB. A shared gain limits both the
mixture and source peaks to 0.9; the mixture is recomputed from the scaled sources.
No validation/test sources enter the mixer. The epoch length remains 5,000 mixtures.
Selection and crops are deterministic for a seed, epoch, and sample index, including
when resuming with zero loader workers. These prepared sources are truncated versions
of the original utterances; remixing does not add new training speakers.

`data.fixed_crops`, `data.segment_seconds`, and `data.dynamic_mixing` in
`config/train.yaml` control this behavior. Training rejects sources shorter than
the fixed crop. Active crops are retried up to `crop_attempts`; exhausting retries
raises an error. Disable dynamic mixing to train on synchronized crops of the
original mixtures. Prepared PCM16 WAVs are memory-mapped and cropped before float
conversion, so training reads only the selected window from each track.
The `speed_perturbation` experiment independently samples each source's playback
speed from 95%, 100%, and 105% before RMS normalization and remixing. Polyphase
resampling changes both speed and pitch, then crops to the fixed training length.
If a speed-up reaches the end of a short source, the remainder is zero-padded.
The output sample rate remains 8 kHz. Augmentation is deterministic by seed, epoch,
and sample index across loader workers; validation and test are never augmented.
Audio is loaded per example rather than caching the full dataset in RAM; workers
prefetch a bounded number of batches. Validation uses complete recordings individually.
Test evaluation reports both full recordings and deterministic centered 10-second
crops (or the full clip when shorter). These metrics describe this selected subset,
not the complete official benchmark. The test set is never used for checkpoint selection.
Evaluation writes each example's results directly to its JSON file, keeping only
running metric totals in memory.

## Model, optimization, and artifacts

`src/afrcnn.py` implements the concatenation-fusion variant from the
[A-FRCNN reference](https://github.com/JusperLee/AFRCNN-For-Speech-Separation),
with a shared convolutional block applied 16 times. Defaults: 512 encoder channels,
21-sample kernel/10-sample stride, bottleneck and hidden widths 512, and five temporal
resolutions. Adjacent resolutions exchange features, then merge at full resolution
inside each refinement. The two-speaker model has 6,134,793 parameters. It uses
global normalization, so inference requires the complete clip. Attribution and the
upstream MIT license are in `THIRD_PARTY_NOTICES.md`.

`src/conv_tasnet.py` preserves the original
[Conv-TasNet](https://arxiv.org/abs/1809.07454) encoder/decoder and dilated temporal
separator. Existing checkpoints retain their original architecture settings.
`src/model.py` selects `afrcnn`, `early_split_afrcnn`, or `conv_tasnet` using `model.name`.
Architecture-specific YAML fields must match the selected constructor; to retrain
the baseline, use the `model` section saved with that baseline run.

`src/earlier_speaker_split/model.py` adds the `early_split_afrcnn` variant. It runs
eight mixture refinement iterations, splits the bottleneck features into two or
three streams through a convolution and GLU, then refines each stream eight more
times. `split_iteration` sets the split point within the total `iterations`.
The same A-FRCNN block and residual projection are shared across both phases and
all speakers. After splitting, sources are folded into the batch dimension so
normalization and temporal processing operate independently for each speaker.
A shared mask head and single-source waveform decoder reconstruct every stream.
This replaces the baseline's joint waveform decoder, so a comparison measures
the combined split-and-shared-decoder change. No attention, speaker embeddings,
cross-speaker module, or auxiliary loss is added. This is a CNN adaptation of the
early split/shared reconstruction idea from
[Separate and Reconstruct](https://arxiv.org/abs/2406.05983), not a SepReformer reproduction.

Speaker refinement uses activation checkpointing during training to reduce memory
by recomputing block activations during backward. Inference runs directly. Sharing
weights does not remove the extra speaker-stream computation; GPU memory and
runtime still need measurement on the 3090. This architecture needs a fresh run.
Evaluation, inference, W&B, and checkpoint loading use the existing entry points.
Run the synthetic checks with:

```fish
/home/arjun/ai_env/bin/python -m scripts.check_earlier_speaker_split
```

Utterance-level PIT selects the best of all source permutations (two for Libri2Mix,
six for Libri3Mix) independently
for each example using the combined objective:

```text
loss = mean(-SI-SDR + scale_loss_weight * scale_error)
scale_error = 10 * log10(1 + sum((prediction - reference)^2) / sum(reference^2))
```

The scale-sensitive waveform error and SI-SDR use the same permutation and valid
samples. `training.scale_loss_weight` defaults to 1.0; setting it to zero restores
the SI-SDR objective. This weight is an initial experiment setting, not a tuned value.
Numerical stabilizers keep scores finite; silent references are rejected.
Logs include separate `train/si_sdr`, `train/scale_loss`, and `train/output_peak`
values, with corresponding validation metrics. `train/loss` is the combined objective;
its negative is no longer a SI-SDR score. Checkpoint selection still uses validation
SI-SDR; validation and test recordings retain their original mixtures.

The new architecture and training settings require a fresh run with `checkpoint.resume: null`. Old
checkpoints still work for inference; the new crop/mixing/loss settings cannot be
resumed into an old experiment. Run `/home/arjun/ai_env/bin/python -m src.train`
to train with W&B using the existing environment. No speech-quality improvement
is established until that run is evaluated.

Adam starts at `1e-3`, with batch size 8 and one-step accumulation. Gradients are
unscaled before clipping at norm 5. Validation SI-SDR selects the best model;
ReduceLROnPlateau halves the rate after its configured patience, and early stopping
ends training after 30 unimproved epochs, allowing more time to recover from a
validation dip. Maximum training length is 200 epochs.
These are baseline settings, not a promise of separation quality on this subset.
Select `experiment` at the top of `config/train.yaml` and run each comparison
separately with `checkpoint.resume: null`:

| Experiment | Scale-loss weight | Speed perturbation | Comparison |
| --- | ---: | --- | --- |
| `earlier_speaker_split` (default) | 1.0 | Off | Early split A-FRCNN; compare with `baseline_200` |
| `baseline_200` | 1.0 | Off | Longer training with the existing objective |
| `si_sdr_only` | 0.0 | Off | Isolates removal of the scale loss |
| `speed_perturbation` | 0.0 | On | Compare with `si_sdr_only` to isolate augmentation |

The `experiments` entries override the common `model`, `data`, and `training` settings when
the training YAML is loaded. Resolved settings, including the experiment name,
are saved with each run and sent to W&B. The optimizer, learning rate, scheduler,
precision, crop length, and effective batch size are identical across these runs.
Select using validation results, then evaluate the selected checkpoint on test.
At the previous run's epoch time, 200 epochs would take about 22 hours;
augmentation and hardware load can change that estimate. There is no wall-clock cutoff.

```fish
/home/arjun/ai_env/bin/python -m src.train
```

Each epoch prints training batch progress and validation example progress at
`training.log_every_steps` intervals, alongside the existing local and W&B metrics.

Each run prints its unique timestamp ID and writes:

```text
out/<run_id>/best.pt
logs/<run_id>/train.yaml
logs/<run_id>/environment.json
logs/<run_id>/metrics.jsonl
logs/<run_id>/result.json
```

A rolling `recovery.pt` is updated at epoch boundaries during training and retained
on interruption. A successful run removes it, retaining `best.pt`. Saves use atomic
replacement. Checkpoints contain model settings/weights, optimizer, scheduler,
scaler, RNG state, dataset hashes, and progress. Interrupted work within an epoch
is rerun. For recovery, set `checkpoint.resume` in `config/train.yaml` to that run's
`recovery.pt`, keeping `best.pt` beside it. Resume creates a new timestamped run and
carries forward the original best model. You may increase the epoch limit or change
device, workers, and logging frequency; incompatible data/model/optimization
changes are rejected. Seeding improves reproducibility; GPU kernels and environment
changes can still affect exact results.

Evaluation writes per-example SI-SDR, mixture baseline SI-SDR, and SI-SDR improvement,
plus utterance-averaged summaries under the checkpoint's log directory. It checks
that the evaluated manifest matches the checkpoint's dataset identity.

Inference restores model settings and sample rate from the checkpoint, averages
multichannel input to mono, and resamples if necessary. It writes one float32 WAV per source
tracks at 8 kHz with the same resampled duration. If predictions exceed the playback
range, a shared attenuation brings their joint peak to 0.95 without clipping or
changing relative speaker levels. SI-SDR ignores overall gain, so it cannot detect
this playback overflow. The CLI prints the applied gain; the demo reports it in
its response. `separate_waveform` still returns raw predictions for evaluation.
Existing output tracks cause an error; choose a fresh output
directory. Whole-clip inference avoids permutation changes between chunks; very long
recordings may require more memory. Streaming/long-recording chunk stitching is outside
this first experiment.

## Verification and limitations

The smoke script uses synthetic modulated tones, including actual FLAC archive
fixtures. It checks gains, resampling, shortest-source lengths, preparation reruns,
partial-example recovery, and the storage guard. It checks all PIT permutations,
scale invariance, padding exclusion, gradients, exact output lengths including
80,000 samples, synthetic overfitting, checkpoint reload, deterministic epoch recovery,
evaluation, stereo resampling, and three-track inference.

`scripts.check_afrcnn` additionally checks two/three-speaker A-FRCNN-16 output
lengths, finite gradients with the joint loss, deterministic preparation/remixing,
tiny synthetic training, checkpoint reload, evaluation, and the correct number of
inference exports. CPU checks do not establish GPU memory requirements or speech
quality. No real-speech A-FRCNN training results have been measured here; the
current crop/mixing/loss recipe is not a reproduction of published benchmark scores.

Results are written to `logs/smoke_result.json`. Synthetic runs have their own
timestamped logs/checkpoints; their temporary fixture dataset is removed afterward.
They establish implementation behavior, not performance on real speech. Real-data
SI-SDR/SI-SDRi and full-subset disk size must be measured after preparation/training.

For future models, keep the `[B,T] -> [B,C,T]` interface and add an explicit branch
in `build_model`; data, PIT, training, and evaluation can remain shared.

LibriSpeech is distributed under [CC BY 4.0](https://www.openslr.org/12).
LibriMix code/metadata attribution and its MIT license are recorded in
`THIRD_PARTY_NOTICES.md`.
