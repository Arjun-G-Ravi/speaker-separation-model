# Three-speaker separation with Conv-TasNet

A from-scratch, noncausal PyTorch Conv-TasNet baseline for mono speech at 8 kHz.
It maps `[batch, samples]` to `[batch, 3, samples]`. A 10-second input produces
three 80,000-sample tracks. Source identities are unordered; speaker 1 does not
identify the same person across different clips. This is an offline model.

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
Select a mixture to play its three ground-truth sources; **Run separation** adds
the model outputs, matched to the reference order using PIT, with per-example
SI-SDR and improvement over the mixture. Predictions are generated on demand and
cached for the current server session. The training history and checkpoint score
are shown separately from the current example's score.

The **Your audio** tab accepts WAV, MP3, FLAC, OGG, M4A, and other formats supported
by the installed ffmpeg. Uploads default to at most 50 MB and 60 seconds. Uploaded
audio is converted to mono/8 kHz, with three playable/downloadable float WAV
outputs; there are no reference tracks or quality scores for uploads. Only one
inference request runs at a time. The model loads once at startup.

Set the checkpoint, device, port, upload limits, and output directory in
`config/demo.yaml`. It defaults to the latest completed run's best checkpoint.
Generated tracks stay under `out/demo/<session>/`; source dataset files are untouched.
The service binds only to localhost. When running on your SSH machine, forward
the port from your laptop (`ssh -L 7860:127.0.0.1:7860 your-server`), then open the
same localhost URL in your laptop browser. No hosting service is required.

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

Preparation pins the [official LibriMix metadata](https://github.com/JorisCos/LibriMix)
to the revision in `config/prepare.yaml`. With seed 42 it selects exactly:

| Output split | Original split | Mixtures | Unique speech utterances |
| --- | --- | ---: | ---: |
| train | train-clean-100 | 500 | 1,500 |
| validation | dev-clean | 50 | 144 |
| test | test-clean | 50 | 146 |

This is the small trial configuration. For the later 5,000/500/500 experiment,
change the three counts in `config/prepare.yaml` and use a separate root, also
updating `data.root` in `config/train.yaml`. Existing larger-run files under
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

Training uses random synchronized 10-second crops, keeping shorter examples and
masking batch padding in the loss. Validation uses complete recordings individually.
Test evaluation reports both full recordings and deterministic centered 10-second
crops (or the full clip when shorter). These metrics describe this selected subset,
not the complete official benchmark. The test set is never used for checkpoint selection.

## Model, optimization, and artifacts

The [Conv-TasNet paper](https://arxiv.org/abs/1809.07454) describes the learned
encoder/decoder and dilated temporal convolution separator implemented here.
Defaults: encoder width 512, 16-sample kernel/8-sample stride, bottleneck and skip
width 128, hidden width 512, kernel 3, eight dilation blocks repeated three times.
Blocks use global layer normalization, PReLU, depthwise convolutions, residual and
skip connections. Three ReLU masks gate the encoded waveform.

Utterance-level PIT selects the best of all six source permutations independently
for each example. The objective is negative mean zero-mean SI-SDR in dB. Numerical
stabilizers keep scores finite; silent references are rejected.

Adam starts at `1e-3`, with batch size 1 and four-step accumulation. Gradients are
unscaled before clipping at norm 5. Validation SI-SDR selects the best model;
ReduceLROnPlateau halves the rate after its configured patience, and early stopping
ends training after ten unimproved epochs. Maximum training length is 100 epochs.
These are baseline settings, not a promise of separation quality on this small subset.

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
multichannel input to mono, and resamples if necessary. It writes three float32 WAV
tracks at 8 kHz with the same resampled duration. Tracks are not independently
normalized or clipped. Existing output tracks cause an error; choose a fresh output
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

Results are written to `logs/smoke_result.json`. Synthetic runs have their own
timestamped logs/checkpoints; their temporary fixture dataset is removed afterward.
They establish implementation behavior, not performance on real speech. Real-data
SI-SDR/SI-SDRi and full-subset disk size must be measured after preparation/training.

For future models, keep the `[B,T] -> [B,3,T]` interface and add an explicit branch
in `build_model`; data, PIT, training, and evaluation can remain shared.

LibriSpeech is distributed under [CC BY 4.0](https://www.openslr.org/12).
LibriMix code/metadata attribution and its MIT license are recorded in
`THIRD_PARTY_NOTICES.md`.
