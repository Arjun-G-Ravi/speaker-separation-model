import html
import logging
import platform
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from scipy.io import wavfile

from src.audio import read_audio
from src.inference import playback_audio, separate_waveform
from src.loss import pairwise_si_sdr, pit_si_sdr
from src.settings import digest, load_config, load_model, project_path, select_device

SETTINGS = load_config('preview.yaml')
CHECKPOINT = SETTINGS['checkpoint']
MAX_SECONDS = SETTINGS['max_seconds']
LOGGER = logging.getLogger(__name__)


def training_chart(rows):
    rows = [dict(r, **{'train/si_sdr': r.get('train/si_sdr', -r['train/loss'])}) for r in rows]
    values = [r['train/si_sdr'] for r in rows] + [r['validation/si_sdr'] for r in rows]
    low, high = np.floor(min(values)) - 1, np.ceil(max(values)) + 1
    def y(value): return 220 - (value - low) / (high - low) * 185
    parts = ['<svg viewBox="0 0 800 265" role="img" aria-label="Training and validation SI-SDR by epoch">']
    for value in np.linspace(low, high, 5):
        parts.append(f'<path d="M55 {y(value):.2f}H780" stroke="#d4dcd4"/><text x="6" y="{y(value)+4:.2f}">{value:.1f}</text>')
    for field, color in [('train/si_sdr', '#67786b'), ('validation/si_sdr', '#11634b')]:
        points = ' '.join(f'{55 + i/max(1,len(rows)-1)*725:.2f},{y(r[field]):.2f}' for i,r in enumerate(rows))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
    parts.append('<text x="55" y="250">Epoch 1</text>')
    parts.append(f'<text x="670" y="250">Epoch {rows[-1]["epoch"]}</text></svg>')
    return ''.join(parts)


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    playback_boost = 10 ** (SETTINGS.get('playback_gain_db', 0) / 20)
    device = select_device(SETTINGS['device'])
    model, state = load_model(CHECKPOINT, device)
    model_name = {
        'afrcnn': 'A-FRCNN',
        'conv_tasnet': 'Conv-TasNet',
        'early_split_afrcnn': 'Early-split A-FRCNN',
    }[state['config']['model']['name']]
    rate = state['config']['data']['sample_rate']
    data_root = project_path(state['config']['data']['root'])
    examples = []
    manifest_hashes = {}
    for split in ('test', 'train'):
        manifest_path = data_root / state['config']['data'][f'{split}_manifest']
        manifest_hashes[split] = digest(manifest_path)
        if manifest_hashes[split] != state['manifest_hashes'][split]:
            raise ValueError(f'{split} manifest does not match the checkpoint')
        records = json.loads(manifest_path.read_text())['examples']
        count = SETTINGS[f'{split}_examples']
        if not isinstance(count, int) or not 1 <= count <= len(records):
            raise ValueError(f'{split}_examples must be between 1 and {len(records)}')
        examples.extend((split, index, row) for index, row in enumerate(records[:count]))
    output = ROOT / 'out' / state['run_id'] / ('inference-preview-' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(output/'inference.log')])
    LOGGER.info('Checkpoint %s; epoch %s; device %s', CHECKPOINT, state['next_epoch'], device)
    LOGGER.info('Output directory: %s', output)
    LOGGER.info('Selected first %s test and %s train entries before inference',
                SETTINGS['test_examples'], SETTINGS['train_examples'])
    (output/'config.yaml').write_text(yaml.safe_dump(SETTINGS))
    training = output / 'training'
    training.mkdir()
    run_logs = ROOT / 'logs' / state['run_id']
    for name in ('train.yaml', 'environment.json', 'metrics.jsonl', 'result.json'):
        if (run_logs / name).exists(): shutil.copy2(run_logs / name, training / name)
    rows = [r for line in (training / 'metrics.jsonl').read_text().splitlines() if 'epoch' in (r := json.loads(line))]
    best = max(rows, key=lambda r:r['validation/si_sdr'])
    last = rows[-1]
    counts = {split:len(json.loads((data_root / state['config']['data'][f'{split}_manifest']).read_text())['examples']) for split in ('train', 'validation', 'test')}
    details = {
        'checkpoint': CHECKPOINT, 'checkpoint_sha256': digest(project_path(CHECKPOINT)),
        'run_id':state['run_id'], 'selected_epoch':state['next_epoch'],
        'completed_epochs':len(rows), 'best_validation_si_sdr_db':best['validation/si_sdr'],
        'final_validation_si_sdr_db':last['validation/si_sdr'],
        'final_training_si_sdr_db':last.get('train/si_sdr', -last['train/loss']),
        'logged_epoch_minutes':sum(r['elapsed_seconds'] for r in rows)/60,
        'peak_training_gpu_memory_bytes':max(r['gpu_peak_bytes'] for r in rows),
        'training_completed':(run_logs/'result.json').exists(),
        'early_stopped':(run_logs/'result.json').exists() and last['epoch'] < state['config']['training']['epochs'] and last['epoch']-best['epoch'] >= state['config']['training']['early_stopping_patience'],
        'dataset_examples':counts, 'parameters':sum(p.numel() for p in model.parameters()),
        'inference_device':str(device), 'inference_precision':'float32',
        'selection':f"First {SETTINGS['test_examples']} test and {SETTINGS['train_examples']} train entries; selected before inference",
        'crop_policy':'Full recording' if not MAX_SECONDS else f'Centered crop up to {MAX_SECONDS} seconds',
        'sample_rate':rate, 'python':platform.python_version(), 'torch':str(torch.__version__),
        'manifest_sha256':manifest_hashes,
    }
    (training/'details.json').write_text(json.dumps(details,indent=2))
    chart = training_chart(rows)
    (training/'learning_curves.svg').write_text(chart)
    scores = []
    for split,index,row in examples:
        started = time.monotonic()
        sample = output / f'{split}_example_{index+1:02d}'
        sample.mkdir()
        source_rate, mixture = read_audio(data_root / row['mixture'])
        targets = np.stack([read_audio(data_root / p)[1] for p in row['sources']])
        length = min(len(mixture), int(MAX_SECONDS * rate)) if MAX_SECONDS else len(mixture)
        start = (len(mixture)-length)//2
        mixture = mixture[start:start+length]
        targets = targets[:,start:start+length]
        mixture, estimate = separate_waveform(model, mixture, source_rate, rate, device)
        target = torch.from_numpy(targets)[None]
        _, order = pit_si_sdr(torch.from_numpy(estimate)[None], target)
        estimate = estimate[torch.argsort(order[0]).numpy()]
        score = pairwise_si_sdr(torch.from_numpy(estimate)[None],target)[0].diag()
        baseline = pairwise_si_sdr(torch.from_numpy(mixture)[None,None].expand_as(target),target)[0].diag()
        wavfile.write(sample/'mixture.wav',rate,mixture.astype(np.float32))
        playback, attenuation = playback_audio(estimate * playback_boost)
        gain = playback_boost * attenuation
        raw_folder = sample/'raw_predictions'
        raw_folder.mkdir()
        for i in range(model.sources):
            wavfile.write(sample/f'reference_speaker_{i+1}.wav',rate,targets[i].astype(np.float32))
            wavfile.write(sample/f'predicted_speaker_{i+1}.wav',rate,playback[i])
            wavfile.write(raw_folder/f'predicted_speaker_{i+1}.wav',rate,estimate[i].astype(np.float32))
        LOGGER.info('Shared playback gain %.9g; raw peak %.3f; playback peak %.3f',
                    gain, np.max(np.abs(estimate)), np.max(np.abs(playback)))
        result = {'example':sample.name, 'split':split, 'mixture_id':row['id'], 'original_samples':row['length'],
                  'crop_start_sample':start, 'samples':length, 'seconds':length/rate,
                  'si_sdr_db':score.mean().item(), 'mixture_si_sdr_db':baseline.mean().item(),
                  'si_sdri_db':(score-baseline).mean().item(), 'speaker_si_sdr_db':score.tolist(),
                  'estimate_to_reference_permutation':order[0].tolist(),
                  'elapsed_seconds':time.monotonic()-started, 'playback_gain':gain,
                  'raw_peak':float(np.max(np.abs(estimate))),
                  'playback_peak':float(np.max(np.abs(playback))),
                  'raw_outside_playback_range_percent':(100*np.mean(np.abs(estimate)>1,axis=1)).tolist()}
        (sample/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        scores.append(result)
        LOGGER.info('%s example %d: %s', split, index+1, json.dumps(result))
    split_metrics = {}
    for split in ('test', 'train'):
        selected = [r for r in scores if r['split'] == split]
        split_metrics[split] = {
            'examples':len(selected),
            'mean_si_sdr_db':sum(r['si_sdr_db'] for r in selected)/len(selected),
            'mean_si_sdri_db':sum(r['si_sdri_db'] for r in selected)/len(selected),
        }
    summary = {'examples':len(scores), 'splits':split_metrics,
               'not_full_test_evaluation':True, 'training':details,'results':scores}
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    def player(path): return f'<audio controls preload="metadata" src="{path}"></audio><a href="{path}" download>Download WAV</a>'
    panels=[]
    for result in scores:
        folder=result['example']
        panels.append(f'<section><h2>{folder.replace("_"," ").title()}</h2><p class="id">{html.escape(result["mixture_id"])}</p><p>{result["seconds"]:.2f} seconds · SI-SDR {result["si_sdr_db"]:.2f} dB · improvement over mixture {result["si_sdri_db"]:.2f} dB</p><h3>Input mixture</h3>{player(folder+"/mixture.wav")}<div class="columns"><strong>Reference / ground truth</strong><strong>Model / PIT-matched estimate</strong></div>')
        for speaker in range(1, model.sources + 1):
            panels.append(f'<div class="columns tracks"><div><h3>Speaker {speaker}</h3>{player(f"{folder}/reference_speaker_{speaker}.wav")}</div><div><h3>Speaker {speaker} · {result["speaker_si_sdr_db"][speaker-1]:.2f} dB</h3>{player(f"{folder}/predicted_speaker_{speaker}.wav")}</div></div>')
        panels.append('</section>')
    document=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Inference preview · {html.escape(model_name)}</title>
<style>body{{margin:0;background:#f4f3ec;color:#19332e;font:16px/1.6 'DejaVu Sans',sans-serif}}main{{max-width:1050px;margin:auto;padding:40px 24px}}h1,h2{{font-family:Georgia,serif;font-weight:400}}h1{{font-size:42px}}h2{{font-size:29px}}h3{{font-size:14px}}a{{color:#11634b;font-size:13px}}section{{border-top:1px solid #c9d2c8;margin-top:36px;padding-top:20px}}.columns{{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-top:24px}}.tracks{{border-top:1px solid #d4dcd4;padding:16px 0}}audio{{display:block;width:100%;margin:10px 0}}.id{{font-family:monospace;overflow-wrap:anywhere}}svg{{width:100%;background:#fffefa}}svg text{{font:12px monospace;fill:#19332e}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #d4dcd4}}.note{{background:#e3ece3;padding:18px}}@media(max-width:640px){{.columns{{grid-template-columns:1fr}}h1{{font-size:32px}}}}a:focus-visible{{outline:3px solid #a84928}}</style>
<main><p class="id">{model_name.upper()} / TRAIN AND TEST LISTENING PREVIEW</p><h1>One mixture. {model.sources} estimated voices.</h1>
<p>{SETTINGS['test_examples']} held-out test examples and {SETTINGS['train_examples']} training examples from run <span class="id">{state['run_id']}</span>, using its best checkpoint (epoch {state['next_epoch']}).</p>
<div class="note"><strong>Test preview: {split_metrics['test']['mean_si_sdr_db']:.2f} dB SI-SDR · {split_metrics['test']['mean_si_sdri_db']:.2f} dB improvement.<br>Training preview: {split_metrics['train']['mean_si_sdr_db']:.2f} dB SI-SDR · {split_metrics['train']['mean_si_sdri_db']:.2f} dB improvement.</strong><br>Training examples contain speech used during training. Scores are reported separately by split. These small previews do not measure full-dataset performance. Outputs are reordered using references for comparison. Both predicted speakers receive the same playback gain, reduced when needed to prevent clipping. SI-SDR does not measure playback loudness or guarantee intelligibility.</div>
<h2>Training details</h2><table><tr><th>Dataset</th><td>{counts['train']} training / {counts['validation']} validation / {counts['test']} test mixtures</td></tr><tr><th>Completed epochs</th><td>{len(rows)}; early stopping: {details['early_stopped']}</td></tr><tr><th>Best validation SI-SDR</th><td>{best['validation/si_sdr']:.2f} dB, epoch {best['epoch']}</td></tr><tr><th>Final training / validation SI-SDR</th><td>{details['final_training_si_sdr_db']:.2f} / {last['validation/si_sdr']:.2f} dB</td></tr><tr><th>Logged epoch time</th><td>{details['logged_epoch_minutes']:.1f} minutes</td></tr><tr><th>Optimizer</th><td>Adam, initial learning rate {state['config']['training']['learning_rate']}, batch {state['config']['training']['batch_size']}, accumulation {state['config']['training']['accumulation_steps']}</td></tr><tr><th>Inference</th><td>{device}, float32, 8 kHz; complete recordings</td></tr></table>
<p>Training uses random crops; validation uses full recordings. Preview policy: {details['crop_policy']}. Training: gray line; validation: green line. Vertical axis: SI-SDR in dB.</p>{chart}<p><a href="training/train.yaml">Training configuration</a> · <a href="training/details.json">Full training details</a> · <a href="training/metrics.jsonl">Raw training logs</a> · <a href="summary.json">Inference metrics</a></p>
{''.join(panels)}</main></html>'''
    (output/'index.html').write_text(document)
    report = ['# Train and test inference preview', '',
              f'Checkpoint: `{CHECKPOINT}` (epoch {state["next_epoch"]}).', '',
              details['selection'], f"Crop policy: {details['crop_policy']}.", '',
              'Training samples contain speech used during training; test samples are held out.',
              'Predictions are PIT-matched to references. Speaker numbers are local to each example.', '',
              '| Example | Split | Seconds | SI-SDR (dB) | Improvement (dB) |',
              '| --- | --- | ---: | ---: | ---: |']
    for r in scores:
        report.append(f'| {r["example"]} | {r["split"]} | {r["seconds"]:.3f} | {r["si_sdr_db"]:.3f} | {r["si_sdri_db"]:.3f} |')
    for split, metrics in split_metrics.items():
        report.append(f"\n{split.title()} preview: {metrics['mean_si_sdr_db']:.3f} dB SI-SDR; {metrics['mean_si_sdri_db']:.3f} dB improvement.")
    report.extend(['', 'These small previews do not measure full-dataset performance.', '',
                   'Open `index.html` to listen. Keep this entire folder for relative audio links.',
                   'Playback WAVs use shared gain with clip protection. Raw predictions are for analysis.',
                   'Training snapshots are in `training/`; per-example scores are in `summary.json`.', '',
                   'Run again from the project root using `config/preview.yaml`:', '',
                   '```fish', '/home/arjun/ai_env/bin/python -m scripts.preview_inference', '```', '',
                   'Each run creates a fresh timestamped preview.'])
    (output/'report.md').write_text('\n'.join(report)+'\n')
    (output/'README.md').write_text('\n'.join(report)+'\n')
    shutil.copy2(__file__, output/'preview_script.py')
    # Check all audio exports, including lengths and finite sample values.
    for result in scores:
        for path in (output/result['example']).glob('*.wav'):
            actual_rate,signal=read_audio(path)
            assert actual_rate==rate and len(signal)==result['samples'] and np.isfinite(signal).all() and np.max(np.abs(signal)) <= 1, path
    LOGGER.info('Verified all %d playback WAV exports: sample rate, duration, finite samples, and safe peaks', len(scores)*(1+2*model.sources))
    LOGGER.info('OUTPUT_DIRECTORY %s', output)
    LOGGER.info('PREVIEW_SUMMARY %s', json.dumps(split_metrics))


if __name__ == '__main__':
    main()
