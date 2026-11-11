const $ = (id) => document.getElementById(id);
const state = { info: null, split: 'train', selected: 0, page: 0, pageSize: 20, mode: 'dataset', file: null, busy: false, request: 0, detail: null };

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}
function showError(error) { $('error').textContent = error.message; $('error').hidden = false; }
function clearError() { $('error').hidden = true; }
function status(message) { $('status').textContent = message; }
function setBusy(busy) {
  state.busy = busy;
  $('run-example').disabled = busy || !state.detail;
  $('run-upload').disabled = busy || !state.file;
  $('run-example').textContent = busy ? 'Separating…' : 'Run separation ↗';
  $('run-upload').textContent = busy ? 'Separating…' : 'Separate audio ↗';
  $('audio-file').disabled = busy;
  $('comparison').classList.toggle('loading', busy);
}
function waveform(peaks, label) {
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', '0 0 800 80');
  svg.setAttribute('preserveAspectRatio', 'none');
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', `${label} waveform`);
  const peak = Math.max(0.001, ...peaks.flat().map(Math.abs));
  const path = document.createElementNS(ns, 'path');
  path.setAttribute('d', peaks.map(([low, high], i) => {
    const x = (i + 0.5) * 800 / peaks.length;
    return `M${x.toFixed(2)},${(40 - high / peak * 35).toFixed(2)}V${(40 - low / peak * 35).toFixed(2)}`;
  }).join(' '));
  path.setAttribute('stroke', 'currentColor');
  path.setAttribute('stroke-width', '1.3');
  svg.append(path);
  return svg;
}
function track(data, className = '') {
  const element = document.createElement('div');
  element.className = `track ${className}`;
  const top = document.createElement('div'); top.className = 'track-top';
  const title = document.createElement('span'); title.textContent = data.label;
  const download = document.createElement('a'); download.href = data.url;
  download.download = `${data.label.replaceAll(' ', '_')}.wav`; download.textContent = 'Download ↓';
  top.append(title, download);
  const audio = document.createElement('audio'); audio.controls = true; audio.preload = 'metadata';
  audio.src = data.url; audio.setAttribute('aria-label', `Play ${data.label}`);
  element.append(top, waveform(data.peaks, data.label), audio);
  return element;
}
function renderComparison(references, result = null) {
  $('comparison').replaceChildren();
  references.forEach((reference, i) => {
    const row = document.createElement('div'); row.className = 'speaker-row';
    row.append(track(reference));
    if (result) row.append(track(result.tracks[i], 'model-track'));
    else {
      const pending = document.createElement('div'); pending.className = 'pending-track';
      const text = document.createElement('div'); text.textContent = `Speaker ${i + 1} estimate`;
      const note = document.createElement('span'); note.textContent = 'Run separation to listen';
      pending.append(text, note); row.append(pending);
    }
    $('comparison').append(row);
  });
  $('sample-metrics').replaceChildren();
  $('sample-metrics').hidden = !result;
  if (result?.metrics) {
    [['SI-SDR', `${result.metrics.si_sdr.toFixed(2)} dB`], ['Improvement over mixture', `${result.metrics.si_sdri.toFixed(2)} dB`], ['Separation time', `${result.seconds.toFixed(2)} s`]].forEach(([label, value]) => {
      const item = document.createElement('span'); const strong = document.createElement('strong');
      strong.textContent = value; item.append(strong, document.createTextNode(` ${label}`)); $('sample-metrics').append(item);
    });
  }
}
function renderList() {
  const query = $('search').value.trim().toLowerCase();
  const rows = state.info.splits[state.split].filter(row => row.id.toLowerCase().includes(query));
  const pages = Math.max(1, Math.ceil(rows.length / state.pageSize));
  state.page = Math.min(state.page, pages - 1);
  $('sample-count').textContent = `${rows.length} examples`;
  $('page').textContent = `${state.page + 1} / ${pages}`;
  $('previous').disabled = state.page === 0;
  $('next').disabled = state.page + 1 === pages;
  $('sample-list').replaceChildren();
  if (!rows.length) { $('sample-list').textContent = 'No matching examples.'; return; }
  rows.slice(state.page * state.pageSize, (state.page + 1) * state.pageSize).forEach(row => {
    const button = document.createElement('button');
    button.className = `sample ${row.index === state.selected ? 'active' : ''}`;
    button.setAttribute('aria-pressed', String(row.index === state.selected));
    const number = document.createElement('span'); number.className = 'number'; number.textContent = String(row.index + 1).padStart(3, '0');
    const identifier = document.createElement('span'); identifier.className = 'identifier';
    identifier.textContent = row.id.replaceAll('_', ' / ');
    const duration = document.createElement('small'); duration.textContent = `${row.duration.toFixed(2)} seconds · 3 speakers`;
    identifier.append(duration); button.append(number, identifier);
    button.addEventListener('click', () => selectExample(row.index));
    $('sample-list').append(button);
  });
}
async function selectExample(index) {
  state.selected = index; state.detail = null;
  const request = ++state.request;
  renderList(); clearError(); $('run-example').disabled = true;
  $('sample-metrics').hidden = true;
  $('dataset-mixture').replaceChildren(); $('comparison').replaceChildren();
  $('mixture-id').textContent = '';
  status('Loading mixture and reference recordings…');
  try {
    const data = await api(`/api/example?split=${state.split}&index=${index}`);
    if (request !== state.request) return;
    state.detail = data;
    $('split-label').textContent = `${state.split === 'train' ? 'Training' : 'Validation'} example`;
    $('example-number').textContent = `${String(index + 1).padStart(3, '0')} / ${state.info.splits[state.split].length}`;
    $('mixture-id').textContent = data.id;
    $('duration').textContent = `${data.mixture.duration.toFixed(2)} seconds`;
    $('dataset-mixture').replaceChildren(track(data.mixture, 'mixture-track'));
    renderComparison(data.references, data.prediction);
    $('run-example').disabled = state.busy;
    status(data.prediction ? 'Saved prediction loaded. Listen to each reference and its matched estimate.' : 'References ready. Run separation to hear what the model learned.');
  } catch (error) { if (request === state.request) { showError(error); status('Could not load this example.'); } }
}
async function runExample() {
  if (state.busy || !state.detail) return;
  const split = state.split, index = state.selected;
  setBusy(true); clearError(); status('Separating the mixture and matching the three estimates to the references…');
  try {
    const result = await api('/api/separate', { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'separation-viewer' }, body: JSON.stringify({ split, index }) });
    if (state.mode === 'dataset' && state.split === split && state.selected === index && state.detail) {
      renderComparison(state.detail.references, result);
      status('Separation complete. Compare the three matched speakers below.');
    }
  } catch (error) { showError(error); status('Separation did not finish. You can try again.'); }
  finally { setBusy(false); }
}
function changeMode(mode) {
  state.mode = mode; clearError();
  const dataset = mode === 'dataset';
  $('dataset-panel').hidden = !dataset; $('library-controls').hidden = !dataset;
  $('upload-panel').hidden = dataset; $('upload-library-note').hidden = dataset;
  for (const name of ['dataset', 'upload']) {
    const active = (name === 'dataset') === dataset;
    $(`${name}-mode`).classList.toggle('active', active);
    $(`${name}-mode`).setAttribute('aria-pressed', String(active));
  }
  $('heading').innerHTML = dataset ? 'One mixture.<br>Three voices.' : 'Your audio.<br>A new perspective.';
  $('example-number').hidden = !dataset;
  status(state.busy ? 'Separation is in progress…' : dataset ? 'Choose an example, then compare the reference and model tracks.' : 'Upload a mixture to get three separated tracks.');
}
function chooseFile(file) {
  if (state.busy) return;
  clearError();
  if (file && file.size > state.info.max_upload_mb * 1024 ** 2) {
    showError(new Error(`Choose a file smaller than ${state.info.max_upload_mb} MB.`)); return;
  }
  state.file = file || null;
  $('selected-file').textContent = file ? `${file.name} · ${(file.size / 1024 ** 2).toFixed(2)} MB` : 'No file selected';
  $('run-upload').disabled = !file;
  $('upload-results').hidden = true;
}
async function runUpload(event) {
  event.preventDefault();
  if (!state.file || state.busy) return;
  setBusy(true); clearError(); status('Uploading and separating your audio. This may take a moment…');
  try {
    const result = await api(`/api/upload?name=${encodeURIComponent(state.file.name)}`, { method: 'POST', headers: { 'Content-Type': 'application/octet-stream', 'X-Requested-With': 'separation-viewer' }, body: state.file });
    $('upload-mixture').replaceChildren(track(result.mixture, 'mixture-track'));
    $('upload-tracks').replaceChildren(...result.tracks.map(t => track(t)));
    $('upload-timing').textContent = `Separated in ${result.seconds.toFixed(2)} seconds. Listen or download each track.`;
    $('upload-results').hidden = false;
    status('Your three separated tracks are ready.');
  } catch (error) { showError(error); status('Could not separate this file. Check the message and try again.'); }
  finally { setBusy(false); }
}
function history(info) {
  if (!info.history.length) { $('history-summary').textContent = 'No epoch history available.'; return; }
  const rows = info.history;
  const values = rows.flatMap(r => [-r['train/loss'], r['validation/si_sdr']]);
  const low = Math.min(...values), high = Math.max(...values) + 0.1;
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 250 100'); svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', 'Training and validation SI-SDR over completed epochs');
  for (const [field, color, dashed] of [['train/loss', 'var(--muted)', true], ['validation/si_sdr', 'var(--accent)', false]]) {
    const line = document.createElementNS(svg.namespaceURI, 'polyline');
    line.setAttribute('points', rows.map((r, i) => `${10 + i / Math.max(1, rows.length - 1) * 230},${90 - ((field === 'train/loss' ? -r[field] : r[field]) - low) / (high - low) * 80}`).join(' '));
    line.setAttribute('fill', 'none'); line.setAttribute('stroke', color); line.setAttribute('stroke-width', '2');
    if (dashed) line.setAttribute('stroke-dasharray', '4 3'); svg.append(line);
  }
  $('history').append(svg);
  const last = rows.at(-1);
  $('history-summary').textContent = `${rows.length} epochs logged. Dashed: training; solid: validation. Last epoch: ${(-last['train/loss']).toFixed(2)} / ${last['validation/si_sdr'].toFixed(2)} dB.`;
}
document.addEventListener('play', event => {
  if (event.target.tagName === 'AUDIO') document.querySelectorAll('audio').forEach(audio => { if (audio !== event.target) audio.pause(); });
}, true);
$('dataset-mode').addEventListener('click', () => changeMode('dataset'));
$('upload-mode').addEventListener('click', () => changeMode('upload'));
$('split').addEventListener('change', () => { state.split = $('split').value; state.page = 0; $('search').value = ''; selectExample(0); });
$('search').addEventListener('input', () => { state.page = 0; renderList(); });
$('previous').addEventListener('click', () => { state.page--; renderList(); });
$('next').addEventListener('click', () => { state.page++; renderList(); });
$('run-example').addEventListener('click', runExample);
$('audio-file').addEventListener('change', event => chooseFile(event.target.files[0]));
$('upload-form').addEventListener('submit', runUpload);
for (const name of ['dragenter', 'dragover']) $('dropzone').addEventListener(name, event => { event.preventDefault(); $('dropzone').classList.add('dragover'); });
for (const name of ['dragleave', 'drop']) $('dropzone').addEventListener(name, event => { event.preventDefault(); $('dropzone').classList.remove('dragover'); });
$('dropzone').addEventListener('drop', event => chooseFile(event.dataTransfer.files[0]));
api('/api/info').then(info => {
  state.info = info;
  $('device').textContent = `${info.device.toUpperCase()} ready`;
  $('checkpoint').textContent = `Best model · epoch ${info.epoch}`;
  $('checkpoint-score').textContent = `${info.validation_si_sdr.toFixed(2)} dB validation SI-SDR`;
  $('checkpoint').title = info.run;
  $('upload-limit').textContent = `WAV, MP3, FLAC, OGG, M4A & more · up to ${info.max_upload_seconds}s / ${info.max_upload_mb} MB`;
  history(info); selectExample(0);
}).catch(error => { showError(error); status('Could not load the listening room. Check the server terminal.'); });
