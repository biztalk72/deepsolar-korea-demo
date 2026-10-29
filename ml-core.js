/** Dependency-free synthetic imagery and a real browser logistic pixel baseline.
 * This is neither the original DeepSolar model nor a Korean satellite benchmark.
 * Class-balanced full-batch training uses training labels only. Threshold is fixed.
 */
export const SIZE = 64;
export const DATASET_VERSION = 'synthetic-v1';
const FEATURE_COUNT = 9;
const MODEL_FORMAT = 'deepsolar-browser-logistic-v1';
const FEATURE_NAMES = ['red', 'green', 'blue', 'blue−red', 'blue−green', 'brightness', 'chroma', 'local texture', 'bias'];
const VALID_SPLITS = new Set(['train', 'validation', 'test']);

function randomGenerator(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6D2B79F5;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const clampByte = value => Math.max(0, Math.min(255, Math.round(value)));
const sigmoid = value => 1 / (1 + Math.exp(-Math.max(-40, Math.min(40, value))));
const now = () => typeof performance === 'undefined' ? Date.now() : performance.now();

/** Every scene is generated locally; names identify synthetic scenarios only. */
export function createDataset(seed = 72) {
  if (!Number.isInteger(seed) || seed < 0 || seed > 4294967295) throw new Error('시드는 0~4294967295 정수여야 합니다.');
  const rand = randomGenerator(seed);
  const tiles = [];
  const sceneNames = ['산업 지붕 · 합성', '농촌 창고 · 합성', '도심 옥상 · 합성', '소형 주택 · 합성'];
  for (let index = 0; index < 24; index++) {
    const sceneIndex = index % 4;
    const rgb = new Uint8ClampedArray(SIZE * SIZE * 3);
    const mask = new Uint8Array(SIZE * SIZE);
    const roofX = 8 + Math.floor(rand() * 6), roofY = 8 + Math.floor(rand() * 5);
    const roofW = sceneIndex === 3 ? 34 : 42;
    const roofH = sceneIndex === 3 ? 35 : 41;
    const base = 115 + rand() * 38;
    const roof = sceneIndex === 1 ? [base + 16, base + 5, base - 14] : [base, base + 4, base + 5];
    const paintRect = (x0, y0, width, height, color, label = 0, noise = 5) => {
      for (let y = Math.max(0, y0); y < Math.min(SIZE, y0 + height); y++) {
        for (let x = Math.max(0, x0); x < Math.min(SIZE, x0 + width); x++) {
          const pixel = y * SIZE + x;
          const grain = (rand() - .5) * noise;
          for (let c = 0; c < 3; c++) rgb[pixel * 3 + c] = clampByte(color[c] + grain);
          mask[pixel] = label;
        }
      }
    };
    for (let y = 0; y < SIZE; y++) {
      for (let x = 0; x < SIZE; x++) {
        const p = y * SIZE + x, noise = (rand() - .5) * 22;
        const road = x > 56 || y > 57;
        const field = sceneIndex === 1;
        const bg = road ? [88, 91, 87] : field ? [113, 122, 78] : [77, 104, 78];
        for (let c = 0; c < 3; c++) rgb[3 * p + c] = clampByte(bg[c] + noise + (field ? (y % 5) * 2 : 0));
      }
    }
    // Building shadow, roof border, and roof. Edges remain distinct at native pixels.
    paintRect(roofX + 3, roofY + 4, roofW + 2, roofH + 2, [55, 62, 62], 0, 7);
    paintRect(roofX - 1, roofY - 1, roofW + 2, roofH + 2, [173, 174, 164], 0, 6);
    paintRect(roofX, roofY, roofW, roofH, roof, 0, 12);
    // Roof seams and a blue non-PV surface create background distractors.
    for (let y = roofY + 6; y < roofY + roofH; y += 8) paintRect(roofX, y, roofW, 1, roof.map(v => v - 12), 0, 4);
    if (index % 3 === 0) paintRect(roofX + roofW - 11, roofY + 3, 7, 8, [55, 98, 122], 0, 8);
    if (index % 4 === 2) paintRect(roofX + 4, roofY + roofH - 9, 8, 5, [67, 74, 78], 0, 6);
    // Three tiles intentionally have no PV. These are genuine negative examples.
    if (index % 8 !== 7) {
      const rows = sceneIndex === 3 ? 2 : 3;
      const cols = sceneIndex === 0 ? 4 : 3;
      const panelW = sceneIndex === 3 ? 5 : 6, panelH = 7;
      const xStart = roofX + 4 + Math.floor(rand() * 3), yStart = roofY + 5 + Math.floor(rand() * 3);
      const tint = rand() * 12 - 6;
      for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
          const px = xStart + col * (panelW + 2), py = yStart + row * (panelH + 2);
          paintRect(px + 1, py + 1, panelW + 1, panelH + 1, [38, 44, 48], 0, 3);
          paintRect(px, py, panelW, panelH, [31 + tint, 55 + tint, 79 + tint], 1, 10);
          for (let yy = py + 2; yy < py + panelH; yy += 3) paintRect(px, yy, panelW, 1, [54 + tint, 75 + tint, 94 + tint], 1, 6);
          if (index % 5 === 1) paintRect(px + panelW - 1, py, 1, panelH, [74, 91, 109], 1, 5);
        }
      }
    }
    // A small uncertain border is explicitly ignored in training AND scoring.
    for (let y = 0; y < 2; y++) for (let x = 0; x < 5; x++) mask[y * SIZE + x] = 255;
    const split = index < 16 ? 'train' : index < 20 ? 'validation' : 'test';
    tiles.push({id: `SYN-${String(index + 1).padStart(3, '0')}`, name: `합성 타일 ${String(index + 1).padStart(2, '0')}`, scene: sceneNames[sceneIndex], split, width: SIZE, height: SIZE, rgb, mask});
  }
  return {id: `${DATASET_VERSION}-${seed}`, seed, tiles};
}

function assertImage(rgb, width, height) {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1 || width * height > 1048576 || !rgb || rgb.length !== width * height * 3) {
    throw new Error('RGB 영상 크기가 올바르지 않습니다.');
  }
}

/** Area averaging followed by nearest-neighbor upsampling; no new detail is invented. */
export function degradeImage(rgb, width, height, factor = 1) {
  assertImage(rgb, width, height);
  if (![1, 2, 4, 8].includes(Number(factor))) throw new Error('해상도 배율은 1, 2, 4, 8 중 하나여야 합니다.');
  factor = Number(factor);
  if (factor === 1) return new Uint8ClampedArray(rgb);
  const result = new Uint8ClampedArray(rgb.length);
  for (let y0 = 0; y0 < height; y0 += factor) {
    for (let x0 = 0; x0 < width; x0 += factor) {
      const xmax = Math.min(width, x0 + factor), ymax = Math.min(height, y0 + factor);
      const sums = [0, 0, 0];
      for (let y = y0; y < ymax; y++) for (let x = x0; x < xmax; x++) {
        const i = 3 * (y * width + x);
        for (let c = 0; c < 3; c++) sums[c] += rgb[i + c];
      }
      const count = (xmax - x0) * (ymax - y0);
      const means = sums.map(v => Math.round(v / count));
      for (let y = y0; y < ymax; y++) for (let x = x0; x < xmax; x++) {
        const i = 3 * (y * width + x);
        for (let c = 0; c < 3; c++) result[i + c] = means[c];
      }
    }
  }
  return result;
}

function features(rgb, width, height) {
  assertImage(rgb, width, height);
  const count = width * height;
  const result = new Float32Array(count * FEATURE_COUNT);
  // Summed-area tables make a cropped 5×5 local box mean deterministic and fast.
  const stride = width + 1;
  const tables = [0, 1, 2].map(() => new Float64Array(stride * (height + 1)));
  for (let y = 0; y < height; y++) {
    const sums = [0, 0, 0];
    for (let x = 0; x < width; x++) {
      const pixel = (y * width + x) * 3, pos = (y + 1) * stride + x + 1;
      for (let c = 0; c < 3; c++) { sums[c] += rgb[pixel + c]; tables[c][pos] = tables[c][pos - stride] + sums[c]; }
    }
  }
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const p = y * width + x, k = p * FEATURE_COUNT;
    const r = rgb[p * 3] / 255, g = rgb[p * 3 + 1] / 255, b = rgb[p * 3 + 2] / 255;
    const x0 = Math.max(0, x - 2), x1 = Math.min(width, x + 3), y0 = Math.max(0, y - 2), y1 = Math.min(height, y + 3);
    const n = (x1 - x0) * (y1 - y0);
    let texture = 0;
    for (let c = 0; c < 3; c++) {
      const t = tables[c];
      const localMean = (t[y1 * stride + x1] - t[y1 * stride + x0] - t[y0 * stride + x1] + t[y0 * stride + x0]) / n;
      texture += Math.abs(rgb[p * 3 + c] - localMean) / 765;
    }
    result.set([r, g, b, b - r, b - g, (r + g + b) / 3, Math.max(r, g, b) - Math.min(r, g, b), texture, 1], k);
  }
  return result;
}

function validateModel(model) {
  if (!model || model.format !== MODEL_FORMAT || !['weights', 'featureMean', 'featureStd'].every(key => model[key]?.length === FEATURE_COUNT && Array.from(model[key]).every(Number.isFinite)) || model.featureStd.some(v => v <= 0)) {
    throw new Error('호환되는 브라우저 경량 모델이 아닙니다.');
  }
}

function probabilities(model, feat) {
  const result = new Float32Array(feat.length / FEATURE_COUNT);
  for (let p = 0; p < result.length; p++) {
    let z = 0;
    for (let j = 0; j < FEATURE_COUNT; j++) z += (feat[p * FEATURE_COUNT + j] - model.featureMean[j]) / model.featureStd[j] * model.weights[j];
    result[p] = sigmoid(z);
  }
  return result;
}

/** Pass an already degraded RGB image when evaluating a resolution experiment. */
export function predictMask(model, rgb, width, height) {
  validateModel(model);
  return probabilities(model, features(rgb, width, height));
}

function metricsFromCounts({tp = 0, fp = 0, fn = 0, tn = 0}) {
  return {precision: tp + fp ? tp / (tp + fp) : 0, recall: tp + fn ? tp / (tp + fn) : 0,
    f1: 2 * tp + fp + fn ? 2 * tp / (2 * tp + fp + fn) : 0,
    iou: tp + fp + fn ? tp / (tp + fp + fn) : 0, tp, fp, fn, tn, pixels: tp + fp + fn + tn};
}

export function evaluate(mask, probs, threshold = .5) {
  if (!mask || !probs || mask.length !== probs.length || !Number.isFinite(threshold) || threshold < 0 || threshold > 1) throw new Error('정답·예측 배열 또는 임계값이 올바르지 않습니다.');
  const counts = {tp: 0, fp: 0, fn: 0, tn: 0};
  for (let i = 0; i < mask.length; i++) {
    const truth = mask[i];
    if (truth === 255) continue;
    if (truth !== 0 && truth !== 1) throw new Error('마스크는 0, 1, 255만 사용할 수 있습니다.');
    if (!Number.isFinite(probs[i]) || probs[i] < 0 || probs[i] > 1) throw new Error('예측 확률이 올바르지 않습니다.');
    if (probs[i] >= threshold) counts[truth ? 'tp' : 'fp']++;
    else counts[truth ? 'fn' : 'tn']++;
  }
  return metricsFromCounts(counts);
}

/** A reproducibility label, not a cryptographic integrity signature. */
export function fingerprintDataset(tiles) {
  let hash = 2166136261;
  const add = value => { hash = Math.imul(hash ^ value, 16777619) >>> 0; };
  for (const tile of tiles) {
    const header = `${tile.id}|${tile.split}|${tile.width}|${tile.height}|`;
    for (let i = 0; i < header.length; i++) add(header.charCodeAt(i));
    for (const value of tile.rgb) add(value);
    for (const value of tile.mask) add(value);
  }
  return `fnv1a-${hash.toString(16).padStart(8, '0')}`;
}

export async function trainModel(tiles, options = {}, onProgress = () => {}) {
  const started = now();
  const epochs = Number(options.epochs ?? 80), learningRate = Number(options.learningRate ?? .2), factor = Number(options.factor ?? 1);
  if (!Number.isInteger(epochs) || epochs < 1 || epochs > 300 || !Number.isFinite(learningRate) || learningRate <= 0 || learningRate > 1 || ![1, 2, 4, 8].includes(factor)) throw new Error('학습 설정이 허용 범위를 벗어났습니다.');
  if (!Array.isArray(tiles) || !tiles.length || tiles.length > 128) throw new Error('학습할 타일이 필요합니다.');
  const prepared = tiles.map(tile => {
    assertImage(tile.rgb, tile.width, tile.height);
    if (!VALID_SPLITS.has(tile.split) || !tile.mask || tile.mask.length !== tile.width * tile.height) throw new Error('타일 분할 또는 마스크 크기가 올바르지 않습니다.');
    for (const label of tile.mask) if (![0, 1, 255].includes(label)) throw new Error('마스크는 0, 1, 255만 사용할 수 있습니다.');
    return {split: tile.split, mask: tile.mask, features: features(degradeImage(tile.rgb, tile.width, tile.height, factor), tile.width, tile.height)};
  });
  const trainTiles = prepared.filter(t => t.split === 'train');
  let count = 0, positives = 0;
  const mean = new Float64Array(FEATURE_COUNT), variance = new Float64Array(FEATURE_COUNT);
  // Statistics see only valid TRAIN pixels. Evaluation labels do not enter fitting.
  for (const tile of trainTiles) for (let p = 0; p < tile.mask.length; p++) {
    if (tile.mask[p] === 255) continue;
    count++; positives += tile.mask[p];
    for (let j = 0; j < FEATURE_COUNT - 1; j++) {
      const value = tile.features[p * FEATURE_COUNT + j];
      const delta = value - mean[j]; mean[j] += delta / count; variance[j] += delta * (value - mean[j]);
    }
  }
  if (!count || !positives || positives === count) throw new Error('학습 마스크에 태양광(1)과 배경(0)이 모두 필요합니다.');
  const std = Array.from(variance, v => Math.max(.01, Math.sqrt(v / count)));
  mean[FEATURE_COUNT - 1] = 0; std[FEATURE_COUNT - 1] = 1;
  const x = new Float32Array(count * FEATURE_COUNT), labels = new Uint8Array(count);
  let at = 0;
  for (const tile of trainTiles) for (let p = 0; p < tile.mask.length; p++) {
    if (tile.mask[p] === 255) continue;
    labels[at] = tile.mask[p];
    for (let j = 0; j < FEATURE_COUNT; j++) x[at * FEATURE_COUNT + j] = (tile.features[p * FEATURE_COUNT + j] - mean[j]) / std[j];
    at++;
  }
  const prevalence = positives / count;
  const positiveWeight = .5 / prevalence, negativeWeight = .5 / (1 - prevalence);
  const model = {format: MODEL_FORMAT, modelKind: 'browser_logistic_pixel_baseline', weights: Array(FEATURE_COUNT).fill(0), featureMean: Array.from(mean), featureStd: std,
    featureNames: FEATURE_NAMES.slice(), threshold: .5, factor, epochs, learningRate, l2: .01, trainingPixels: count,
    datasetFingerprint: fingerprintDataset(tiles), warning: '합성 데이터 브라우저 경량 기준선. 원본 DeepSolar·TinyUNet 모델이 아니며 한국 위성 영상 성능을 뜻하지 않습니다.'};
  const history = [], interval = Math.max(1, Math.ceil(epochs / 20));
  const scoreSplit = split => {
    const sums = {tp: 0, fp: 0, fn: 0, tn: 0};
    for (const tile of prepared.filter(t => t.split === split)) {
      const result = evaluate(tile.mask, probabilities(model, tile.features), model.threshold);
      for (const key of ['tp', 'fp', 'fn', 'tn']) sums[key] += result[key];
    }
    return metricsFromCounts(sums);
  };
  const trainingLoss = () => {
    let loss = 0;
    for (let p = 0; p < count; p++) {
      let z = 0;
      for (let j = 0; j < FEATURE_COUNT; j++) z += x[p * FEATURE_COUNT + j] * model.weights[j];
      const prob = sigmoid(z), label = labels[p], weight = label ? positiveWeight : negativeWeight;
      loss -= weight * (label ? Math.log(Math.max(1e-12, prob)) : Math.log(Math.max(1e-12, 1 - prob)));
    }
    return loss / count;
  };
  for (let epoch = 1; epoch <= epochs; epoch++) {
    const grad = new Float64Array(FEATURE_COUNT);
    for (let p = 0; p < count; p++) {
      const base = p * FEATURE_COUNT;
      let z = 0;
      for (let j = 0; j < FEATURE_COUNT; j++) z += x[base + j] * model.weights[j];
      const error = (sigmoid(z) - labels[p]) * (labels[p] ? positiveWeight : negativeWeight);
      for (let j = 0; j < FEATURE_COUNT; j++) grad[j] += x[base + j] * error;
    }
    for (let j = 0; j < FEATURE_COUNT; j++) model.weights[j] -= learningRate * (grad[j] / count + (j < FEATURE_COUNT - 1 ? model.l2 * model.weights[j] : 0));
    if (epoch === 1 || epoch % interval === 0 || epoch === epochs) {
      const point = {epoch, epochs, loss: trainingLoss(), val: scoreSplit('validation'), elapsedMs: Math.round(now() - started)};
      history.push(point); onProgress(point);
      // Yield genuine computation progress to the worker event loop, without timers that simulate training.
      await new Promise(resolve => setTimeout(resolve, 0));
    }
  }
  return {model, metrics: {train: scoreSplit('train'), validation: scoreSplit('validation'), test: scoreSplit('test')}, history,
    options: {epochs, learningRate, factor}, trainingPixels: count, elapsedMs: Math.round(now() - started)};
}
