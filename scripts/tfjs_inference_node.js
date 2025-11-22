const fs = require('fs/promises');
const path = require('path');
const crypto = require('crypto');
const util = require('util');
util.isNullOrUndefined =
  typeof util.isNullOrUndefined === 'function'
    ? util.isNullOrUndefined
    : (value) => value === null || value === undefined;
util.isArray = (...args) => Array.isArray(...args);

const tf = require('@tensorflow/tfjs-node');

function parseArgs(argv) {
  const args = {
    model: 'web/models/tfjs_baseline/tensorflow_automl_model/model-197536060022980608_tf-js_2023-05-04T05_50_49.038047Z_model.json',
    manifest: 'outputs/frame_manifest.json',
    framesDir: 'outputs/frames/highres',
    output: 'outputs/detections/pilot_plant.json',
    labels: 'web/models/tfjs_baseline/tensorflow_automl_model/model-197536060022980608_tf-js_2023-05-04T05_50_49.038047Z_dict.txt',
    videoName: 'Pilot_plant.mp4',
    thresholds: null,
    maxDets: 200,
    iouThreshold: 0.5,
    inputWidth: 224,
    inputHeight: 224,
  };

  for (let i = 0; i < argv.length; i++) {
    const key = argv[i];
    if (!key.startsWith('--')) continue;
    const value = argv[i + 1];
    switch (key) {
      case '--model':
        args.model = value;
        i++;
        break;
      case '--manifest':
        args.manifest = value;
        i++;
        break;
      case '--frames-dir':
        args.framesDir = value;
        i++;
        break;
      case '--output':
        args.output = value;
        i++;
        break;
      case '--labels':
        args.labels = value;
        i++;
        break;
      case '--video-name':
        args.videoName = value;
        i++;
        break;
      case '--thresholds':
        args.thresholds = value;
        i++;
        break;
      case '--max-dets':
        args.maxDets = Number(value);
        i++;
        break;
      case '--iou-threshold':
        args.iouThreshold = Number(value);
        i++;
        break;
      case '--input-width':
        args.inputWidth = Number(value);
        i++;
        break;
      case '--input-height':
        args.inputHeight = Number(value);
        i++;
        break;
      default:
        break;
    }
  }
  return args;
}

async function loadThresholds(thresholdOption) {
  if (!thresholdOption) {
    return {};
  }
  try {
    const candidate = path.resolve(thresholdOption);
    const buf = await fs.readFile(candidate, 'utf-8');
    return JSON.parse(buf);
  } catch (err) {
    // Try parse as JSON literal.
    try {
      return JSON.parse(thresholdOption);
    } catch (inner) {
      console.warn('Unable to parse thresholds option, falling back to defaults.');
      return {};
    }
  }
}

async function loadLabels(dictPath) {
  const content = await fs.readFile(dictPath, 'utf-8');
  return content
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
}

function getThreshold(label, thresholdMap, defaultValue = 0.5) {
  if (!thresholdMap || Object.keys(thresholdMap).length === 0) {
    return defaultValue;
  }
  if (label in thresholdMap) {
    return thresholdMap[label];
  }
  if ('_default' in thresholdMap) {
    return thresholdMap._default;
  }
  return defaultValue;
}

function makeDetId(frameIndex, bbox, clsLabel) {
  const hash = crypto
    .createHash('md5')
    .update(`${frameIndex}:${clsLabel}:${bbox.join(',')}`)
    .digest('hex');
  return hash.slice(0, 16);
}

function computeIoU(boxA, boxB) {
  const [yminA, xminA, ymaxA, xmaxA] = boxA;
  const [yminB, xminB, ymaxB, xmaxB] = boxB;
  const interYMin = Math.max(yminA, yminB);
  const interXMin = Math.max(xminA, xminB);
  const interYMax = Math.min(ymaxA, ymaxB);
  const interXMax = Math.min(xmaxA, xmaxB);
  const interArea = Math.max(0, interYMax - interYMin) * Math.max(0, interXMax - interXMin);
  const areaA = Math.max(0, ymaxA - yminA) * Math.max(0, xmaxA - xminA);
  const areaB = Math.max(0, ymaxB - yminB) * Math.max(0, xmaxB - xminB);
  const unionArea = areaA + areaB - interArea + 1e-9;
  return interArea / unionArea;
}

function applyNms(detections, maxDetections, iouThreshold) {
  const sorted = [...detections].sort((a, b) => b.conf - a.conf);
  const kept = [];
  for (const det of sorted) {
    let keep = true;
    for (const existing of kept) {
      if (det.cls === existing.cls && computeIoU(det.bbox, existing.bbox) > iouThreshold) {
        keep = false;
        break;
      }
    }
    if (keep) {
      kept.push(det);
      if (kept.length >= maxDetections) {
        break;
      }
    }
  }
  return kept;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const manifestPath = path.resolve(args.manifest);
  const framesDir = path.resolve(args.framesDir);
  const modelPath = path.resolve(args.model);
  const outputPath = path.resolve(args.output);
  const labelPath = path.resolve(args.labels);
  const thresholdMap = await loadThresholds(args.thresholds);
  const labels = await loadLabels(labelPath);

  const manifestBuf = await fs.readFile(manifestPath, 'utf-8');
  const manifest = JSON.parse(manifestBuf);

  const model = await tf.loadGraphModel(`file://${modelPath}`);
  console.log(`Loaded model from ${modelPath}`);

  const results = [];
  const normalizeScalar = tf.scalar(255);
  for (const entry of manifest) {
    const framePath = path.join(framesDir, entry.filename);
    const imgBuffer = await fs.readFile(framePath);
    let decoded = tf.node.decodeImage(imgBuffer, 3).toFloat();
    if (
      decoded.shape[0] !== args.inputHeight ||
      decoded.shape[1] !== args.inputWidth
    ) {
      const resized = tf.image.resizeBilinear(
        decoded,
        [args.inputHeight, args.inputWidth],
        true
      );
      decoded.dispose();
      decoded = resized;
    }
    const normalized = decoded.div(normalizeScalar);
    const inputTensor = normalized.expandDims(0);
    normalized.dispose();
    decoded.dispose();

    const frameStart = process.hrtime.bigint();
    const pred = await model.executeAsync(inputTensor);
    if (entry === manifest[0]) {
      if (Array.isArray(pred)) {
        console.log(
          'Output tensor shapes:',
          pred.map((tensor) => tensor.shape)
        );
      } else {
        console.log('Output tensor shape:', pred.shape);
      }
    }
    const frameEnd = process.hrtime.bigint();
    const elapsedMs = Number(frameEnd - frameStart) / 1e6;

    const detections = [];

    let boxesTensor;
    let scoresTensor;
    if (Array.isArray(pred)) {
      if (pred.length === 1) {
        boxesTensor = pred[0];
      } else {
        for (const tensor of pred) {
          const shape = tensor.shape || [];
          const rank = shape.length;
          const lastDim = shape[rank - 1];
          if ((rank === 2 && lastDim === 4) || (rank >= 3 && lastDim === 4)) {
            boxesTensor = tensor;
          } else if (rank >= 2) {
            scoresTensor = tensor;
          }
        }
      }
    } else {
      boxesTensor = pred;
    }

    if (!boxesTensor || !scoresTensor) {
      console.warn(
        'Unable to determine boxes/scores tensors for frame',
        entry.frame_index,
        'shapes',
        pred && Array.isArray(pred) ? pred.map((t) => t.shape) : boxesTensor?.shape
      );
      if (Array.isArray(pred)) {
        pred.forEach((tensor) => tensor.dispose());
      } else {
        pred.dispose();
      }
      inputTensor.dispose();
      continue;
    }

    const boxes = await boxesTensor.array();
    const scores = await scoresTensor.array();
    const boxList =
      Array.isArray(boxes[0]) && Array.isArray(boxes[0][0]) ? boxes[0] : boxes;
    const scoreList =
      Array.isArray(scores[0]) && Array.isArray(scores[0][0]) ? scores[0] : scores;
    const detectionsCount = Math.min(boxList.length || 0, scoreList.length || 0);

    for (let i = 0; i < detectionsCount; i++) {
      const classScores = scoreList[i] || [];
      const probabilities = classScores.map((logit) => 1 / (1 + Math.exp(-logit)));
      const backgroundOffset =
        labels.length && labels[0].toLowerCase() === 'background' ? 1 : 0;
      let bestIdx = -1;
      let bestScore = -Infinity;
      for (let idx = backgroundOffset; idx < probabilities.length; idx++) {
        const value = probabilities[idx];
        if (value > bestScore) {
          bestScore = value;
          bestIdx = idx;
        }
      }

      if (bestIdx < 0) {
        continue;
      }

      const label = labels[bestIdx] || `class_${bestIdx}`;
      const threshold = getThreshold(label, thresholdMap);
      if (bestScore < threshold) {
        continue;
      }

      const bbox = boxList[i];
      const detId = makeDetId(entry.frame_index, bbox, label);
      detections.push({
        id: detId,
        cls: label,
        conf: bestScore,
        bbox,
        logits: classScores,
      });
    }

    const kept = applyNms(detections, args.maxDets, args.iouThreshold);

    console.log(
      `Frame ${entry.frame_index} @ ${entry.timestamp_sec}s -> ${kept.length} dets (${elapsedMs.toFixed(
        2
      )} ms)`
    );

    results.push({
      i: entry.frame_index,
      t: entry.timestamp_sec,
      dets: kept,
    });

    if (Array.isArray(pred)) {
      pred.forEach((tensor) => tensor.dispose());
    } else {
      pred.dispose();
    }
    inputTensor.dispose();
  }
  normalizeScalar.dispose();

  const outputPayload = {
    video: args.videoName,
    frames: results,
  };
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.writeFile(outputPath, JSON.stringify(outputPayload, null, 2));
  console.log(`Saved detections to ${outputPath}`);
}

main().catch((err) => {
  console.error('Inference failed', err);
  process.exit(1);
});
