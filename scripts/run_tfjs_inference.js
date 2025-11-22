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
    framesDir: 'outputs/frames',
    output: 'outputs/detections/pilot_plant.json',
    labels: 'web/models/tfjs_baseline/tensorflow_automl_model/model-197536060022980608_tf-js_2023-05-04T05_50_49.038047Z_dict.txt',
    videoName: 'Pilot_plant.mp4',
    thresholds: null,
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
  for (const entry of manifest) {
    const framePath = path.join(framesDir, entry.filename);
    const imgBuffer = await fs.readFile(framePath);
    let imageTensor = tf.node.decodeImage(imgBuffer, 3);
    let inputTensor = imageTensor
      .toFloat()
      .div(tf.scalar(255))
      .expandDims(0);

    const frameStart = process.hrtime.bigint();
    const pred = await model.executeAsync(inputTensor);
    const frameEnd = process.hrtime.bigint();
    const elapsedMs = Number(frameEnd - frameStart) / 1e6;

    const detections = [];

    const boxesTensor = Array.isArray(pred) ? pred[0] : pred;
    const classesTensor = Array.isArray(pred) ? pred[1] : null;
    const scoresTensor = Array.isArray(pred) ? pred[2] : null;
    let boxes = await boxesTensor.array();
    let classes = classesTensor ? await classesTensor.array() : [];
    let scores = scoresTensor ? await scoresTensor.array() : [];

    const detectionsCount = boxes[0]?.length || 0;
    for (let i = 0; i < detectionsCount; i++) {
      const score = scores[0]?.[i] ?? 0;
      const classIdx = Math.round(classes[0]?.[i] ?? -1);
      const label = labels[classIdx] || `class_${classIdx}`;
      const threshold = getThreshold(label, thresholdMap);
      if (score < threshold) {
        continue;
      }
      const bbox = boxes[0][i];
      const detId = makeDetId(entry.frame_index, bbox, label);
      const conf = score;
      const logit = Math.log(conf / Math.max(1 - conf, 1e-6));
      detections.push({
        id: detId,
        cls: label,
        conf,
        bbox,
        logit,
      });
    }

    console.log(
      `Frame ${entry.frame_index} @ ${entry.timestamp_sec}s -> ${detections.length} dets (${elapsedMs.toFixed(
        2
      )} ms)`
    );

    results.push({
      i: entry.frame_index,
      t: entry.timestamp_sec,
      dets: detections,
    });

    if (Array.isArray(pred)) {
      pred.forEach((tensor) => tensor.dispose());
    } else {
      pred.dispose();
    }
    inputTensor.dispose();
    imageTensor.dispose();
  }

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
