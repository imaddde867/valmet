const MODEL_URL = "models/tfjs_baseline/tensorflow_automl_model/model-197536060022980608_tf-js_2023-05-04T05_50_49.038047Z_model.json";
const MANIFEST_URL = "../outputs/frame_manifest.json";
const FRAMES_DIR = "../outputs/frames/";

let model;

async function loadModel() {
  model = await tf.loadGraphModel(MODEL_URL);
  document.getElementById("log").innerText = "Model Loaded!";
}

async function run() {
  try {
    await loadModel();
  } catch (err) {
    document.getElementById("log").innerText = `Model load failed: ${err}`;
    throw err;
  }

  const manifest = await fetch(MANIFEST_URL).then((r) => r.json());

  const canvas = document.getElementById("frameCanvas");
  const ctx = canvas.getContext("2d");

  const detections = [];

  for (const entry of manifest) {
    const img = new Image();
    img.src = FRAMES_DIR + entry.filename;
    await img.decode();

    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);

    const input = tf.browser
      .fromPixels(canvas)
      .expandDims(0)
      .toFloat()
      .div(255);

    const pred = await model.executeAsync(input);
    const boxes = pred[0].arraySync();
    const classes = pred[1].arraySync();
    const scores = pred[2].arraySync();

    detections.push({
      frame_index: entry.frame_index,
      timestamp_sec: entry.timestamp_sec,
      dets: boxes.map((b, i) => ({
        bbox: b,
        cls: classes[i],
        conf: scores[i],
      })),
    });

    pred.forEach((tensor) => tensor.dispose());
    tf.dispose(input);
  }

  const out = {
    video: "Pilot_plant.mp4",
    frames: detections,
  };

  download("pilot_plant.json", JSON.stringify(out, null, 2));

  document.getElementById("log").innerText = "DONE!";
}

function download(name, text) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: "application/json" }));
  a.download = name;
  a.click();
}

run();
