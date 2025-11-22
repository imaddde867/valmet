const VIDEO_SRC = "../assets/videos/Pilot_plant.mp4";
const DETECTIONS_SRC = "../outputs/detections/pilot_plant.json";
const DEFAULT_FPS = 30; // matches Pilot_plant.mp4 metadata

const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const confSlider = document.getElementById("conf-slider");
const confValue = document.getElementById("conf-value");
const statsDiv = document.getElementById("stats");
const filtersDiv = document.getElementById("class-filters");

const detectionsByFrame = new Map();
const enabledClasses = new Set();
const allClasses = new Set();

function getColor(cls) {
  const colors = ["#ff6b6b", "#4ecdc4", "#ffd166", "#1a8fe3", "#c77dff", "#ff9f1c"];
  const idx = Math.abs(cls.split("").reduce((acc, ch) => acc + ch.charCodeAt(0), 0)) % colors.length;
  return colors[idx];
}

async function loadDetections() {
  const resp = await fetch(DETECTIONS_SRC);
  if (!resp.ok) {
    throw new Error(`Failed to load detections (${resp.status})`);
  }
  const payload = await resp.json();
  (payload.frames || []).forEach((frame) => {
    const frameIdx = frame.i ?? frame.frame_index;
    if (typeof frameIdx !== "number") {
      return;
    }
    const dets = (frame.dets || frame.detections || []).map((det) => ({
      cls: det.cls || det.class,
      conf: det.conf ?? det.score ?? 0,
      bbox: det.bbox || det.box || [],
    }));
    detectionsByFrame.set(frameIdx, dets);
    dets.forEach((det) => det.cls && allClasses.add(det.cls));
  });
}

function buildClassFilters() {
  filtersDiv.innerHTML = "";
  allClasses.forEach((cls) => {
    enabledClasses.add(cls);
    const label = document.createElement("label");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = true;
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) {
        enabledClasses.add(cls);
      } else {
        enabledClasses.delete(cls);
      }
      drawFrame();
    });
    label.appendChild(checkbox);
    label.appendChild(document.createTextNode(` ${cls}`));
    filtersDiv.appendChild(label);
    filtersDiv.appendChild(document.createElement("br"));
  });
}

function updateConfValue() {
  confValue.textContent = Number(confSlider.value).toFixed(2);
}

function getCurrentFrameIndex() {
  return Math.round(video.currentTime * DEFAULT_FPS);
}

function projectBox(box) {
  if (!Array.isArray(box) || box.length < 4) {
    return null;
  }
  const [yMin, xMin, yMax, xMax] = box;
  const clamp = (val) => Math.min(1, Math.max(0, val));
  const left = clamp(xMin) * canvas.width;
  const top = clamp(yMin) * canvas.height;
  const right = clamp(xMax) * canvas.width;
  const bottom = clamp(yMax) * canvas.height;
  return {
    x: left,
    y: top,
    w: Math.max(0, right - left),
    h: Math.max(0, bottom - top),
  };
}

function drawFrame() {
  if (!canvas.width || !canvas.height) {
    return;
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const threshold = Number(confSlider.value);
  const frameIdx = getCurrentFrameIndex();
  const dets = (detectionsByFrame.get(frameIdx) || []).filter(
    (det) => det.conf >= threshold && enabledClasses.has(det.cls)
  );

  ctx.lineWidth = 2;
  ctx.font = "12px Inter, sans-serif";
  ctx.textBaseline = "top";

  dets.forEach((det) => {
    const projected = projectBox(det.bbox);
    if (!projected) {
      return;
    }
    const color = getColor(det.cls);
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.strokeRect(projected.x, projected.y, projected.w, projected.h);
    const label = `${det.cls} ${(det.conf * 100).toFixed(1)}%`;
    ctx.fillText(label, projected.x + 4, projected.y + 4);
  });

  updateStats(frameIdx, dets);

  if (!video.paused && !video.ended) {
    requestAnimationFrame(drawFrame);
  }
}

function updateStats(frameIdx, dets) {
  if (!dets.length) {
    statsDiv.textContent = `Frame ${frameIdx}: no detections at threshold`;
    return;
  }
  const counts = dets.reduce((acc, det) => {
    acc[det.cls] = (acc[det.cls] || 0) + 1;
    return acc;
  }, {});
  const summary = Object.entries(counts)
    .map(([cls, count]) => `${cls}×${count}`)
    .join(", ");
  statsDiv.textContent = `Frame ${frameIdx}: ${dets.length} detections (${summary})`;
}

async function init() {
  video.src = VIDEO_SRC;
  await loadDetections();
  buildClassFilters();
  updateConfValue();
}

confSlider.addEventListener("input", () => {
  updateConfValue();
  drawFrame();
});

video.addEventListener("loadedmetadata", () => {
  canvas.width = video.videoWidth || video.clientWidth;
  canvas.height = video.videoHeight || video.clientHeight;
});

video.addEventListener("play", () => {
  requestAnimationFrame(drawFrame);
});

video.addEventListener("pause", () => {
  drawFrame();
});

video.addEventListener("seeked", () => {
  drawFrame();
});

init().catch((err) => {
  statsDiv.textContent = `Failed to load assets: ${err.message}`;
});
