import { api } from "./api.js";
import { CaptureController, tipForShotCount, counterClass, MIN_RECOMMENDED_PHOTOS } from "./capture.js";
import { MeshViewer, SplatViewer } from "./viewer.js";

const screens = {
  sessions: document.getElementById("screen-sessions"),
  capture: document.getElementById("screen-capture"),
  mode: document.getElementById("screen-mode"),
  progress: document.getElementById("screen-progress"),
  viewer: document.getElementById("screen-viewer"),
};
const topbarTitle = document.getElementById("topbar-title");
const btnBack = document.getElementById("btn-back");

const state = {
  screenStack: [],
  sessionId: null,
  photoCount: 0,
  activeViewer: null,
  jobPollTimer: null,
  scalePoints: [],
};

function showScreen(name, { push = true, title } = {}) {
  Object.values(screens).forEach((el) => el.classList.remove("active"));
  screens[name].classList.add("active");
  if (push) state.screenStack.push(name);
  btnBack.hidden = state.screenStack.length <= 1;
  topbarTitle.textContent = title || "Phone 3D Capture";
  if (name !== "capture") capture.stop();
  if (name !== "viewer") teardownViewer();
}

btnBack.addEventListener("click", () => {
  state.screenStack.pop();
  const prev = state.screenStack.pop() || "sessions";
  showScreen(prev);
});

// ---------- Sessions list ----------

async function refreshSessions() {
  const list = document.getElementById("session-list");
  list.innerHTML = "";
  const sessions = await api.listSessions();
  for (const s of sessions) {
    const li = document.createElement("li");
    li.className = "session-item";
    li.innerHTML = `
      <div>
        <div class="name">${escapeHtml(s.name)}</div>
        <div class="meta">${s.photo_count} photos &middot; ${new Date(s.created_at).toLocaleString()}</div>
      </div>
      <span class="status-pill ${s.status}">${s.status}</span>
    `;
    li.addEventListener("click", () => openSession(s.id));
    list.appendChild(li);
  }
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

document.getElementById("form-new-session").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const input = document.getElementById("input-session-name");
  const session = await api.createSession(input.value.trim() || "Untitled capture");
  input.value = "";
  await startCapture(session.id);
});

async function openSession(sessionId) {
  const session = await api.getSession(sessionId);
  state.sessionId = sessionId;
  if (session.status === "done") {
    await showResult(sessionId);
  } else if (session.status === "running" || session.status === "queued") {
    showScreen("progress", { title: session.name });
    pollJob(sessionId);
  } else if (session.photo_count > 0) {
    showScreen("mode", { title: session.name });
    updatePhotoCountWarning(session.photo_count);
  } else {
    await startCapture(sessionId);
  }
}

// ---------- Capture ----------

const capture = new CaptureController(
  document.getElementById("camera-video"),
  document.getElementById("capture-canvas")
);

async function startCapture(sessionId) {
  state.sessionId = sessionId;
  showScreen("capture");
  await renderThumbs();
  try {
    await capture.start();
  } catch (err) {
    alert("Could not access the camera: " + err.message);
  }
}

document.getElementById("btn-shutter").addEventListener("click", async () => {
  const blob = await capture.capturePhoto();
  const filename = `photo_${Date.now()}.jpg`;
  const result = await api.uploadPhoto(state.sessionId, blob, filename);
  state.photoCount = result.photo_count;
  updateCaptureHud();
  addThumb(blob);
});

document.getElementById("btn-switch-camera").addEventListener("click", () => capture.switchCamera());

document.getElementById("btn-done-capturing").addEventListener("click", async () => {
  await capture.stop();
  showScreen("mode");
  updatePhotoCountWarning(state.photoCount);
});

function updateCaptureHud() {
  const counter = document.getElementById("shot-counter");
  counter.textContent = `${state.photoCount} photo${state.photoCount === 1 ? "" : "s"}`;
  counter.className = `shot-counter ${counterClass(state.photoCount)}`;
  document.getElementById("capture-tip").textContent = tipForShotCount(state.photoCount);
}

function addThumb(blob) {
  const img = document.createElement("img");
  img.src = URL.createObjectURL(blob);
  document.getElementById("thumb-strip").appendChild(img);
}

async function renderThumbs() {
  const strip = document.getElementById("thumb-strip");
  strip.innerHTML = "";
  const { photos } = await api.listPhotos(state.sessionId);
  state.photoCount = photos.length;
  for (const name of photos) {
    const img = document.createElement("img");
    img.src = api.photoUrl(state.sessionId, name);
    strip.appendChild(img);
  }
  updateCaptureHud();
}

function updatePhotoCountWarning(count) {
  const el = document.getElementById("photo-count-warning");
  if (count < MIN_RECOMMENDED_PHOTOS) {
    el.textContent = `Only ${count} photos captured - ${MIN_RECOMMENDED_PHOTOS}-150+ is recommended for reliable reconstruction.`;
  } else {
    el.textContent = `${count} photos captured. Good coverage.`;
  }
}

// ---------- Mode selection ----------

document.querySelectorAll(".mode-card").forEach((card) => {
  card.addEventListener("click", async () => {
    const mode = card.dataset.mode;
    await api.reconstruct(state.sessionId, mode);
    showScreen("progress");
    pollJob(state.sessionId);
  });
});

// ---------- Progress ----------

function pollJob(sessionId) {
  clearInterval(state.jobPollTimer);
  const tick = async () => {
    const job = await api.getJob(sessionId);
    document.getElementById("progress-fill").style.width = `${job.percent}%`;
    document.getElementById("progress-message").textContent = job.message;
    document.getElementById("progress-stage").textContent = job.stage;
    if (job.status === "done") {
      clearInterval(state.jobPollTimer);
      await showResult(sessionId);
    } else if (job.status === "error") {
      clearInterval(state.jobPollTimer);
      document.getElementById("progress-message").textContent = "Failed: " + job.message;
    }
  };
  tick();
  state.jobPollTimer = setInterval(tick, 2000);
}

// ---------- Viewer ----------

function teardownViewer() {
  if (state.activeViewer) {
    state.activeViewer.dispose();
    state.activeViewer = null;
  }
  document.getElementById("scale-tool").hidden = true;
  state.scalePoints = [];
}

async function showResult(sessionId) {
  showScreen("viewer");
  const job = await api.getJob(sessionId);
  const files = job.result_files || [];
  const splatFile = files.find((f) => f.endsWith("splat.ply"));
  const meshFile = files.find((f) => f === "model.ply") || files.find((f) => f === "model.obj");

  const container = document.getElementById("viewer-container");
  container.innerHTML = "";

  const scaleToggle = document.getElementById("btn-toggle-scale");
  const downloadBtn = document.getElementById("btn-download");

  if (splatFile) {
    const viewer = new SplatViewer(container);
    await viewer.load(api.resultUrl(sessionId, splatFile));
    state.activeViewer = viewer;
    scaleToggle.hidden = true; // scale calibration only implemented for meshes
    downloadBtn.href = api.resultUrl(sessionId, splatFile);
    downloadBtn.setAttribute("download", splatFile);
  } else if (meshFile) {
    const viewer = new MeshViewer(container);
    await viewer.load(api.resultUrl(sessionId, meshFile));
    state.activeViewer = viewer;
    scaleToggle.hidden = false;
    downloadBtn.href = api.resultUrl(sessionId, meshFile);
    downloadBtn.setAttribute("download", meshFile);
  } else {
    container.innerHTML = '<p class="muted" style="padding:1rem">No result files.</p>';
  }
}

document.getElementById("btn-toggle-scale").addEventListener("click", () => {
  const tool = document.getElementById("scale-tool");
  const willShow = tool.hidden;
  tool.hidden = !willShow;
  state.scalePoints = [];
  document.getElementById("scale-points-status").textContent = "Pick point A";
  document.getElementById("btn-apply-scale").disabled = true;
  if (willShow && state.activeViewer instanceof MeshViewer) {
    state.activeViewer.enablePicking((point) => {
      state.scalePoints.push(point);
      if (state.scalePoints.length === 1) {
        document.getElementById("scale-points-status").textContent = "Pick point B";
      } else if (state.scalePoints.length === 2) {
        document.getElementById("scale-points-status").textContent = "Points A & B set";
        document.getElementById("btn-apply-scale").disabled = false;
        state.activeViewer.disablePicking();
      }
    });
  } else if (state.activeViewer instanceof MeshViewer) {
    state.activeViewer.disablePicking();
  }
});

document.getElementById("btn-apply-scale").addEventListener("click", async () => {
  const distance = parseFloat(document.getElementById("scale-distance").value);
  if (!distance || state.scalePoints.length !== 2) return;
  const [a, b] = state.scalePoints;
  await api.applyScale(state.sessionId, [a.x, a.y, a.z], [b.x, b.y, b.z], distance);
  await showResult(state.sessionId);
});

// ---------- Boot ----------

showScreen("sessions", { push: true });
refreshSessions();
