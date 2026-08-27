import { api } from "./api.js";
import {
  CaptureController,
  tipForElapsedSeconds,
  timerClass,
  MIN_RECORD_SECONDS,
  MAX_RECORD_SECONDS,
} from "./capture.js";
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

// ---------- Capture (record a short orbit video, server extracts frames) ----------

const capture = new CaptureController(document.getElementById("camera-video"));
let recordTimerHandle = null;

async function startCapture(sessionId) {
  state.sessionId = sessionId;
  showScreen("capture");
  resetCaptureUi();
  try {
    await capture.start();
  } catch (err) {
    alert("Could not access the camera: " + err.message);
  }
}

function resetCaptureUi() {
  document.getElementById("thumb-strip").innerHTML = "";
  document.getElementById("extract-status").hidden = true;
  document.getElementById("btn-done-capturing").hidden = true;
  document.getElementById("btn-record").disabled = false;
  document.getElementById("btn-record").classList.remove("recording");
  setRecTimerText("Ready to record", "");
  document.getElementById("capture-tip").textContent = "";
}

function setRecTimerText(text, cssClass) {
  const el = document.getElementById("rec-timer");
  el.textContent = text;
  el.className = `shot-counter ${cssClass}`;
}

document.getElementById("btn-record").addEventListener("click", async () => {
  if (!capture.isRecording()) {
    capture.startRecording();
    document.getElementById("btn-record").classList.add("recording");
    document.getElementById("btn-record").setAttribute("aria-label", "Stop recording");
    document.getElementById("btn-done-capturing").hidden = true;
    document.getElementById("extract-status").hidden = true;
    document.getElementById("thumb-strip").innerHTML = "";
    recordTimerHandle = setInterval(() => {
      const elapsed = capture.elapsedSeconds();
      setRecTimerText(`${elapsed.toFixed(0)}s`, timerClass(elapsed));
      document.getElementById("capture-tip").textContent = tipForElapsedSeconds(elapsed);
      if (elapsed >= MAX_RECORD_SECONDS) document.getElementById("btn-record").click();
    }, 250);
    return;
  }

  clearInterval(recordTimerHandle);
  document.getElementById("btn-record").classList.remove("recording");
  document.getElementById("btn-record").disabled = true;
  document.getElementById("capture-tip").textContent = "";

  const { blob, filename } = await capture.stopRecording();
  const statusEl = document.getElementById("extract-status");
  statusEl.hidden = false;
  statusEl.textContent = "Uploading and extracting frames...";
  try {
    const result = await api.uploadVideo(state.sessionId, blob, filename);
    state.photoCount = result.photo_count;
    statusEl.textContent = `Extracted ${result.photo_count} sharp frames from your orbit.`;
    await renderThumbs();
    document.getElementById("btn-done-capturing").hidden = false;
  } catch (err) {
    statusEl.textContent = "Couldn't process that video: " + err.message + ". Try recording again.";
  } finally {
    setRecTimerText("Ready to record", "");
    document.getElementById("btn-record").disabled = false;
    document.getElementById("btn-record").setAttribute("aria-label", "Start recording");
  }
});

document.getElementById("btn-switch-camera").addEventListener("click", () => capture.switchCamera());

document.getElementById("btn-done-capturing").addEventListener("click", async () => {
  await capture.stop();
  showScreen("mode");
  updatePhotoCountWarning(state.photoCount);
});

async function renderThumbs() {
  const strip = document.getElementById("thumb-strip");
  strip.innerHTML = "";
  const { photos } = await api.listPhotos(state.sessionId);
  for (const name of photos) {
    const img = document.createElement("img");
    img.src = api.photoUrl(state.sessionId, name);
    strip.appendChild(img);
  }
}

function updatePhotoCountWarning(count) {
  const el = document.getElementById("photo-count-warning");
  if (count < 30) {
    el.textContent = `Only ${count} frames extracted - try a longer, slower orbit (${MIN_RECORD_SECONDS}-${MAX_RECORD_SECONDS}s) for reliable reconstruction.`;
  } else {
    el.textContent = `${count} frames extracted from your orbit. Good coverage.`;
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

  const downloadPhotosBtn = document.getElementById("btn-download-photos");
  downloadPhotosBtn.href = api.photosZipUrl(sessionId);
  downloadPhotosBtn.setAttribute("download", `${sessionId}_photos.zip`);
  document.getElementById("reprocess-status").textContent = "";

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

// ---------- Reprocess (download photos -> process on a GPU machine -> upload result back) ----------

document.getElementById("btn-upload-result").addEventListener("click", () => {
  document.getElementById("input-upload-result").click();
});

document.getElementById("input-upload-result").addEventListener("change", async (ev) => {
  const file = ev.target.files[0];
  ev.target.value = ""; // allow re-selecting the same filename later
  if (!file) return;

  const statusEl = document.getElementById("reprocess-status");
  statusEl.textContent = `Uploading ${file.name}...`;
  try {
    const result = await api.uploadResult(state.sessionId, file);
    // showResult() re-renders the viewer and resets this status text, so
    // the success message has to be set *after* it resolves, not before -
    // otherwise it flashes and is immediately wiped.
    await showResult(state.sessionId);
    statusEl.textContent = `Uploaded. Now showing ${result.result_files.join(", ")}.`;
  } catch (err) {
    statusEl.textContent = "Upload failed: " + err.message;
  }
});

// ---------- Boot ----------

showScreen("sessions", { push: true });
refreshSessions();
