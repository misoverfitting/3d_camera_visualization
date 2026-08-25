// Guided phone-camera capture: live video preview + a short recorded orbit
// video, with live coaching based on the practical capture tips (lighting /
// surfaces / coverage / scale) from the "Phone-Based 3D Capture: The 2026
// Landscape" report. Recording a slow orbit is far less fiddly than tapping
// a shutter 60-150 times, and tends to give more even coverage too - the
// server extracts and picks the sharp frames it needs from the video
// (see server/app/pipeline/video_extract.py).

export const MIN_RECORD_SECONDS = 15;
export const RECOMMENDED_RECORD_SECONDS = 30;
export const MAX_RECORD_SECONDS = 60;

const PREFERRED_MIME_TYPES = [
  "video/webm;codecs=vp9",
  "video/webm;codecs=vp8",
  "video/webm",
  "video/mp4",
];

function pickSupportedMimeType() {
  if (typeof MediaRecorder === "undefined") return "";
  for (const type of PREFERRED_MIME_TYPES) {
    if (MediaRecorder.isTypeSupported(type)) return type;
  }
  return ""; // let the browser pick its own default
}

function extensionForMimeType(mimeType) {
  if (mimeType.includes("mp4")) return ".mp4";
  return ".webm";
}

export class CaptureController {
  constructor(videoEl) {
    this.video = videoEl;
    this.stream = null;
    this.facingMode = "environment";
    this.recorder = null;
    this.chunks = [];
    this.recordingStartedAt = null;
  }

  async start() {
    await this.stop();
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: this.facingMode, width: { ideal: 1920 }, height: { ideal: 1440 } },
      audio: false,
    });
    this.video.srcObject = this.stream;
    await this.video.play();
  }

  async stop() {
    if (this.recorder && this.recorder.state !== "inactive") {
      this.recorder.stop();
    }
    if (this.stream) {
      this.stream.getTracks().forEach((t) => t.stop());
      this.stream = null;
    }
  }

  async switchCamera() {
    const wasRecording = this.isRecording();
    if (wasRecording) this.cancelRecording();
    this.facingMode = this.facingMode === "environment" ? "user" : "environment";
    await this.start();
  }

  isRecording() {
    return !!this.recorder && this.recorder.state === "recording";
  }

  startRecording() {
    if (!this.stream) throw new Error("Camera not started");
    const mimeType = pickSupportedMimeType();
    this.recorder = mimeType
      ? new MediaRecorder(this.stream, { mimeType })
      : new MediaRecorder(this.stream);
    this.chunks = [];
    this.recorder.addEventListener("dataavailable", (ev) => {
      if (ev.data.size > 0) this.chunks.push(ev.data);
    });
    this.recorder.start();
    this.recordingStartedAt = Date.now();
  }

  /** Stops recording and resolves with { blob, filename }. */
  stopRecording() {
    return new Promise((resolve, reject) => {
      if (!this.recorder) return reject(new Error("Not recording"));
      const mimeType = this.recorder.mimeType || "video/webm";
      this.recorder.addEventListener(
        "stop",
        () => {
          const blob = new Blob(this.chunks, { type: mimeType });
          this.chunks = [];
          resolve({ blob, filename: `orbit${extensionForMimeType(mimeType)}` });
        },
        { once: true }
      );
      this.recorder.stop();
    });
  }

  cancelRecording() {
    if (this.recorder && this.recorder.state !== "inactive") this.recorder.stop();
    this.chunks = [];
    this.recordingStartedAt = null;
  }

  elapsedSeconds() {
    if (!this.recordingStartedAt) return 0;
    return (Date.now() - this.recordingStartedAt) / 1000;
  }
}

/** Returns a live coaching tip for the current recording duration, cycling
 * through orbit/height reminders (report guidance: orbit at multiple
 * heights, heavy overlap - much easier to nudge for continuously during a
 * recording than it was between individual shutter taps). */
export function tipForElapsedSeconds(seconds) {
  if (seconds < 2) return "Center the object, then start orbiting slowly";
  const cyclePosition = Math.floor(seconds / 8) % 3;
  if (seconds >= MAX_RECORD_SECONDS - 3) return "Wrap up the orbit and tap Stop";
  if (cyclePosition === 0) return "Keep orbiting slowly, all the way around";
  if (cyclePosition === 1) return "Now raise your phone a bit higher";
  return "Now lower your phone a bit";
}

export function timerClass(seconds) {
  if (seconds < MIN_RECORD_SECONDS) return "low";
  if (seconds < RECOMMENDED_RECORD_SECONDS) return "mid";
  return "good";
}
