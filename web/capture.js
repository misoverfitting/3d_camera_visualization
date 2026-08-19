// Guided phone-camera capture: live video + shot-by-shot coaching, based on
// the practical capture tips (lighting / surfaces / coverage / scale) from
// the "Phone-Based 3D Capture: The 2026 Landscape" report.

export const MIN_RECOMMENDED_PHOTOS = 60;
export const MAX_RECOMMENDED_PHOTOS = 150;

export class CaptureController {
  constructor(videoEl, canvasEl) {
    this.video = videoEl;
    this.canvas = canvasEl;
    this.stream = null;
    this.facingMode = "environment";
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
    if (this.stream) {
      this.stream.getTracks().forEach((t) => t.stop());
      this.stream = null;
    }
  }

  async switchCamera() {
    this.facingMode = this.facingMode === "environment" ? "user" : "environment";
    await this.start();
  }

  /** Captures the current video frame as a JPEG Blob. */
  async capturePhoto() {
    const w = this.video.videoWidth;
    const h = this.video.videoHeight;
    this.canvas.width = w;
    this.canvas.height = h;
    const ctx = this.canvas.getContext("2d");
    ctx.drawImage(this.video, 0, 0, w, h);
    return new Promise((resolve) => this.canvas.toBlob(resolve, "image/jpeg", 0.92));
  }
}

/** Returns a live coaching tip for the current shot count, or "" for none.
 * Cycles through orbit/height/overlap reminders so a long capture session
 * keeps getting nudged toward the report's "orbit at multiple heights,
 * heavy overlap" guidance rather than one static ring of photos. */
export function tipForShotCount(count) {
  if (count === 0) return "Center the object, then start orbiting slowly";
  if (count > 0 && count % 15 === 0) {
    const cycle = Math.floor(count / 15) % 3;
    if (cycle === 0) return "Now raise your phone higher and continue orbiting";
    if (cycle === 1) return "Now lower your phone and continue orbiting";
    return "Keep overlap heavy - move a little between each shot";
  }
  if (count < MIN_RECOMMENDED_PHOTOS) return "";
  if (count === MIN_RECOMMENDED_PHOTOS) return "Good coverage! Keep going or tap Done";
  return "";
}

export function counterClass(count) {
  if (count < 20) return "low";
  if (count < MIN_RECOMMENDED_PHOTOS) return "mid";
  return "good";
}
