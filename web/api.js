const API_BASE = "/api";

async function asJson(respPromise) {
  const resp = await respPromise;
  if (!resp.ok) {
    let detail = resp.statusText;
    try {
      const body = await resp.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      /* ignore */
    }
    throw new Error(`${resp.status}: ${detail}`);
  }
  return resp.json();
}

export const api = {
  createSession(name) {
    return asJson(
      fetch(`${API_BASE}/sessions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      })
    );
  },
  listSessions() {
    return asJson(fetch(`${API_BASE}/sessions`));
  },
  getSession(id) {
    return asJson(fetch(`${API_BASE}/sessions/${id}`));
  },
  deleteSession(id) {
    return asJson(fetch(`${API_BASE}/sessions/${id}`, { method: "DELETE" }));
  },
  uploadPhoto(sessionId, blob, filename) {
    const form = new FormData();
    form.append("file", blob, filename);
    return asJson(
      fetch(`${API_BASE}/sessions/${sessionId}/photos`, { method: "POST", body: form })
    );
  },
  listPhotos(sessionId) {
    return asJson(fetch(`${API_BASE}/sessions/${sessionId}/photos`));
  },
  deletePhoto(sessionId, filename) {
    return asJson(
      fetch(`${API_BASE}/sessions/${sessionId}/photos/${encodeURIComponent(filename)}`, {
        method: "DELETE",
      })
    );
  },
  photoUrl(sessionId, filename) {
    return `${API_BASE}/sessions/${sessionId}/photos/${encodeURIComponent(filename)}`;
  },
  reconstruct(sessionId, mode) {
    return asJson(
      fetch(`${API_BASE}/sessions/${sessionId}/reconstruct`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode }),
      })
    );
  },
  getJob(sessionId) {
    return asJson(fetch(`${API_BASE}/sessions/${sessionId}/job`));
  },
  resultUrl(sessionId, filename) {
    return `${API_BASE}/sessions/${sessionId}/result/${encodeURIComponent(filename)}`;
  },
  applyScale(sessionId, pointA, pointB, realWorldDistanceM) {
    return asJson(
      fetch(`${API_BASE}/sessions/${sessionId}/scale`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          point_a: pointA,
          point_b: pointB,
          real_world_distance_m: realWorldDistanceM,
        }),
      })
    );
  },
};
