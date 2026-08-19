// Two independent in-browser viewers:
//   MeshViewer  - plain three.js scene for OBJ/PLY meshes (the "accurate" path)
//   SplatViewer - wraps the open-source GaussianSplats3D library (the "compelling" path)
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";

export class MeshViewer {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x05070d);

    this.camera = new THREE.PerspectiveCamera(45, 1, 0.001, 1000);
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dir = new THREE.DirectionalLight(0xffffff, 0.8);
    dir.position.set(1, 1, 1);
    this.scene.add(dir);

    this.mesh = null;
    this._pickCallback = null;
    this.raycaster = new THREE.Raycaster();
    this._onClick = (ev) => this._handleClick(ev);
    this.renderer.domElement.addEventListener("click", this._onClick);

    this._resizeObserver = new ResizeObserver(() => this._resize());
    this._resizeObserver.observe(container);
    this._resize();

    this._animate = () => {
      this._frameId = requestAnimationFrame(this._animate);
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    };
    this._animate();
  }

  _resize() {
    const w = this.container.clientWidth || 1;
    const h = this.container.clientHeight || 1;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  async load(url) {
    const isObj = url.toLowerCase().endsWith(".obj");
    const geometry = isObj ? await this._loadObjGeometry(url) : await this._loadPlyGeometry(url);
    this._setMeshGeometry(geometry);
  }

  async _loadPlyGeometry(url) {
    const loader = new PLYLoader();
    const geometry = await loader.loadAsync(url);
    geometry.computeVertexNormals();
    return geometry;
  }

  async _loadObjGeometry(url) {
    const loader = new OBJLoader();
    const obj = await loader.loadAsync(url);
    let geometry = null;
    obj.traverse((child) => {
      if (child.isMesh && !geometry) geometry = child.geometry;
    });
    return geometry;
  }

  _setMeshGeometry(geometry) {
    if (this.mesh) {
      this.scene.remove(this.mesh);
      this.mesh.geometry.dispose();
      this.mesh.material.dispose();
    }
    const hasColor = !!geometry.getAttribute("color");
    const material = new THREE.MeshStandardMaterial({
      vertexColors: hasColor,
      color: hasColor ? 0xffffff : 0x8899aa,
      side: THREE.DoubleSide,
      roughness: 0.9,
    });
    this.mesh = new THREE.Mesh(geometry, material);
    this.scene.add(this.mesh);
    this._frameCamera(geometry);
  }

  _frameCamera(geometry) {
    geometry.computeBoundingSphere();
    const sphere = geometry.boundingSphere;
    const center = sphere.center;
    const radius = Math.max(sphere.radius, 1e-4);
    this.controls.target.copy(center);
    this.camera.position.set(center.x + radius * 1.6, center.y + radius * 1.1, center.z + radius * 1.6);
    this.camera.near = radius / 100;
    this.camera.far = radius * 100;
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  /** callback(point: {x,y,z}) fires on every click while picking is enabled,
   * with the point in the mesh's raw (untransformed) local coordinates -
   * the same space the server-side .ply file uses. */
  enablePicking(callback) {
    this._pickCallback = callback;
  }

  disablePicking() {
    this._pickCallback = null;
  }

  _handleClick(ev) {
    if (!this._pickCallback || !this.mesh) return;
    const rect = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((ev.clientX - rect.left) / rect.width) * 2 - 1,
      -((ev.clientY - rect.top) / rect.height) * 2 + 1
    );
    this.raycaster.setFromCamera(ndc, this.camera);
    const hits = this.raycaster.intersectObject(this.mesh, false);
    if (hits.length > 0) {
      const p = hits[0].point;
      this._pickCallback({ x: p.x, y: p.y, z: p.z });
    }
  }

  dispose() {
    cancelAnimationFrame(this._frameId);
    this._resizeObserver.disconnect();
    this.renderer.domElement.removeEventListener("click", this._onClick);
    this.controls.dispose();
    this.renderer.dispose();
    if (this.container.contains(this.renderer.domElement)) {
      this.container.removeChild(this.renderer.domElement);
    }
  }
}

export class SplatViewer {
  constructor(container) {
    this.container = container;
    this.viewer = null;
  }

  async load(url) {
    // Loaded lazily: this is a large module and most sessions only ever
    // use the mesh viewer, so we don't pay for it unless splat mode is used.
    const GaussianSplats3D = await import("@mkkellogg/gaussian-splats-3d");
    this.viewer = new GaussianSplats3D.Viewer({
      rootElement: this.container,
      // Avoids requiring cross-origin-isolation (COOP/COEP) headers, which
      // this simple deployment doesn't set up; costs a bit of sort perf.
      sharedMemoryForWorkers: false,
      cameraUp: [0, -1, 0],
      initialCameraPosition: [0, -1.2, 2.5],
      initialCameraLookAt: [0, 0, 0],
    });
    await this.viewer.addSplatScene(url, {
      splatAlphaRemovalThreshold: 5,
      showLoadingUI: true,
    });
    this.viewer.start();
  }

  async dispose() {
    if (this.viewer) {
      await this.viewer.dispose();
      this.viewer = null;
    }
  }
}
