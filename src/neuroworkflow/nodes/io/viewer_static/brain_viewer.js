// brain_viewer.js — Self-contained Three.js brain connectivity viewer.
//
// Usage:
//   import { initBrainViewer } from './brain_viewer.js';
//   const { dispose } = initBrainViewer(containerElement, dataUrl);
//
// containerElement must have defined dimensions (width + height).
// dataUrl is the URL of a connectivity_data.json file served via HTTP.
// dispose() tears down Three.js and removes all injected DOM.

// Requires an importmap in the host page that maps 'three' and 'three/addons/'
// to the Three.js CDN. See brain_viewer.html for the importmap, and
// viewer/EMBEDDING.md for how to add it to a React/Django host page.
import * as THREE from 'three';
import { TrackballControls }    from 'three/addons/controls/TrackballControls.js';
import { LineSegments2 }        from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial }         from 'three/addons/lines/LineMaterial.js';

// ── module-level constants (shared across all instances) ──────────────────────
const BASE_R      = 0.35;
const MAX_STRANDS = 5;
const CURVE_PTS   = 12;
const ARC_FRAC    = 0.20;
const SPREAD_FRAC = 0.15;

const COL_L        = new THREE.Color(0x4488ff);
const COL_R        = new THREE.Color(0xff7733);
const COL_NEUT     = new THREE.Color(0x6699bb);
const COL_BOLD_HI  = new THREE.Color(1.0, 0.45, 0.05);
const COL_DIM      = new THREE.Color(0x1a1a1a);
const COL_BOLD_DIM = COL_BOLD_HI.clone().lerp(COL_DIM, 0.78);
const STIM_COLS    = ['#ff8833', '#4499ff', '#44dd88', '#cc66ff', '#ffee44'];

const CONN_GROUPS = [
  { lw: 2.5, op: 0.95, from: 0.0, to: 0.2 },
  { lw: 2.0, op: 0.75, from: 0.2, to: 0.4 },
  { lw: 1.5, op: 0.50, from: 0.4, to: 0.6 },
  { lw: 1.0, op: 0.28, from: 0.6, to: 0.8 },
  { lw: 1.0, op: 0.12, from: 0.8, to: 1.0 },
];

// Shared sphere geometry/material (one copy per page, not per instance).
const _sgeom = new THREE.SphereGeometry(1, 22, 22);
const _smat  = new THREE.MeshPhongMaterial({ shininess: 55 });

// ── CSS (injected once into document.head) ────────────────────────────────────
const BCV_CSS = `
.bcv-root { display:flex; width:100%; height:100%; overflow:hidden; background:#111; color:#ccc;
            font-family:system-ui,-apple-system,sans-serif; font-size:13px; position:relative; }
.bcv-root * { box-sizing:border-box; margin:0; padding:0; }
.bcv-panel { width:230px; min-width:200px; background:#181818; border-right:1px solid #2a2a2a;
             padding:16px 14px; display:flex; flex-direction:column; gap:20px;
             overflow-y:auto; flex-shrink:0; }
.bcv-canvas-wrap { flex:1; position:relative; overflow:hidden; }
.bcv-canvas-wrap canvas { display:block; }
.bcv-root h1 { font-size:13px; font-weight:600; color:#888; letter-spacing:0.06em; text-transform:uppercase; }
.bcv-section { display:flex; flex-direction:column; gap:10px; }
.bcv-sec-title { font-size:10px; text-transform:uppercase; letter-spacing:0.1em; color:#555;
                 border-bottom:1px solid #2a2a2a; padding-bottom:4px; }
.bcv-root label { display:flex; flex-direction:column; gap:4px; color:#aaa; font-size:12px; }
.bcv-root input[type=range] { width:100%; height:4px; accent-color:#5588ff; cursor:pointer; }
.bcv-val  { color:#7799ff; font-size:11px; font-weight:600; }
.bcv-info { font-size:11px; color:#555; line-height:1.6; }
.bcv-row  { display:flex; gap:6px; flex-wrap:wrap; }
.bcv-colorbar { height:6px; border-radius:3px;
                background:linear-gradient(to right,#2244cc,#aaccff,#eeeeee,#ffaaaa,#cc2222); }
.bcv-cblabels { display:flex; justify-content:space-between; font-size:10px; color:#555; margin-top:2px; }
.bcv-sel-name { font-weight:600; color:#eee; font-size:13px; margin-bottom:2px; }
.bcv-root button { background:#222; border:1px solid #333; color:#aaa; padding:5px 10px;
                   border-radius:4px; cursor:pointer; font-size:11px; white-space:nowrap; }
.bcv-root button:hover { background:#2a2a3a; color:#ddd; }
.bcv-root button.on { background:#1e3060; border-color:#4466aa; color:#99bbff; }
.bcv-tooltip { display:none; position:fixed; background:rgba(0,0,0,0.85); color:#eee;
               padding:4px 9px; border-radius:4px; font-size:12px; pointer-events:none;
               z-index:9999; border:1px solid #444; white-space:nowrap; }
.bcv-status  { display:none; position:absolute; top:14px; left:50%; transform:translateX(-50%);
               background:rgba(0,0,0,0.75); padding:7px 18px; border-radius:20px;
               font-size:12px; color:#aaa; pointer-events:none; white-space:nowrap; }
.bcv-bold-controls.disabled { opacity:0.35; pointer-events:none; }
.bcv-trace-host { position:relative; width:100%; height:72px; margin-top:6px;
                  border-radius:3px; background:#0a0a0a; overflow:hidden; }
.bcv-trace-host canvas { display:block; width:100%; height:100%; }
.bcv-btn-pop { position:absolute; top:3px; right:3px; padding:1px 5px; font-size:10px;
               opacity:0.45; z-index:2; }
.bcv-trace-host:hover .bcv-btn-pop { opacity:1; }
.bcv-trace-stub { display:none; margin-top:6px; font-size:11px; color:#555; line-height:1.6; }
.bcv-float { display:none; position:absolute; z-index:50; width:520px; height:300px;
             min-width:240px; min-height:150px; background:#181818; border:1px solid #3a3a3a;
             border-radius:6px; box-shadow:0 10px 34px rgba(0,0,0,0.65);
             flex-direction:column; overflow:hidden; resize:both; }
.bcv-float.open { display:flex; }
.bcv-float-head { display:flex; align-items:center; justify-content:space-between; gap:10px;
                  padding:5px 8px; background:#202020; border-bottom:1px solid #2a2a2a;
                  cursor:move; user-select:none; font-size:10px; color:#888;
                  letter-spacing:0.1em; text-transform:uppercase; }
.bcv-float-body { flex:1; min-height:0; padding:8px; display:flex; }
.bcv-float-body .bcv-trace-host { height:100%; margin-top:0; }
.bcv-float .bcv-btn-pop { display:none; }
`;

function injectCSS() {
  if (document.getElementById('bcv-styles')) return;
  const s = document.createElement('style');
  s.id = 'bcv-styles';
  s.textContent = BCV_CSS;
  document.head.appendChild(s);
}

// ── HTML template (injected into container) ───────────────────────────────────
const BCV_HTML = `
<div class="bcv-panel">
  <h1>&#129504; Connectivity</h1>

  <div class="bcv-section">
    <div class="bcv-sec-title">Connections</div>
    <label>Threshold &#8212; hide weakest
      <input type="range" class="bcv-sld-thresh" min="0" max="100" value="90" step="1">
      <span class="bcv-val bcv-lbl-thresh">top 10%</span>
    </label>
    <div class="bcv-colorbar"></div>
    <div class="bcv-cblabels"><span>weak</span><span>strong</span></div>
    <div class="bcv-info bcv-lbl-nconn">&#8212;</div>
  </div>

  <div class="bcv-section">
    <div class="bcv-sec-title">BOLD Signal</div>
    <div class="bcv-bold-controls disabled">
      <div class="bcv-row">
        <button class="bcv-btn-play">&#9654; Play</button>
        <button class="bcv-btn-bold-reset">&#9198;</button>
      </div>
      <label style="margin-top:6px">Speed
        <input type="range" class="bcv-sld-speed" min="-2" max="2" value="0" step="0.25">
        <span class="bcv-val bcv-lbl-speed">1.00&#xD7;</span>
      </label>
      <label>Time
        <input type="range" class="bcv-sld-time" min="0" max="19" value="0" step="1">
        <span class="bcv-val bcv-lbl-time">&#8212;</span>
      </label>
      <div class="bcv-trace-host" style="display:none">
        <canvas class="bcv-bold-trace"></canvas>
        <button class="bcv-btn-pop" title="Detach into a floating window">&#9974;</button>
      </div>
      <div class="bcv-trace-stub">Trace detached &#8212; see floating window.</div>
    </div>
    <div class="bcv-info bcv-lbl-bold-info">No BOLD data loaded</div>
  </div>

  <div class="bcv-section bcv-sel-section" style="display:none">
    <div class="bcv-sec-title">Selected region</div>
    <div class="bcv-sel-name"></div>
    <div class="bcv-info bcv-sel-details"></div>
    <button class="bcv-btn-deselect" style="margin-top:2px">&#10005; Clear selection</button>
  </div>

  <div class="bcv-section">
    <div class="bcv-sec-title">Display</div>
    <label>Sphere size
      <input type="range" class="bcv-sld-sphere-scale" min="0.3" max="3" value="2" step="0.05">
      <span class="bcv-val bcv-lbl-sphere-scale">2.00&#xD7;</span>
    </label>
    <label>Line width
      <input type="range" class="bcv-sld-line-width" min="0.2" max="4" value="0.2" step="0.1">
      <span class="bcv-val bcv-lbl-line-width">0.2&#xD7;</span>
    </label>
    <div class="bcv-row">
      <button class="bcv-btn-hemi">L/R colours</button>
      <button class="bcv-btn-area">Area size</button>
    </div>
    <div class="bcv-row">
      <button class="bcv-btn-brain-mesh">Brain mesh</button>
      <button class="bcv-btn-halos">Stimulated</button>
    </div>
    <div class="bcv-row bcv-tract-row" style="display:none">
      <button class="bcv-btn-tracts on">Fiber tracts</button>
    </div>
    <div class="bcv-row">
      <button class="bcv-btn-dim">Dim others</button>
      <button class="bcv-btn-reset-cam">Reset camera</button>
    </div>
    <div class="bcv-row">
      <button class="bcv-btn-copy-chat">&#128203; Copy state for Chat</button>
    </div>
  </div>

  <div class="bcv-info">Left-drag &middot; Scroll &middot; Right-drag to pan<br>Click region to focus</div>
</div>

<div class="bcv-canvas-wrap">
  <div class="bcv-tooltip"></div>
  <div class="bcv-status">Loading&#x2026;</div>
</div>

<div class="bcv-float">
  <div class="bcv-float-head">
    <span>BOLD activity</span>
    <button class="bcv-btn-dock" title="Dock back into the panel">&#8600; Dock</button>
  </div>
  <div class="bcv-float-body"></div>
</div>
`;

const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);

// ── public API ────────────────────────────────────────────────────────────────
export function initBrainViewer(container, dataUrl) {
  injectCSS();
  container.classList.add('bcv-root');
  container.innerHTML = BCV_HTML;

  // Shorthand for container-scoped queries.
  const q = cls => container.querySelector('.' + cls);

  // ── element refs ─────────────────────────────────────────────────────────────
  const cWrap          = q('bcv-canvas-wrap');
  const tooltip        = q('bcv-tooltip');
  const statusEl       = q('bcv-status');
  const sldThresh      = q('bcv-sld-thresh');
  const lblThresh      = q('bcv-lbl-thresh');
  const lblNconn       = q('bcv-lbl-nconn');
  const boldControls   = q('bcv-bold-controls');
  const btnPlay        = q('bcv-btn-play');
  const btnBoldReset   = q('bcv-btn-bold-reset');
  const sldSpeed       = q('bcv-sld-speed');
  const lblSpeed       = q('bcv-lbl-speed');
  const sldTime        = q('bcv-sld-time');
  const lblTime        = q('bcv-lbl-time');
  const boldTrace      = q('bcv-bold-trace');
  const traceHost      = q('bcv-trace-host');
  const traceStub      = q('bcv-trace-stub');
  const btnPop         = q('bcv-btn-pop');
  const boldPanelSlot  = traceHost.parentNode;
  const floatWin       = q('bcv-float');
  const floatHead      = q('bcv-float-head');
  const floatBody      = q('bcv-float-body');
  const btnDock        = q('bcv-btn-dock');
  const lblBoldInfo    = q('bcv-lbl-bold-info');
  const selSection     = q('bcv-sel-section');
  const selName        = q('bcv-sel-name');
  const selDetails     = q('bcv-sel-details');
  const btnDeselect    = q('bcv-btn-deselect');
  const sldSphereScale = q('bcv-sld-sphere-scale');
  const lblSphereScale = q('bcv-lbl-sphere-scale');
  const sldLineWidth   = q('bcv-sld-line-width');
  const lblLineWidth   = q('bcv-lbl-line-width');
  const btnHemi        = q('bcv-btn-hemi');
  const btnArea        = q('bcv-btn-area');
  const btnBrainMesh   = q('bcv-btn-brain-mesh');
  const btnHalos       = q('bcv-btn-halos');
  const btnDim         = q('bcv-btn-dim');
  const btnResetCam    = q('bcv-btn-reset-cam');
  const btnCopyChat    = q('bcv-btn-copy-chat');
  const btnTracts      = q('bcv-btn-tracts');
  const tractRow       = q('bcv-tract-row');

  // ── instance state ────────────────────────────────────────────────────────────
  let D;
  let scene, cam, renderer, ctl, ray;
  let regMesh, connGroups = [], lineMats = [];
  let brainMeshObj = null, haloMesh = null;
  const brainCtr = new THREE.Vector3();
  let camHome, tgtHome;

  let showHemi, showArea;
  let showMesh       = true;
  let showHalos      = true;
  let showTracts     = true;
  let dimNonSelected = false;
  let threshold      = 90;
  let sphereScale    = 2.0;
  let lineScale      = 0.2;

  let selectedRegion = -1;
  let traceImageData = null;
  let traceW = 0, traceH = 0, traceDpr = 1;  // logical (CSS px) canvas size + backing-store ratio
  let floatPlaced = false;

  let boldActive = false, boldPlaying = false;
  let boldT = 0, boldSpeed = 1.0;
  let boldMin, boldRange;

  let rebuildId       = null;
  let buildGeneration = 0;
  let lastTs          = 0;
  let disposed        = false;

  // Per-instance scratch vectors (avoids GC churn in buildConnections).
  const _pA = new THREE.Vector3(), _pB  = new THREE.Vector3();
  const _mid = new THREE.Vector3(), _arc = new THREE.Vector3();
  const _dir = new THREE.Vector3(), _perp = new THREE.Vector3(), _perp2 = new THREE.Vector3();

  let logWMin, logWRange;

  // ── helpers ───────────────────────────────────────────────────────────────────
  function show(msg) {
    statusEl.textContent = msg;
    statusEl.style.display = msg ? 'block' : 'none';
  }

  function setBtn(el, on) { el.classList.toggle('on', on); }

  // ── init ──────────────────────────────────────────────────────────────────────
  async function init() {
    show('Fetching data…');
    try {
      const r = await fetch(dataUrl, { cache: 'no-store' });
      if (!r.ok) throw new Error('HTTP ' + r.status);
      D = await r.json();
    } catch (e) {
      show(`Cannot load ${dataUrl} — serve via HTTP, not file://`);
      return;
    }
    if (disposed) return;

    showHemi = D.meta.hemi_colors;
    showArea = D.meta.area_spheres;
    setBtn(btnHemi, showHemi);
    setBtn(btnArea, showArea);

    setupRenderer();
    setupScene();
    buildBrainCenter();
    buildBrainMesh();
    buildRegions();
    buildHalos();
    fitCamera();
    setupBold();
    setupUI();
    setupClick();
    show('');
    requestAnimationFrame(loop);
    await buildConnections();
  }

  // ── renderer / scene ──────────────────────────────────────────────────────────
  function setupRenderer() {
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(cWrap.clientWidth, cWrap.clientHeight);
    renderer.setClearColor(0x111111);
    cWrap.appendChild(renderer.domElement);

    cam = new THREE.PerspectiveCamera(48, cWrap.clientWidth / cWrap.clientHeight, 0.01, 2000);

    ctl = new TrackballControls(cam, renderer.domElement);
    ctl.rotateSpeed = 2.5;
    ctl.zoomSpeed   = 1.2;
    ctl.panSpeed    = 0.8;
    ctl.staticMoving = false;
    ctl.dynamicDampingFactor = 0.15;

    ray = new THREE.Raycaster();
    ray.params.Line = { threshold: 0.1 };

    new ResizeObserver(() => {
      const w = cWrap.clientWidth, h = cWrap.clientHeight;
      cam.aspect = w / h;
      cam.updateProjectionMatrix();
      renderer.setSize(w, h);
      ctl.handleResize();
      lineMats.forEach(({ mat }) => mat.resolution.set(w, h));
    }).observe(cWrap);
  }

  function setupScene() {
    scene = new THREE.Scene();
    scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const d = new THREE.DirectionalLight(0xffffff, 0.9);
    d.position.set(8, 14, 10);
    scene.add(d);
    const d2 = new THREE.DirectionalLight(0x4466aa, 0.3);
    d2.position.set(-6, -8, -4);
    scene.add(d2);
  }

  // ── brain mesh ────────────────────────────────────────────────────────────────
  function buildBrainMesh() {
    if (!D.mesh) { btnBrainMesh.style.display = 'none'; return; }
    const { v, f } = D.mesh;
    const flatV = new Float32Array(v.length * 3);
    v.forEach(([x, y, z], i) => { flatV[i*3] = x; flatV[i*3+1] = y; flatV[i*3+2] = z; });
    const flatF = new Uint32Array(f.flat());

    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(flatV, 3));
    geom.setIndex(new THREE.BufferAttribute(flatF, 1));
    geom.computeVertexNormals();

    const mat = new THREE.MeshPhongMaterial({
      color: 0xaaddff, transparent: true, opacity: 0.15,
      side: THREE.BackSide, depthWrite: false,
    });
    brainMeshObj = new THREE.Mesh(geom, mat);
    scene.add(brainMeshObj);
    setBtn(btnBrainMesh, showMesh);
  }

  // ── regions ───────────────────────────────────────────────────────────────────
  function regionRadius(i) {
    if (!showArea) return BASE_R * sphereScale;
    const t = (D.regions[i].area - D.meta.area_min) / (D.meta.area_max - D.meta.area_min + 1e-9);
    return BASE_R * sphereScale * (0.45 + 1.1 * t);
  }

  function regionBaseColor(i) {
    const base = showHemi
      ? (D.regions[i].hemi === 'L' ? COL_L : COL_R)
      : COL_NEUT;
    if (dimNonSelected && selectedRegion >= 0 && i !== selectedRegion)
      return base.clone().lerp(COL_DIM, 0.78);
    return base;
  }

  function buildRegions() {
    if (regMesh) scene.remove(regMesh);
    const n = D.regions.length;
    regMesh = new THREE.InstancedMesh(_sgeom, _smat, n);
    regMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    const dm = new THREE.Object3D();
    for (let i = 0; i < n; i++) {
      const r = D.regions[i];
      dm.position.set(r.x, r.y, r.z);
      dm.scale.setScalar(regionRadius(i));
      dm.updateMatrix();
      regMesh.setMatrixAt(i, dm.matrix);
      regMesh.setColorAt(i, regionBaseColor(i));
    }
    regMesh.instanceMatrix.needsUpdate = true;
    regMesh.instanceColor.needsUpdate  = true;
    scene.add(regMesh);
  }

  function refreshRegionColors() {
    const dm = new THREE.Object3D();
    for (let i = 0; i < D.regions.length; i++) {
      regMesh.setColorAt(i, regionBaseColor(i));
      const r = D.regions[i];
      dm.position.set(r.x, r.y, r.z);
      dm.scale.setScalar(regionRadius(i));
      dm.updateMatrix();
      regMesh.setMatrixAt(i, dm.matrix);
    }
    regMesh.instanceMatrix.needsUpdate = true;
    regMesh.instanceColor.needsUpdate  = true;
    if (haloMesh) {
      const stim = D.meta.stimulated_regions;
      stim.forEach((ri, k) => {
        const r = D.regions[ri];
        dm.position.set(r.x, r.y, r.z);
        dm.scale.setScalar(regionRadius(ri) * 1.6);
        dm.updateMatrix();
        haloMesh.setMatrixAt(k, dm.matrix);
      });
      haloMesh.instanceMatrix.needsUpdate = true;
    }
  }

  // ── halos ─────────────────────────────────────────────────────────────────────
  function buildHalos() {
    const stim = D.meta.stimulated_regions;
    if (!stim || stim.length === 0) { btnHalos.style.display = 'none'; return; }
    const hGeom = new THREE.SphereGeometry(1, 14, 10);
    const hMat  = new THREE.MeshBasicMaterial({
      color: 0xffdd33, transparent: true, opacity: 0.30, side: THREE.BackSide,
    });
    haloMesh = new THREE.InstancedMesh(hGeom, hMat, stim.length);
    const dm = new THREE.Object3D();
    stim.forEach((ri, k) => {
      const r = D.regions[ri];
      dm.position.set(r.x, r.y, r.z);
      dm.scale.setScalar(regionRadius(ri) * 1.6);
      dm.updateMatrix();
      haloMesh.setMatrixAt(k, dm.matrix);
    });
    haloMesh.instanceMatrix.needsUpdate = true;
    scene.add(haloMesh);
    setBtn(btnHalos, showHalos);
  }

  // ── connections ───────────────────────────────────────────────────────────────
  function buildBrainCenter() {
    D.regions.forEach(r => brainCtr.add(new THREE.Vector3(r.x, r.y, r.z)));
    brainCtr.divideScalar(D.regions.length);
  }

  function nStrands(w) {
    return Math.max(1, Math.round((w / D.meta.weight_max) * MAX_STRANDS));
  }

  function connColor(w) {
    const t = Math.max(0, Math.min(1, (Math.log(w) - logWMin) / logWRange));
    const c = new THREE.Color();
    if (t < 0.5) {
      const s = t * 2;
      c.setRGB(0.13 + 0.62 * s, 0.27 + 0.53 * s, 0.9 - 0.05 * s);
    } else {
      const s = (t - 0.5) * 2;
      c.setRGB(0.75 + 0.2 * s, 0.8 - 0.67 * s, 0.85 - 0.78 * s);
    }
    return c;
  }

  async function buildConnections() {
    const gen = ++buildGeneration;

    connGroups.forEach(obj => { scene.remove(obj); obj.geometry.dispose(); });
    connGroups = [];
    lineMats.forEach(({ mat }) => mat.dispose());
    lineMats.length = 0;

    logWMin   = Math.log(D.meta.weight_min_nz);
    logWRange = Math.log(D.meta.weight_max) - logWMin;

    const total = D.connections.length;
    const nShow = Math.ceil(total * (1 - threshold / 100));
    let active  = D.connections.slice(0, nShow);
    if (selectedRegion >= 0)
      active = active.filter(([i, j]) => i === selectedRegion || j === selectedRegion);

    const n   = active.length;
    const res = new THREE.Vector2(cWrap.clientWidth, cWrap.clientHeight);

    lblNconn.textContent = `${n.toLocaleString()} / ${total.toLocaleString()} — building…`;

    await new Promise(r => setTimeout(r, 0));
    if (gen !== buildGeneration) return;

    const nG   = CONN_GROUPS.length;
    const gPos = Array.from({ length: nG }, () => []);
    const gCol = Array.from({ length: nG }, () => []);
    const CHUNK = 150;

    for (let ci = 0; ci < n; ci++) {
      const gi  = Math.min(Math.floor(ci / n * nG), nG - 1);
      const pos = gPos[gi], col = gCol[gi];
      const [i, j, w] = active[ci];
      const ri = D.regions[i], rj = D.regions[j];

      const cc = connColor(w);
      const ns = nStrands(w);

      // Use anatomical fiber tract when available and enabled, fall back to CatmullRom spline.
      const tractKey = `${Math.min(i, j)},${Math.max(i, j)}`;
      const tractPts = showTracts && D.tracts && D.tracts[tractKey];

      if (tractPts) {
        // Anchor the fiber to region centres as first/last CatmullRom control points.
        // The stored tract coordinates are the actual white-matter fiber; adding the
        // node positions at each end makes the curve start/end exactly at the spheres
        // while smoothly approaching the real anatomical path — no hard snapping.
        _pA.set(ri.x, ri.y, ri.z);
        _pB.set(rj.x, rj.y, rj.z);
        _mid.addVectors(_pA, _pB).multiplyScalar(0.5);
        const ctrl = [
          _pA.clone(),
          ...tractPts.map(([x, y, z]) => new THREE.Vector3(x, y, z)),
          _pB.clone(),
        ];
        const tpts = new THREE.CatmullRomCurve3(ctrl).getPoints(tractPts.length * 2);

        const spread = _pA.distanceTo(_pB) * ARC_FRAC * SPREAD_FRAC;
        _dir.subVectors(_pB, _pA).normalize();
        _arc.subVectors(_mid, brainCtr).normalize();
        _perp.crossVectors(_dir, _arc);
        if (_perp.lengthSq() < 1e-8) _perp.set(0, 1, 0);
        _perp.normalize();
        _perp2.crossVectors(_dir, _perp).normalize();

        for (let k = 0; k < ns; k++) {
          const angle = (k / Math.max(ns, 1)) * Math.PI * 2;
          const ox = Math.cos(angle) * spread, oy = Math.sin(angle) * spread;
          for (let s = 0; s < tpts.length - 1; s++) {
            const v0 = tpts[s].clone().addScaledVector(_perp, ox).addScaledVector(_perp2, oy);
            const v1 = tpts[s + 1].clone().addScaledVector(_perp, ox).addScaledVector(_perp2, oy);
            pos.push(v0.x, v0.y, v0.z);  col.push(cc.r, cc.g, cc.b);
            pos.push(v1.x, v1.y, v1.z);  col.push(cc.r, cc.g, cc.b);
          }
        }
      } else {
        // Geometric fallback: arced CatmullRom spline between region centres.
        _pA.set(ri.x, ri.y, ri.z);
        _pB.set(rj.x, rj.y, rj.z);
        _mid.addVectors(_pA, _pB).multiplyScalar(0.5);
        const len    = _pA.distanceTo(_pB);
        const arcAmt = len * ARC_FRAC;
        const spread = arcAmt * SPREAD_FRAC;
        _arc.subVectors(_mid, brainCtr).normalize();
        _dir.subVectors(_pB, _pA).normalize();
        _perp.crossVectors(_dir, _arc);
        if (_perp.lengthSq() < 1e-8) _perp.set(0, 1, 0);
        _perp.normalize();
        _perp2.crossVectors(_dir, _perp).normalize();
        const arcMidBase = _mid.clone().addScaledVector(_arc, arcAmt);

        for (let k = 0; k < ns; k++) {
          const angle  = (k / Math.max(ns, 1)) * Math.PI * 2;
          const arcMid = arcMidBase.clone()
            .addScaledVector(_perp,  Math.cos(angle) * spread)
            .addScaledVector(_perp2, Math.sin(angle) * spread);
          const pts = new THREE.CatmullRomCurve3([_pA.clone(), arcMid, _pB.clone()])
                        .getPoints(CURVE_PTS - 1);
          for (let s = 0; s < pts.length - 1; s++) {
            const v0 = pts[s], v1 = pts[s + 1];
            pos.push(v0.x, v0.y, v0.z);  col.push(cc.r, cc.g, cc.b);
            pos.push(v1.x, v1.y, v1.z);  col.push(cc.r, cc.g, cc.b);
          }
        }
      }

      if ((ci + 1) % CHUNK === 0) {
        await new Promise(r => setTimeout(r, 0));
        if (gen !== buildGeneration) return;
      }
    }

    CONN_GROUPS.forEach(({ lw, op }, gi) => {
      if (gPos[gi].length === 0) return;
      const geom = new LineSegmentsGeometry();
      geom.setPositions(new Float32Array(gPos[gi]));
      geom.setColors(new Float32Array(gCol[gi]));
      const mat = new LineMaterial({
        vertexColors: true, linewidth: lw * lineScale,
        transparent: true, opacity: op,
        resolution: res.clone(),
      });
      const obj = new LineSegments2(geom, mat);
      scene.add(obj);
      connGroups.push(obj);
      lineMats.push({ mat, baseLw: lw });
    });

    lblNconn.textContent = `${n.toLocaleString()} / ${total.toLocaleString()} connections`;
  }

  // ── camera ────────────────────────────────────────────────────────────────────
  function fitCamera() {
    const box = new THREE.Box3();
    D.regions.forEach(r => box.expandByPoint(new THREE.Vector3(r.x, r.y, r.z)));
    const ctr = new THREE.Vector3(), sz = new THREE.Vector3();
    box.getCenter(ctr);
    box.getSize(sz);
    const dist = (Math.max(sz.x, sz.y, sz.z) / 2) / Math.tan((cam.fov * Math.PI / 180) / 2) * 1.7;
    cam.position.set(ctr.x, ctr.y, ctr.z + dist);
    ctl.target.copy(ctr);
    ctl.update();
    camHome = cam.position.clone();
    tgtHome = ctl.target.clone();
  }

  // ── selection ─────────────────────────────────────────────────────────────────
  function selectRegion(idx) {
    selectedRegion = idx;
    refreshRegionColors();
    if (boldActive) updateBold(boldT);
    buildConnections();
    updateSelectionPanel();
    drawBoldTrace();
  }

  function setupClick() {
    let mdX = 0, mdY = 0;
    const mv = new THREE.Vector2();
    renderer.domElement.addEventListener('mousedown', e => { mdX = e.clientX; mdY = e.clientY; });
    renderer.domElement.addEventListener('click', e => {
      if (!e.isTrusted) return;
      if (Math.hypot(e.clientX - mdX, e.clientY - mdY) > 5) return;
      const rect = renderer.domElement.getBoundingClientRect();
      mv.x =  ((e.clientX - rect.left) / rect.width)  * 2 - 1;
      mv.y = -((e.clientY - rect.top)  / rect.height) * 2 + 1;
      ray.setFromCamera(mv, cam);
      const hits = ray.intersectObject(regMesh);
      const idx  = hits.length > 0 ? hits[0].instanceId : -1;
      if (idx !== selectedRegion) selectRegion(idx);
    });
  }

  function updateSelectionPanel() {
    if (selectedRegion < 0) { selSection.style.display = 'none'; return; }
    selSection.style.display = '';
    const r     = D.regions[selectedRegion];
    selName.textContent = r.name;
    const total = D.connections.length;
    const nShow = Math.ceil(total * (1 - threshold / 100));
    const nConn = D.connections.slice(0, nShow)
      .filter(([i, j]) => i === selectedRegion || j === selectedRegion).length;
    selDetails.textContent =
      `${r.hemi === 'L' ? 'Left' : 'Right'} hemisphere · ${r.area} mm² · ${nConn} connections`;
  }

  // ── BOLD ──────────────────────────────────────────────────────────────────────
  function setupBold() {
    if (!D.bold) return;
    const { time, data } = D.bold;
    const nT = time.length, nR = D.regions.length;

    boldMin   = new Float32Array(nR);
    boldRange = new Float32Array(nR);
    for (let r = 0; r < nR; r++) {
      let mn = Infinity, mx = -Infinity;
      for (let t = 0; t < nT; t++) { mn = Math.min(mn, data[t][r]); mx = Math.max(mx, data[t][r]); }
      boldMin[r]   = mn;
      boldRange[r] = Math.max(mx - mn, 1e-9);
    }

    boldActive = true;
    boldControls.classList.remove('disabled');
    sldTime.max = nT - 1;
    lblBoldInfo.textContent = `${nT} frames · ${time[0]}–${time[nT-1]} ms · ${nR} regions`;
    traceHost.style.display = 'block';
    updateBold(0);
    resizeTrace();
  }

  function updateBold(t) {
    if (!boldActive || !D.bold) return;
    const { time, data } = D.bold;
    const nT  = data.length;
    const ti  = Math.min(Math.floor(t), nT - 1);
    const tf  = t - Math.floor(t);
    const ti2 = Math.min(ti + 1, nT - 1);

    const dm = new THREE.Object3D();
    const cc = new THREE.Color();
    for (let i = 0; i < D.regions.length; i++) {
      const v   = data[ti][i] + (data[ti2][i] - data[ti][i]) * tf;
      const val = Math.max(0, Math.min(1, (v - boldMin[i]) / boldRange[i]));
      const r   = D.regions[i];
      dm.position.set(r.x, r.y, r.z);
      const nonSelected = dimNonSelected && selectedRegion >= 0 && i !== selectedRegion;
      dm.scale.setScalar(regionRadius(i) * (0.72 + 0.55 * val));
      dm.updateMatrix();
      regMesh.setMatrixAt(i, dm.matrix);
      cc.lerpColors(regionBaseColor(i), nonSelected ? COL_BOLD_DIM : COL_BOLD_HI, val);
      regMesh.setColorAt(i, cc);
    }
    regMesh.instanceMatrix.needsUpdate = true;
    regMesh.instanceColor.needsUpdate  = true;

    const tDisp = time[ti] + (time[ti2] - time[ti]) * tf;
    lblTime.textContent = tDisp.toFixed(0) + ' ms';
    if (!sldTime.matches(':active')) sldTime.value = ti;
    drawTraceCursor();
  }

  // ── BOLD trace chart ──────────────────────────────────────────────────────────
  // Match the backing store to the host's current size; redraw only on a real change.
  function resizeTrace() {
    const dpr = window.devicePixelRatio || 1;
    const w   = traceHost.clientWidth, h = traceHost.clientHeight;
    if (w < 1 || h < 1) return;
    const bw = Math.round(w * dpr), bh = Math.round(h * dpr);
    if (boldTrace.width === bw && boldTrace.height === bh) return;
    boldTrace.width = bw; boldTrace.height = bh;
    traceW = w; traceH = h; traceDpr = dpr;
    drawBoldTrace();
  }

  function drawBoldTrace() {
    if (!D.bold || traceW < 1) return;
    const { data } = D.bold;
    const nT = data.length, nR = D.regions.length;
    const ctx = boldTrace.getContext('2d');
    const W = traceW, H = traceH;
    ctx.setTransform(traceDpr, 0, 0, traceDpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    function tracePath(ri) {
      const vals = data.map(row => row[ri]);
      const mn   = Math.min(...vals);
      const rng  = Math.max(Math.max(...vals) - mn, 1e-9);
      ctx.beginPath();
      for (let t = 0; t < nT; t++) {
        const x = (t / (nT - 1)) * (W - 2) + 1;
        const y = (H - 4) - ((vals[t] - mn) / rng) * (H - 8) + 2;
        t === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    const stim = D.meta.stimulated_regions || [];
    ctx.lineWidth = 0.6; ctx.globalAlpha = 0.18; ctx.strokeStyle = '#aaaaaa';
    for (let ri = 0; ri < nR; ri++) {
      if (stim.includes(ri) || ri === selectedRegion) continue;
      tracePath(ri);
    }
    ctx.lineWidth = 1.5; ctx.globalAlpha = 0.85;
    stim.forEach((ri, k) => {
      if (ri === selectedRegion) return;
      ctx.strokeStyle = STIM_COLS[k % STIM_COLS.length];
      tracePath(ri);
    });
    if (selectedRegion >= 0) {
      ctx.lineWidth = 2; ctx.globalAlpha = 1; ctx.strokeStyle = '#ffffff';
      tracePath(selectedRegion);
    }
    ctx.globalAlpha = 1;
    // Cached backdrop is in device pixels; putImageData ignores the transform.
    traceImageData  = ctx.getImageData(0, 0, boldTrace.width, boldTrace.height);
    drawTraceCursor();
  }

  function drawTraceCursor() {
    if (!D.bold || !traceImageData) return;
    const ctx = boldTrace.getContext('2d');
    ctx.putImageData(traceImageData, 0, 0);
    ctx.setTransform(traceDpr, 0, 0, traceDpr, 0, 0);
    const nT = D.bold.data.length;
    const x  = (Math.min(boldT, nT - 1) / (nT - 1)) * (traceW - 2) + 1;
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(255,255,255,0.55)';
    ctx.lineWidth   = 1;
    ctx.setLineDash([3, 2]);
    ctx.moveTo(x, 0);
    ctx.lineTo(x, traceH);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // ── detachable trace window ───────────────────────────────────────────────────
  function setupTraceWindow() {
    let pending = false;
    new ResizeObserver(() => {
      if (pending) return;
      pending = true;
      requestAnimationFrame(() => { pending = false; resizeTrace(); });
    }).observe(traceHost);

    btnPop.addEventListener('click', () => {
      if (!floatPlaced) {           // first open: park it near the top-left of the 3D view
        const root = container.getBoundingClientRect();
        floatWin.style.left = Math.round(cWrap.getBoundingClientRect().left - root.left + 24) + 'px';
        floatWin.style.top  = '24px';
        floatPlaced = true;
      }
      floatBody.appendChild(traceHost);
      floatWin.classList.add('open');
      traceStub.style.display = 'block';
      resizeTrace();
    });

    btnDock.addEventListener('click', () => {
      floatWin.classList.remove('open');
      boldPanelSlot.appendChild(traceHost);
      traceStub.style.display = 'none';
      resizeTrace();
    });

    // Drag by the header, clamped so the title bar can never leave the viewer.
    let dragging = false, offX = 0, offY = 0;
    floatHead.addEventListener('mousedown', e => {
      if (e.target.closest('button')) return;
      const root = container.getBoundingClientRect();
      const win  = floatWin.getBoundingClientRect();
      dragging = true;
      offX = e.clientX - win.left;
      offY = e.clientY - win.top;
      e.preventDefault();
      const move = ev => {
        if (!dragging) return;
        const maxX = root.width  - floatWin.offsetWidth;
        const maxY = root.height - floatHead.offsetHeight;
        floatWin.style.left = clamp(ev.clientX - root.left - offX, 0, Math.max(maxX, 0)) + 'px';
        floatWin.style.top  = clamp(ev.clientY - root.top  - offY, 0, Math.max(maxY, 0)) + 'px';
      };
      const up = () => {
        dragging = false;
        window.removeEventListener('mousemove', move);
        window.removeEventListener('mouseup', up);
      };
      window.addEventListener('mousemove', move);
      window.addEventListener('mouseup', up);
    });
  }

  // ── tooltip ───────────────────────────────────────────────────────────────────
  function setupTooltip() {
    const mv = new THREE.Vector2();
    cWrap.addEventListener('mousemove', e => {
      const rect = renderer.domElement.getBoundingClientRect();
      mv.x =  ((e.clientX - rect.left) / rect.width)  * 2 - 1;
      mv.y = -((e.clientY - rect.top)  / rect.height) * 2 + 1;
      ray.setFromCamera(mv, cam);
      const hits = ray.intersectObject(regMesh);
      if (hits.length > 0) {
        tooltip.textContent    = D.regions[hits[0].instanceId].name;
        tooltip.style.display  = 'block';
        tooltip.style.left     = (e.clientX + 12) + 'px';
        tooltip.style.top      = (e.clientY - 24) + 'px';
      } else {
        tooltip.style.display = 'none';
      }
    });
    cWrap.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });
  }

  // ── UI ────────────────────────────────────────────────────────────────────────
  function setupUI() {
    setupTooltip();
    setupTraceWindow();

    sldThresh.addEventListener('input', e => {
      threshold = +e.target.value;
      const pct = 100 - threshold;
      lblThresh.textContent = pct === 0 ? 'none' : `top ${pct}%`;
      clearTimeout(rebuildId);
      rebuildId = setTimeout(() => { buildConnections(); updateSelectionPanel(); }, 130);
    });

    btnPlay.addEventListener('click', () => {
      if (!boldActive) return;
      boldPlaying = !boldPlaying;
      btnPlay.textContent = boldPlaying ? '⏸ Pause' : '▶ Play';
    });

    btnBoldReset.addEventListener('click', () => {
      boldT = 0; boldPlaying = false;
      btnPlay.textContent = '▶ Play';
      updateBold(0);
    });

    sldSpeed.addEventListener('input', e => {
      boldSpeed = Math.pow(2, +e.target.value);
      lblSpeed.textContent = boldSpeed.toFixed(2) + '\xd7';
    });

    sldTime.addEventListener('input', e => {
      if (!e.isTrusted) return;
      boldT = +e.target.value;
      boldPlaying = false;
      btnPlay.textContent = '▶ Play';
      updateBold(boldT);
    });

    btnHemi.addEventListener('click', () => {
      showHemi = !showHemi; setBtn(btnHemi, showHemi);
      refreshRegionColors();
      if (boldActive) updateBold(boldT);
      drawBoldTrace();
    });

    btnArea.addEventListener('click', () => {
      showArea = !showArea; setBtn(btnArea, showArea);
      refreshRegionColors();
      if (boldActive) updateBold(boldT);
    });

    btnBrainMesh.addEventListener('click', () => {
      if (!brainMeshObj) return;
      showMesh = !showMesh;
      brainMeshObj.visible = showMesh;
      setBtn(btnBrainMesh, showMesh);
    });

    btnHalos.addEventListener('click', () => {
      if (!haloMesh) return;
      showHalos = !showHalos;
      haloMesh.visible = showHalos;
      setBtn(btnHalos, showHalos);
    });

    if (D.tracts) {
      tractRow.style.display = '';
      btnTracts.addEventListener('click', () => {
        showTracts = !showTracts;
        setBtn(btnTracts, showTracts);
        buildConnections();
      });
    }

    btnDeselect.addEventListener('click', () => selectRegion(-1));

    sldSphereScale.addEventListener('input', e => {
      sphereScale = +e.target.value;
      lblSphereScale.textContent = sphereScale.toFixed(2) + '\xd7';
      refreshRegionColors();
      if (boldActive) updateBold(boldT);
    });

    sldLineWidth.addEventListener('input', e => {
      lineScale = +e.target.value;
      lblLineWidth.textContent = lineScale.toFixed(1) + '\xd7';
      lineMats.forEach(({ mat, baseLw }) => { mat.linewidth = baseLw * lineScale; });
    });

    btnDim.addEventListener('click', () => {
      dimNonSelected = !dimNonSelected;
      setBtn(btnDim, dimNonSelected);
      refreshRegionColors();
      if (boldActive) updateBold(boldT);
    });

    btnResetCam.addEventListener('click', () => {
      cam.position.copy(camHome);
      ctl.target.copy(tgtHome);
      ctl.update();
    });

    btnCopyChat.addEventListener('click', copyStateForChat);
  }

  // ── animation loop ────────────────────────────────────────────────────────────
  function loop(ts) {
    if (disposed) return;
    requestAnimationFrame(loop);
    const dt = Math.min((ts - lastTs) / 1000, 0.1);
    lastTs = ts;
    if (boldPlaying && boldActive && D.bold) {
      boldT = (boldT + dt * boldSpeed * 2.0) % D.bold.data.length;
      updateBold(boldT);
    }
    ctl.update();
    renderer.render(scene, cam);
  }

  // ── dispose ───────────────────────────────────────────────────────────────────
  function dispose() {
    disposed = true;
    clearTimeout(rebuildId);
    buildGeneration++;   // cancels any in-flight buildConnections

    if (renderer) {
      connGroups.forEach(obj => obj.geometry.dispose());
      lineMats.forEach(({ mat }) => mat.dispose());
      if (brainMeshObj) { brainMeshObj.geometry.dispose(); brainMeshObj.material.dispose(); }
      if (haloMesh)     { haloMesh.geometry.dispose(); haloMesh.material.dispose(); }
      if (regMesh)      regMesh.dispose();
      if (ctl)          ctl.dispose();
      renderer.dispose();
    }
    container.innerHTML = '';
    container.classList.remove('bcv-root');
  }

  // ── state snapshot (for the Jupyter Chat assistant) ────────────────────────────
  // Serialise everything the user currently sees — selection, thresholds, toggles,
  // BOLD playback — into a structured object the LLM can read. Reuses the same
  // visible-connection maths as buildConnections()/updateSelectionPanel().
  function buildStateSnapshot() {
    const r2 = v => Math.round(v * 100) / 100;

    const total = D.connections.length;
    const nShow = Math.ceil(total * (1 - threshold / 100));
    let visible = D.connections.slice(0, nShow);
    if (selectedRegion >= 0)
      visible = visible.filter(([i, j]) => i === selectedRegion || j === selectedRegion);
    const nVisible = visible.length;

    let selected = null;
    if (selectedRegion >= 0) {
      const r = D.regions[selectedRegion];
      // D.connections is sorted by weight desc, so the first matches are strongest.
      const top = D.connections
        .filter(([i, j]) => i === selectedRegion || j === selectedRegion)
        .slice(0, 5)
        .map(([i, j, w]) => ({
          region: D.regions[i === selectedRegion ? j : i].name,
          weight: w,
        }));
      selected = {
        index: selectedRegion,
        name: r.name,
        hemisphere: r.hemi === 'L' ? 'left' : 'right',
        area_mm2: r.area,
        visible_connections: nVisible,
        strongest_connections: top,
      };
    }

    let bold = null;
    if (boldActive && D.bold) {
      const { time, data } = D.bold;
      const nT  = data.length;
      const ti  = Math.min(Math.floor(boldT), nT - 1);
      const ti2 = Math.min(ti + 1, nT - 1);
      const tf  = boldT - Math.floor(boldT);
      const tDisp = time[ti] + (time[ti2] - time[ti]) * tf;
      bold = {
        playing: boldPlaying,
        frame: ti,
        n_frames: nT,
        time_ms: Math.round(tDisp),
        time_range_ms: [time[0], time[nT - 1]],
        speed: boldSpeed,
      };
    }

    const stimulated = (D.meta.stimulated_regions || []).map(i => D.regions[i].name);

    return {
      dataset: {
        species:       D.meta.species,
        n_regions:     D.meta.n_regions,
        n_connections: D.meta.n_connections,
        weight_min_nz: D.meta.weight_min_nz,
        weight_max:    D.meta.weight_max,
        area_min:      D.meta.area_min,
        area_max:      D.meta.area_max,
      },
      selected,
      display: {
        threshold_percentile: threshold,
        showing_top_percent:  100 - threshold,
        visible_connections:  nVisible,
        total_connections:    total,
        hemisphere_colors:    showHemi,
        area_scaled_spheres:  showArea,
        brain_mesh:           showMesh,
        fiber_tracts:         showTracts && !!D.tracts,
        stimulated_halos:     showHalos,
        dim_non_selected:     dimNonSelected,
        sphere_scale:         sphereScale,
        line_width_scale:     lineScale,
      },
      bold,
      stimulated_regions: stimulated,
      camera: {
        position: [r2(cam.position.x), r2(cam.position.y), r2(cam.position.z)],
        target:   [r2(ctl.target.x),   r2(ctl.target.y),   r2(ctl.target.z)],
      },
    };
  }

  function formatStateSnapshot(s) {
    const L = [];
    L.push('# Brain Connectivity Viewer — current state');
    L.push('');
    L.push(`Dataset: ${s.dataset.n_regions} regions, ${s.dataset.n_connections} connections (${s.dataset.species || 'brain'} connectome).`);
    L.push('');

    if (s.selected) {
      const sel = s.selected;
      L.push(`## Selected region: ${sel.name}`);
      L.push(`- Hemisphere: ${sel.hemisphere}`);
      L.push(`- Surface area: ${sel.area_mm2} mm²`);
      L.push(`- Visible connections touching it: ${sel.visible_connections}`);
      if (sel.strongest_connections.length) {
        L.push('- Strongest connections:');
        sel.strongest_connections.forEach(c => L.push(`  - ${c.region} (weight ${c.weight})`));
      }
    } else {
      L.push('## Selection: none (whole network shown)');
    }
    L.push('');

    const d = s.display;
    L.push('## Display');
    L.push(`- Connection threshold: showing top ${d.showing_top_percent}% strongest (${d.visible_connections} of ${d.total_connections} connections visible)`);
    L.push(`- Hemisphere colours: ${d.hemisphere_colors ? 'on' : 'off'} · area-scaled spheres: ${d.area_scaled_spheres ? 'on' : 'off'}`);
    L.push(`- Brain mesh: ${d.brain_mesh ? 'on' : 'off'} · fiber tracts: ${d.fiber_tracts ? 'on' : 'off'} · stimulated halos: ${d.stimulated_halos ? 'on' : 'off'} · dim non-selected: ${d.dim_non_selected ? 'on' : 'off'}`);
    L.push('');

    if (s.bold) {
      const b = s.bold;
      L.push('## BOLD activity');
      L.push(`- ${b.playing ? 'Playing' : 'Paused'} at frame ${b.frame + 1}/${b.n_frames} (t = ${b.time_ms} ms; range ${b.time_range_ms[0]}–${b.time_range_ms[1]} ms) · speed ${b.speed}×`);
      L.push('');
    }

    if (s.stimulated_regions.length) {
      L.push(`## Stimulated regions: ${s.stimulated_regions.join(', ')}`);
      L.push('');
    }

    L.push('```json');
    L.push(JSON.stringify(s, null, 2));
    L.push('```');
    return L.join('\n');
  }

  function copyStateForChat() {
    const text = formatStateSnapshot(buildStateSnapshot());
    const done = () => { show('Copied — paste into Chat'); setTimeout(() => show(''), 2200); };
    const fallback = () =>
      window.prompt('Copy the viewer state below, then paste it into Chat:', text);
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done).catch(fallback);
    } else {
      fallback();
    }
  }

  init();
  return { dispose };
}
