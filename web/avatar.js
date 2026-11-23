import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ─────────────────────────────────────────────────────────────────────────────
// Renderer / Scene / Camera
// ─────────────────────────────────────────────────────────────────────────────
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.shadowMap.enabled = true;
document.body.appendChild(renderer.domElement);

const scene  = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a12);
scene.fog = new THREE.FogExp2(0x0a0a12, 0.06);

const camera = new THREE.PerspectiveCamera(
  28, window.innerWidth / window.innerHeight, 0.1, 100
);
camera.position.set(0, 1.72, 2.4);
camera.lookAt(0, 1.6, 0);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 1.6, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.minDistance = 0.8;
controls.maxDistance = 5;
controls.update();

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ─────────────────────────────────────────────────────────────────────────────
// Lights
// ─────────────────────────────────────────────────────────────────────────────
scene.add(new THREE.AmbientLight(0xffffff, 0.5));

const key = new THREE.DirectionalLight(0xfff4e0, 2.5);
key.position.set(1.5, 3.5, 2.5);
key.castShadow = true;
scene.add(key);

const fill = new THREE.DirectionalLight(0xd0e8ff, 1.0);
fill.position.set(-2, 2, 1.5);
scene.add(fill);

const rim = new THREE.DirectionalLight(0xaac8ff, 0.6);
rim.position.set(0, 3, -3);
scene.add(rim);

// Ground
const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(12, 12),
  new THREE.MeshStandardMaterial({ color: 0x08080f, roughness: 1 })
);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

// ─────────────────────────────────────────────────────────────────────────────
// Face canvas texture  (512×512)
// ─────────────────────────────────────────────────────────────────────────────
const TEX_W = 512, TEX_H = 512;
const faceCanvas = document.createElement('canvas');
faceCanvas.width  = TEX_W;
faceCanvas.height = TEX_H;
const fCtx = faceCanvas.getContext('2d');
const faceTex = new THREE.CanvasTexture(faceCanvas);

// Skin palette
const SKIN_L   = '#f2c896';
const SKIN_M   = '#e8b07a';
const SKIN_D   = '#c8885a';
const LIP_UP   = '#c0706a';
const LIP_LO   = '#d48880';
const TEETH    = '#f5f0eb';
const EYE_W    = '#f0f0f0';
const IRIS_C   = '#3a5060';
const PUPIL_C  = '#0a0a0c';
const BROW_C   = '#6a4030';

function drawFace(mouthOpen, blinkL, blinkR, browRaise, viseme) {
  const W = TEX_W, H = TEX_H;
  fCtx.clearRect(0, 0, W, H);

  // ── Skin base ───────────────────────────────────────────────────────────
  const skinGrad = fCtx.createRadialGradient(256, 230, 40, 256, 280, 270);
  skinGrad.addColorStop(0, SKIN_L);
  skinGrad.addColorStop(0.7, SKIN_M);
  skinGrad.addColorStop(1,   SKIN_D);
  fCtx.fillStyle = skinGrad;
  fCtx.beginPath();
  fCtx.ellipse(256, 256, 215, 240, 0, 0, Math.PI * 2);
  fCtx.fill();

  // ── Subtle jaw shadow ────────────────────────────────────────────────────
  const jawGrad = fCtx.createLinearGradient(256, 400, 256, 500);
  jawGrad.addColorStop(0, 'rgba(0,0,0,0)');
  jawGrad.addColorStop(1, 'rgba(0,0,0,0.18)');
  fCtx.fillStyle = jawGrad;
  fCtx.beginPath();
  fCtx.ellipse(256, 256, 215, 240, 0, 0, Math.PI * 2);
  fCtx.fill();

  // ── Eyebrows ─────────────────────────────────────────────────────────────
  const browY = 185 - browRaise * 10;
  drawBrow(fCtx, 178, browY, -0.18, BROW_C);
  drawBrow(fCtx, 334, browY,  0.18, BROW_C);

  // ── Eyes ─────────────────────────────────────────────────────────────────
  drawEye(fCtx, 178, 222, 32, 22, blinkL);
  drawEye(fCtx, 334, 222, 32, 22, blinkR);

  // ── Nose ─────────────────────────────────────────────────────────────────
  fCtx.strokeStyle = SKIN_D;
  fCtx.lineWidth   = 2.5;
  fCtx.lineCap     = 'round';
  fCtx.beginPath();
  fCtx.moveTo(248, 278); fCtx.quadraticCurveTo(242, 320, 236, 330);
  fCtx.stroke();
  fCtx.beginPath();
  fCtx.moveTo(264, 278); fCtx.quadraticCurveTo(270, 320, 276, 330);
  fCtx.stroke();
  fCtx.beginPath();
  fCtx.ellipse(256, 332, 22, 10, 0, 0, Math.PI);
  fCtx.stroke();

  // ── Mouth ────────────────────────────────────────────────────────────────
  drawMouth(fCtx, 256, 390, 78, mouthOpen, viseme);

  faceTex.needsUpdate = true;
}

// ── Eye helper ───────────────────────────────────────────────────────────────
function drawEye(ctx, cx, cy, rw, rh, blink) {
  const blinkScale = blink;     // 0=open, 1=closed
  const openH = rh * (1 - blinkScale * 0.97);

  // White sclera
  ctx.fillStyle = EYE_W;
  ctx.beginPath();
  ctx.ellipse(cx, cy, rw, Math.max(1, openH), 0, 0, Math.PI * 2);
  ctx.fill();

  if (blinkScale < 0.95) {
    // Iris
    const irisR = rw * 0.62;
    const irisGrad = ctx.createRadialGradient(cx, cy - 2, 2, cx, cy, irisR);
    irisGrad.addColorStop(0, '#6a8aa0');
    irisGrad.addColorStop(0.6, IRIS_C);
    irisGrad.addColorStop(1, '#1a2830');
    ctx.fillStyle = irisGrad;
    ctx.beginPath();
    ctx.arc(cx, cy, Math.min(irisR, openH * 0.95), 0, Math.PI * 2);
    ctx.fill();

    // Pupil
    ctx.fillStyle = PUPIL_C;
    ctx.beginPath();
    ctx.arc(cx, cy, irisR * 0.48, 0, Math.PI * 2);
    ctx.fill();

    // Specular
    ctx.fillStyle = 'rgba(255,255,255,0.75)';
    ctx.beginPath();
    ctx.ellipse(cx - irisR * 0.28, cy - irisR * 0.3, irisR * 0.18, irisR * 0.14, -0.4, 0, Math.PI * 2);
    ctx.fill();
  }

  // Upper eyelid shadow
  const lidGrad = ctx.createLinearGradient(cx, cy - rh, cx, cy);
  lidGrad.addColorStop(0,   'rgba(100,60,40,0.22)');
  lidGrad.addColorStop(0.5, 'rgba(100,60,40,0.05)');
  lidGrad.addColorStop(1,   'rgba(100,60,40,0)');
  ctx.fillStyle = lidGrad;
  ctx.beginPath();
  ctx.ellipse(cx, cy, rw, Math.max(1, openH), 0, 0, Math.PI * 2);
  ctx.fill();

  // Lashes (top arc)
  ctx.strokeStyle = '#1a0a08';
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  ctx.ellipse(cx, cy, rw + 2, Math.max(1, openH * blinkScale < 0.5 ? openH + 2 : openH * 0.5 + 1), 0, Math.PI, Math.PI * 2);
  ctx.stroke();
}

// ── Brow helper ──────────────────────────────────────────────────────────────
function drawBrow(ctx, cx, cy, angle, color) {
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(angle);
  ctx.strokeStyle = color;
  ctx.lineWidth   = 5;
  ctx.lineCap     = 'round';
  ctx.beginPath();
  ctx.moveTo(-26, 0);
  ctx.quadraticCurveTo(0, -7, 26, 2);
  ctx.stroke();
  ctx.restore();
}

// ── Mouth helper ─────────────────────────────────────────────────────────────
function drawMouth(ctx, cx, cy, halfW, open, viseme) {
  // open: 0=closed, 1=fully open
  // Shape coefficients per viseme:
  //   [openH_scale, wideScale, roundness]
  const shapes = {
    'viseme_sil': [0.0,  1.0,  0.5],
    'viseme_aa':  [1.0,  1.0,  0.7],
    'viseme_E':   [0.55, 1.25, 0.4],
    'viseme_I':   [0.25, 1.15, 0.35],
    'viseme_O':   [0.85, 0.75, 0.95],
    'viseme_U':   [0.7,  0.65, 0.98],
    'viseme_PP':  [0.0,  0.85, 0.5],
    'viseme_FF':  [0.15, 0.9,  0.4],
    'viseme_DD':  [0.2,  0.95, 0.5],
    'viseme_kk':  [0.3,  1.0,  0.55],
    'viseme_CH':  [0.4,  0.9,  0.65],
    'viseme_SS':  [0.15, 1.0,  0.38],
    'viseme_TH':  [0.2,  1.0,  0.42],
    'viseme_nn':  [0.1,  0.9,  0.45],
    'viseme_RR':  [0.45, 0.8,  0.75],
  };
  const [oh, ws, rn] = shapes[viseme] ?? shapes['viseme_sil'];
  const openH  = halfW * 0.7  * oh * open;
  const lipW   = halfW * ws;

  // ── Upper lip ────────────────────────────────────────────────────────────
  const lipThick = 10 + open * 4;
  const upperY   = cy - openH;
  const lowerY   = cy + openH;

  // Mouth cavity (inside) — drawn first so lips sit on top
  if (openH > 2) {
    // Dark interior
    ctx.fillStyle = '#1a0808';
    ctx.beginPath();
    ctx.ellipse(cx, cy, lipW * 0.88, openH * 0.95, 0, 0, Math.PI * 2);
    ctx.fill();

    // Teeth strip (upper)
    if (openH > 6) {
      const teethH = Math.min(openH * 0.45, 14);
      const teethW = lipW * 0.78;
      ctx.fillStyle = TEETH;
      ctx.beginPath();
      ctx.ellipse(cx, upperY + teethH * 0.5, teethW, teethH, 0, 0, Math.PI * 2);
      ctx.fill();

      // Tooth dividers
      ctx.strokeStyle = 'rgba(200,190,180,0.4)';
      ctx.lineWidth = 1;
      for (let i = -1; i <= 1; i++) {
        ctx.beginPath();
        ctx.moveTo(cx + i * teethW * 0.3, upperY);
        ctx.lineTo(cx + i * teethW * 0.3, upperY + teethH);
        ctx.stroke();
      }

      // Lower teeth (darker, smaller)
      if (openH > 12) {
        const ltH = teethH * 0.7;
        ctx.fillStyle = '#ddd8d0';
        ctx.beginPath();
        ctx.ellipse(cx, lowerY - ltH * 0.5, teethW * 0.72, ltH, 0, Math.PI, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  // ── Lip shapes (bezier) ──────────────────────────────────────────────────
  const ctrlBulge = lipThick * rn * 1.1;

  // Upper lip
  const ulGrad = ctx.createLinearGradient(cx, upperY - lipThick * 0.8, cx, upperY + lipThick * 0.5);
  ulGrad.addColorStop(0, '#a85055');
  ulGrad.addColorStop(0.5, LIP_UP);
  ulGrad.addColorStop(1, '#b86060');
  ctx.fillStyle = ulGrad;
  ctx.beginPath();
  ctx.moveTo(cx - lipW, cy);
  ctx.bezierCurveTo(
    cx - lipW * 0.5, upperY - ctrlBulge,
    cx + lipW * 0.5, upperY - ctrlBulge,
    cx + lipW, cy
  );
  // Cupid's bow
  ctx.bezierCurveTo(
    cx + lipW * 0.4, upperY + lipThick * 0.3,
    cx + lipW * 0.12, upperY - lipThick * 0.1,
    cx, upperY - lipThick * 0.2
  );
  ctx.bezierCurveTo(
    cx - lipW * 0.12, upperY - lipThick * 0.1,
    cx - lipW * 0.4,  upperY + lipThick * 0.3,
    cx - lipW, cy
  );
  ctx.closePath();
  ctx.fill();

  // Upper lip highlight
  ctx.fillStyle = 'rgba(255,200,180,0.18)';
  ctx.beginPath();
  ctx.ellipse(cx, upperY - lipThick * 0.1, lipW * 0.4, lipThick * 0.28, 0, 0, Math.PI * 2);
  ctx.fill();

  // Lower lip
  const llGrad = ctx.createLinearGradient(cx, lowerY - lipThick * 0.3, cx, lowerY + lipThick);
  llGrad.addColorStop(0, LIP_LO);
  llGrad.addColorStop(0.6, '#c87870');
  llGrad.addColorStop(1, '#b06050');
  ctx.fillStyle = llGrad;
  ctx.beginPath();
  ctx.moveTo(cx - lipW, cy);
  ctx.bezierCurveTo(
    cx - lipW * 0.45, lowerY + ctrlBulge * 1.1,
    cx + lipW * 0.45, lowerY + ctrlBulge * 1.1,
    cx + lipW, cy
  );
  ctx.bezierCurveTo(
    cx + lipW * 0.35, lowerY + lipThick * 0.25,
    cx - lipW * 0.35, lowerY + lipThick * 0.25,
    cx - lipW, cy
  );
  ctx.closePath();
  ctx.fill();

  // Lower lip highlight
  ctx.fillStyle = 'rgba(255,210,190,0.22)';
  ctx.beginPath();
  ctx.ellipse(cx, lowerY + lipThick * 0.3, lipW * 0.32, lipThick * 0.22, 0, 0, Math.PI * 2);
  ctx.fill();

  // Mouth corner shadows
  ctx.fillStyle = 'rgba(100,50,40,0.28)';
  ctx.beginPath(); ctx.arc(cx - lipW, cy, 5, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath(); ctx.arc(cx + lipW, cy, 5, 0, Math.PI * 2); ctx.fill();
}

// ─────────────────────────────────────────────────────────────────────────────
// 3D Avatar geometry
// ─────────────────────────────────────────────────────────────────────────────
const HEAD_R = 0.42;
const avatarGroup = new THREE.Group();
avatarGroup.position.y = 1.0;
scene.add(avatarGroup);

// Head sphere — front half uses face texture, back half skin colour
const headGeo = new THREE.SphereGeometry(HEAD_R, 64, 64);
// Split UVs so face fills front hemisphere (we'll just use the full sphere
// but map the texture tightly to the front)
const headMat = new THREE.MeshStandardMaterial({
  map: faceTex,
  roughness: 0.7,
  metalness: 0.0,
});
const head = new THREE.Mesh(headGeo, headMat);
head.castShadow = true;
head.position.y = 0.52;
avatarGroup.add(head);

// Back-of-head (hemisphere cap behind) — same skin tone, no texture
const backMat = new THREE.MeshStandardMaterial({
  color: new THREE.Color(SKIN_M),
  roughness: 0.75,
});
// We'll tint the head material's back-face via a second mesh
const backGeo  = new THREE.SphereGeometry(HEAD_R * 0.995, 32, 32);
const backMesh = new THREE.Mesh(backGeo, backMat);
backMesh.position.y = 0.52;
avatarGroup.add(backMesh);

// Hair
const hairMat = new THREE.MeshStandardMaterial({ color: 0x2a1808, roughness: 0.9 });

// Top hair cap
const hairTop = new THREE.Mesh(
  new THREE.SphereGeometry(HEAD_R * 1.02, 32, 32, 0, Math.PI * 2, 0, Math.PI * 0.52),
  hairMat
);
hairTop.position.y = 0.52;
hairTop.castShadow = true;
avatarGroup.add(hairTop);

// Side hair flaps
for (const side of [-1, 1]) {
  const flap = new THREE.Mesh(
    new THREE.SphereGeometry(HEAD_R * 0.35, 16, 16),
    hairMat
  );
  flap.scale.set(0.55, 1.4, 0.4);
  flap.position.set(side * HEAD_R * 0.9, 0.52 + HEAD_R * 0.08, 0);
  avatarGroup.add(flap);
}

// Neck
const neck = new THREE.Mesh(
  new THREE.CylinderGeometry(0.105, 0.12, 0.28, 24),
  new THREE.MeshStandardMaterial({ color: new THREE.Color(SKIN_M), roughness: 0.75 })
);
neck.position.y = 0.17;
neck.castShadow = true;
avatarGroup.add(neck);

// Body (simple torso silhouette)
const bodyMat = new THREE.MeshStandardMaterial({ color: 0x1a1a2a, roughness: 0.8 });

const torso = new THREE.Mesh(
  new THREE.CapsuleGeometry(0.25, 0.55, 8, 16),
  bodyMat
);
torso.position.y = -0.27;
torso.castShadow = true;
avatarGroup.add(torso);

// Shoulders
for (const side of [-1, 1]) {
  const sh = new THREE.Mesh(
    new THREE.SphereGeometry(0.16, 16, 16),
    bodyMat
  );
  sh.position.set(side * 0.38, -0.12, 0);
  sh.castShadow = true;
  avatarGroup.add(sh);
}

// ─────────────────────────────────────────────────────────────────────────────
// Animation state
// ─────────────────────────────────────────────────────────────────────────────
// Viseme weights received from WebSocket
const targetWeights  = {};
const currentWeights = {};
const ALL_VISEMES = [
  'viseme_sil','viseme_aa','viseme_E','viseme_I','viseme_O','viseme_U',
  'viseme_PP','viseme_FF','viseme_DD','viseme_kk','viseme_CH',
  'viseme_SS','viseme_TH','viseme_nn','viseme_RR',
];
for (const v of ALL_VISEMES) {
  targetWeights[v]  = v === 'viseme_sil' ? 1 : 0;
  currentWeights[v] = v === 'viseme_sil' ? 1 : 0;
}

let amplitude = 0;

// Idle state
let blinkLTarget = 0, blinkRTarget = 0;
let blinkLCurrent = 0, blinkRCurrent = 0;
let nextBlink = performance.now() + rand(2000, 5000);
let blinking  = false;
let blinkStart = 0;
const BLINK_MS = 130;

let swayT = 0;

// ─────────────────────────────────────────────────────────────────────────────
// WebSocket
// ─────────────────────────────────────────────────────────────────────────────
const statusEl = document.getElementById('status');
const barEl    = document.getElementById('amplitude-bar');

let ws = null;
function connectWS() {
  ws = new WebSocket('ws://localhost:8765');
  ws.onopen  = () => { statusEl.textContent = 'connected'; statusEl.className = 'active'; };
  ws.onclose = () => { statusEl.textContent = 'reconnecting…'; setTimeout(connectWS, 2000); };
  ws.onerror = () => {};
  ws.onmessage = (evt) => {
    try {
      const msg = JSON.parse(evt.data);
      if (msg.type === 'visemes') {
        for (const v of ALL_VISEMES) targetWeights[v] = 0;
        for (const [k, v] of Object.entries(msg.weights)) {
          if (k in targetWeights) targetWeights[k] = v;
        }
        amplitude = msg.amplitude ?? 0;
        barEl.style.width = (amplitude * 100).toFixed(1) + '%';
      }
    } catch (_) {}
  };
}
connectWS();

// Hide overlay immediately (no external model needed)
document.getElementById('overlay').style.display = 'none';

// ─────────────────────────────────────────────────────────────────────────────
// Render loop
// ─────────────────────────────────────────────────────────────────────────────
const clock = new THREE.Clock();

renderer.setAnimationLoop(() => {
  const dt  = clock.getDelta() * 1000; // ms
  const now = performance.now();
  swayT += dt * 0.001;

  // ── Lerp viseme weights ──────────────────────────────────────────────────
  const k = 0.14;  // lerp factor per frame (~16ms)
  for (const v of ALL_VISEMES) {
    currentWeights[v] += (targetWeights[v] - currentWeights[v]) * k;
  }

  // Dominant viseme for face drawing
  let domViseme = 'viseme_sil';
  let domWeight = 0;
  for (const v of ALL_VISEMES) {
    if (v !== 'viseme_sil' && currentWeights[v] > domWeight) {
      domWeight = currentWeights[v];
      domViseme = v;
    }
  }
  const mouthOpen = amplitude < 0.03 ? 0 : Math.min(1, domWeight);

  // ── Blink ────────────────────────────────────────────────────────────────
  if (!blinking && now >= nextBlink) {
    blinking = true; blinkStart = now;
  }
  if (blinking) {
    const t = (now - blinkStart) / BLINK_MS;
    const v = t < 0.4 ? t / 0.4 : t < 1 ? 1 - (t - 0.4) / 0.6 : 0;
    blinkLTarget = blinkRTarget = Math.max(0, v);
    if (t >= 1) { blinking = false; nextBlink = now + rand(2500, 6000); }
  } else {
    blinkLTarget = blinkRTarget = 0;
  }
  blinkLCurrent += (blinkLTarget - blinkLCurrent) * 0.22;
  blinkRCurrent += (blinkRTarget - blinkRCurrent) * 0.22;

  // ── Brow raise (subtle idle) ─────────────────────────────────────────────
  const browRaise = (Math.sin(swayT * 0.35) * 0.5 + 0.5) * 0.25;

  // ── Head sway ────────────────────────────────────────────────────────────
  head.rotation.y  = Math.sin(swayT * 0.22) * 0.04;
  head.rotation.x  = Math.sin(swayT * 0.17) * 0.025 - 0.02;
  head.rotation.z  = Math.sin(swayT * 0.13) * 0.018;
  hairTop.rotation.copy(head.rotation);
  neck.rotation.y  = head.rotation.y * 0.4;

  // ── Redraw face canvas ───────────────────────────────────────────────────
  drawFace(mouthOpen, blinkLCurrent, blinkRCurrent, browRaise, domViseme);

  controls.update();
  renderer.render(scene, camera);
});

// ─────────────────────────────────────────────────────────────────────────────
function rand(a, b) { return a + Math.random() * (b - a); }
