'use strict';

import * as LJS from 'littlejsengine';
const { vec2, hsl, PI } = LJS;

// Hand tracking + crab overlay for KaniAttack
// - Uses MediaPipe Hand Landmarker to get 5 fingertip positions
// - Spawns a central crab body with 5 limb tips
// - Each limb tip is connected to the center with a distance joint
// - Each limb tip also has a TargetJoint that is driven toward the fingertip world position

let handLandmarker: any = null;
let videoEl: HTMLVideoElement | null = null;
let running = false;
// timestamp of last detection call (ms)
let lastDetectTime = 0;
let loopRunning = false;

// Video capture parameters (tweakable)
export const VIDEO_WIDTH = 320;
export const VIDEO_HEIGHT = 240;
export const VIDEO_ELEMENT_SCALE = 0.5;
export const VIDEO_FPS = 15;
export const VIDEO_FACING_MODE = 'user';
export const VIDEO_OPACITY = '0.8';

const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm';

const fingertipIndices = [4, 8, 12, 16, 20]; // thumb, index, middle, ring, pinky

// last known fingertip world positions (mapped into game world coords)
const fingerWorld = [vec2(20, 10), vec2(20, 10), vec2(20, 10), vec2(20, 10), vec2(20, 10)];

// Debug overlay: canvas and last normalized landmarks from MediaPipe
let debugCanvas: HTMLCanvasElement | null = null;
let debugCtx: CanvasRenderingContext2D | null = null;
let lastLandmarks: any[] = [];

// Crab physics objects
let crabCenter: any = null;
const tipObjects: any[] = [];
const tipTargetJoints: any[] = [];
const tipDistanceJoints: any[] = [];

export async function initHandTracking() {
  if (handLandmarker) return;

  try {
    // create a hidden low-res video element and configure it with constants
    videoEl = document.createElement('video');
    videoEl.className = 'hand-landmark-video';
    videoEl.style.position = 'fixed';
    videoEl.style.left = '10px';
    videoEl.style.top = '10px';
    videoEl.style.width = VIDEO_WIDTH * VIDEO_ELEMENT_SCALE + 'px';
    videoEl.style.height = VIDEO_HEIGHT * VIDEO_ELEMENT_SCALE + 'px';
    videoEl.style.opacity = VIDEO_OPACITY;
    videoEl.style.zIndex = '1000';
    videoEl.style.pointerEvents = 'none';
    // mirror video horizontally so movement feels like a mirror
    const mirrorTransform = 'scaleX(-1)';
    videoEl.style.transform = mirrorTransform;
    videoEl.setAttribute('playsinline', '');
    // set actual element pixel size to match capture resolution
    videoEl.width = VIDEO_WIDTH;
    videoEl.height = VIDEO_HEIGHT;
    document.body.appendChild(videoEl);

    // create an overlay canvas for drawing debug landmarks
    debugCanvas = document.createElement('canvas');
    debugCanvas.className = 'hand-landmark-debug';
    debugCanvas.width = VIDEO_WIDTH;
    debugCanvas.height = VIDEO_HEIGHT;
    debugCanvas.style.position = 'fixed';
    debugCanvas.style.left = videoEl.style.left;
    debugCanvas.style.top = videoEl.style.top;
    debugCanvas.style.width = VIDEO_WIDTH * VIDEO_ELEMENT_SCALE + 'px';
    debugCanvas.style.height = VIDEO_HEIGHT * VIDEO_ELEMENT_SCALE + 'px';
    debugCanvas.style.zIndex = '1001';
    debugCanvas.style.pointerEvents = 'none';
    debugCanvas.style.transform = mirrorTransform;
    document.body.appendChild(debugCanvas);
    debugCtx = debugCanvas.getContext('2d');

    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: VIDEO_WIDTH,
        height: VIDEO_HEIGHT,
        frameRate: VIDEO_FPS,
        facingMode: VIDEO_FACING_MODE,
      },
      audio: false,
    });
    videoEl.srcObject = stream;
    await videoEl.play();

    // dynamically import the mediapipe tasks bundle
    const m = await import('@mediapipe/tasks-vision');
    const { FilesetResolver, HandLandmarker } = m;

    const vision = await FilesetResolver.forVisionTasks(WASM_URL);
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URL },
      runningMode: 'VIDEO',
      numHands: 1,
      minHandDetectionConfidence: 0.3,
      minHandPresenceConfidence: 0.3,
      minTrackingConfidence: 0.3,
    });

    running = true;
    if (!loopRunning) {
      loopRunning = true;
      requestAnimationFrame(detectLoop);
    }
  } catch (e) {
    console.warn('initHandTracking failed', e);
  }
}

function detectLoop(now?: number) {
  if (!running || !handLandmarker || !videoEl) {
    requestAnimationFrame(detectLoop);
    return;
  }

  const t = now || performance.now();
  const interval = 1000 / VIDEO_FPS;
  if (t - lastDetectTime >= interval && videoEl.readyState >= 2) {
    lastDetectTime = t;
    try {
      const res = handLandmarker.detectForVideo(videoEl, t);
      if (res && res.landmarks && res.landmarks.length) {
        const lm = res.landmarks[0];
        // copy landmarks for debug overlay (normalized coordinates)
        lastLandmarks = lm.map((p: any) => ({ x: p.x, y: p.y, z: p.z ?? 0 }));
        for (let i = 0; i < 5; i++) {
          const idx = fingertipIndices[i];
          const l = lm[idx];
          if (l) fingerWorld[i] = lmToWorld(l);
        }
      } else {
        lastLandmarks = [];
      }
    } catch (e) {
      console.warn('detectForVideo error', e);
    }
  }

  drawDebugOverlay();

  requestAnimationFrame(detectLoop);
}

function lmToWorld(lm: any) {
  // lm.x/lm.y are normalized [0..1] with origin at top-left
  // map to game world: mirror X (so movement feels like a mirror) and scale to world units
  const x = (1 - lm.x) * 40;
  const y = (1 - lm.y) * 20;
  return vec2(x, y);
}

export function initCrab(groundObject: any) {
  // destroy existing crab
  if (crabCenter) {
    tipTargetJoints.forEach((j) => j && j.destroy());
    tipDistanceJoints.forEach((j) => j && j.destroy());
    tipObjects.forEach((o) => o && o.destroy());
    crabCenter.destroy();
    tipObjects.length = 0;
    tipTargetJoints.length = 0;
    tipDistanceJoints.length = 0;
    crabCenter = null;
  }

  // center body (static kinematic anchor)
  crabCenter = new LJS.Box2dObject(vec2(20, 10), vec2(1), 0, 0, hsl(0.6, 0.6, 0.45), LJS.box2d.bodyTypeStatic);
  crabCenter.addCircle(1);
  crabCenter.setMass(1);

  // create 5 limb tips around the center
  for (let i = 0; i < 5; i++) {
    const angle = (i / 5) * PI * 2;
    const startPos = crabCenter.pos.add(vec2(Math.cos(angle), Math.sin(angle)).scale(3));
    const tip = new LJS.Box2dObject(startPos, vec2(0.6), 0, 0, hsl(i / 5, 0.8, 0.5), LJS.box2d.bodyTypeDynamic);
    tip.addCircle(0.6, vec2(), 1, 0.5, 0.2);
    tip.setLinearDamping(3);
    tip.setAngularDamping(4);
    tipObjects.push(tip);

    // distance joint to act like a limb
    const dj = new LJS.Box2dDistanceJoint(crabCenter, tip, crabCenter.pos, tip.pos, false);
    tipDistanceJoints.push(dj);

    // target joint driven by fingertip position; attached to the global ground object (exists)
    const tj = new LJS.Box2dTargetJoint(tip, groundObject, tip.pos);
    tipTargetJoints.push(tj);
  }
}

function drawDebugOverlay() {
  if (!debugCtx || !debugCanvas) return;
  const ctx = debugCtx;
  const w = debugCanvas.width;
  const h = debugCanvas.height;
  ctx.clearRect(0, 0, w, h);
  if (!lastLandmarks || !lastLandmarks.length) return;

  // finger chains (indices within MediaPipe 21-point model)
  const fingers = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
    [17, 18, 19, 20],
  ];

  ctx.lineWidth = 2;
  for (const finger of fingers) {
    ctx.beginPath();
    for (let i = 0; i < finger.length; i++) {
      const p = lastLandmarks[finger[i]];
      if (!p) continue;
      const x = p.x * w;
      const y = p.y * h;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = 'rgba(0,200,255,0.9)';
    ctx.stroke();
  }

  // draw points (bigger for fingertips)
  for (let i = 0; i < lastLandmarks.length; i++) {
    const p = lastLandmarks[i];
    if (!p) continue;
    const x = p.x * w;
    const y = p.y * h;
    ctx.beginPath();
    ctx.arc(x, y, fingertipIndices.includes(i) ? 6 : 3, 0, Math.PI * 2);
    ctx.fillStyle = fingertipIndices.includes(i) ? 'rgba(255,0,0,0.95)' : 'rgba(0,255,0,0.6)';
    ctx.fill();
  }
}

export function updateCrab() {
  // update target joints toward last known fingertip world positions
  for (let i = 0; i < 5; i++) {
    const tj = tipTargetJoints[i];
    if (tj) tj.setTarget(fingerWorld[i]);
  }
}

export function renderCrab() {
  if (!crabCenter) return;
  for (let i = 0; i < 5; i++) {
    const tip = tipObjects[i];
    if (!tip) continue;
    // draw limb line
    LJS.drawLine(crabCenter.pos, tip.pos, 0.12, LJS.BLACK);
    // draw fingertip debug target as a thin line from tip to the target
    LJS.drawLine(tip.pos, fingerWorld[i], 0.06, hsl(i / 5, 1, 0.6, 0.7));
  }
}
