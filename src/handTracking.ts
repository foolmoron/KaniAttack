'use strict';

import { HandLandmarker } from '@mediapipe/tasks-vision';
import * as LJS from 'littlejsengine';
const { vec2, hsl, PI } = LJS;

interface Box2dNode extends LJS.Box2dObject {
  prevVel?: LJS.Vector2;
  touching?: boolean;
}

// Hand tracking + crab overlay for KaniAttack
// - Uses MediaPipe Hand Landmarker to get 4 fingertip positions (ignores thumb)
// - Spawns a central crab body with 4 limb tips
// - Each limb tip is connected to the center with a distance joint
// - Each limb tip also has a TargetJoint that is driven toward the fingertip world position

let handLandmarker: HandLandmarker | null = null;
let videoEl: HTMLVideoElement | null = null;
let running = false;
// timestamp of last detection call (ms)
let lastDetectTime = 0;
let loopRunning = false;

// Video capture parameters (tweakable)
export const VIDEO_WIDTH = 160;
export const VIDEO_HEIGHT = 120;
export const VIDEO_ELEMENT_SCALE = 1.0;
export const VIDEO_FPS = 24;
export const VIDEO_FACING_MODE = 'user';
export const VIDEO_OPACITY = '0.8';

const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.34/wasm';

// Use only the 4 non-thumb fingertips: index, middle, ring, pinky
const fingertipIndices = [8, 12, 16, 20]; // index, middle, ring, pinky

// last known fingertip world positions (mapped into game world coords)
const fingerWorld = [vec2(20, 10), vec2(20, 10), vec2(20, 10), vec2(20, 10)];
// previous raw fingertip world positions (used to compute per-tick landmark deltas)
const prevFingerWorld = [vec2(20, 10), vec2(20, 10), vec2(20, 10), vec2(20, 10)];

// Debug overlay: canvas and last normalized landmarks from MediaPipe
let debugCanvas: HTMLCanvasElement | null = null;
let debugCtx: CanvasRenderingContext2D | null = null;
let lastLandmarks: any[] = [];

// Tuning constants for springs and targets
// make limb distance joints stiffer so collisions better transmit force to center
const DIST_JOINT_FREQUENCY = 10; // Hz
const DIST_JOINT_DAMPING = 0.2; // damping ratio
// tip target joints should respond quickly to finger extensions
const TARGET_JOINT_FREQUENCY = 15; // Hz (tip joints)
// increased max force so tip target joints can push off the world and
// transfer significant impulses to the crab center
// (lower default to make interactions less explosive)
const TARGET_JOINT_MAX_FORCE = 800;
const CENTER_TARGET_FREQUENCY = 8; // Hz (center spring)
const CENTER_TARGET_MAX_FORCE = 800;
const SMOOTH_ALPHA = 0.6; // exponential smoothing for landmark targets
// Pull-to-center tuning
const PULL_TRANSFER = 0.35; // fraction of tip pull impulse applied to center
const PULL_DAMPING_RATIO = 0.7; // damping ratio used when estimating tip damping
const PULL_MIN_DISTANCE = 0.05; // ignore tiny tip-target offsets
// fraction of collision-derived impulse applied to center (reduce explosions)
const COLLISION_TRANSFER = 0.5;

// multiplier applied to fingertip offset from hand center when computing
// tip targets. Values >1 make fingers reach further from the detected center,
// values <1 pull them closer.
const FINGER_LENGTH_MULTIPLIER = 0.65;
const FINGER_OFFSET = vec2(0, -3);

// maximum linear speed for the crab center (world units/sec). Set to 0 to disable.
const CENTER_MAX_VELOCITY = 10.0;

// Crab physics objects
let crabCenter: LJS.Box2dObject | null = null;
// kinematic anchor used to attach limb roots so joints don't move the center
let rootAnchor: LJS.Box2dObject | null = null;
// limbNodes[limbIndex] = array of 4 nodes (node0..node3 where node3 is the tip)
const limbNodes: Box2dNode[][] = [];
// limbDistanceJoints[limbIndex] = array of 4 distance joints (center->node0, node0->node1, node1->node2, node2->node3)
const limbDistanceJoints: LJS.Box2dDistanceJoint[][] = [];
// target joints for each tip (node3)
const tipTargetJoints: LJS.Box2dTargetJoint[] = [];
// smoothed landmark targets used to drive target joints
const smoothedFingerWorld = [vec2(20, 10), vec2(20, 10), vec2(20, 10), vec2(20, 10)];
// computed world targets for tips (mapped relative to the crab center)
const tipWorldTargets = [vec2(20, 10), vec2(20, 10), vec2(20, 10), vec2(20, 10)];
// smoothed hand center target and center target joint
let smoothedHandCenter = vec2(20, 10);
let centerTargetJoint: LJS.Box2dObject | null = null;
// raw detected hand center (from landmarks)
let rawHandWorld = vec2(20, 10);

export async function initHandTracking() {
  if (handLandmarker) {
    return;
  }

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
      baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
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
      const videoFrame = new VideoFrame(videoEl, {
        timestamp: t,
      });
      const res = handLandmarker.detectForVideo(videoFrame, t);
      videoFrame.close();
      if (res && res.landmarks && res.landmarks.length) {
        const lm = res.landmarks[0];
        // copy landmarks for debug overlay (normalized coordinates)
        lastLandmarks = lm.map((p: any) => ({ x: p.x, y: p.y, z: p.z ?? 0 }));
        for (let i = 0; i < 4; i++) {
          const idx = fingertipIndices[i];
          const l = lm[idx];
          if (l) {
            fingerWorld[i] = lmToWorld(l);
          }
        }
        // landmark index 0 (wrist / isolated palm point) as approximate hand center
        const centerIdx = 0;
        if (lm[centerIdx]) {
          rawHandWorld = lmToWorld(lm[centerIdx]);
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

function computeTipPullImpulse(tipIndex: number) {
  const tip = limbNodes[tipIndex] && limbNodes[tipIndex][3];
  if (!tip) return vec2(0, 0);
  const target = tipWorldTargets[tipIndex] || smoothedFingerWorld[tipIndex];
  const disp = target.subtract(tip.pos);
  const dist = disp.length();
  if (dist < PULL_MIN_DISTANCE) return vec2(0, 0);

  const tipMass = tip.getMass() || 1;
  const freq = TARGET_JOINT_FREQUENCY;
  const wn = 2 * PI * freq;
  const k = tipMass * wn * wn;
  const c = 4 * PI * PULL_DAMPING_RATIO * tipMass * freq;

  const vel = tip.getLinearVelocity();
  const spring = disp.copy().scale(k);
  const damp = vel.copy().scale(c);
  const force = spring.add(damp.scale(-1));

  const maxF = TARGET_JOINT_MAX_FORCE || 0;
  if (maxF > 0) {
    const mag = force.length();
    if (mag > maxF) {
      force.scale(maxF / mag);
    }
  }

  const dt = (globalThis as any).timeDelta || 1 / 60;
  return force.scale(dt);
}

export function initCrab(groundObject: LJS.Box2dObject) {
  // destroy existing crab (skip if already destroyed by engine cleanup)
  if (crabCenter && !crabCenter.destroyed) {
    tipTargetJoints.forEach((j) => j && j.destroy());
    if (centerTargetJoint) centerTargetJoint.destroy();
    limbDistanceJoints.forEach((arr) => arr.forEach((j) => j && j.destroy()));
    limbNodes.forEach((arr) => arr.forEach((o) => o && o.destroy()));
    if (rootAnchor) {
      rootAnchor.destroy();
      rootAnchor = null;
    }
    crabCenter.destroy();
    limbNodes.length = 0;
    limbDistanceJoints.length = 0;
    tipTargetJoints.length = 0;
    centerTargetJoint = null;
    crabCenter = null;
  }

  // center body (dynamic movable ball)
  crabCenter = new LJS.Box2dObject(vec2(20, 10), vec2(2), 0, 0, hsl(0.6, 0.6, 0.45), LJS.box2d.bodyTypeDynamic);
  crabCenter.addCircle(1.8);
  // higher mass so limbs must push to lift it (center moved only by impulses)
  crabCenter.setMass(12);
  crabCenter.setLinearDamping(1);
  crabCenter.setAngularDamping(2);
  // ensure center collides with limb nodes (category 1)
  crabCenter.setFilterData(2, 2);

  // create a kinematic root anchor placed at the center position. Joints
  // will attach limbs to this anchor so joint forces don't move the center.
  rootAnchor = new LJS.Box2dObject(
    crabCenter.pos.copy(),
    vec2(0.2),
    0,
    0,
    hsl(0, 0, 0, 0),
    LJS.box2d.bodyTypeKinematic
  );
  rootAnchor.setGravityScale(0);

  // create 4 limbs; each limb has 4 nodes (0..3) where node3 is the tip
  const nodeDistances = [2.5, 4.0, 5.5, 7.0];
  for (let i = 0; i < 4; i++) {
    const angle = (i / 4) * PI * 2;
    const dir = vec2(Math.cos(angle), Math.sin(angle));
    const nodes: Box2dNode[] = [];
    const joints: LJS.Box2dDistanceJoint[] = [];

    // create 4 nodes for this limb
    for (let j = 0; j < 4; j++) {
      const pos = crabCenter.pos.add(dir.scale(nodeDistances[j]));
      const node: Box2dNode = new LJS.Box2dObject(
        pos,
        vec2(0.6),
        0,
        0,
        hsl(i / 4, 0.8, 0.5),
        LJS.box2d.bodyTypeDynamic
      );
      node.addCircle(0.6, vec2(), 1, 0.5, 0.2);
      node.setLinearDamping(3);
      node.setAngularDamping(4);
      node.setBullet(true);
      // limb nodes should not be affected by global gravity—keep them free-floating
      node.setGravityScale(0);
      // make limb nodes light so collisions mainly affect tips, not the center
      node.setMass(0.4);
      // store previous velocity to estimate collision impulse on contact
      node.prevVel = vec2(0, 0);
      // set filter so limb nodes don't collide with other limb nodes (category 2, ignore 2)
      node.setFilterData(2, 2);
      nodes.push(node);
    }

    // rootAnchor -> node0 (anchor is kinematic so joints won't move the center)
    const dj0 = new LJS.Box2dDistanceJoint(rootAnchor!, nodes[0], rootAnchor!.pos, nodes[0].pos, false);
    dj0.setFrequency(DIST_JOINT_FREQUENCY);
    dj0.setDampingRatio(DIST_JOINT_DAMPING);
    joints.push(dj0);

    // node0 -> node1, node1 -> node2, node2 -> node3
    for (let j = 0; j < 3; j++) {
      const a = nodes[j];
      const b = nodes[j + 1];
      const dj = new LJS.Box2dDistanceJoint(a, b, a.pos, b.pos, false);
      dj.setFrequency(DIST_JOINT_FREQUENCY);
      dj.setDampingRatio(DIST_JOINT_DAMPING);
      joints.push(dj);
    }

    // tip target joint attached to the world/ground so the joint doesn't
    // directly apply forces to the crab center. Impulses from tip contacts
    // are detected and transferred artificially to the center.
    // Attach the tip target joint to the kinematic `rootAnchor` so the
    // joint doesn't apply forces to the crab center and doesn't suppress
    // collisions between the tip and the ground object.
    const tj = new LJS.Box2dTargetJoint(nodes[3], rootAnchor, nodes[3].pos);
    if (tj.setMaxForce) {
      tj.setMaxForce(TARGET_JOINT_MAX_FORCE);
    }
    if (tj.setFrequency) {
      tj.setFrequency(TARGET_JOINT_FREQUENCY);
    }
    tipTargetJoints.push(tj);

    nodes[3].touching = false;

    // install contact callbacks on the tip to track touching and transfer collision impulses
    const tip = nodes[3];
    tip.beginContact = function (other: LJS.Box2dObject) {
      if (!crabCenter || !other) {
        return;
      }
      for (const arr of limbNodes) {
        if (arr.includes(other)) {
          return;
        }
      }
      if (other === rootAnchor) {
        return;
      }
      const preVel = tip.prevVel || tip.getLinearVelocity();
      const speed = Math.hypot(preVel.x || 0, preVel.y || 0);
      if (speed > 0.05) {
        const impulseVec = preVel.copy().scale(-tip.getMass());
        crabCenter.applyAcceleration(impulseVec.copy().scale(COLLISION_TRANSFER), tip.pos);
      }
      tip.touching = true;
    };

    tip.endContact = function (other: LJS.Box2dObject) {
      for (const arr of limbNodes) {
        if (arr.includes(other)) {
          return;
        }
      }
      if (other === rootAnchor) {
        return;
      }
      tip.touching = false;
    };

    limbNodes.push(nodes);
    limbDistanceJoints.push(joints);
  }

  // NOTE: do not create a center target joint here — the crab center should
  // be moved via physics interactions from the limb collisions and joints.
  centerTargetJoint = null;
}

function drawDebugOverlay() {
  if (!debugCtx || !debugCanvas) {
    return;
  }
  const ctx = debugCtx;
  const w = debugCanvas.width;
  const h = debugCanvas.height;
  ctx.clearRect(0, 0, w, h);
  if (!lastLandmarks || !lastLandmarks.length) {
    return;
  }

  // finger chains (indices within MediaPipe 21-point model)
  const fingers = [
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
      if (!p) {
        continue;
      }
      const x = p.x * w;
      const y = p.y * h;
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.strokeStyle = 'rgba(0,200,255,0.9)';
    ctx.stroke();
  }

  // draw points (bigger for fingertips)
  for (let i = 0; i < lastLandmarks.length; i++) {
    const p = lastLandmarks[i];
    if (!p) {
      continue;
    }
    const x = p.x * w;
    const y = p.y * h;
    ctx.beginPath();
    ctx.arc(x, y, fingertipIndices.includes(i) ? 6 : 3, 0, Math.PI * 2);
    ctx.fillStyle = fingertipIndices.includes(i) ? 'rgba(255,0,0,0.95)' : 'rgba(0,255,0,0.6)';
    ctx.fill();
  }
}

export function updateCrab() {
  // smooth detected hand center (use rawHandWorld computed from landmarks)
  smoothedHandCenter = smoothedHandCenter.scale(1 - SMOOTH_ALPHA).add(rawHandWorld.scale(SMOOTH_ALPHA));

  // update smoothed fingertip targets and drive tip joints.
  // Tip targets are computed relative to the hand center and then mapped
  // onto the crab's center position so global hand translation doesn't
  // move the crab unless the fingertips "crawl" relative to the wrist.
  for (let i = 0; i < 4; i++) {
    const raw = fingerWorld[i] || vec2(20, 10);
    // how far the raw landmark actually moved since last tick
    const rawDelta = raw.subtract(prevFingerWorld[i]);
    prevFingerWorld[i] = raw.copy();
    // exponential smoothing: new = old*(1-alpha) + raw*alpha
    smoothedFingerWorld[i] = smoothedFingerWorld[i].scale(1 - SMOOTH_ALPHA).add(raw.scale(SMOOTH_ALPHA));
    const tj = tipTargetJoints[i];
    // offset of fingertip from detected hand center; apply global multiplier
    const offset = smoothedFingerWorld[i]
      .add(smoothedHandCenter.scale(-1))
      .scale(FINGER_LENGTH_MULTIPLIER)
      .add(FINGER_OFFSET);
    // map offset into game world relative to the crab center
    const target = crabCenter ? crabCenter.pos.add(offset) : smoothedFingerWorld[i];
    if (tj) {
      tj.setTarget(target);
    }
    tipWorldTargets[i] = target;
    // apply pull impulse only when the tip is grounded, scaled by raw landmark movement
    const tipObj = limbNodes[i] && limbNodes[i][3];
    if (tipObj && tipObj.touching && crabCenter) {
      const landmarkSpeed = rawDelta.length();
      if (landmarkSpeed > 0.0001) {
        const imp = computeTipPullImpulse(i);
        if (imp.length() > 0.00001) {
          crabCenter.applyAcceleration(imp.scale(-PULL_TRANSFER * landmarkSpeed), tipObj.pos);
        }
      }
    }
  }

  // Do not directly drive the crab center to the absolute hand center anymore.
  // The crab should be moved via physics interactions (fingertips crawling).
  // keep the kinematic root anchor at the crab center so limb roots follow
  if (rootAnchor && crabCenter) {
    rootAnchor.setPosition(crabCenter.pos);
  }
  // enforce a hard max on the crab center linear velocity to avoid runaway speeds
  if (crabCenter && CENTER_MAX_VELOCITY > 0) {
    const v = crabCenter.getLinearVelocity();
    const speed = Math.hypot(v.x || 0, v.y || 0);
    if (speed > CENTER_MAX_VELOCITY) {
      crabCenter.setLinearVelocity(v.scale(CENTER_MAX_VELOCITY / speed));
    }
  }

  // record current velocities to estimate impulses at the next contact
  for (const nodes of limbNodes) {
    for (const n of nodes) {
      n.prevVel = n.getLinearVelocity().copy();
    }
  }
}

export function renderCrab() {
  if (!crabCenter) {
    return;
  }

  // draw center ball
  LJS.drawCircle(crabCenter.pos, 1.8, hsl(0.6, 0.6, 0.45));

  // draw spring from hand center to crab center
  LJS.drawLine(crabCenter.pos, smoothedHandCenter, 0.12, hsl(0.1, 0.8, 0.6, 0.6));

  for (let i = 0; i < 4; i++) {
    const nodes = limbNodes[i];
    if (!nodes || nodes.length === 0) {
      continue;
    }

    // line from center to first node
    LJS.drawLine(crabCenter.pos, nodes[0].pos, 0.12, LJS.BLACK);

    for (let j = 0; j < nodes.length; j++) {
      const node = nodes[j];
      if (!node) {
        continue;
      }
      // connection to next node
      if (j < nodes.length - 1) {
        LJS.drawLine(node.pos, nodes[j + 1].pos, 0.12, LJS.BLACK);
      }
      // draw node as small circle
      LJS.drawCircle(node.pos, 0.5, hsl(i / 4, 0.8, 0.5));
    }

    // draw fingertip debug target as a thin line from tip to the computed target
    const tip = nodes[nodes.length - 1];
    if (tip) {
      LJS.drawLine(tip.pos, tipWorldTargets[i] || smoothedFingerWorld[i], 0.06, hsl(i / 4, 1, 0.6, 0.7));
    }
  }
}
