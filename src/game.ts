/*
    Little JS Box2d Demo
    - Demonstrates how to use Box2D with LittleJS
    - Several scenes to demonstrate Box2D features
    - Every type of shape and joint
    - Contact begin and end callbacks
    - Raycasting and querying
    - Collision filtering
    - User Interaction
*/

'use strict';

// import LittleJS module
import * as LJS from 'littlejsengine';
import * as GameObjects from './gameObjects';
import * as Scenes from './scenes';
import tilesPng from './tiles.png';
import * as HandTracking from './handTracking';
const { vec2, hsl } = LJS;

// use HD textures
LJS.setCanvasPixelated(false);
LJS.setTilesPixelated(false);

///////////////////////////////////////////////////////////////////////////////
// game variables

const sceneCount = 12;
const startScene = 3;
export let spriteAtlas,
  groundObject,
  mouseJoint,
  repeatSpawnTimer = new LJS.Timer();
const sound_click = new LJS.Sound([0.2, 0.1, , , , 0.01, , , , , , , , , , , , , , , -500]);

// Pause / overlay state (controlled by hand tracking)
let pauseReason = 'none' as HandTracking.HandStatus;
let overlayVisible = false;
let overlayMessage = '';
let overlayAlpha = 0;
let overlayShowUntil = 0;
const OVERLAY_MIN_MS = 300;

export function setGamePaused(paused: boolean, reason?: HandTracking.HandStatus) {
  pauseReason = reason || 'none';
  if (paused) {
    overlayMessage = reasonToMessage(pauseReason);
    showHandOverlay(overlayMessage);
    LJS.setPaused(true);
  } else {
    LJS.setPaused(false);
    hideHandOverlay();
  }
}

function reasonToMessage(reason?: HandTracking.HandStatus) {
  if (reason === 'bad') return 'HAND OBSCURED!!\nMove your hand more clearly in front of the camera';
  return 'CONTROL WITH YOUR HAND!!\nHold hand about 12 inches away from the camera';
}

function showHandOverlay(message: string) {
  overlayVisible = true;
  overlayMessage = message;
  overlayShowUntil = performance.now() + OVERLAY_MIN_MS;
}

function hideHandOverlay() {
  overlayVisible = false;
  overlayShowUntil = performance.now() + OVERLAY_MIN_MS;
}

function drawHandOverlay() {
  // simple fade in/out
  overlayAlpha += overlayVisible ? 0.15 : -0.15;
  overlayAlpha = Math.max(0, Math.min(1, overlayAlpha));
  if (!overlayVisible && overlayAlpha <= 0) return;

  const w = Math.min(1000, LJS.mainCanvasSize.x - 200);
  const h = 140;
  const center = vec2(LJS.mainCanvasSize.x / 2, LJS.mainCanvasSize.y / 2);

  // background box (screen-space)
  LJS.drawCanvas2D(
    center,
    vec2(w, h),
    0,
    false,
    (ctx) => {
      ctx.fillStyle = `rgba(0,0,0,${0.6 * overlayAlpha})`;
      ctx.fillRect(-0.5, -0.5, 1, 1);
    },
    true
  );

  // text (screen-space)
  const fontSize = Math.floor(Math.min(48, LJS.mainCanvasSize.y * 0.04));
  LJS.drawTextScreen(overlayMessage, center, fontSize, LJS.rgb(1, 1, 1, overlayAlpha), fontSize * 0.08, LJS.BLACK);
}

///////////////////////////////////////////////////////////////////////////////
function setScene(scene) {
  // setup
  LJS.setCameraPos(vec2(20, 10));
  LJS.setGravity(vec2(0, -20));

  // destroy old scene
  LJS.engineObjectsDestroy();
  mouseJoint = 0;

  // create walls
  groundObject = GameObjects.spawnBox(vec2(0, -4), vec2(1e3, 8), hsl(0, 0, 0.2), LJS.box2d.bodyTypeStatic);
  GameObjects.spawnBox(vec2(-4, 0), vec2(8, 1e3), LJS.BLACK, LJS.box2d.bodyTypeStatic);
  GameObjects.spawnBox(vec2(44, 0), vec2(8, 1e3), LJS.BLACK, LJS.box2d.bodyTypeStatic);
  GameObjects.spawnBox(vec2(0, 100), vec2(1e3, 8), LJS.BLACK, LJS.box2d.bodyTypeStatic);

  // load the scene
  Scenes.loadScene(scene);
  // initialize crab overlay for this scene
  HandTracking.initCrab(groundObject);
}

///////////////////////////////////////////////////////////////////////////////
async function gameInit() {
  // start up LittleJS Box2D plugin
  await LJS.box2dInit();
  // start hand tracking (hidden video + model). Fire-and-forget.
  HandTracking.initHandTracking();
  // subscribe to hand status updates to control pause/overlay
  HandTracking.onHandStatusChange((s) => {
    if (s !== 'ok') {
      setGamePaused(true, s);
    } else {
      setGamePaused(false);
    }
  });
  //LJS.box2dSetDebug(true); // enable box2d debug draw

  // create a table of all sprites
  const gameTile = (i) => LJS.tile(i, 124, 0, 2);
  spriteAtlas = {
    circle: gameTile(0),
    dot: gameTile(1),
    circleOutline: gameTile(2),
    squareOutline: gameTile(3),
    wheel: gameTile(4),
    gear: gameTile(5),
    squareOutline2: gameTile(6),
  };

  // startup the demo
  LJS.setCanvasClearColor(hsl(0, 0, 0.8));
  setScene(startScene);
}

///////////////////////////////////////////////////////////////////////////////
function gameUpdate() {
  // scale canvas to fit based on 1080p
  LJS.setCameraScale((LJS.mainCanvasSize.y * 48) / 1080);

  // mouse controls
  if (mouseJoint) {
    // update mouse joint
    mouseJoint.setTarget(LJS.mousePos);
    if (LJS.mouseWasReleased(0)) {
      // release object
      sound_click.play(LJS.mousePos, 1, 0.5);
      mouseJoint.destroy();
      mouseJoint = 0;
    }
  } else if (LJS.mouseWasPressed(0)) {
    // grab object
    sound_click.play(LJS.mousePos);
    const object = LJS.box2d.pointCast(LJS.mousePos);
    if (object) mouseJoint = new LJS.Box2dTargetJoint(object, groundObject, LJS.mousePos);
  }

  // controls
  if (LJS.mouseIsDown(1) || LJS.keyIsDown('KeyZ')) {
    const isSet = repeatSpawnTimer.isSet();
    if (!isSet || repeatSpawnTimer.elapsed()) {
      // spawn continuously after a delay
      isSet || repeatSpawnTimer.set(0.5);
      GameObjects.spawnRandomObject(LJS.mousePos);
    }
  } else repeatSpawnTimer.unset();
  if (LJS.mouseWasPressed(2) || LJS.keyWasPressed('KeyX')) GameObjects.explosion(LJS.mousePos);

  if (LJS.keyWasPressed('KeyR')) setScene(Scenes.scene); // reset scene
  if (LJS.keyWasPressed('ArrowUp') || LJS.keyWasPressed('ArrowDown')) {
    // change scene
    const upPressed = LJS.keyWasPressed('ArrowUp');
    setScene(LJS.mod(Scenes.scene + (upPressed ? 1 : -1), sceneCount));
  }
  // update crab overlay (finger-driven targets)
  HandTracking.updateCrab();
}

///////////////////////////////////////////////////////////////////////////////
function gameUpdatePost() {}

///////////////////////////////////////////////////////////////////////////////
function gameRender() {
  if (Scenes.scene == 5) {
    // raycast test
    const count = 100;
    const distance = 10;
    for (let i = count; i--; ) {
      const start = LJS.mousePos;
      const end = start.add(vec2(distance, 0).rotate((i / count) * LJS.PI * 2));
      const result = LJS.box2d.raycast(start, end);
      const color = result ? hsl(0, 1, 0.5, 0.5) : hsl(0.5, 1, 0.5, 0.5);
      LJS.drawLine(start, result ? result.point : end, 0.1, color);
    }
  }
  // render crab overlay (lines + debug)
  HandTracking.renderCrab();
}

///////////////////////////////////////////////////////////////////////////////
function gameRenderPost() {
  if (mouseJoint) {
    // draw mouse joint
    const ab = mouseJoint.getAnchorB();
    LJS.drawTile(ab, vec2(0.3), spriteAtlas.circle, LJS.BLACK);
    LJS.drawLine(LJS.mousePos, ab, 0.1, LJS.BLACK);
  }

  // // draw demo info
  // const pos = vec2(LJS.mainCanvasSize.x / 2, 50);
  // drawText('LittleJS Box2D Demo', 65, 70);
  // drawText(Scenes.sceneName, 50, 100);
  // if (Scenes.scene == 0) {
  //   drawText('Mouse Left = Grab');
  //   drawText('Mouse Middle or Z = Spawn');
  //   drawText('Mouse Right or X = Explode');
  //   drawText('Arrows Up/Down = Change Scene');
  // }
  // if (Scenes.scene == 3) {
  //   drawText('Right = Accelerate');
  //   drawText('Left = Reverse');
  // }

  function drawText(text, size = 40, gap = 50) {
    LJS.drawTextScreen(text, pos, size, LJS.WHITE, size * 0.1);
    pos.y += gap;
  }
  // render hand overlay (if any)
  drawHandOverlay();
}

///////////////////////////////////////////////////////////////////////////////
// Startup LittleJS Engine with Box2D

LJS.engineInit(gameInit, gameUpdate, gameUpdatePost, gameRender, gameRenderPost, [tilesPng]);
