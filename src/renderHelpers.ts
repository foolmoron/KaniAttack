'use strict';

import * as LJS from 'littlejsengine';
const { vec2 } = LJS;

// Rendering helpers for oriented rounded rectangles, triangles, and googly eyes
export function drawRoundedRect(center, size, angle, radius, fill, stroke, strokeWidth = 0.06) {
  if (!size || !size.x || !size.y) return;
  const w = Math.abs(size.x);
  const h = Math.abs(size.y);
  LJS.drawCanvas2D(
    center,
    size,
    angle || 0,
    false,
    (ctx) => {
      const rx = Math.min(0.5, Math.abs(radius / w));
      const ry = Math.min(0.5, Math.abs(radius / h));
      ctx.beginPath();
      ctx.moveTo(-0.5 + rx, -0.5);
      ctx.lineTo(0.5 - rx, -0.5);
      ctx.quadraticCurveTo(0.5, -0.5, 0.5, -0.5 + ry);
      ctx.lineTo(0.5, 0.5 - ry);
      ctx.quadraticCurveTo(0.5, 0.5, 0.5 - rx, 0.5);
      ctx.lineTo(-0.5 + rx, 0.5);
      ctx.quadraticCurveTo(-0.5, 0.5, -0.5, 0.5 - ry);
      ctx.lineTo(-0.5, -0.5 + ry);
      ctx.quadraticCurveTo(-0.5, -0.5, -0.5 + rx, -0.5);
      ctx.closePath();
      if (fill) {
        ctx.fillStyle = fill;
        ctx.fill();
      }
      if (stroke && strokeWidth > 0) {
        ctx.lineWidth = strokeWidth;
        ctx.strokeStyle = stroke;
        ctx.stroke();
      }
    },
    false
  );
}

export function drawSegmentRoundedRect(a, b, thickness, radius, fill, stroke, strokeWidth = 0.06) {
  if (!a || !b) return;
  const mid = a.add(b).scale(0.5);
  const length = a.distance(b);
  if (length <= 0) return;
  const angle = -Math.atan2(b.y - a.y, b.x - a.x);
  const size = vec2(length, thickness);
  drawRoundedRect(mid, size, angle, radius, fill, stroke, strokeWidth);
}

export function drawIsoscelesTriangle(tip, baseWidth, height, angle, fill, stroke, strokeWidth = 0.06) {
  if (!tip) return;
  const dx = Math.cos(angle);
  const dy = Math.sin(angle);
  const baseCenter = tip.add(vec2(-dx * height, -dy * height));
  const half = baseWidth / 2;
  const left = baseCenter.add(vec2(-dy * half, dx * half));
  const right = baseCenter.add(vec2(dy * half, -dx * half));
  LJS.drawPoly([tip, right, left], fill, strokeWidth, stroke);
}

export function drawGooglyEyes(center, size, lookAt, eyeOffsetX = 0.22, eyeOffsetY = 0.12) {
  if (!center || !size) return;
  const eyeRadius = Math.min(size.x, size.y) * 0.16;
  const pupilRadius = eyeRadius * 0.5;
  const maxPupilOffset = Math.max(0, eyeRadius - pupilRadius - 0.02);

  const leftEye = center.add(vec2(-size.x * eyeOffsetX, size.y * eyeOffsetY));
  const rightEye = center.add(vec2(size.x * eyeOffsetX, size.y * eyeOffsetY));

  const dir = lookAt ? lookAt.subtract(center) : vec2(0, 0);
  const ndir = dir.length() ? dir.normalize() : vec2(0, 0);
  const pupilOffset = ndir.scale(maxPupilOffset);

  const leftPupil = leftEye.add(pupilOffset);
  const rightPupil = rightEye.add(pupilOffset);

  // eye whites
  LJS.drawCircle(leftEye, eyeRadius, LJS.WHITE, 0.02, LJS.BLACK);
  LJS.drawCircle(rightEye, eyeRadius, LJS.WHITE, 0.02, LJS.BLACK);
  // pupils
  LJS.drawCircle(leftPupil, pupilRadius, LJS.BLACK);
  LJS.drawCircle(rightPupil, pupilRadius, LJS.BLACK);
}
