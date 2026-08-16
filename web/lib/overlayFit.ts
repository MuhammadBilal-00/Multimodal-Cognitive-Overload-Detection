// Maps MediaPipe's normalized (0..1) landmark coordinates onto an overlay
// canvas that sits on top of a `object-cover` <video>.
//
// The overlay used to assume the stream was exactly 4:3 (a hardcoded 640x480
// canvas stretched over the container). WebcamFeed asks for 640x480, but
// getUserMedia treats width/height as *ideal*, not exact — plenty of laptop
// webcams hand back 16:9 instead. `object-cover` then crops the video to fill
// the 4:3 box, while the landmarks kept mapping to the full box, so the dots
// drifted off the face horizontally.
//
// This replicates object-cover: scale to cover, centre, and let the overflow
// fall outside the box — exactly what the browser does to the video pixels.
//
// Feature extraction is unaffected and always was: computeFeatures() is fed
// video.videoWidth/videoHeight directly (hooks/usePipeline.ts).
export interface OverlayFit {
  /** normalized x (0..1 of the video frame) -> canvas px */
  x: (nx: number) => number;
  /** normalized y (0..1 of the video frame) -> canvas px */
  y: (ny: number) => number;
}

export function coverFit(
  videoW: number, videoH: number, canvasW: number, canvasH: number,
  mirrored: boolean,
): OverlayFit {
  // Degenerate before the first frame arrives: fall back to a plain stretch
  // rather than dividing by zero.
  if (!videoW || !videoH) {
    return {
      x: (nx) => (mirrored ? 1 - nx : nx) * canvasW,
      y: (ny) => ny * canvasH,
    };
  }
  const scale = Math.max(canvasW / videoW, canvasH / videoH);
  const drawW = videoW * scale;
  const drawH = videoH * scale;
  const offsetX = (canvasW - drawW) / 2; // negative when cropped horizontally
  const offsetY = (canvasH - drawH) / 2;
  return {
    x: (nx) => offsetX + (mirrored ? 1 - nx : nx) * drawW,
    y: (ny) => offsetY + ny * drawH,
  };
}
