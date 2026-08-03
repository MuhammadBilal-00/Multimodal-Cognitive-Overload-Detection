'use client';
import { useEffect, useRef, useState } from 'react';

type CamState = 'starting' | 'active' | 'denied' | 'nocamera' | 'busy' | 'unsupported' | 'error';

const MESSAGES: Record<Exclude<CamState, 'active'>, string> = {
  starting: 'Starting camera…',
  denied: 'Camera permission denied. Allow camera access in the browser address bar, then reload.',
  nocamera: 'No camera found on this device.',
  busy: 'Camera is in use by another application. Close it and reload.',
  unsupported: 'Camera access isn’t supported here — this page needs HTTPS (or localhost) and a device with a camera.',
  error: 'Could not start the camera.',
};

export default function WebcamFeed({
  onVideoReady,
  mirrored = true,
}: {
  onVideoReady: (v: HTMLVideoElement) => void;
  mirrored?: boolean;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [state, setState] = useState<CamState>('starting');

  useEffect(() => {
    let stream: MediaStream | null = null;
    let cancelled = false;
    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, frameRate: 30 },
          audio: false,
        });
        if (cancelled) { stream.getTracks().forEach((t) => t.stop()); return; }
        const v = videoRef.current!;
        v.srcObject = stream;
        await v.play();
        setState('active');
        onVideoReady(v);
      } catch (e) {
        const err = e as DOMException;
        if (err.name === 'NotAllowedError') setState('denied');
        else if (err.name === 'NotFoundError') setState('nocamera');
        else if (err.name === 'NotReadableError') setState('busy');
        else if (err.name === 'NotSupportedError') setState('unsupported');
        else setState('error');
      }
    })();
    return () => {
      cancelled = true;
      stream?.getTracks().forEach((t) => t.stop()); // camera light must turn off
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="relative w-full aspect-[4/3] bg-black rounded-xl overflow-hidden">
      {/* Raw frame goes to the landmarker; mirroring is CSS-only (contract 4.2) */}
      <video
        ref={videoRef}
        playsInline
        muted
        className="h-full w-full object-cover"
        style={mirrored ? { transform: 'scaleX(-1)' } : undefined}
      />
      {state !== 'active' && (
        <div className="absolute inset-0 grid place-items-center bg-zinc-950/90 p-6 text-center text-zinc-300">
          {MESSAGES[state]}
        </div>
      )}
    </div>
  );
}
