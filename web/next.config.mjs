/** @type {import('next').NextConfig} */
const nextConfig = {
  // Without COOP/COEP the page is not cross-origin isolated and ort-web WASM
  // silently drops to a single thread (brief section B1).
  //
  // The CSP started life as the single directive `connect-src 'self'` —
  // enforcing the privacy claim at the browser policy level rather than
  // trusting every dependency's behavior. Verified necessary, not
  // theoretical: @mediapipe/tasks-vision has an undocumented,
  // non-configurable usage-telemetry call baked into its bundle (POST to
  // odml.pa.googleapis.com every ~60s; see docs/privacy.md) with no opt-out
  // in its public BaseOptions API.
  //
  // A 2026-08-29 audit pointed out the honest limits of connect-src alone:
  // it governs fetch/XHR/WebSocket/sendBeacon but NOT <img> pixels,
  // <script>/<iframe> loads, form posts, or WebRTC — so "no code path can
  // reach any host" was an overclaim. The policy below closes those routes:
  // default-src 'self' covers every unlisted fetch surface;
  // script-src needs 'unsafe-inline' (Next.js app-router inline flight
  // data/hydration scripts) and 'wasm-unsafe-eval' (Chromium requires it to
  // compile the MediaPipe + onnxruntime WASM under any CSP);
  // style-src 'unsafe-inline' for Next's inlined critical CSS;
  // worker-src blob: for ort-web's thread-pool workers;
  // img-src data:/blob: for canvas-derived images. frame-ancestors 'none'
  // (anti-embedding), object-src 'none', base-uri/form-action 'self'
  // complete the lockdown. Verified live: the J2 e2e run reaches `live`
  // with real predictions under exactly this policy.
  headers: async () => [
    {
      source: '/(.*)',
      headers: [
        { key: 'Cross-Origin-Opener-Policy', value: 'same-origin' },
        { key: 'Cross-Origin-Embedder-Policy', value: 'require-corp' },
        {
          key: 'Content-Security-Policy',
          value: [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval'",
            "style-src 'self' 'unsafe-inline'",
            "img-src 'self' data: blob:",
            "media-src 'self' blob:",
            "connect-src 'self'",
            "worker-src 'self' blob:",
            "object-src 'none'",
            "base-uri 'self'",
            "form-action 'self'",
            "frame-ancestors 'none'",
          ].join('; '),
        },
        // The app's entire purpose is camera analysis on THIS origin only —
        // deny every other powerful feature, and deny camera to any
        // embedded context (belt-and-braces with frame-ancestors 'none').
        { key: 'Permissions-Policy', value: 'camera=(self), microphone=(), geolocation=()' },
        { key: 'Referrer-Policy', value: 'no-referrer' },
        { key: 'X-Content-Type-Options', value: 'nosniff' },
      ],
    },
  ],
};

export default nextConfig;
