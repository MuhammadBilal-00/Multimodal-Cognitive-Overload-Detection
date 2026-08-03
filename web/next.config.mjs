/** @type {import('next').NextConfig} */
const nextConfig = {
  // Without COOP/COEP the page is not cross-origin isolated and ort-web WASM
  // silently drops to a single thread (brief section B1).
  //
  // connect-src 'self': enforces the privacy claim at the browser policy
  // level rather than trusting every dependency's behavior. Verified
  // necessary, not theoretical — @mediapipe/tasks-vision has an
  // undocumented, non-configurable usage-telemetry call baked into its
  // bundle (POST to odml.pa.googleapis.com every ~60s; see docs/privacy.md)
  // with no opt-out in its public BaseOptions API. This header makes that
  // request (and any other cross-origin request, from any current or future
  // dependency) fail closed instead of silently succeeding.
  headers: async () => [
    {
      source: '/(.*)',
      headers: [
        { key: 'Cross-Origin-Opener-Policy', value: 'same-origin' },
        { key: 'Cross-Origin-Embedder-Policy', value: 'require-corp' },
        { key: 'Content-Security-Policy', value: "connect-src 'self'" },
      ],
    },
  ],
};

export default nextConfig;
