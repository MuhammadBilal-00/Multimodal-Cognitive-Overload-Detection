/** @type {import('next').NextConfig} */
const nextConfig = {
  // Without COOP/COEP the page is not cross-origin isolated and ort-web WASM
  // silently drops to a single thread (brief section B1).
  headers: async () => [
    {
      source: '/(.*)',
      headers: [
        { key: 'Cross-Origin-Opener-Policy', value: 'same-origin' },
        { key: 'Cross-Origin-Embedder-Policy', value: 'require-corp' },
      ],
    },
  ],
};

export default nextConfig;
