/** @type {import('next').NextConfig} */
const nextConfig = {
  // onnxruntime-web and tasks-vision are browser-only; nothing special is
  // required as long as they are only imported inside client components
  // (dynamic import in src/engine.js).
  reactStrictMode: true,
};

export default nextConfig;
