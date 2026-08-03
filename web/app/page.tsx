'use client';
import dynamic from 'next/dynamic';

// onnxruntime-web and @mediapipe/tasks-vision are browser-only (WASM, GL,
// getUserMedia); ssr:false keeps their module graph out of the server
// bundle entirely — onnxruntime-web's package resolution otherwise pulls
// in a Node-targeted entry point when the server tries to import it for
// SSR, which fails production minification (not a browser-vs-WASM issue,
// a server-bundling-a-browser-lib issue).
const CognitiveApp = dynamic(() => import('../components/CognitiveApp'), { ssr: false });

export default function Home() {
  return <CognitiveApp />;
}
