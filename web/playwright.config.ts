import { defineConfig, devices } from '@playwright/test';

// J1 parity gate only (web/tests/e2e/**). Kept separate from vitest's
// "test" script — different runner, needs a real browser + a running app.
// PORT 3100: distinct from `next dev`'s default 3000 and from the Python
// Playwright drivers' ports (collect_benchmark.py 3124, e2e_app_test.py
// 3123) so none of these can collide if run back to back.
const PORT = 3100;

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 300_000, // 100 MediaPipe WASM detections per run; slow on CI CPUs
  fullyParallel: false,
  retries: 0,
  reporter: [['list']],
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    trace: 'retain-on-failure',
  },
  webServer: {
    command: `npm run dev -- -p ${PORT}`,
    url: `http://127.0.0.1:${PORT}`,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
  ],
});
