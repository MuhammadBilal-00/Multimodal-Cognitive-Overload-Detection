import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['tests/**/*.test.ts'],
    exclude: ['tests/e2e/**'], // Playwright Test, not vitest — run via `npm run test:parity`
  },
});
