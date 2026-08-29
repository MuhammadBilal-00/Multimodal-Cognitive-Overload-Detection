// Test-only static file server for the J1 parity gate (__parity-test__
// route). Serves ml/tests/fixtures/* — outside web/public — because those
// fixtures (parity_expected.json, parity_frames/*.png) are DAiSEE-derived
// and must stay out of the committed/public asset tree (see .gitignore).
// A Node.js route handler can reach them via the filesystem; a static
// public/ file cannot, and Next never exposes node_modules-style absolute
// paths outside the project root.
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { NextResponse } from 'next/server';

const FIXTURES_ROOT = path.resolve(process.cwd(), '..', 'ml', 'tests', 'fixtures');

const CONTENT_TYPES: Record<string, string> = {
  '.png': 'image/png',
  '.json': 'application/json',
};

export async function GET(
  _req: Request,
  { params }: { params: { path: string[] } },
) {
  // Test-only: a fixture server for licence-restricted DAiSEE-derived data
  // has no business existing in a deployed instance. The J1 gate runs
  // against `next dev` (playwright.config.ts webServer), so disabling this
  // outside development costs nothing and closes the exposure.
  if (process.env.NODE_ENV === 'production') {
    return new NextResponse('not available in production', { status: 404 });
  }
  const filePath = path.resolve(FIXTURES_ROOT, path.join(...params.path));
  // Path-traversal guard: resolved path must stay inside FIXTURES_ROOT.
  if (filePath !== FIXTURES_ROOT && !filePath.startsWith(FIXTURES_ROOT + path.sep)) {
    return new NextResponse('forbidden', { status: 403 });
  }
  const contentType = CONTENT_TYPES[path.extname(filePath)];
  if (!contentType) return new NextResponse('unsupported file type', { status: 400 });
  try {
    const data = await readFile(filePath);
    return new NextResponse(data, { headers: { 'Content-Type': contentType } });
  } catch {
    return new NextResponse('not found', { status: 404 });
  }
}
