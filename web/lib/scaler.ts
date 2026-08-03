import { FEATURE_NAMES } from './features';

export interface Scaler {
  mean: number[];
  std: number[];
  feature_names: string[];
  pitch_centre: number;
  version: string;
}

export function validateScaler(s: unknown): Scaler {
  const sc = s as Scaler;
  if (!Array.isArray(sc.mean) || sc.mean.length !== 13) throw new Error('scaler.mean must be 13 floats');
  if (!Array.isArray(sc.std) || sc.std.length !== 13) throw new Error('scaler.std must be 13 floats');
  if (!Array.isArray(sc.feature_names) || sc.feature_names.length !== 13) {
    throw new Error('scaler.feature_names must be 13 strings');
  }
  sc.feature_names.forEach((n, i) => {
    if (n !== FEATURE_NAMES[i]) {
      throw new Error(`scaler feature_names[${i}]="${n}" != contract "${FEATURE_NAMES[i]}" — refusing to run`);
    }
  });
  if (typeof sc.pitch_centre !== 'number') throw new Error('scaler.pitch_centre missing');
  return sc;
}

export async function loadScaler(url = '/model/scaler.json'): Promise<Scaler> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`scaler.json fetch failed: ${res.status}`);
  return validateScaler(await res.json());
}
