import type { DemoIndex } from '../store';

export async function loadDemoIndex(): Promise<DemoIndex> {
  const res = await fetch('/demo/index.json');
  if (!res.ok) throw new Error(`Failed to load demo index: ${res.status}`);
  return res.json();
}
