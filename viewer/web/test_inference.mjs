// Verify the JS pipeline against reference.json (torch outputs from
// export_weights.py). Run: node viewer/web/test_inference.mjs

import { readFileSync } from "node:fs";
import { OthelloGPT } from "./model.js";
import { makeEncoder, topFeatures } from "./sae.js";

const dir = new URL("./weights/", import.meta.url);
const read = f => readFileSync(new URL(f, dir));
const arrayBuf = f => { const b = read(f); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };

const manifest = JSON.parse(read("manifest.json"));
const ref = JSON.parse(read("reference.json"));

let failed = false;
const check = (ok, msg) => { console.log(`${ok ? "PASS" : "FAIL"}  ${msg}`); if (!ok) failed = true; };

// 1. model forward vs torch resid_post5
const gpt = new OthelloGPT(manifest.model, arrayBuf(manifest.model.file));
const t0 = performance.now();
const resid = gpt.run(ref.tokens);
const t1 = performance.now();
let d2 = 0, n2 = 0, maxAbs = 0;
for (let i = 0; i < resid.length; i++) {
  const d = resid[i] - ref.resid_post5[i];
  d2 += d * d; n2 += ref.resid_post5[i] ** 2;
  maxAbs = Math.max(maxAbs, Math.abs(d));
}
const rel = Math.sqrt(d2 / n2);
check(rel < 0.02, `resid_post5 rel err ${rel.toExponential(2)} (max abs ${maxAbs.toExponential(2)}), ` +
  `${ref.tokens.length} tokens in ${(t1 - t0).toFixed(0)}ms`);

// 2. each SAE encoder on the torch resid (isolates encode correctness),
//    then on the JS resid (end-to-end top-k agreement)
const torchResid = Float32Array.from(ref.resid_post5);
for (const m of manifest.saes) {
  const encode = makeEncoder(m, arrayBuf(m.file), manifest.cos_eps);
  const expected = ref.saes[m.display];

  const f = encode(torchResid);
  const { l0 } = topFeatures(f, 10);
  let worst = 0;
  for (const [idx, val] of expected.top) {
    worst = Math.max(worst, Math.abs(f[idx] - val) / Math.max(Math.abs(val), 1e-6));
  }
  check(worst < 0.05 && Math.abs(l0 - expected.l0) <= 2,
    `${m.display.padEnd(12)} encode: top-10 worst rel err ${worst.toExponential(2)}, ` +
    `l0 ${l0} (torch ${expected.l0})`);

  const e2e = topFeatures(encode(resid), 10);
  const refSet = new Set(expected.top.map(([i]) => i));
  const overlap = e2e.top.filter(t => refSet.has(t.index)).length;
  check(overlap >= 8, `${m.display.padEnd(12)} end-to-end: top-10 overlap ${overlap}/10`);
}

process.exit(failed ? 1 : 0);
