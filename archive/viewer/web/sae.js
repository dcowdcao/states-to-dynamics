// SAE encoders ported from circuits/dictionary_learning and src/models — one
// closure per checkpoint kind. encode(x) takes the raw resid vector and returns
// the full feature vector (standardization applied inside, as in
// StandardizedDictionary).

import { tensors } from "./weights_io.js";

function sparsemax(z) {
  const n = z.length;
  const idx = Array.from(z.keys()).sort((a, b) => z[b] - z[a]);
  let S = 0, Sk = 0, k = 0;
  for (let i = 1; i <= n; i++) {
    const zi = z[idx[i - 1]];
    S += zi;
    if (1 + i * zi > S) { k = i; Sk = S; } else break;
  }
  const tau = (Sk - 1) / k;
  const f = new Float32Array(n);
  for (let i = 0; i < n; i++) f[i] = Math.max(z[i] - tau, 0);
  return f;
}

// out[f] = sum_i W[f, i] * v[i]  (W row-major [F, D])
function matvec(W, v, F, D, out) {
  for (let f = 0; f < F; f++) {
    let s = 0;
    const o = f * D;
    for (let i = 0; i < D; i++) s += W[o + i] * v[i];
    out[f] = s;
  }
}

export function makeEncoder(m, buffer, cosEps) {
  const w = tensors(m.tensors, buffer);
  const F = m.dict_size, D = w.std_mean.length;
  const mean = w.std_mean, scale = m.scalars.std_scale;
  const xs = new Float32Array(D), sub = new Float32Array(D);

  let aeNorm = null;
  if (m.kind === "cosine_spade") {
    aeNorm = new Float32Array(F);
    for (let f = 0; f < F; f++) {
      let s = 0;
      for (let i = 0; i < D; i++) s += w.Ae[f * D + i] ** 2;
      aeNorm[f] = Math.max(Math.sqrt(s), cosEps);
    }
  }

  return function encode(x) {
    for (let i = 0; i < D; i++) xs[i] = scale * (x[i] - mean[i]);
    const f = new Float32Array(F);

    switch (m.kind) {
      case "standard": {  // relu(W (x - bias) + b_enc)
        for (let i = 0; i < D; i++) sub[i] = xs[i] - w.bias[i];
        matvec(w["encoder.weight"], sub, F, D, f);
        for (let j = 0; j < F; j++) f[j] = Math.max(f[j] + w["encoder.bias"][j], 0);
        return f;
      }
      case "gated": {  // (x_enc + gate_bias > 0) * relu(exp(r_mag) x_enc + mag_bias)
        for (let i = 0; i < D; i++) sub[i] = xs[i] - w.decoder_bias[i];
        matvec(w["encoder.weight"], sub, F, D, f);
        for (let j = 0; j < F; j++) {
          f[j] = f[j] + w.gate_bias[j] > 0
            ? Math.max(Math.exp(w.r_mag[j]) * f[j] + w.mag_bias[j], 0) : 0;
        }
        return f;
      }
      case "topk": {  // top k of relu(W (x - b_dec) + b_enc)
        for (let i = 0; i < D; i++) sub[i] = xs[i] - w.b_dec[i];
        matvec(w["encoder.weight"], sub, F, D, f);
        for (let j = 0; j < F; j++) f[j] = Math.max(f[j] + w["encoder.bias"][j], 0);
        const top = Array.from(f.keys()).sort((a, b) => f[b] - f[a]).slice(0, m.scalars.k);
        const out = new Float32Array(F);
        for (const j of top) out[j] = f[j];
        return out;
      }
      case "batch_topk": {  // relu(W (x - b_dec) + b_enc) masked by > threshold
        for (let i = 0; i < D; i++) sub[i] = xs[i] - w.b_dec[i];
        matvec(w["encoder.weight"], sub, F, D, f);
        for (let j = 0; j < F; j++) {
          const v = Math.max(f[j] + w["encoder.bias"][j], 0);
          f[j] = v > m.scalars.threshold ? v : 0;
        }
        return f;
      }
      case "spade": {  // sparsemax(-lambda * ||x - Ae||^2)
        for (let j = 0; j < F; j++) {
          let d2 = 0;
          const o = j * D;
          for (let i = 0; i < D; i++) { const d = xs[i] - w.Ae[o + i]; d2 += d * d; }
          f[j] = -m.scalars.lambda * d2;
        }
        return sparsemax(f);
      }
      case "cosine_spade": {  // sparsemax(cos(x, Ae))
        let xn = 0;
        for (let i = 0; i < D; i++) xn += xs[i] ** 2;
        xn = Math.max(Math.sqrt(xn), cosEps);
        for (let j = 0; j < F; j++) {
          let s = 0;
          const o = j * D;
          for (let i = 0; i < D; i++) s += w.Ae[o + i] * xs[i];
          f[j] = s / (xn * aeNorm[j]);
        }
        return sparsemax(f);
      }
      default:
        throw new Error(`unknown SAE kind ${m.kind}`);
    }
  };
}

export function topFeatures(f, k) {
  const active = [];
  for (let j = 0; j < f.length; j++) if (f[j] > 0) active.push(j);
  active.sort((a, b) => f[b] - f[a]);
  return {
    l0: active.length,
    top: active.slice(0, k).map(j => ({ index: j, activation: f[j] })),
  };
}
