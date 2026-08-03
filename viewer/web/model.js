// OthelloGPT forward pass (layers 0..5, LNPre, exact gelu), mirroring
// transformer_lens's processed weights. KV-cached per position with prefix
// reuse, so placing one piece computes one token (~40 MFLOP).

import { tensors } from "./weights_io.js";

function erf(x) {
  const s = Math.sign(x);
  x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                  - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}

const gelu = x => 0.5 * x * (1 + erf(x / Math.SQRT2));

function lnPre(x, out, eps) {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let sq = 0;
  for (let i = 0; i < n; i++) { const c = x[i] - mean; out[i] = c; sq += c * c; }
  const scale = Math.sqrt(sq / n + eps);
  for (let i = 0; i < n; i++) out[i] /= scale;
}

export class OthelloGPT {
  constructor(manifest, buffer) {
    this.cfg = manifest.cfg;
    this.w = tensors(manifest.tensors, buffer);
    const { n_layers, n_ctx, n_heads, d_head, d_model } = this.cfg;
    this.kCache = Array.from({ length: n_layers }, () => new Float32Array(n_ctx * n_heads * d_head));
    this.vCache = Array.from({ length: n_layers }, () => new Float32Array(n_ctx * n_heads * d_head));
    this.resids = new Float32Array(n_ctx * d_model);  // resid_post of the last layer
    this.cachedTokens = [];
  }

  // tokens: int array (1..60), length <= n_ctx. Returns the final layer's
  // resid_post at the last position (view into the cache — copy if kept).
  run(tokens) {
    const D = this.cfg.d_model;
    let p = 0;
    while (p < this.cachedTokens.length && p < tokens.length && this.cachedTokens[p] === tokens[p]) p++;
    this.cachedTokens.length = p;
    for (let pos = p; pos < tokens.length; pos++) this._token(tokens[pos], pos);
    return this.resids.subarray((tokens.length - 1) * D, tokens.length * D);
  }

  _token(token, pos) {
    const { d_model: D, n_heads: H, d_head: dh, d_mlp: M, n_layers: L, eps } = this.cfg;
    const w = this.w;
    const x = new Float32Array(D), h = new Float32Array(D);
    const q = new Float32Array(H * dh), z = new Float32Array(H * dh);
    const scores = new Float32Array(pos + 1), mlp = new Float32Array(M);

    const WE = w["embed.W_E"], WP = w["pos_embed.W_pos"];
    for (let i = 0; i < D; i++) x[i] = WE[token * D + i] + WP[pos * D + i];

    for (let l = 0; l < L; l++) {
      const WQ = w[`blocks.${l}.attn.W_Q`], bQ = w[`blocks.${l}.attn.b_Q`];
      const WK = w[`blocks.${l}.attn.W_K`], bK = w[`blocks.${l}.attn.b_K`];
      const WV = w[`blocks.${l}.attn.W_V`], bV = w[`blocks.${l}.attn.b_V`];
      const WO = w[`blocks.${l}.attn.W_O`], bO = w[`blocks.${l}.attn.b_O`];
      const kc = this.kCache[l], vc = this.vCache[l];

      lnPre(x, h, eps);
      for (let hd = 0; hd < H; hd++) {
        const qo = hd * dh, ko = (pos * H + hd) * dh, base = hd * D * dh;
        for (let j = 0; j < dh; j++) { q[qo + j] = bQ[qo + j]; kc[ko + j] = bK[qo + j]; vc[ko + j] = bV[qo + j]; }
        for (let i = 0; i < D; i++) {
          const hi = h[i], o = base + i * dh;
          for (let j = 0; j < dh; j++) {
            q[qo + j] += hi * WQ[o + j];
            kc[ko + j] += hi * WK[o + j];
            vc[ko + j] += hi * WV[o + j];
          }
        }
      }
      for (let hd = 0; hd < H; hd++) {
        const qo = hd * dh;
        let maxs = -Infinity;
        for (let t = 0; t <= pos; t++) {
          let s = 0;
          const ko = (t * H + hd) * dh;
          for (let j = 0; j < dh; j++) s += q[qo + j] * kc[ko + j];
          s /= Math.sqrt(dh);
          scores[t] = s;
          if (s > maxs) maxs = s;
        }
        let sum = 0;
        for (let t = 0; t <= pos; t++) { scores[t] = Math.exp(scores[t] - maxs); sum += scores[t]; }
        for (let j = 0; j < dh; j++) {
          let a = 0;
          for (let t = 0; t <= pos; t++) a += scores[t] * vc[(t * H + hd) * dh + j];
          z[qo + j] = a / sum;
        }
      }
      for (let i = 0; i < D; i++) x[i] += bO[i];
      for (let c = 0; c < H * dh; c++) {
        const zv = z[c], o = c * D;  // W_O[hd, j, :] rows are contiguous in hd*dh order
        for (let i = 0; i < D; i++) x[i] += zv * WO[o + i];
      }

      const Wi = w[`blocks.${l}.mlp.W_in`], bi = w[`blocks.${l}.mlp.b_in`];
      const Wo = w[`blocks.${l}.mlp.W_out`], bo = w[`blocks.${l}.mlp.b_out`];
      lnPre(x, h, eps);
      mlp.fill(0);
      for (let i = 0; i < D; i++) {
        const hi = h[i], o = i * M;
        for (let m = 0; m < M; m++) mlp[m] += hi * Wi[o + m];
      }
      for (let i = 0; i < D; i++) x[i] += bo[i];
      for (let m = 0; m < M; m++) {
        const v = gelu(mlp[m] + bi[m]), o = m * D;
        for (let i = 0; i < D; i++) x[i] += v * Wo[o + i];
      }
    }
    this.resids.set(x, pos * D);
    this.cachedTokens.push(token);
  }
}
